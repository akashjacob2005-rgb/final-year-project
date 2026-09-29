"""Train the cognitive-decline model on words AND speech behaviour.

    python ml/train_audio_model.py

What is new compared to ml/train_language_model.py, which this does not replace:

1. TIMING FEATURES ARE LEARNED, NOT DESCRIBED.
   speech_rate_wpm, pause_count, long_pause_count, pause_ratio and friends were
   already being computed at inference and then discarded, because the text-only
   HuggingFace mirror had no audio to fit a weight against. With real Pitt audio
   they become predictors.

2. THE TEXT IS ASR OUTPUT, NOT CLINICIAN TRANSCRIPTS.
   The deployed model is trained on typed CHAT transcripts and then fed Whisper
   output in production. That mismatch is the top caveat in metrics.json. Here
   training and serving see the same kind of text.

3. PAUSES ARE MARKED WHERE THEY HAPPENED, NOT JUST COUNTED.
   pause_ratio cannot tell "the ... uh ... thing" apart from one long pause to
   gather a thought. speech.pause_encoded_transcript splices shortpause /
   longpause into the word sequence, so with ngram_range=(1, 2) the model gets
   `longpause cookie` as a feature and learns hesitation-before-this-word.
   Only the TF-IDF block sees that column; the handcrafted linguistic features
   stay on the clean transcript so the markers cannot skew them.

4. CROSS-VALIDATION IS GROUPED BY PARTICIPANT.
   Pitt is longitudinal: 002-0, 002-1 and 002-2 are one person at yearly visits.
   RepeatedStratifiedKFold would put the same voice in train and test, and with
   dementia participants attending more often the leak is class-correlated, so it
   inflates the score in the flattering direction. StratifiedGroupKFold fixes it.
   The old ungrouped number is computed too and reported as `leakage_demonstration`
   — the gap between them is a result worth showing, not an embarrassment.

EXPECT A LOWER AUC THAN THE CURRENT 0.905. Real ASR error, and a leak that was
never being measured, both push it down. That is the honest number.

The artifact is written to a NEW filename (audio_model.joblib), so the existing
language_model.joblib and everything that loads it are untouched.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from features import FEATURE_NAMES, extract_batch  # noqa: E402
from train_language_model import (  # noqa: E402  — reused, not reimplemented
    SEED,
    TFIDF_KWARGS,
    make_estimator,
)

DATASET = HERE / "data" / "pitt" / "dataset.csv"
FEATURE_CACHE = HERE / "data" / "pitt" / "features_asr.csv"
ARTIFACTS = HERE / "artifacts"
MODEL_PATH = ARTIFACTS / "audio_model.joblib"
METRICS_PATH = ARTIFACTS / "metrics_audio.json"

# Must match backend/app/ml/speech.py::_timing_features, in that order.
TIMING_FEATURE_NAMES = [
    "duration_seconds",
    "word_count",
    "speech_rate_wpm",
    "pause_count",
    "long_pause_count",
    "total_pause_seconds",
    "mean_pause_seconds",
    "pause_ratio",
    "articulation_rate_wpm",
]

MIN_WORDS = 25  # same floor the inference path enforces

TEXT_COL = "transcript"
PAUSE_COL = "transcript_pauses"

# (display name, feature mode, which text column feeds the vectorizer).
#
# Only the TF-IDF block ever sees the pause-marked column. The handcrafted
# features stay on the clean transcript in every arm: `shortpause` / `longpause`
# are not words the speaker said, so letting spaCy parse them would inflate
# total_words, register as nouns, and quietly skew every ratio built on top —
# type_token_ratio, pronoun_noun_ratio, idea_density, all of them. The markers
# are a signal for the n-gram model, not for the linguistic measures.
ARMS = [
    ("text only (linguistic + tfidf)", "text", TEXT_COL),
    ("timing only (how they spoke)", "timing", TEXT_COL),
    ("text + timing", "text+timing", TEXT_COL),
    ("pause-marked text", "text", PAUSE_COL),
    ("pause-marked text + timing", "text+timing", PAUSE_COL),
]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_data():
    if not DATASET.exists():
        raise SystemExit(
            "No dataset. Run:\n"
            "    python ml/download_pitt.py\n"
            "    python ml/build_dataset.py"
        )

    df = pd.read_csv(DATASET)
    df["transcript"] = df["transcript"].fillna("").astype(str)

    before = len(df)
    df = df[df["transcript"].str.split().str.len() >= MIN_WORDS].reset_index(drop=True)
    if before != len(df):
        print(f"Dropped {before - len(df)} sessions under {MIN_WORDS} words")

    texts = df[TEXT_COL].tolist()
    y = df["label"].to_numpy()
    groups = df["participant_id"].astype(str).to_numpy()
    timing = df[TIMING_FEATURE_NAMES].to_numpy(dtype=float)

    # Datasets built before the pause encoder existed have no marked column.
    # Degrade to the clean text so the pause arms still run — they simply come
    # out identical to their unmarked twins, which is an obvious tell in the
    # results table rather than a crash.
    if PAUSE_COL in df.columns:
        texts_pauses = df[PAUSE_COL].fillna("").astype(str).tolist()
        texts_pauses = [p or t for p, t in zip(texts_pauses, texts)]
    else:
        print(f"WARNING: {DATASET.name} has no '{PAUSE_COL}' column; "
              "pause arms will duplicate the unmarked ones. Rebuild with "
              "ml/build_dataset.py --force to fix.")
        texts_pauses = list(texts)
    by_col = {TEXT_COL: texts, PAUSE_COL: texts_pauses}

    if FEATURE_CACHE.exists():
        hand = pd.read_csv(FEATURE_CACHE)
        if len(hand) == len(texts) and list(hand.columns) == FEATURE_NAMES:
            print(f"Using cached features: {FEATURE_CACHE}")
            return by_col, y, groups, hand, timing, df
        print("Feature cache stale, recomputing")

    # Always from the clean column — see the ARMS comment.
    print(f"Extracting linguistic features from {len(texts)} transcripts...")
    t0 = time.time()
    hand = extract_batch(texts)
    hand.to_csv(FEATURE_CACHE, index=False)
    print(f"  done in {time.time() - t0:.1f}s")
    return by_col, y, groups, hand, timing, df


def build_matrix(mode, texts, hand, timing, vec=None, hs=None, ts=None, fit=False):
    """Assemble the design matrix. Mirrors train_language_model.build_matrix,
    extended with a timing block."""
    parts = []
    if "text" in mode:
        H = np.nan_to_num(hand.to_numpy(dtype=float), posinf=0.0, neginf=0.0)
        if fit:
            hs = StandardScaler().fit(H)
        parts.append(hs.transform(H))
        if fit:
            vec = TfidfVectorizer(**TFIDF_KWARGS).fit(texts)
        parts.append(vec.transform(texts).toarray())
    if "timing" in mode:
        T = np.nan_to_num(np.asarray(timing, dtype=float), posinf=0.0, neginf=0.0)
        if fit:
            ts = StandardScaler().fit(T)
        parts.append(ts.transform(T))
    return np.hstack(parts), vec, hs, ts


def grouped_splits(y, groups, n_splits=5, n_repeats=4, seed=SEED):
    """Repeated StratifiedGroupKFold.

    sklearn has no RepeatedStratifiedGroupKFold, so repeat over seeds by hand —
    same variance-estimation rationale as the 5x4 scheme in the original script.
    """
    X = np.zeros(len(y))
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        yield from cv.split(X, y, groups)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate(name, mode, texts, y, groups, hand, timing, splits, text_column=TEXT_COL):
    """-> (summary, out-of-fold probabilities).

    The OOF vector is averaged over the repeats, matching
    train_language_model.evaluate. It is the only honest basis for a ROC curve or
    a threshold choice: the deployed model is refit on every sample, so scoring
    training data with it gives an optimistic probability.
    """
    texts_arr = np.array(texts, dtype=object)
    rows = []
    oof_prob = np.zeros(len(y))
    oof_count = np.zeros(len(y))

    for tr, te in splits:
        Xtr, vec, hs, ts = build_matrix(
            mode, texts_arr[tr].tolist(), hand.iloc[tr], timing[tr], fit=True
        )
        Xte, *_ = build_matrix(
            mode, texts_arr[te].tolist(), hand.iloc[te], timing[te], vec=vec, hs=hs, ts=ts
        )
        clf = make_estimator("logreg").fit(Xtr, y[tr])
        prob = clf.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)

        # Accumulated before the single-class guard below: the prediction is valid
        # even when the fold cannot yield an AUC, and discarding it would leave
        # holes in the OOF vector.
        oof_prob[te] += prob
        oof_count[te] += 1

        # A fold can be single-class once grouping constrains the split.
        if len(np.unique(y[te])) < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "roc_auc": roc_auc_score(y[te], prob),
                "accuracy": accuracy_score(y[te], pred),
                "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
                "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
                "f1": f1_score(y[te], pred, zero_division=0),
            }
        )

    d = pd.DataFrame(rows)
    summary = {
        "name": name,
        "mode": mode,
        "text_column": text_column,
        "n_folds": len(rows),
        **{f"{c}_mean": float(d[c].mean()) for c in d.columns},
        **{f"{c}_std": float(d[c].std()) for c in d.columns},
        "roc_auc_ci95": [
            float(d["roc_auc"].mean() - 1.96 * d["roc_auc"].sem()),
            float(d["roc_auc"].mean() + 1.96 * d["roc_auc"].sem()),
        ],
    }
    return summary, oof_prob / np.maximum(oof_count, 1)


def threshold_sweep(y: np.ndarray, oof: np.ndarray) -> list[dict]:
    """Sensitivity/specificity across decision thresholds, from OOF predictions.

    The 0.5 default is an artefact of logistic regression, not a screening
    decision. A screen that misses cases is worse than one that over-refers, so
    the operating point deserves to be chosen from evidence and stated openly
    rather than inherited.
    """
    out = []
    for t in np.arange(0.20, 0.81, 0.05):
        pred = (oof >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        out.append(
            {
                "threshold": round(float(t), 2),
                "sensitivity": round(float(sens), 3),
                "specificity": round(float(spec), 3),
                "youden_j": round(float(sens + spec - 1), 3),
                "accuracy": round(float((tp + tn) / len(y)), 3),
            }
        )
    return out


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    by_col, y, groups, hand, timing, df = load_data()

    n_part = len(set(groups))
    print(
        f"\nDataset: {len(y)} sessions from {n_part} participants "
        f"| dementia={int(y.sum())} control={int((1 - y).sum())}"
    )
    print(f"         {len(y) / n_part:.2f} visits per participant\n")

    print("=" * 74)
    print(f"{'arm':34}{'ROC-AUC':>18}{'acc':>7}{'sens':>7}{'spec':>7}")
    print("=" * 74)

    results, oof_by_name = [], {}
    for name, mode, textcol in ARMS:
        r, oof = evaluate(name, mode, by_col[textcol], y, groups, hand, timing,
                          list(grouped_splits(y, groups)), text_column=textcol)
        results.append(r)
        oof_by_name[name] = oof
        lo, hi = r["roc_auc_ci95"]
        print(
            f"{name:34}{r['roc_auc_mean']:.3f} [{lo:.3f}-{hi:.3f}]"
            f"{r['accuracy_mean']:7.3f}{r['sensitivity_mean']:7.3f}{r['specificity_mean']:7.3f}"
        )
    print("=" * 74)

    top = max(results, key=lambda r: r["roc_auc_mean"])
    by_name = {r["name"]: r for r in results}

    # --- did the pause markers actually earn their place? ----------------
    # With ~285 participants a fold tests on ~57 people, so a real gain of a few
    # thousandths can sit entirely inside the noise. Overlapping intervals are
    # reported as overlapping, not spun as a win.
    baseline = by_name["text + timing"]
    paused = by_name["pause-marked text + timing"]
    delta = paused["roc_auc_mean"] - baseline["roc_auc_mean"]
    lo_b, hi_b = baseline["roc_auc_ci95"]
    lo_p, hi_p = paused["roc_auc_ci95"]
    overlap = lo_p <= hi_b and lo_b <= hi_p
    verdict = (
        "inconclusive at this sample size (95% CIs overlap)" if overlap
        else ("pause markers help" if delta > 0 else "pause markers hurt")
    )
    print(
        f"\nPause-marker check: {paused['roc_auc_mean']:.4f} vs "
        f"{baseline['roc_auc_mean']:.4f} unmarked ({delta:+.4f}) -> {verdict}."
    )

    # Extra machinery has to EARN its place, and "scored 0.0006 higher" is not
    # earning it when the script's own significance check says the two cannot be
    # separated. Relying on max() to break the tie was wrong: it only preferred the
    # simpler arm on an EXACT tie, which never happens with float means, so the
    # pause-marked arm shipped on noise.
    #
    # Scoped deliberately to the pause-marker pair rather than applied as a general
    # "simplest arm within overlapping CIs" rule. That general rule would also drop
    # the timing block, because text-only overlaps the top arm too — a much bigger
    # decision than this one, and not one to make by side effect.
    best = top
    if top["text_column"] == PAUSE_COL and overlap:
        simpler = by_name.get(top["name"].replace("pause-marked text", "text"))
        if simpler is not None:
            best = simpler
            print(
                f"  -> Deploying the UNMARKED arm '{simpler['name']}' instead.\n"
                "     The pause encoder is real inference machinery (the vectorizer\n"
                "     must be fed transcript_pauses, matching how it was fitted), and\n"
                f"     it buys {delta:+.4f} AUC that this corpus cannot distinguish\n"
                "     from zero."
            )
    if overlap and best is top:
        print(
            "  The point estimate favours whichever arm is higher, but the\n"
            "  intervals overlap: this corpus cannot separate them. Report the\n"
            "  uncertainty rather than the difference."
        )

    # --- what grouping cost us, stated plainly --------------------------
    # Run the ungrouped comparison on the arm being deployed, so the inflation
    # figure describes the actual model rather than a sibling of it.
    ungrouped, _ = evaluate(
        f"{best['name']}, UNGROUPED", best["mode"], by_col[best["text_column"]],
        y, groups, hand, timing,
        list(RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEED)
             .split(np.zeros(len(y)), y)),
        text_column=best["text_column"],
    )
    inflation = ungrouped["roc_auc_mean"] - best["roc_auc_mean"]
    print(
        f"\nLeakage check: ungrouped CV scores {ungrouped['roc_auc_mean']:.3f} "
        f"vs {best['roc_auc_mean']:.3f} grouped (+{inflation:.3f} inflation).\n"
        "  The same participant appears at several yearly visits; splitting at\n"
        "  random lets the model recognise the speaker instead of the condition."
    )

    # --- refit and calibrate on everything ------------------------------
    print(f"\nDeploying: {best['name']}")
    texts = by_col[best["text_column"]]
    X, vec, hs, ts = build_matrix(best["mode"], texts, hand, timing, fit=True)

    # Calibration folds must be grouped too, or the calibrator leaks participants
    # even when the outer evaluation does not.
    cal_splits = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        .split(X, y, groups)
    )
    calibrated = CalibratedClassifierCV(
        make_estimator("logreg"), cv=cal_splits, method="sigmoid"
    ).fit(X, y)

    joblib.dump(
        {
            "model": calibrated,
            "vectorizer": vec,
            "scaler": hs,
            "timing_scaler": ts,
            "mode": best["mode"],
            "estimator": "logreg",
            "feature_names": FEATURE_NAMES,
            "timing_feature_names": TIMING_FEATURE_NAMES,
            "tfidf_kwargs": TFIDF_KWARGS,
            # Which text the VECTORIZER was fitted on. Inference must feed the
            # same kind: transcript_pauses means speech.pause_encoded_transcript
            # output, not the plain transcript. The handcrafted block is always
            # built from the clean transcript regardless.
            "text_column": best["text_column"],
            "hand_features_from": TEXT_COL,
            "text_source": "faster-whisper ASR (matches inference)",
            "whisper_model": "base.en",
            "cv": "StratifiedGroupKFold(5) x 4 seeds, grouped by participant_id",
            "trained_on": "DementiaBank Pitt Cookie Theft (TalkBank; restricted, not redistributed)",
            "n_samples": int(len(y)),
            "n_participants": int(n_part),
            "version": "2.0.0",
        },
        MODEL_PATH,
    )

    # Out-of-fold predictions for the deployed arm: the only honest basis for a
    # ROC curve, a calibration plot, or a threshold choice. Saved alongside the
    # labels so a plot cannot accidentally be drawn against the in-sample refit.
    best_oof = oof_by_name[best["name"]]
    np.save(ARTIFACTS / "audio_oof_probabilities.npy", best_oof)
    np.save(ARTIFACTS / "audio_labels.npy", y)
    sweep = threshold_sweep(y, best_oof)
    at_default = next(s for s in sweep if s["threshold"] == 0.50)
    best_j = max(sweep, key=lambda s: s["youden_j"])
    print(
        f"\nThreshold: default 0.50 gives sens {at_default['sensitivity']:.3f} / "
        f"spec {at_default['specificity']:.3f}.\n"
        f"  Best Youden J at {best_j['threshold']:.2f}: sens "
        f"{best_j['sensitivity']:.3f} / spec {best_j['specificity']:.3f}.\n"
        "  A screen that misses cases is worse than one that over-refers, so the\n"
        "  operating point is a clinical choice, not a default. Full sweep is in\n"
        "  metrics_audio.json."
    )

    metrics = {
        "deployed_model": best["name"],
        "deployed_text_column": best["text_column"],
        "threshold_sweep": sweep,
        "threshold_note": (
            "Sensitivity/specificity by decision threshold, computed from "
            "out-of-fold predictions. The model reports a calibrated probability; "
            "0.5 is logistic regression's default, not a screening decision. "
            "Sensitivity matters more than specificity for a screen, so a lower "
            "threshold is defensible and should be stated rather than assumed."
        ),
        "evaluation": "mean over StratifiedGroupKFold(5) x 4 seeds, grouped by participant",
        "all_models": results,
        "pause_marker_check": {
            "unmarked_roc_auc": baseline["roc_auc_mean"],
            "pause_marked_roc_auc": paused["roc_auc_mean"],
            "delta": delta,
            "ci95_overlap": bool(overlap),
            "verdict": verdict,
            "top_scoring_arm": top["name"],
            "deployed_arm": best["name"],
            "deployed_simpler_on_overlap": bool(best is not top),
            "explanation": (
                "Pause markers are spliced into the transcript where the silence "
                "actually fell, so the n-gram model can learn 'hesitated before "
                "this word' instead of only an aggregate pause rate. With ~30 "
                "participants per test fold a few thousandths of AUC sit inside "
                "the noise; when the intervals overlap the comparison is reported "
                "as inconclusive rather than as an improvement."
            ),
        },
        "leakage_demonstration": {
            "ungrouped_roc_auc": ungrouped["roc_auc_mean"],
            "grouped_roc_auc": best["roc_auc_mean"],
            "inflation": inflation,
            "explanation": (
                "Pitt is longitudinal. Ungrouped cross-validation places the same "
                "participant in train and test, so the model can recognise the "
                "speaker rather than the condition. The grouped figure is the "
                "honest one."
            ),
        },
        "dataset": {
            "source": "DementiaBank Pitt Cookie Theft, audio (TalkBank, restricted access)",
            "n_total": int(len(y)),
            "n_participants": int(n_part),
            "n_dementia": int(y.sum()),
            "n_control": int((1 - y).sum()),
            "text_source": "faster-whisper base.en, same path as inference",
        },
        "caveats": [
            "Small corpus, and the effective sample size is the participant count, "
            "not the session count.",
            "Trained on picture-description speech. Valid only for that task.",
            "Labels are diagnostic group, not severity, so this is not validated "
            "for distinguishing early decline from normal ageing.",
            "Pitt audio is 1980s-90s clinic recordings; users record on a modern "
            "microphone in lossy Opus. Timing features derive from word timestamps "
            "and transfer across that gap reasonably; spectral voice-quality "
            "features would not, which is why none are used.",
            "This is a screening indicator, not a diagnosis.",
        ],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved -> {MODEL_PATH}")
    print(f"Saved -> {METRICS_PATH}")
    print(
        "\nThe running app is unaffected: it still loads language_model.joblib.\n"
        "Wiring this artifact into the backend is a separate, deliberate step."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
