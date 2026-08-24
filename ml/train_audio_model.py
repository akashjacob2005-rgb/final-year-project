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

3. CROSS-VALIDATION IS GROUPED BY PARTICIPANT.
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

ARMS = [
    ("text only (linguistic + tfidf)", "text"),
    ("timing only (how they spoke)", "timing"),
    ("text + timing", "text+timing"),
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

    texts = df["transcript"].tolist()
    y = df["label"].to_numpy()
    groups = df["participant_id"].astype(str).to_numpy()
    timing = df[TIMING_FEATURE_NAMES].to_numpy(dtype=float)

    if FEATURE_CACHE.exists():
        hand = pd.read_csv(FEATURE_CACHE)
        if len(hand) == len(texts) and list(hand.columns) == FEATURE_NAMES:
            print(f"Using cached features: {FEATURE_CACHE}")
            return texts, y, groups, hand, timing, df
        print("Feature cache stale, recomputing")

    print(f"Extracting linguistic features from {len(texts)} transcripts...")
    t0 = time.time()
    hand = extract_batch(texts)
    hand.to_csv(FEATURE_CACHE, index=False)
    print(f"  done in {time.time() - t0:.1f}s")
    return texts, y, groups, hand, timing, df


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
def evaluate(name, mode, texts, y, groups, hand, timing, splits):
    texts_arr = np.array(texts, dtype=object)
    rows = []

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
    return {
        "name": name,
        "mode": mode,
        "n_folds": len(rows),
        **{f"{c}_mean": float(d[c].mean()) for c in d.columns},
        **{f"{c}_std": float(d[c].std()) for c in d.columns},
        "roc_auc_ci95": [
            float(d["roc_auc"].mean() - 1.96 * d["roc_auc"].sem()),
            float(d["roc_auc"].mean() + 1.96 * d["roc_auc"].sem()),
        ],
    }


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    texts, y, groups, hand, timing, df = load_data()

    n_part = len(set(groups))
    print(
        f"\nDataset: {len(y)} sessions from {n_part} participants "
        f"| dementia={int(y.sum())} control={int((1 - y).sum())}"
    )
    print(f"         {len(y) / n_part:.2f} visits per participant\n")

    print("=" * 74)
    print(f"{'arm':34}{'ROC-AUC':>18}{'acc':>7}{'sens':>7}{'spec':>7}")
    print("=" * 74)

    results = []
    for name, mode in ARMS:
        r = evaluate(name, mode, texts, y, groups, hand, timing,
                     list(grouped_splits(y, groups)))
        results.append(r)
        lo, hi = r["roc_auc_ci95"]
        print(
            f"{name:34}{r['roc_auc_mean']:.3f} [{lo:.3f}-{hi:.3f}]"
            f"{r['accuracy_mean']:7.3f}{r['sensitivity_mean']:7.3f}{r['specificity_mean']:7.3f}"
        )
    print("=" * 74)

    best = max(results, key=lambda r: r["roc_auc_mean"])

    # --- what grouping cost us, stated plainly --------------------------
    ungrouped = evaluate(
        "text+timing, UNGROUPED", "text+timing", texts, y, groups, hand, timing,
        list(RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEED)
             .split(np.zeros(len(y)), y)),
    )
    grouped_auc = next(r for r in results if r["mode"] == "text+timing")["roc_auc_mean"]
    inflation = ungrouped["roc_auc_mean"] - grouped_auc
    print(
        f"\nLeakage check: ungrouped CV scores {ungrouped['roc_auc_mean']:.3f} "
        f"vs {grouped_auc:.3f} grouped (+{inflation:.3f} inflation).\n"
        "  The same participant appears at several yearly visits; splitting at\n"
        "  random lets the model recognise the speaker instead of the condition."
    )

    # --- refit and calibrate on everything ------------------------------
    print(f"\nDeploying: {best['name']}")
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

    metrics = {
        "deployed_model": best["name"],
        "evaluation": "mean over StratifiedGroupKFold(5) x 4 seeds, grouped by participant",
        "all_models": results,
        "leakage_demonstration": {
            "ungrouped_roc_auc": ungrouped["roc_auc_mean"],
            "grouped_roc_auc": grouped_auc,
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
