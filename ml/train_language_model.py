"""Train the cognitive-decline language classifier on DementiaBank Pitt transcripts.

Compares four model configurations under repeated stratified cross-validation,
picks the best by mean ROC-AUC, refits with probability calibration, and saves
the artifact plus an honest metrics report.

Deliberate design choice: NO custom classes are pickled. The saved artifact holds
only stock scikit-learn objects (vectorizer / scaler / classifier). Feature
extraction is done explicitly in code at both training and inference time, so the
artifact can never fail to unpickle because of a module path change.

Usage:
    python ml/train_language_model.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_NAMES, extract_batch  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data" / "transcripts.csv"
FEATURE_CACHE = HERE / "data" / "features.csv"
ARTIFACTS = HERE / "artifacts"
SEED = 42

TFIDF_KWARGS = dict(
    ngram_range=(1, 2),
    min_df=3,
    max_features=1500,
    sublinear_tf=True,
    strip_accents="unicode",
)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_data() -> tuple[list[str], np.ndarray, pd.DataFrame]:
    if not DATA.exists():
        raise SystemExit("Missing data. Run: python ml/download_data.py")

    df = pd.read_csv(DATA)
    texts = df["text"].astype(str).tolist()
    y = df["label"].to_numpy()

    # spaCy parsing of 498 documents takes ~30s; cache it.
    if FEATURE_CACHE.exists():
        hand = pd.read_csv(FEATURE_CACHE)
        if len(hand) == len(texts) and list(hand.columns) == FEATURE_NAMES:
            print(f"Using cached features: {FEATURE_CACHE}")
            return texts, y, hand
        print("Feature cache stale, recomputing")

    print(f"Extracting linguistic features from {len(texts)} transcripts...")
    t0 = time.time()
    hand = extract_batch(texts)
    hand.to_csv(FEATURE_CACHE, index=False)
    print(f"  done in {time.time() - t0:.1f}s -> {FEATURE_CACHE}")
    return texts, y, hand


# --------------------------------------------------------------------------
# matrix construction
# --------------------------------------------------------------------------
def build_matrix(mode, texts, hand, vec=None, scaler=None, fit=False):
    """Assemble the design matrix for a given feature mode."""
    parts = []
    if mode in ("hand", "combined"):
        H = np.nan_to_num(hand.to_numpy(dtype=float), posinf=0.0, neginf=0.0)
        if fit:
            scaler = StandardScaler().fit(H)
        parts.append(scaler.transform(H))
    if mode in ("tfidf", "combined"):
        if fit:
            vec = TfidfVectorizer(**TFIDF_KWARGS).fit(texts)
        parts.append(vec.transform(texts).toarray())
    return np.hstack(parts), vec, scaler


def make_estimator(kind):
    if kind == "logreg":
        return LogisticRegression(
            max_iter=3000, C=1.0, class_weight="balanced", random_state=SEED
        )
    if kind == "gbm":
        # sklearn's histogram gradient booster rather than LightGBM/XGBoost:
        # same algorithm family, but no OpenMP/native library to install, which
        # keeps both this machine and the deployment container simple.
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=SEED,
        )
    raise ValueError(kind)


CONFIGS = [
    ("handcrafted + logreg", "hand", "logreg"),
    ("tfidf + logreg", "tfidf", "logreg"),
    ("combined + logreg", "combined", "logreg"),
    ("handcrafted + gbm", "hand", "gbm"),
    ("combined + gbm", "combined", "gbm"),
]

# TF-IDF tends to win on cross-validation here, but its vocabulary is learned
# from clinician-typed CHAT transcripts, while at inference we feed Whisper ASR
# output. Handcrafted features (pronoun ratio, idea density, content units) are
# far less sensitive to exact word forms and transcription conventions, so the
# combined model should degrade less under that domain shift. If combined is
# within this AUC margin of the winner, deploy combined instead and say why.
ROBUSTNESS_MARGIN = 0.03


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def cross_validate(name, mode, kind, texts, y, hand, n_splits=5, n_repeats=4):
    """Repeated stratified CV. Repeats give a variance estimate — with only 498
    samples a single split is close to meaningless."""
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=SEED
    )
    texts_arr = np.array(texts, dtype=object)
    rows = []
    oof_prob = np.zeros(len(y))
    oof_count = np.zeros(len(y))

    for tr, te in cv.split(texts_arr, y):
        Xtr, vec, scaler = build_matrix(
            mode, texts_arr[tr].tolist(), hand.iloc[tr], fit=True
        )
        Xte, _, _ = build_matrix(
            mode, texts_arr[te].tolist(), hand.iloc[te], vec=vec, scaler=scaler
        )
        clf = make_estimator(kind).fit(Xtr, y[tr])
        prob = clf.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)

        oof_prob[te] += prob
        oof_count[te] += 1

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
        "estimator": kind,
        "n_folds": len(rows),
        **{f"{c}_mean": float(d[c].mean()) for c in d.columns},
        **{f"{c}_std": float(d[c].std()) for c in d.columns},
        # 95% CI on the mean AUC across folds
        "roc_auc_ci95": [
            float(d["roc_auc"].mean() - 1.96 * d["roc_auc"].sem()),
            float(d["roc_auc"].mean() + 1.96 * d["roc_auc"].sem()),
        ],
    }
    return summary, oof_prob / np.maximum(oof_count, 1)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    texts, y, hand = load_data()
    print(f"\nDataset: {len(y)} transcripts | dementia={int(y.sum())} control={int((1-y).sum())}")

    print("\n" + "=" * 78)
    print(f"{'model':26} {'ROC-AUC':>16} {'acc':>7} {'sens':>7} {'spec':>7}")
    print("=" * 78)

    results, oofs = [], {}
    for name, mode, kind in CONFIGS:
        summary, oof = cross_validate(name, mode, kind, texts, y, hand)
        results.append(summary)
        oofs[name] = oof
        lo, hi = summary["roc_auc_ci95"]
        print(
            f"{name:26} {summary['roc_auc_mean']:.3f} "
            f"[{lo:.3f}-{hi:.3f}] {summary['accuracy_mean']:7.3f} "
            f"{summary['sensitivity_mean']:7.3f} {summary['specificity_mean']:7.3f}"
        )
    print("=" * 78)

    best = max(results, key=lambda r: r["roc_auc_mean"])
    print(f"\nBest by CV ROC-AUC: {best['name']}  ({best['roc_auc_mean']:.3f})")

    # Deployment choice is NOT simply argmax(AUC). Differences of a few
    # thousandths are far inside the fold-to-fold noise, so among models that
    # are statistically indistinguishable from the winner we take the one that
    # is most robust and most explainable. Preference order, best first:
    deploy_preference = ["combined + logreg", "combined + gbm"]

    deployed, rationale = best, "highest cross-validated ROC-AUC"
    by_name = {r["name"]: r for r in results}
    for candidate in deploy_preference:
        r = by_name.get(candidate)
        if r and best["roc_auc_mean"] - r["roc_auc_mean"] <= ROBUSTNESS_MARGIN:
            if r["name"] == best["name"]:
                break
            deployed = r
            reasons = [
                f"statistically indistinguishable from the CV winner "
                f"({best['name']}, {best['roc_auc_mean']:.3f} vs {r['roc_auc_mean']:.3f})",
                "includes handcrafted linguistic features, which are less "
                "sensitive to the transcript-vs-ASR domain shift at inference",
            ]
            if r["estimator"] == "logreg":
                reasons.append(
                    "linear coefficients give directly interpretable per-feature "
                    "contributions for the results page"
                )
            rationale = "; ".join(reasons)
            print(f"Deploying instead: {r['name']} ({r['roc_auc_mean']:.3f})")
            print(f"  reason: {rationale}")
            break

    best = deployed

    # --- refit on all data, with calibration -----------------------------
    print("Refitting on full dataset with probability calibration...")
    X, vec, scaler = build_matrix(best["mode"], texts, hand, fit=True)
    calibrated = CalibratedClassifierCV(
        make_estimator(best["estimator"]), cv=5, method="sigmoid"
    ).fit(X, y)

    joblib.dump(
        {
            "model": calibrated,
            "vectorizer": vec,
            "scaler": scaler,
            "mode": best["mode"],
            "estimator": best["estimator"],
            "feature_names": FEATURE_NAMES,
            "tfidf_kwargs": TFIDF_KWARGS,
            "trained_on": "DementiaBank Pitt (MearaHe/dementiabank mirror)",
            "n_samples": int(len(y)),
            "version": "1.0.0",
        },
        ARTIFACTS / "language_model.joblib",
    )

    metrics = {
        "deployed_model": best["name"],
        "deployed_because": rationale,
        "best_cv_model": max(results, key=lambda r: r["roc_auc_mean"])["name"],
        "evaluation": "mean over 20 folds (5-fold stratified x 4 repeats)",
        "all_models": results,
        "dataset": {
            "source": "DementiaBank Pitt Cookie Theft transcripts",
            "n_total": int(len(y)),
            "n_dementia": int(y.sum()),
            "n_control": int((1 - y).sum()),
        },
        "caveats": [
            "498 samples is small; fold-to-fold variance is substantial.",
            "Trained on picture-description speech. Valid only for that task.",
            "Transcripts are clinician-transcribed; ASR output at inference "
            "introduces a domain shift not captured by these metrics.",
            "This is a screening indicator, not a diagnosis.",
        ],
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    np.save(ARTIFACTS / "oof_probabilities.npy", oofs[best["name"]])
    np.save(ARTIFACTS / "labels.npy", y)

    print(f"\nSaved -> {ARTIFACTS / 'language_model.joblib'}")
    print(f"Saved -> {ARTIFACTS / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
