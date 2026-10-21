"""Offline ensemble of per-session predictions from two or more architectures.

Averages the session probabilities saved by `train_mri.py --save-preds`
(preds_<arch>.csv) and recomputes the same session-level metrics per fold.
Because every arch is trained under identical subject-stratified folds
(same seed), rows join exactly on (fold, session) — the script refuses to
run if they don't.

Usage (from repo root):
    ml/mri/.venv/bin/python ml/mri/ensemble_eval.py \
        ml/artifacts/preds_resnet18.csv ml/artifacts/preds_dinov2_s.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def fold_metrics(df: pd.DataFrame, prob_cols: list[str]) -> dict:
    y_true = df["label"].to_numpy()
    y_prob = df[prob_cols].to_numpy()
    y_pred = y_prob.argmax(axis=1)
    n_classes = len(prob_cols)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    try:
        auc = (roc_auc_score(y_true, y_prob[:, 1]) if n_classes == 2
               else roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    avg = "binary" if n_classes == 2 else "macro"
    return {
        "sessions": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "f1": float(f1_score(y_true, y_pred, average=avg)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "auc": float(auc),
        "per_class_recall": (cm.diagonal() / cm.sum(axis=1).clip(min=1)).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help="two or more preds_<arch>.csv files")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ap.add_argument("--write-preds", default=None, metavar="DIR",
                    help="also write the averaged predictions as an experiment dir "
                         "(preds_sessions.csv + summary.json) so compare_experiments.py "
                         "can compare and --calibrate the ensemble like any single model")
    args = ap.parse_args()
    if len(args.preds) < 2:
        ap.error("need at least two preds CSVs to ensemble")

    frames = [pd.read_csv(p) for p in args.preds]
    prob_cols = sorted(c for c in frames[0].columns if c.startswith("p_"))
    keys = frames[0][["fold", "session", "label"]]
    for i, f in enumerate(frames[1:], 1):
        merged = keys.merge(f[["fold", "session", "label"]], on=["fold", "session", "label"])
        if len(merged) != len(keys) or len(f) != len(keys):
            raise SystemExit(f"{args.preds[i]} does not cover the same (fold, session) "
                             "set — all archs must be trained with the same --folds/seed")

    ens = frames[0].copy()
    for f in frames[1:]:
        aligned = ens[["fold", "session"]].merge(f, on=["fold", "session"])
        ens[prob_cols] = ens[prob_cols].to_numpy() + aligned[prob_cols].to_numpy()
    ens[prob_cols] /= len(frames)

    per_fold = [fold_metrics(g, prob_cols) for _, g in ens.groupby("fold")]

    def agg(key):
        vals = [f[key] for f in per_fold]
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    summary = {
        "members": [Path(p).stem.replace("preds_", "") for p in args.preds],
        "folds": len(per_fold),
        "accuracy": agg("accuracy"),
        "f1": agg("f1"),
        "macro_f1": agg("macro_f1"),
        "auc": agg("auc"),
        "per_fold": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}
                     for f in per_fold],
    }
    print(json.dumps({k: summary[k] for k in ("members", "accuracy", "f1", "auc")}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"written: {args.out}")
    if args.write_preds:
        write_experiment(ens, prob_cols, summary, Path(args.write_preds))


def write_experiment(ens: pd.DataFrame, prob_cols: list[str], summary: dict, out_dir: Path) -> None:
    """Averaged OOF predictions in compare_experiments.py's format. Fold keys
    "<seed>-<fold>" (repeated CV) are split back into repeat + fold."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = ens.copy()
    key = df["fold"].astype(str)
    if key.str.contains("-").any():
        parts = key.str.split("-", n=1, expand=True)
        seeds = sorted(parts[0].astype(int).unique())
        df["repeat"] = parts[0].astype(int).map({s: i for i, s in enumerate(seeds)})
        df["fold"] = parts[1].astype(int)
    else:
        df["repeat"] = 0
        df["fold"] = key.astype(int)
    df["seed"] = 0
    df["source"] = df["session"].map(lambda s: "oasis1" if str(s).upper().startswith("OAS1") else "oasis3")
    cols = ["repeat", "fold", "seed", "session", "source", "label"] + prob_cols
    df[cols].to_csv(out_dir / "preds_sessions.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "experiment": out_dir.name, "arch": "ensemble:" + "+".join(summary["members"]),
        "members": summary["members"], "cv": {k: summary[k] for k in ("accuracy", "f1", "auc")},
    }, indent=2))
    print(f"ensemble experiment written: {out_dir} (compare_experiments.py {out_dir.name} --calibrate {out_dir.name})")


if __name__ == "__main__":
    main()
