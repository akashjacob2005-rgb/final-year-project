"""E0: scanner/source confound audit for mixed OASIS-3 + OASIS-1 training.

Reads the per-session prediction CSVs saved by `train_mri.py --save-preds`
and answers, per architecture:

  1. Per-source performance — does the model work on BOTH sources, or is the
     pooled number hiding a failure on one of them?
  2. Pooled-vs-within-source AUC gap — if the pooled AUC is clearly higher
     than the within-source AUCs, part of the pooled score comes from the
     model separating SOURCES (whose class prevalences differ: OASIS-3 is
     ~67% impaired, OASIS-1 ~42%) rather than separating CLASSES.
  3. OASIS-3-only AUC vs the v2 baseline — did adding OASIS-1 help, hurt, or
     leave the original cohort unchanged?

Pure analysis, no training. Writes a JSON + markdown report.

Usage:
    python ml/mri/source_audit.py \
        --preds-dir ~/Downloads/neuroguard_results_v3 \
        --v2-baseline-auc 0.844 \
        --out-dir ml/experiments/e0_source_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def _source(session: str) -> str:
    return "oasis1" if session.startswith("OAS1") else "oasis3"


def _metrics(df: pd.DataFrame) -> dict:
    y, p = df["label"].to_numpy(), df["p_1"].to_numpy()
    yhat = (p >= 0.5).astype(int)
    out = {
        "n": int(len(df)),
        "base_rate_impaired": round(float(y.mean()), 3),
        "accuracy": round(float((y == yhat).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, yhat)), 4),
    }
    out["auc"] = round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None
    # mean of per-fold AUCs (the number train_mri.py's CV summary reports)
    fold_aucs = [roc_auc_score(g["label"], g["p_1"])
                 for _, g in df.groupby("fold") if g["label"].nunique() > 1]
    out["auc_mean_over_folds"] = round(float(np.mean(fold_aucs)), 4) if fold_aucs else None
    return out


def audit(preds_csv: Path, v2_baseline_auc: float) -> dict:
    df = pd.read_csv(preds_csv)
    df["source"] = df["session"].map(_source)

    pooled = _metrics(df)
    per_source = {s: _metrics(g) for s, g in df.groupby("source")}

    # weighted mean of within-source AUCs vs pooled AUC: the confound gap
    ws = [(per_source[s]["auc"], per_source[s]["n"]) for s in per_source
          if per_source[s]["auc"] is not None]
    within = float(np.average([a for a, _ in ws], weights=[n for _, n in ws]))
    gap = round(pooled["auc"] - within, 4)

    oas3_auc = per_source.get("oasis3", {}).get("auc")
    return {
        "preds": preds_csv.name,
        "pooled": pooled,
        "per_source": per_source,
        "within_source_auc_weighted": round(within, 4),
        "pooled_minus_within_gap": gap,
        "oasis3_auc_vs_v2_baseline": {
            "oasis3_within_v3": oas3_auc,
            "v2_baseline": v2_baseline_auc,
            "delta": round(oas3_auc - v2_baseline_auc, 4) if oas3_auc else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--v2-baseline-auc", type=float, default=0.844)
    ap.add_argument("--out-dir", default="ml/experiments/e0_source_audit")
    args = ap.parse_args()

    preds_dir = Path(args.preds_dir).expanduser()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {p.stem.replace("preds_", ""): audit(p, args.v2_baseline_auc)
               for p in sorted(preds_dir.glob("preds_*.csv"))}

    # verdict on the screening model (resnet18)
    r = results.get("resnet18")
    verdict = []
    if r:
        gap = r["pooled_minus_within_gap"]
        verdict.append(
            f"Confound gap (resnet18): pooled AUC {r['pooled']['auc']} vs "
            f"within-source {r['within_source_auc_weighted']} -> gap {gap:+.4f}. "
            + ("SOURCE CONFOUND MATERIAL (>0.02): use source-balanced sampling."
               if gap > 0.02 else "Gap small: pooled scores are not source-driven.")
        )
        d = r["oasis3_auc_vs_v2_baseline"]["delta"]
        verdict.append(
            f"OASIS-3 cohort within v3: AUC {r['oasis3_auc_vs_v2_baseline']['oasis3_within_v3']} "
            f"vs v2 baseline {args.v2_baseline_auc} -> delta {d:+.4f}. "
            + ("Mixing HURT the original cohort (<-0.02): consider source-balanced loss."
               if d is not None and d < -0.02 else
               "Original cohort not harmed by mixing.")
        )
    report = {"results": results, "verdict": verdict}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    lines = ["# E0 — source confound audit\n"]
    for arch, r in results.items():
        lines.append(f"## {arch}")
        lines.append(f"- pooled: AUC {r['pooled']['auc']}, bal-acc {r['pooled']['balanced_accuracy']}, n={r['pooled']['n']}")
        for s, m in r["per_source"].items():
            lines.append(f"- {s}: AUC {m['auc']}, bal-acc {m['balanced_accuracy']}, "
                         f"n={m['n']}, base-rate {m['base_rate_impaired']}")
        lines.append(f"- pooled-vs-within gap: {r['pooled_minus_within_gap']:+.4f}\n")
    lines.append("## Verdict")
    lines += [f"- {v}" for v in verdict]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
