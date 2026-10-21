"""Compare MRI experiments honestly, from their out-of-fold predictions.

Reads ml/experiments/<name>/preds_sessions.csv (written by
train_mri.py --experiment) and reports, per experiment:

  * pooled out-of-fold metrics (every session predicted once per repeat):
    accuracy, balanced accuracy, AUC, per-class recall, bootstrap 95% CI
    over sessions, plus the mean +/- std over folds from summary.json
  * CROSS-FITTED decision threshold + temperature: for each fold k the
    threshold / temperature are fitted on the OOF rows of the OTHER folds and
    applied to fold k, so the reported "calibrated" accuracy is itself honest
  * paired comparison against --baseline on identical (repeat, fold) splits:
    per-fold AUC deltas, fraction of folds improved, paired bootstrap CI of
    the pooled AUC delta, McNemar test on per-session correctness
  * slice -> session aggregation variants from preds_slices.csv (mean, median,
    trimmed mean, top-k), which need no retraining

and writes <experiments-root>/comparison.json. With --calibrate NAME it also
writes <name>/calibration.json (temperature + threshold fitted on ALL OOF rows
of that experiment) for promote.py to bake into the deployed metrics.

Usage (from repo root):
    python ml/mri/compare_experiments.py                       # all experiments
    python ml/mri/compare_experiments.py e02_recipe_v2 e03_... --baseline e01_baseline_honest
    python ml/mri/compare_experiments.py --calibrate e04_swa
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
N_BOOT = 2000
EPS = 1e-7


# ---------------------------------------------------------------- helpers
def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def binary_metrics(y: np.ndarray, p1: np.ndarray, threshold: float = 0.5) -> dict:
    yhat = (p1 >= threshold).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    rec1 = tp / max(tp + fn, 1)
    rec0 = tn / max(tn + fp, 1)
    try:
        auc = float(roc_auc_score(y, p1))
    except ValueError:
        auc = float("nan")
    return {
        "n": int(len(y)),
        "accuracy": (tp + tn) / max(len(y), 1),
        "balanced_accuracy": (rec0 + rec1) / 2,
        "auc": auc,
        "recall_normal": rec0,
        "recall_impaired": rec1,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "confusion": [[tn, fp], [fn, tp]],
    }


def fit_temperature(y: np.ndarray, p1: np.ndarray) -> float:
    """Temperature T minimising NLL of sigmoid(logit(p)/T) — one scalar, so it
    cannot overfit, and it leaves the ranking (AUC) untouched."""
    z = logit(p1)

    def nll(t: float) -> float:
        q = np.clip(sigmoid(z / t), EPS, 1 - EPS)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    # Bounded so a (near-)separable fold cannot drive T -> 0 and turn every
    # displayed percentage into 0% / 100%.
    res = minimize_scalar(nll, bounds=(0.25, 10.0), method="bounded")
    return float(res.x)


def fit_threshold(y: np.ndarray, p1: np.ndarray, objective: str = "balanced_accuracy") -> float:
    """Threshold on the (calibrated) impaired probability maximising the
    objective; ties resolved towards 0.5."""
    cands = np.unique(np.concatenate([[0.5], np.round(p1, 4)]))
    best_t, best_v = 0.5, -1.0
    for t in cands:
        v = binary_metrics(y, p1, t)[objective]
        if v > best_v + 1e-9 or (abs(v - best_v) <= 1e-9 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_v = float(t), v
    return best_t


def apply_calibration(p1: np.ndarray, temperature: float) -> np.ndarray:
    return sigmoid(logit(p1) / temperature)


def bootstrap_ci(y: np.ndarray, p1: np.ndarray, key: str, threshold: float = 0.5,
                 n_boot: int = N_BOOT, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        vals.append(binary_metrics(y[idx], p1[idx], threshold)[key])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar on discordant sessions (b right & a wrong vs a right & b wrong)."""
    b_only = int((correct_b & ~correct_a).sum())
    a_only = int((correct_a & ~correct_b).sum())
    n = a_only + b_only
    p = float(binomtest(b_only, n, 0.5).pvalue) if n else 1.0
    return {"candidate_only_correct": b_only, "baseline_only_correct": a_only, "p_value": p}


# ------------------------------------------------------------- experiment
class Experiment:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = path.name
        self.summary = json.loads((path / "summary.json").read_text()) if (path / "summary.json").exists() else {}
        self.config = json.loads((path / "config.json").read_text()) if (path / "config.json").exists() else {}
        df = pd.read_csv(path / "preds_sessions.csv")
        prob_cols = sorted(c for c in df.columns if c.startswith("p_"))
        if len(prob_cols) != 2:
            raise SystemExit(f"{self.name}: compare_experiments supports the binary task only")
        if "repeat" not in df.columns:
            df["repeat"] = 0
        if "seed" not in df.columns:
            df["seed"] = 0
        # Seed-ensemble per fold: average the seeds' probabilities (the
        # headline the training script also reports).
        keys = [c for c in ("repeat", "fold", "session", "source", "label") if c in df.columns]
        self.df = df.groupby(keys, as_index=False)[prob_cols].mean()
        self.n_seeds = int(df["seed"].nunique())
        self.slices = pd.read_csv(path / "preds_slices.csv") if (path / "preds_slices.csv").exists() else None

    @property
    def repeats(self) -> list[int]:
        return sorted(self.df["repeat"].unique())

    def pooled(self, repeat: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        g = self.df[self.df["repeat"] == repeat].sort_values("session").reset_index(drop=True)
        return g["label"].to_numpy(), g["p_1"].to_numpy(), g

    def cross_fitted(self, repeat: int, objective: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Calibrated probabilities + decisions for every session of a repeat,
        each fold's parameters fitted on the OTHER folds of the same repeat."""
        g = self.df[self.df["repeat"] == repeat].sort_values("session").reset_index(drop=True)
        p_cal = np.zeros(len(g))
        yhat = np.zeros(len(g), dtype=int)
        fits = []
        for k in sorted(g["fold"].unique()):
            m = (g["fold"] == k).to_numpy()
            y_o, p_o = g.loc[~m, "label"].to_numpy(), g.loc[~m, "p_1"].to_numpy()
            t = fit_temperature(y_o, p_o)
            thr = fit_threshold(y_o, apply_calibration(p_o, t), objective)
            p_cal[m] = apply_calibration(g.loc[m, "p_1"].to_numpy(), t)
            yhat[m] = (p_cal[m] >= thr).astype(int)
            fits.append({"fold": int(k), "temperature": round(t, 4), "threshold": round(thr, 4)})
        return p_cal, yhat, fits

    def report(self, objective: str) -> dict:
        per_rep, per_rep_cal, fits_all = [], [], []
        for r in self.repeats:
            y, p, _ = self.pooled(r)
            per_rep.append(binary_metrics(y, p))
            p_cal, yhat, fits = self.cross_fitted(r, objective)
            cal = binary_metrics(y, p_cal, 0.5)  # AUC unchanged; decisions from yhat
            cal_dec = _metrics_from_decisions(y, yhat)
            cal.update(cal_dec)
            per_rep_cal.append(cal)
            fits_all += [{"repeat": int(r), **f} for f in fits]

        def mean_std(rows: list[dict], key: str) -> dict:
            vals = [x[key] for x in rows if not (isinstance(x[key], float) and math.isnan(x[key]))]
            return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

        y0, p0, _ = self.pooled(self.repeats[0])
        out = {
            "name": self.name,
            "arch": self.summary.get("arch", self.config.get("arch")),
            "repeats": len(self.repeats),
            "folds": int(self.df["fold"].nunique()),
            "seeds_per_fold": self.n_seeds,
            "sessions": int(self.df[self.df["repeat"] == self.repeats[0]]["session"].nunique()),
            "git_sha": self.config.get("git_sha"),
            "argv": " ".join(self.config.get("argv", [])),
            "fold_mean": {k: self.summary.get("cv", {}).get(k) for k in ("accuracy", "balanced_accuracy", "auc")},
            "pooled_oof_argmax": {k: mean_std(per_rep, k) for k in
                                  ("accuracy", "balanced_accuracy", "auc", "recall_normal", "recall_impaired", "f1")},
            "pooled_oof_cross_fitted": {k: mean_std(per_rep_cal, k) for k in
                                        ("accuracy", "balanced_accuracy", "recall_normal", "recall_impaired", "f1")},
            "bootstrap_ci_repeat0": {
                "auc": bootstrap_ci(y0, p0, "auc"),
                "accuracy": bootstrap_ci(y0, p0, "accuracy"),
                "balanced_accuracy": bootstrap_ci(y0, p0, "balanced_accuracy"),
            },
            "cross_fitted_params": fits_all,
            "train_minutes": self.summary.get("cv", {}).get("train_minutes"),
        }
        if "source" in self.df.columns and self.df["source"].nunique() > 1:
            out["per_source"] = {}
            for src, gs in self.df.groupby("source"):
                rows = []
                for r in self.repeats:
                    g = gs[gs["repeat"] == r]
                    if g["label"].nunique() > 1:
                        rows.append(binary_metrics(g["label"].to_numpy(), g["p_1"].to_numpy()))
                if rows:
                    out["per_source"][src] = {k: mean_std(rows, k) for k in ("accuracy", "balanced_accuracy", "auc")}
                    out["per_source"][src]["sessions"] = int(gs["session"].nunique())
        if self.slices is not None:
            out["aggregation_variants"] = self.aggregation_variants()
        return out

    def aggregation_variants(self) -> dict:
        """Session score from slice probabilities under different poolings.
        Free to evaluate; adopting one changes a single line in the backend."""
        s = self.slices.copy()
        if "repeat" not in s.columns:
            s["repeat"] = 0
        keys = ["repeat", "fold", "session", "label"]
        s = s.groupby(keys + ["slice_idx"], as_index=False)["p_1"].mean()  # over seeds
        variants = {
            "mean": lambda v: float(np.mean(v)),
            "median": lambda v: float(np.median(v)),
            "trimmed_mean_2": lambda v: float(np.mean(np.sort(v)[2:-2])) if len(v) > 4 else float(np.mean(v)),
            "top4_mean": lambda v: float(np.mean(np.sort(v)[-4:])),
            "mean_logit": lambda v: float(sigmoid(np.mean(logit(np.asarray(v))))),
        }
        out = {}
        for name, fn in variants.items():
            rows = []
            for r, g in s.groupby("repeat"):
                agg = g.groupby(["session", "label"])["p_1"].apply(lambda v: fn(v.to_numpy())).reset_index()
                rows.append(binary_metrics(agg["label"].to_numpy(), agg["p_1"].to_numpy()))
            out[name] = {k: round(float(np.mean([x[k] for x in rows])), 4) for k in ("accuracy", "balanced_accuracy", "auc")}
        return out


def _metrics_from_decisions(y: np.ndarray, yhat: np.ndarray) -> dict:
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    rec1, rec0 = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {"accuracy": (tp + tn) / max(len(y), 1), "balanced_accuracy": (rec0 + rec1) / 2,
            "recall_normal": rec0, "recall_impaired": rec1,
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "confusion": [[tn, fp], [fn, tp]]}


def paired(base: Experiment, cand: Experiment, objective: str) -> dict | None:
    common = sorted(set(base.repeats) & set(cand.repeats))
    if not common:
        return None
    fold_deltas, auc_deltas_ci, mcn = [], [], []
    for r in common:
        yb, pb, gb = base.pooled(r)
        yc, pc, gc = cand.pooled(r)
        if not gb["session"].equals(gc["session"]) or not gb["fold"].equals(gc["fold"]):
            return {"error": "fold membership differs — compare only runs with the same --folds/--fold-seed/manifest"}
        for k in sorted(gb["fold"].unique()):
            m = (gb["fold"] == k).to_numpy()
            if len(set(yb[m])) < 2:
                continue
            fold_deltas.append(float(roc_auc_score(yc[m], pc[m]) - roc_auc_score(yb[m], pb[m])))
        rng = np.random.default_rng(1)
        n = len(yb)
        boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            if len(set(yb[idx])) < 2:
                continue
            boots.append(roc_auc_score(yc[idx], pc[idx]) - roc_auc_score(yb[idx], pb[idx]))
        auc_deltas_ci.append([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])
        _, yhat_b, _ = base.cross_fitted(r, objective)
        _, yhat_c, _ = cand.cross_fitted(r, objective)
        mcn.append(mcnemar(yhat_b == yb, yhat_c == yc))
    pooled_delta = float(np.mean([binary_metrics(*cand.pooled(r)[:2])["auc"] - binary_metrics(*base.pooled(r)[:2])["auc"]
                                  for r in common]))
    bal_delta = float(np.mean([
        _metrics_from_decisions(cand.pooled(r)[0], cand.cross_fitted(r, objective)[1])["balanced_accuracy"]
        - _metrics_from_decisions(base.pooled(r)[0], base.cross_fitted(r, objective)[1])["balanced_accuracy"]
        for r in common]))
    return {
        "baseline": base.name,
        "repeats_compared": len(common),
        "pooled_auc_delta": round(pooled_delta, 4),
        "pooled_auc_delta_ci_per_repeat": [[round(a, 4), round(b, 4)] for a, b in auc_deltas_ci],
        "fold_auc_deltas": [round(d, 4) for d in fold_deltas],
        "folds_improved": f"{sum(d > 0 for d in fold_deltas)}/{len(fold_deltas)}",
        "cross_fitted_balanced_accuracy_delta": round(bal_delta, 4),
        "mcnemar_per_repeat": mcn,
        "verdict": verdict(pooled_delta, auc_deltas_ci, bal_delta, fold_deltas),
    }


def verdict(auc_delta: float, cis: list, bal_delta: float, fold_deltas: list) -> str:
    """The adoption rule from docs/ARCH_COMPARISON.md, applied mechanically."""
    ci_excludes_zero = all(lo > 0 for lo, _ in cis)
    if auc_delta >= 0.02 and ci_excludes_zero and bal_delta >= 0.02:
        return "ADOPT: AUC +%.3f (CI excludes 0), balanced acc +%.3f" % (auc_delta, bal_delta)
    if auc_delta > 0 and sum(d > 0 for d in fold_deltas) > len(fold_deltas) / 2:
        return "promising but not significant — more repeats or a bigger change needed"
    return "no improvement over baseline"


def fmt(ms: dict | None) -> str:
    if not ms or ms.get("mean") is None:
        return "   -   "
    return f"{ms['mean']:.3f}±{ms['std']:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", help="experiment names (default: all under --experiments-root)")
    ap.add_argument("--experiments-root", default=str(EXPERIMENTS_ROOT))
    ap.add_argument("--baseline", default=None, help="experiment name to compare the others against")
    ap.add_argument("--objective", choices=["balanced_accuracy", "accuracy"], default="balanced_accuracy",
                    help="what the decision threshold maximises (balanced = equal weight to normal recall)")
    ap.add_argument("--calibrate", default=None,
                    help="write <name>/calibration.json (temperature+threshold fitted on all OOF rows)")
    ap.add_argument("--out", default=None, help="comparison JSON path (default <root>/comparison.json)")
    args = ap.parse_args()

    root = Path(args.experiments_root)
    names = args.experiments or sorted(p.name for p in root.iterdir()
                                       if (p / "preds_sessions.csv").exists())
    if args.baseline and args.baseline not in names:
        names.insert(0, args.baseline)
    if not names:
        raise SystemExit(f"no experiments with preds_sessions.csv under {root}")
    exps = {n: Experiment(root / n) for n in names}

    reports = {n: e.report(args.objective) for n, e in exps.items()}
    base = exps.get(args.baseline) if args.baseline else None
    if base is not None:
        for n, e in exps.items():
            if n != base.name:
                reports[n]["vs_baseline"] = paired(base, e, args.objective)

    print(f"\n{'experiment':<28} {'fold acc':>13} {'pooled acc':>13} {'bal acc':>13} {'AUC':>13} "
          f"{'cal bal acc':>13} {'rec N/I (cal)':>14}  vs baseline")
    for n, rep in reports.items():
        cf = rep["pooled_oof_cross_fitted"]
        vs = rep.get("vs_baseline") or {}
        vs_s = "" if not vs else (vs.get("error") or
                                  f"ΔAUC {vs['pooled_auc_delta']:+.3f} ({vs['folds_improved']} folds) — {vs['verdict']}")
        print(f"{n:<28} {fmt(rep['fold_mean'].get('accuracy')):>13} {fmt(rep['pooled_oof_argmax']['accuracy']):>13} "
              f"{fmt(rep['pooled_oof_argmax']['balanced_accuracy']):>13} {fmt(rep['pooled_oof_argmax']['auc']):>13} "
              f"{fmt(cf['balanced_accuracy']):>13} {cf['recall_normal']['mean']:.2f}/{cf['recall_impaired']['mean']:.2f}"
              f"{'':>6} {vs_s}")
        if "per_source" in rep:
            for src, m in rep["per_source"].items():
                print(f"  {src:<26} {'':>13} {fmt(m['accuracy']):>13} {fmt(m['balanced_accuracy']):>13} {fmt(m['auc']):>13}   (n={m['sessions']})")
        if "aggregation_variants" in rep:
            best = max(rep["aggregation_variants"].items(), key=lambda kv: kv[1]["auc"])
            print("  aggregation: " + ", ".join(f"{k} auc {v['auc']:.3f}" for k, v in rep["aggregation_variants"].items())
                  + f"  -> best {best[0]}")
    print("\nfold acc = mean±std over folds (what mri_metrics.json reports); pooled = every session once per repeat;\n"
          "cal = decision threshold + temperature cross-fitted on the other folds (honest calibrated accuracy).")

    out = Path(args.out) if args.out else root / "comparison.json"
    out.write_text(json.dumps({"objective": args.objective, "baseline": args.baseline,
                               "experiments": reports}, indent=2))
    print(f"written: {out}")

    if args.calibrate:
        e = exps[args.calibrate]
        y = e.df["label"].to_numpy()
        p = e.df["p_1"].to_numpy()
        t = fit_temperature(y, p)
        thr = fit_threshold(y, apply_calibration(p, t), args.objective)
        cf = reports[args.calibrate]["pooled_oof_cross_fitted"]
        arch = str(e.summary.get("arch") or e.config.get("arch") or "resnet18")
        members = e.summary.get("members") or [arch]
        cal = {
            "members": members,
            "temperature": round(t, 4),
            "threshold": round(thr, 4),
            "objective": args.objective,
            "fitted_on": f"all out-of-fold rows of {e.name} ({len(y)} session-predictions, {len(e.repeats)} repeat(s))",
            "cross_fitted_estimate": {k: cf[k] for k in ("accuracy", "balanced_accuracy", "recall_normal", "recall_impaired")},
            "note": ("Applied at inference to the SESSION probability (mean of slice softmaxes, "
                     "exactly what was fitted): p_cal = sigmoid(logit(p_impaired) / temperature); "
                     "impaired if p_cal >= threshold. Ranking (AUC) is unchanged by calibration."),
        }
        (e.path / "calibration.json").write_text(json.dumps(cal, indent=2))
        print(f"calibration for {e.name}: T={t:.3f} threshold={thr:.3f} -> {e.path / 'calibration.json'}")


if __name__ == "__main__":
    main()
