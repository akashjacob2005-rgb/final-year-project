"""Train the connectivity model with honest CV, and deploy whichever model wins.

    python ml/graph/train_graph.py --data-dir ~/Downloads/7th_sem/PP-2/data/processed_graph [--sweep]

Input: processed_graph/manifest.csv + fc.npy (+ sc.npy) per session, from
ml/graph/preprocess.py or stream_pipeline.py. CDR labels 0 / 0.5 / >=1.

Task (--label-mode): binary (default) = Normal (CDR 0) vs Impaired (CDR >= 0.5),
matching the MRI pathway; three = Normal / MCI / Dementia.

Evaluation
  * subject-grouped, label-stratified K-fold: no subject in both train and test.
  * the GAT early-stops on an INNER split of the training subjects; with
    --sweep it also picks its hyper-parameters (top-k edges, hidden size) on
    that inner split. The outer fold is scored exactly once.
  * class-weighted loss, fixed seed.
  * baseline on the SAME folds: logistic regression on the Fisher-z FC upper
    triangle.

Deployment: whichever model has the higher mean outer-fold AUC is deployed —
the GAT only if it actually beats the baseline. The winner is refitted on
all subjects and exported for numpy inference (gat_numpy.py or
linear_numpy.py), after a parity check against torch / sklearn.
graph_metrics.json reports both models and which one is deployed.

Guard: no artifacts below --min-sessions sessions or with a class missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED = 42
VERSION = "2.0.0"
LABELS = {
    "binary": (["Normal (CDR 0)", "Impaired (CDR >= 0.5)"], ["Normal", "Impaired"]),
    "three": (["Normal (CDR 0)", "MCI (CDR 0.5)", "Dementia (CDR >= 1)"], ["Normal", "MCI", "Dementia"]),
}
SWEEP = [{"top_k": k, "hidden": h} for k in (5, 10, 20) for h in (16, 32)]


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_matrices(data_dir: Path, use_sc: str, label_mode: str):
    rows = list(csv.DictReader(open(data_dir / "manifest.csv")))
    if not rows:
        raise SystemExit(f"{data_dir / 'manifest.csv'} lists no sessions")
    have_sc = all(r.get("sc") for r in rows)
    if use_sc == "yes" and not have_sc:
        raise SystemExit("--use-sc yes but some sessions have no sc.npy")
    with_sc = use_sc == "yes" or (use_sc == "auto" and have_sc)
    fcs = np.array([np.load(data_dir / r["fc"]) for r in rows])
    scs = [np.load(data_dir / r["sc"]) for r in rows] if with_sc else None
    y = np.array([int(r["label"]) for r in rows])
    if label_mode == "binary":
        y = (y >= 1).astype(int)
    subjects = np.array([r["subject_id"] for r in rows])
    meta = {"atlas": rows[0].get("atlas", ""), "with_sc": with_sc,
            "mean_fd": [float(r["mean_fd"]) for r in rows if r.get("mean_fd")]}
    return fcs, scs, y, subjects, meta


def build_data(fcs, scs, y, top_k, threshold):
    from graph.gat import to_data
    from graph.graph_build import build_graph

    graphs = [build_graph(fc, None if scs is None else scs[i], top_k=top_k, threshold=threshold,
                          fisher_z=True) for i, fc in enumerate(fcs)]
    return graphs, [to_data(g, int(lbl)) for g, lbl in zip(graphs, y)]


def region_names(data_dir: Path, atlas: str, n: int) -> list[str]:
    p = data_dir / "regions.json"
    names = json.loads(p.read_text()) if p.exists() else []
    if len(names) != n:
        try:
            from graph.atlas import load_atlas
            names = load_atlas(atlas)[1]
        except Exception:
            names = []
    if len(names) != n:
        print(f"WARNING: no region names for {n} regions; using 'Region i'")
        names = [f"Region {i + 1}" for i in range(n)]
    return names


def metrics(y: np.ndarray, p: np.ndarray, n_classes: int) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

    pred = p.argmax(axis=1)
    labels = list(range(n_classes))
    cm = confusion_matrix(y, pred, labels=labels)
    recall = (cm.diagonal() / cm.sum(axis=1).clip(min=1)).tolist()
    try:
        if n_classes == 2:
            auc = roc_auc_score(y, p[:, 1])
        else:
            auc = roc_auc_score(y, p, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        auc = float("nan")
    return {"sessions": int(len(y)), "accuracy": float((pred == y).mean()),
            "balanced_accuracy": float(np.mean(recall)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
            "auc": float(auc), "per_class_recall": recall, "confusion_matrix": cm.tolist()}


def agg(folds: list[dict], key: str) -> dict:
    v = [f[key] for f in folds if not np.isnan(f[key])]
    return {"mean": round(float(np.mean(v)), 4) if v else None,
            "std": round(float(np.std(v)), 4) if v else None}


def summarise(folds: list[dict]) -> dict:
    return {
        **{k: agg(folds, k) for k in ("accuracy", "balanced_accuracy", "macro_f1", "auc")},
        "per_class_recall": np.mean([f["per_class_recall"] for f in folds], axis=0).round(4).tolist(),
        "confusion_matrix": np.sum([f["confusion_matrix"] for f in folds], axis=0).tolist(),
        "per_fold": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()} for f in folds],
    }


def class_weights(y: np.ndarray, n_classes: int):
    import torch

    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    return torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32)


def predict(model, data_list) -> np.ndarray:
    import torch
    from torch_geometric.loader import DataLoader

    model.eval()
    out = []
    with torch.no_grad():
        for b in DataLoader(data_list, batch_size=64):
            out.append(torch.softmax(model(b.x, b.edge_index, b.edge_attr, b.batch), dim=1).numpy())
    return np.vstack(out)


def train_model(train, args, in_dim, hidden, n_classes, val=None, epochs=None):
    """Train; with `val`, early-stop on its loss. Returns (model, best_epoch, val_auc)."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader

    from graph.gat import GraphGAT

    model = GraphGAT(in_dim, hidden, args.heads, args.layers, args.dropout, n_classes=n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    w = class_weights(np.array([int(d.y) for d in train]), n_classes)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    for epoch in range(1, (epochs or args.epochs) + 1):
        model.train()
        for b in loader:
            opt.zero_grad()
            F.cross_entropy(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y, weight=w).backward()
            opt.step()
        if val is None:
            continue
        p = predict(model, val)
        yv = np.array([int(d.y) for d in val])
        loss = float(F.nll_loss(torch.log(torch.from_numpy(p).clamp_min(1e-9)), torch.from_numpy(yv), weight=w))
        if loss < best - 1e-4:
            best, best_epoch, bad = loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_auc = float("nan")
    if val is not None:
        yv = np.array([int(d.y) for d in val])
        val_auc = metrics(yv, predict(model, val), n_classes)["auc"]
    return model, (best_epoch or epochs or args.epochs), val_auc


def baseline_model(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C, class_weight="balanced"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="processed_graph/ with manifest.csv")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "artifacts"))
    ap.add_argument("--label-mode", choices=list(LABELS), default="binary")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--use-sc", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--inner-val", type=float, default=0.2)
    ap.add_argument("--sweep", action="store_true",
                    help="pick top-k / hidden per outer fold on the inner split (nested, honest)")
    ap.add_argument("--baseline-C", type=float, default=0.1)
    ap.add_argument("--min-sessions", type=int, default=30)
    ap.add_argument("--deploy", choices=["auto", "gat", "logreg"], default="auto",
                    help="auto (default) deploys the CV winner; forcing a model is for testing and is "
                         "recorded as such in graph_metrics.json")
    args = ap.parse_args()
    if args.layers not in (2, 3):
        raise SystemExit("--layers must be 2 or 3")

    from sklearn.model_selection import StratifiedGroupKFold

    from graph import gat_numpy, linear_numpy
    from graph.gat import export_weights

    classes, short = LABELS[args.label_mode]
    n_classes = len(classes)
    set_seed(args.seed)
    data_dir, out_dir = Path(args.data_dir).expanduser(), Path(args.out_dir).expanduser()
    fcs, scs, y, subjects, meta = load_matrices(data_dir, args.use_sc, args.label_mode)
    n_regions = fcs.shape[1]
    counts = {short[c]: int((y == c).sum()) for c in range(n_classes)}
    print(f"{len(y)} sessions, {len(set(subjects))} subjects, {n_regions} regions, task {args.label_mode}, "
          f"classes {counts}, SC {'used' if meta['with_sc'] else 'absent (FC only)'}")

    configs = SWEEP if args.sweep else [{"top_k": args.top_k, "hidden": args.hidden}]
    data_by_k = {k: build_data(fcs, scs, y, k, args.threshold) for k in sorted({c["top_k"] for c in configs})}
    in_dim = data_by_k[configs[0]["top_k"]][1][0].x.shape[1]
    X_base = linear_numpy.fc_features(fcs)

    outer = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    gat_folds, base_folds, chosen = [], [], []
    t0 = time.time()
    for i, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y, subjects)):
        inner_k = max(2, int(round(1 / args.inner_val)))
        in_tr, in_va = next(StratifiedGroupKFold(n_splits=inner_k, shuffle=True, random_state=args.seed + i)
                            .split(np.zeros(len(tr)), y[tr], subjects[tr]))
        best = None
        for cfg in configs:
            set_seed(args.seed + i)
            data = data_by_k[cfg["top_k"]][1]
            model, epoch, inner_auc = train_model([data[j] for j in tr[in_tr]], args, in_dim, cfg["hidden"],
                                                  n_classes, val=[data[j] for j in tr[in_va]])
            if best is None or (inner_auc > best[2]):
                best = (cfg, model, inner_auc, epoch)
        cfg, model, inner_auc, epoch = best
        chosen.append({**cfg, "epoch": epoch, "inner_auc": round(inner_auc, 4)})
        gat_folds.append(metrics(y[te], predict(model, [data_by_k[cfg["top_k"]][1][j] for j in te]), n_classes))

        base = baseline_model(args.baseline_C).fit(X_base[tr], y[tr])
        pb = np.zeros((len(te), n_classes))
        pb[:, base.classes_] = base.predict_proba(X_base[te])
        base_folds.append(metrics(y[te], pb, n_classes))
        print(f"[fold {i}] GAT acc {gat_folds[-1]['accuracy']:.3f} bal {gat_folds[-1]['balanced_accuracy']:.3f} "
              f"AUC {gat_folds[-1]['auc']:.3f} | baseline acc {base_folds[-1]['accuracy']:.3f} "
              f"bal {base_folds[-1]['balanced_accuracy']:.3f} AUC {base_folds[-1]['auc']:.3f} "
              f"(chosen top-k {cfg['top_k']}, hidden {cfg['hidden']}, epoch {epoch})", flush=True)

    cv, baseline = summarise(gat_folds), summarise(base_folds)
    show = ("accuracy", "balanced_accuracy", "macro_f1", "auc")
    print("\nGAT     :", {k: cv[k] for k in show})
    print("baseline:", {k: baseline[k] for k in show})
    majority = max(counts.values()) / len(y)
    print(f"majority-class accuracy: {majority:.3f}; chance AUC 0.5")

    gat_auc, base_auc = cv["auc"]["mean"] or 0.0, baseline["auc"]["mean"] or 0.0
    deploy = "gat" if gat_auc > base_auc else "logreg"
    reason = (f"GAT mean CV AUC {gat_auc:.3f} > baseline {base_auc:.3f}" if deploy == "gat" else
              f"the GAT did not beat the baseline (CV AUC {gat_auc:.3f} vs {base_auc:.3f}), so the "
              "logistic-regression model is deployed")
    if args.deploy != "auto" and args.deploy != deploy:
        deploy = args.deploy
        reason = f"FORCED with --deploy {deploy} (CV AUC GAT {gat_auc:.3f}, baseline {base_auc:.3f})"
    print(f"deploying: {deploy} — {reason}")

    # ---- deployable artifacts --------------------------------------------
    if len(y) < args.min_sessions or min(counts.values()) == 0:
        raise SystemExit(f"NOT writing artifacts: {len(y)} sessions / classes {counts} is below "
                         f"--min-sessions {args.min_sessions} or missing a class")
    names = region_names(data_dir, meta["atlas"], n_regions)
    config = {"model_type": deploy, "atlas": meta["atlas"], "region_names": names, "n_regions": n_regions,
              "with_sc": meta["with_sc"], "fisher_z": True, "label_mode": args.label_mode,
              "classes": classes, "class_short": short, "version": VERSION}
    if deploy == "gat":
        top_cfg = Counter((c["top_k"], c["hidden"]) for c in chosen).most_common(1)[0][0]
        final_epochs = int(np.median([c["epoch"] for c in chosen if (c["top_k"], c["hidden"]) == top_cfg]))
        config.update({"top_k": top_cfg[0], "threshold": args.threshold, "hidden": top_cfg[1],
                       "heads": args.heads, "layers": args.layers})
        set_seed(args.seed)
        graphs, data = data_by_k.get(top_cfg[0]) or build_data(fcs, scs, y, top_cfg[0], args.threshold)
        final, _, _ = train_model(data, args, in_dim, top_cfg[1], n_classes, epochs=final_epochs)
        final.eval()
        weights = export_weights(final)
        worst = max(float(np.abs(predict(final, [d])[0] - gat_numpy.forward(
            g, {k: v.astype(np.float64) for k, v in weights.items()}, config)["probabilities"]).max())
            for g, d in list(zip(graphs, data))[:20])
        deployed_desc = (f"GAT retrained on all {len(y)} sessions for {final_epochs} epochs "
                         f"(top-k {top_cfg[0]}, hidden {top_cfg[1]})")
    else:
        final = baseline_model(args.baseline_C).fit(X_base, y)
        scaler, lr = final.named_steps["standardscaler"], final.named_steps["logisticregression"]
        weights = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                   "coef": lr.coef_, "intercept": lr.intercept_}
        worst = max(float(np.abs(final.predict_proba(X_base[j:j + 1])[0]
                                 - linear_numpy.forward(fcs[j], weights)["probabilities"]).max())
                    for j in range(min(20, len(y))))
        deployed_desc = f"logistic regression retrained on all {len(y)} sessions (C={args.baseline_C})"
    if worst > 1e-4:
        raise SystemExit(f"numpy forward pass disagrees with the trained model by {worst:.2e}; not writing")
    print(f"numpy parity: max |dp| = {worst:.1e}")

    sess_counts = {str(c): int((y == c).sum()) for c in range(n_classes)}
    task = "Normal vs Impaired" if args.label_mode == "binary" else "Normal / MCI / Dementia"
    metrics_json = {
        "model": ("Graph Attention Network" if deploy == "gat" else
                  "Logistic regression on functional connectivity (outperformed the GAT in CV)"),
        "model_type": deploy,
        "task": f"{task} from brain connectivity (rs-fMRI FC" + (" + DTI SC)" if meta["with_sc"] else ", no DTI)"),
        "classes": classes,
        "dataset": {"name": "OASIS-3 resting-state fMRI" + (" + DTI" if meta["with_sc"] else ""),
                    "subjects": int(len(set(subjects))), "sessions": int(len(y)),
                    "session_class_counts": sess_counts, "atlas": meta["atlas"], "regions": n_regions,
                    "split": f"subject-grouped, stratified {args.folds}-fold cross-validation",
                    "majority_class_accuracy": round(majority, 4),
                    "mean_fd_mm": round(float(np.mean(meta["mean_fd"])), 3) if meta["mean_fd"] else None},
        "graph": {"with_sc": meta["with_sc"], "sweep": args.sweep, "chosen_per_fold": chosen},
        # cv = the DEPLOYED model's numbers; the other model is kept for comparison.
        "cv": {"folds": args.folds, **(cv if deploy == "gat" else baseline)},
        "gat_cv": {"folds": args.folds, **cv},
        "baseline": {"model": "logistic regression on Fisher-z FC upper triangle (same folds)", **baseline},
        "deployed": {"which": deployed_desc, "reason": reason},
        "caveats": [
            "Labels come from CDR staging, not a clinical diagnosis; CDR 0.5 is only a proxy for MCI.",
            "Single cohort (OASIS-3): scanners, protocol and population are not independently validated.",
            f"Cortical atlas only ({meta['atlas'] or 'atlas'}): the hippocampus is not a node.",
            "Trained without DTI: structural connectivity inputs are ignored." if not meta["with_sc"]
            else "Structural connectivity is a precomputed DTI matrix in v1.",
            "Uploads must be motion-corrected and in MNI space; native-space BOLD gives meaningless graphs.",
            "Highlighted connections show what drove the score, not which connection is abnormal.",
            "A screening aid for research and education, not a diagnosis.",
        ],
        "version": VERSION,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "graph_model.npz", **{k: np.asarray(v) for k, v in weights.items()})
    (out_dir / "graph_config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "graph_metrics.json").write_text(json.dumps(metrics_json, indent=2))
    print(f"saved {out_dir}/graph_model.npz, graph_config.json, graph_metrics.json "
          f"({deploy}; {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
