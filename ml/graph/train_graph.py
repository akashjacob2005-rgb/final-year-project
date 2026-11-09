"""Train the connectivity-graph GAT (Normal / MCI / Dementia) with honest CV.

    python ml/graph/train_graph.py --data-dir ~/Downloads/7th_sem/PP-2/data/processed_graph

Input: processed_graph/manifest.csv + fc.npy (+ sc.npy) per session, from
ml/graph/preprocess.py. Labels: 0 = CDR 0, 1 = CDR 0.5, 2 = CDR >= 1.

Evaluation
  * subject-grouped, label-stratified K-fold (StratifiedGroupKFold): no
    subject's sessions appear in both training and test folds.
  * early stopping on an INNER split of the training subjects only; the outer
    fold is scored exactly once (the reported numbers never select an epoch).
  * class-weighted cross-entropy, fixed seed.
  * baseline on the SAME folds: logistic regression on the Fisher-z upper
    triangle of the FC matrix. The GAT has to beat this to justify itself.

Deployment
  The final model is retrained on all subjects for the median best epoch
  found in CV, exported to numpy (graph_model.npz + graph_config.json), and
  checked against the torch forward pass (max |dp| < 1e-4) before anything is
  written. graph_metrics.json mirrors mri_metrics.json's shape.

Guard: artifacts are refused for fewer than --min-sessions sessions or when a
class is missing — a model trained on a handful of scans must never be served
as if it were evaluated. Use --out-dir elsewhere for smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED = 42
VERSION = "1.0.0"
CLASS_SHORT = ["Normal", "MCI", "Dementia"]


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(data_dir: Path, use_sc: str, top_k, threshold):
    from graph.graph_build import build_graph

    rows = list(csv.DictReader(open(data_dir / "manifest.csv")))
    if not rows:
        raise SystemExit(f"{data_dir / 'manifest.csv'} lists no sessions")
    have_sc = all(r.get("sc") for r in rows)
    if use_sc == "yes" and not have_sc:
        raise SystemExit("--use-sc yes but some sessions have no sc.npy")
    with_sc = use_sc == "yes" or (use_sc == "auto" and have_sc)
    graphs, fcs, labels, subjects, sessions = [], [], [], [], []
    for r in rows:
        fc = np.load(data_dir / r["fc"])
        sc = np.load(data_dir / r["sc"]) if with_sc else None
        graphs.append(build_graph(fc, sc, top_k=top_k, threshold=threshold))
        fcs.append(fc)
        labels.append(int(r["label"]))
        subjects.append(r["subject_id"])
        sessions.append(r["session"])
    meta = {"atlas": rows[0].get("atlas", ""), "with_sc": with_sc,
            "mean_fd": [float(r["mean_fd"]) for r in rows if r.get("mean_fd")]}
    return graphs, np.array(fcs), np.array(labels), np.array(subjects), sessions, meta


def region_names(data_dir: Path, atlas: str, n: int) -> list[str]:
    p = data_dir / "regions.json"
    if p.exists():
        names = json.loads(p.read_text())
    else:
        try:
            from graph.atlas import load_atlas
            names = load_atlas(atlas)[1]
        except Exception:
            names = []
    if len(names) != n:
        print(f"WARNING: no region names for {n} regions; using 'Region i'")
        names = [f"Region {i + 1}" for i in range(n)]
    return names


def fc_features(fcs: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(fcs.shape[1], k=1)
    return np.arctanh(np.clip(fcs[:, iu[0], iu[1]], -0.999, 0.999))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

    pred = p.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    recall = (cm.diagonal() / cm.sum(axis=1).clip(min=1)).tolist()
    present = sorted(set(y.tolist()))
    try:
        if len(present) == 3:
            auc = roc_auc_score(y, p, multi_class="ovr", average="macro")
        elif len(present) == 2:  # a fold missing a class: AUC over the two present
            auc = roc_auc_score((y == present[1]).astype(int), p[:, present[1]])
        else:
            auc = float("nan")
    except ValueError:
        auc = float("nan")
    return {"sessions": int(len(y)), "accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
            "auc": float(auc), "per_class_recall": recall, "confusion_matrix": cm.tolist()}


def agg(folds: list[dict], key: str) -> dict:
    v = [f[key] for f in folds if not np.isnan(f[key])]
    return {"mean": round(float(np.mean(v)), 4) if v else None,
            "std": round(float(np.std(v)), 4) if v else None}


def summarise(folds: list[dict]) -> dict:
    return {
        "accuracy": agg(folds, "accuracy"), "macro_f1": agg(folds, "macro_f1"), "auc": agg(folds, "auc"),
        "per_class_recall": np.mean([f["per_class_recall"] for f in folds], axis=0).round(4).tolist(),
        "confusion_matrix": np.sum([f["confusion_matrix"] for f in folds], axis=0).tolist(),
        "per_fold": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()} for f in folds],
    }


def class_weights(y: np.ndarray):
    import torch

    counts = np.bincount(y, minlength=3).astype(float)
    counts[counts == 0] = 1.0
    return torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32)


def predict(model, data_list) -> np.ndarray:
    import torch
    from torch_geometric.loader import DataLoader

    model.eval()
    out = []
    with torch.no_grad():
        for b in DataLoader(data_list, batch_size=64):
            out.append(torch.softmax(model(b.x, b.edge_index, b.edge_attr, b.batch), dim=1).numpy())
    return np.vstack(out)


def train_model(train, args, in_dim, val=None, epochs=None):
    """Train; with `val`, early-stop on its loss and return (model, best_epoch)."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader

    from graph.gat import GraphGAT

    model = GraphGAT(in_dim, args.hidden, args.heads, args.layers, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    w = class_weights(np.array([int(d.y) for d in train]))
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
    return model, (best_epoch or epochs or args.epochs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="processed_graph/ with manifest.csv")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "artifacts"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--use-sc", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--inner-val", type=float, default=0.2)
    ap.add_argument("--min-sessions", type=int, default=30,
                    help="refuse to write deployable artifacts below this many sessions")
    args = ap.parse_args()
    if args.layers not in (2, 3):
        raise SystemExit("--layers must be 2 or 3")

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from graph import gat_numpy
    from graph.gat import CLASSES, export_weights, to_data

    set_seed(args.seed)
    data_dir, out_dir = Path(args.data_dir).expanduser(), Path(args.out_dir).expanduser()
    graphs, fcs, y, subjects, sessions, meta = load_dataset(data_dir, args.use_sc, args.top_k, args.threshold)
    n_regions = fcs.shape[1]
    counts = {CLASS_SHORT[c]: int((y == c).sum()) for c in range(3)}
    print(f"{len(y)} sessions, {len(set(subjects))} subjects, {n_regions} regions, "
          f"classes {counts}, SC {'used' if meta['with_sc'] else 'absent (FC only)'}")
    data = [to_data(g, int(lbl)) for g, lbl in zip(graphs, y)]
    in_dim = data[0].x.shape[1]
    X_base = fc_features(fcs)

    outer = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    gat_folds, base_folds, best_epochs = [], [], []
    t0 = time.time()
    for i, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y, subjects)):
        set_seed(args.seed + i)
        inner_k = max(2, int(round(1 / args.inner_val)))
        in_tr, in_va = next(StratifiedGroupKFold(n_splits=inner_k, shuffle=True, random_state=args.seed + i)
                            .split(np.zeros(len(tr)), y[tr], subjects[tr]))
        model, best_epoch = train_model([data[j] for j in tr[in_tr]], args, in_dim,
                                        val=[data[j] for j in tr[in_va]])
        best_epochs.append(best_epoch)
        gat_folds.append(metrics(y[te], predict(model, [data[j] for j in te])))

        base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced"))
        base.fit(X_base[tr], y[tr])
        pb = np.zeros((len(te), 3))
        pb[:, base.classes_] = base.predict_proba(X_base[te])
        base_folds.append(metrics(y[te], pb))
        print(f"[fold {i}] GAT acc {gat_folds[-1]['accuracy']:.3f} F1 {gat_folds[-1]['macro_f1']:.3f} "
              f"AUC {gat_folds[-1]['auc']:.3f} | baseline acc {base_folds[-1]['accuracy']:.3f} "
              f"F1 {base_folds[-1]['macro_f1']:.3f} (best epoch {best_epoch})")

    cv, baseline = summarise(gat_folds), summarise(base_folds)
    print("\nGAT     :", {k: cv[k] for k in ("accuracy", "macro_f1", "auc")})
    print("baseline:", {k: baseline[k] for k in ("accuracy", "macro_f1", "auc")})

    # ---- deployable artifacts --------------------------------------------
    if len(y) < args.min_sessions or min(counts.values()) == 0:
        raise SystemExit(f"NOT writing artifacts: {len(y)} sessions / classes {counts} is below "
                         f"--min-sessions {args.min_sessions} or missing a class")
    final_epochs = int(np.median(best_epochs))
    set_seed(args.seed)
    final, _ = train_model(data, args, in_dim, epochs=final_epochs)
    final.eval()
    weights = export_weights(final)
    config = {"atlas": meta["atlas"], "region_names": region_names(data_dir, meta["atlas"], n_regions),
              "n_regions": n_regions, "top_k": args.top_k, "threshold": args.threshold,
              "with_sc": meta["with_sc"], "hidden": args.hidden, "heads": args.heads,
              "layers": args.layers, "classes": CLASSES, "version": VERSION}

    # Parity: the numpy forward pass the backend runs must equal torch.
    worst = 0.0
    for g, d in list(zip(graphs, data))[:20]:
        pt = predict(final, [d])[0]
        pn = gat_numpy.forward(g, {k: v.astype(np.float64) for k, v in weights.items()}, config)["probabilities"]
        worst = max(worst, float(np.abs(pt - pn).max()))
    if worst > 1e-4:
        raise SystemExit(f"numpy forward pass disagrees with torch by {worst:.2e}; not writing artifacts")
    print(f"numpy/torch parity: max |dp| = {worst:.1e}")

    sess_counts = {str(c): int((y == c).sum()) for c in range(3)}
    metrics_json = {
        "model": f"Graph Attention Network ({args.layers} x GATConv, {args.heads} heads, edge_dim=2) "
                 f"+ mean/attention readout, 3-class head",
        "task": "3-class classification of brain connectivity graphs (rs-fMRI FC"
                + (" + DTI SC" if meta["with_sc"] else ", no DTI") + ")",
        "classes": CLASSES,
        "dataset": {"name": "OASIS-3 resting-state fMRI" + (" + DTI" if meta["with_sc"] else ""),
                    "subjects": int(len(set(subjects))), "sessions": int(len(y)),
                    "session_class_counts": sess_counts, "atlas": meta["atlas"], "regions": n_regions,
                    "split": f"subject-grouped, stratified {args.folds}-fold cross-validation",
                    "mean_fd_mm": round(float(np.mean(meta["mean_fd"])), 3) if meta["mean_fd"] else None},
        "graph": {"edges": f"top-{args.top_k} |FC| per node" if args.top_k else "",
                  "threshold": args.threshold, "with_sc": meta["with_sc"]},
        "cv": {"folds": args.folds, **cv},
        "baseline": {"model": "logistic regression on Fisher-z FC upper triangle (same folds)", **baseline},
        "deployed": {"which": f"retrained on all {len(y)} sessions for {final_epochs} epochs "
                              "(median early-stopping epoch from the inner splits)"},
        "training": {"optimizer": f"AdamW lr={args.lr} wd={args.weight_decay}", "class_weighted_loss": True,
                     "early_stopping": f"inner {args.inner_val:.0%} subject split, patience {args.patience}",
                     "seed": args.seed, "minutes": round((time.time() - t0) / 60, 1)},
        "caveats": [
            "Labels come from CDR staging (0 / 0.5 / >=1), not a clinical diagnosis; CDR 0.5 is a proxy for MCI.",
            "Single cohort (OASIS-3): scanners, protocol and population are not independently validated.",
            f"Results depend on the parcellation ({meta['atlas'] or 'atlas'}); another atlas needs retraining.",
            "Structural connectivity is a precomputed DTI matrix in v1" if meta["with_sc"]
            else "Trained without DTI: structural connectivity inputs are ignored (all zero).",
            "Uploads must be motion-corrected and in MNI space; native-space BOLD gives meaningless graphs.",
            "Attention shows where the model looked, not which connection is abnormal.",
            "A screening aid for research and education, not a diagnosis.",
        ],
        "version": VERSION,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "graph_model.npz", **weights)
    (out_dir / "graph_config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "graph_metrics.json").write_text(json.dumps(metrics_json, indent=2))
    torch.save({"state_dict": final.state_dict(), "config": config, "in_dim": in_dim}, out_dir / "graph_model.pt")
    print(f"saved {out_dir}/graph_model.npz, graph_config.json, graph_metrics.json, graph_model.pt")


if __name__ == "__main__":
    main()
