"""Train the OASIS-3 MRI slice classifier.

Input:  slice export produced by make_slices.py (manifest_v2.csv, 16 axial
        slices per MR session) or the original 8-slice export (manifest.csv).
        Labels derive from the CDR score closest in time to the scan:
            0 -> CDR 0.0   cognitively normal
            1 -> CDR 0.5   very mild impairment
            2 -> CDR >= 1  dementia

Label modes:
    binary (default)  0 = Normal (CDR 0), 1 = Impaired (CDR >= 0.5)
    three             the raw 3-class problem, kept for the report's comparison

Model:  ResNet-18 pretrained on ImageNet; by default only layer4 + head are
        fine-tuned (126-subject training sets overfit a fully unfrozen net).

Evaluation is BY SUBJECT and BY SESSION: every slice of a person lands in one
split only, and a session's prediction is the mean of its slice probabilities
(with horizontal-flip test-time augmentation). With --folds K the script runs
subject-stratified K-fold cross-validation and reports mean +/- std across
folds — the honest headline numbers for a dataset this small. The deployed
artifact is the fold-0 model (trained on folds 1..K-1, early-stopped on fold 0).

Usage:
    python ml/mri/train_mri.py --data-dir /path/to/data/processed_v2 --folds 5
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 42
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = {
    "three": ["CDR 0 (normal)", "CDR 0.5 (very mild)", "CDR >=1 (dementia)"],
    "binary": ["No impairment signs (CDR 0)", "Signs of impairment (CDR >= 0.5)"],
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SliceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path, train: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        self.root = root
        base = [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.RandomAffine(0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
        self.tf = transforms.Compose((aug + base) if train else base)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int):
        row = self.frame.iloc[i]
        img = Image.open(self.root / row["relpath"]).convert("L")
        return self.tf(img), int(row["label"]), i


def load_manifest(data_dir: Path, label_mode: str) -> pd.DataFrame:
    manifest = data_dir / "manifest_v2.csv"
    if not manifest.exists():
        manifest = data_dir / "manifest.csv"
    df = pd.read_csv(manifest)
    df["relpath"] = df["path"].str.replace(r"^.*?slices/", "slices/", regex=True)
    df["session"] = df["relpath"].str.split("/").str[1]
    if label_mode == "binary":
        df["label"] = (df["label"] >= 1).astype(int)
    missing = [p for p in df["relpath"] if not (data_dir / p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} slice files missing, e.g. {missing[0]}")
    return df


def subject_folds(df: pd.DataFrame, k: int, seed: int = SEED) -> list[set]:
    """K subject groups, stratified by each subject's max label."""
    subj_label = df.groupby("subject_id")["label"].max()
    rng = random.Random(seed)
    folds: list[list] = [[] for _ in range(k)]
    for label in sorted(subj_label.unique()):
        subjects = sorted(subj_label[subj_label == label].index)
        rng.shuffle(subjects)
        for i, s in enumerate(subjects):
            folds[i % k].append(s)
    return [set(f) for f in folds]


def make_model(n_classes: int, finetune: str) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    if finetune == "layer4":
        for name, p in model.named_parameters():
            if not (name.startswith("layer4") or name.startswith("fc")):
                p.requires_grad = False
    return model


@torch.no_grad()
def session_eval(model, loader, frame, device, n_classes: int):
    """Session-level metrics with horizontal-flip test-time augmentation."""
    model.eval()
    probs = np.zeros((len(frame), n_classes), dtype=np.float64)
    for x, _, idx in loader:
        x = x.to(device)
        p = torch.softmax(model(x), dim=1)
        p = p + torch.softmax(model(torch.flip(x, dims=[3])), dim=1)
        probs[idx.numpy()] = (p / 2).cpu().numpy()

    by_session = defaultdict(list)
    for i, row in frame.reset_index(drop=True).iterrows():
        by_session[row["session"]].append(i)

    y_true, y_prob = [], []
    for _, rows in sorted(by_session.items()):
        y_true.append(int(frame.iloc[rows[0]]["label"]))
        y_prob.append(probs[rows].mean(axis=0))
    y_true = np.array(y_true)
    y_prob = np.vstack(y_prob)
    y_pred = y_prob.argmax(axis=1)

    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_recall = (cm.diagonal() / cm.sum(axis=1).clip(min=1)).tolist()
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    avg = "binary" if n_classes == 2 else "macro"
    return {
        "sessions": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1": float(f1_score(y_true, y_pred, average=avg)),
        "auc": float(auc),
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
    }


def train_one(train_frame, val_frame, data_dir, device, args, n_classes, tag=""):
    """Train a model on train_frame, early-stopping on val_frame session AUC."""
    loaders = {}
    for name, frame, is_train in (("train", train_frame, True), ("val", val_frame, False)):
        ds = SliceDataset(frame, data_dir, train=is_train)
        loaders[name] = DataLoader(ds, batch_size=args.batch_size, shuffle=is_train, num_workers=2)

    model = make_model(n_classes, args.finetune).to(device)

    counts = train_frame.groupby("label").size().reindex(range(n_classes), fill_value=1)
    weights = torch.tensor((counts.sum() / (n_classes * counts)).values, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=1e-4)

    best_score, best_state, best_val, bad, history = -1.0, None, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y, _ in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
        train_loss = running / len(train_frame)

        val = session_eval(model, loaders["val"], val_frame, device, n_classes)
        score = val["auc"] if not np.isnan(val["auc"]) else val["macro_f1"]
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_accuracy": round(val["accuracy"], 4), "val_auc": round(val["auc"], 4)})
        print(f"{tag}epoch {epoch:02d}  loss {train_loss:.4f}  val acc {val['accuracy']:.3f}  val auc {val['auc']:.3f}")

        if score > best_score:
            best_score, bad, best_val = score, 0, val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"{tag}early stop at epoch {epoch}")
                break

    return best_state, best_val, history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dir containing manifest(_v2).csv and slices/")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "artifacts"))
    ap.add_argument("--label-mode", choices=["binary", "three"], default="binary")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--finetune", choices=["all", "layer4"], default="layer4")
    args = ap.parse_args()

    set_seed()
    device = pick_device(args.device)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = CLASS_NAMES[args.label_mode]
    n_classes = len(classes)
    df = load_manifest(data_dir, args.label_mode)
    n_slices = int(df.groupby("session").size().mode()[0])
    sess_labels = df.groupby("session")["label"].first()
    print(f"device={device}  label_mode={args.label_mode}  slices/session={n_slices}")
    print(f"{df.subject_id.nunique()} subjects, {df.session.nunique()} sessions, {len(df)} slices")
    print("session class counts:", sess_labels.value_counts().sort_index().to_dict())

    folds = subject_folds(df, args.folds)
    fold_results, deployed_state, deployed_val = [], None, None
    for i, fold_subjects in enumerate(folds):
        val_frame = df[df.subject_id.isin(fold_subjects)]
        train_frame = df[~df.subject_id.isin(fold_subjects)]
        set_seed(SEED + i)
        state, val, _ = train_one(train_frame, val_frame, data_dir, device, args,
                                  n_classes, tag=f"[fold {i}] ")
        fold_results.append(val)
        print(f"[fold {i}] sessions={val['sessions']} acc={val['accuracy']:.3f} "
              f"f1={val['f1']:.3f} auc={val['auc']:.3f}")
        if i == 0:
            deployed_state, deployed_val = state, val

    def agg(key):
        vals = [f[key] for f in fold_results]
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    cv = {
        "folds": args.folds,
        "accuracy": agg("accuracy"),
        "f1": agg("f1"),
        "macro_f1": agg("macro_f1"),
        "auc": agg("auc"),
        "per_fold": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items() if k != "confusion_matrix"}
            for f in fold_results
        ],
    }
    print("\nCV summary:", json.dumps({k: cv[k] for k in ("accuracy", "f1", "auc")}, indent=2))

    # ---- artifacts -------------------------------------------------------
    torch.save({"state_dict": deployed_state, "img_size": IMG_SIZE, "classes": classes},
               out_dir / "mri_model.pt")

    metrics = {
        "model": f"ResNet-18 (ImageNet transfer learning), {n_classes}-class head, finetune={args.finetune}",
        "task": f"{args.label_mode} classification of T1w axial MRI slices, aggregated per session",
        "label_mode": args.label_mode,
        "classes": classes,
        "n_slices": n_slices,
        "dataset": {
            "name": "OASIS-3 (T1w MRI), labels = CDR nearest the scan date",
            "subjects": int(df.subject_id.nunique()),
            "sessions": int(df.session.nunique()),
            "slices": int(len(df)),
            "session_class_counts": {str(k): int(v) for k, v in sess_labels.value_counts().sort_index().items()},
            "split": f"subject-stratified {args.folds}-fold cross-validation",
        },
        "aggregation": f"session probability = mean of its {n_slices} slice probabilities, with hflip TTA",
        "cv": cv,
        "deployed": {
            "which": "fold-0 model (trained on the other folds, early-stopped on fold 0)",
            "fold0_validation": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in deployed_val.items()},
        },
        "training": {"optimizer": f"Adam lr={args.lr} wd=1e-4", "class_weighted_loss": True,
                     "augmentation": "hflip, rot8, translate5%, jitter", "seed": SEED},
        "caveats": [
            "Labels come from CDR staging, not biopsy-confirmed diagnosis.",
            "Slices are 2D projections of a 3D volume; a 3D CNN could use more context.",
            "OASIS-3 is a largely North-American research cohort; other scanners/populations unverified.",
            "Out-of-distribution inputs (non-brain, CT, non-T1w) are not rejected.",
            "Reported separately from the behavioural risk score: no dataset pairs MRI with the app's six tests.",
        ],
        "version": "2.0.0",
    }
    (out_dir / "mri_metrics.json").write_text(json.dumps(metrics, indent=2))

    from export_onnx import export as export_onnx

    export_onnx(out_dir)
    print(f"saved: {out_dir}/mri_model.pt, mri_model.onnx, mri_metrics.json")


if __name__ == "__main__":
    main()
