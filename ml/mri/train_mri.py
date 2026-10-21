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

Model:  selected with --arch (all evaluated under the SAME folds/seed/eval):
    resnet18 (default)  ImageNet CNN; only layer4 + head fine-tuned
                        (126-subject training sets overfit a fully unfrozen
                        net). This is the deployed architecture.
    vit_b16             google/vit-base-patch16-224-in21k (HF transformers);
                        last 2 encoder blocks + head fine-tuned.
    dinov2_s            facebook/dinov2-small (HF transformers); frozen
                        backbone + linear head on the CLS token — the
                        literature-backed small-data recipe. --finetune all
                        additionally unfreezes the last 2 blocks.
Transformer archs write arch-suffixed artifacts (mri_metrics_<arch>.json,
mri_model_<arch>.pt) and never touch the deployed mri_model.pt / ONNX export.
--save-preds dumps per-session probabilities per fold for offline ensembling
(see ensemble_eval.py).

Evaluation is BY SUBJECT and BY SESSION: every slice of a person lands in one
split only, and a session's prediction is the mean of its slice probabilities
(with horizontal-flip test-time augmentation). With --folds K the script runs
subject-stratified K-fold cross-validation and reports mean +/- std across
folds — the honest headline numbers for a dataset this small.

Two modes:

  LEGACY (default, no --experiment): behaves exactly as before. Early-stops on
  the reported fold (optimistic!), keeps the fold-0 model as the deployed
  artifact and, for resnet18, re-exports ml/artifacts/mri_model.onnx.

  EXPERIMENT (--experiment NAME): writes everything to
  ml/experiments/NAME/ and never touches ml/artifacts/. Designed for honest
  comparisons:
    --sched cosine --no-early-stop   fixed epoch budget, last epoch reported
    --inner-val 0.15                 early-stop / pick epochs on an INNER
                                     split of the training subjects instead
    --repeats R                      repeated CV with different fold seeds
    --seeds N                        N models per fold (seed ensemble)
    --final-fit                      retrain on ALL subjects with the same
                                     recipe -> final/model.onnx for promote.py
  Outputs: config.json, summary.json, preds_sessions.csv,
  preds_slices.csv (--save-slice-preds), history.json, final/.
  Analyse with compare_experiments.py; deploy with promote.py.

Usage:
    python ml/mri/train_mri.py --data-dir /path/to/data/processed_v2 --folds 5
    python ml/mri/train_mri.py --data-dir ... --experiment e01_baseline_honest \
        --sched cosine --no-early-stop --epochs 25 --repeats 3 --save-slice-preds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
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

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "ml" / "experiments"

# Per-arch config: HF checkpoint (None = torchvision resnet18), the input
# normalisation the checkpoint was pretrained with, and a sensible default lr
# (head-only training wants a larger step than partial fine-tuning).
ARCHS = {
    "resnet18": {"hf": None, "norm": (IMAGENET_MEAN, IMAGENET_STD), "lr": 3e-4},
    "vit_b16": {"hf": "google/vit-base-patch16-224-in21k",
                "norm": ([0.5] * 3, [0.5] * 3), "lr": 1e-4},
    "dinov2_s": {"hf": "facebook/dinov2-small",
                 "norm": (IMAGENET_MEAN, IMAGENET_STD), "lr": 1e-3},
}

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


class RandomGamma:
    """Intensity gamma jitter on a greyscale PIL image (MRI contrast varies by
    scanner/sequence far more than natural-image brightness does)."""

    def __init__(self, lo: float = 0.8, hi: float = 1.25) -> None:
        self.lo, self.hi = lo, hi

    def __call__(self, img: Image.Image) -> Image.Image:
        g = random.uniform(self.lo, self.hi)
        lut = [int(255.0 * ((i / 255.0) ** g) + 0.5) for i in range(256)]
        return img.point(lut)


def build_transform(train: bool, norm: tuple, aug: str = "v1"):
    base = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(*norm),
    ]
    if not train:
        return transforms.Compose(base)
    if aug == "v1":
        pre = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.RandomAffine(0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
        return transforms.Compose(pre + base)
    # v2: crop-scale jitter + gamma + slightly stronger geometry. Still mild:
    # anatomy must stay anatomy, and inference sees unaugmented slices.
    pre = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomAffine(10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        RandomGamma(0.8, 1.25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
    ]
    rest = [t for t in base if not isinstance(t, transforms.Resize)]
    return transforms.Compose(pre + rest)


class SliceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path, train: bool,
                 norm: tuple = (IMAGENET_MEAN, IMAGENET_STD), aug: str = "v1",
                 cache: dict | None = None) -> None:
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.tf = build_transform(train, norm, aug)
        # Optional shared in-RAM cache of decoded greyscale arrays: PNG decode
        # dominates epoch time on Colab (2 workers, shared CPU), not the GPU.
        self.cache = cache

    def __len__(self) -> int:
        return len(self.frame)

    def _load(self, relpath: str) -> Image.Image:
        if self.cache is not None:
            arr = self.cache.get(relpath)
            if arr is None:
                arr = np.asarray(Image.open(self.root / relpath).convert("L"))
                self.cache[relpath] = arr
            return Image.fromarray(arr, mode="L")
        return Image.open(self.root / relpath).convert("L")

    def __getitem__(self, i: int):
        row = self.frame.iloc[i]
        return self.tf(self._load(row["relpath"])), int(row["label"]), i


def preload_cache(df: pd.DataFrame, root: Path) -> dict:
    cache: dict = {}
    t0 = time.time()
    for rel in df["relpath"]:
        cache[rel] = np.asarray(Image.open(root / rel).convert("L"))
    mb = sum(a.nbytes for a in cache.values()) / 1e6
    print(f"cached {len(cache)} slices in RAM ({mb:.0f} MB, {time.time() - t0:.0f}s)")
    return cache


def infer_source(session: str) -> str:
    """Dataset of origin from the session id (OAS1_xxxx vs OAS3xxxx)."""
    return "oasis1" if session.upper().startswith("OAS1") else "oasis3"


def load_manifest(data_dir: Path, label_mode: str) -> pd.DataFrame:
    manifest = data_dir / "manifest_v2.csv"
    if not manifest.exists():
        manifest = data_dir / "manifest.csv"
    df = pd.read_csv(manifest)
    df["relpath"] = df["path"].str.replace(r"^.*?slices/", "slices/", regex=True)
    df["session"] = df["relpath"].str.split("/").str[1]
    if "source" not in df.columns:
        df["source"] = df["session"].map(infer_source)
    if label_mode == "binary":
        df["label"] = (df["label"] >= 1).astype(int)
    missing = [p for p in df["relpath"] if not (data_dir / p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} slice files missing, e.g. {missing[0]}")
    return df


def manifest_hash(data_dir: Path) -> str:
    manifest = data_dir / "manifest_v2.csv"
    if not manifest.exists():
        manifest = data_dir / "manifest.csv"
    return hashlib.md5(manifest.read_bytes()).hexdigest()[:12]


def subject_folds(df: pd.DataFrame, k: int, seed: int = SEED,
                  source_aware: bool = False) -> list[set]:
    """K subject groups, stratified by each subject's max label (and, with
    source_aware, by dataset of origin so every fold sees the same mix)."""
    subj_label = df.groupby("subject_id")["label"].max()
    if source_aware:
        subj_source = df.groupby("subject_id")["source"].first()
        strata = subj_source.astype(str) + "/" + subj_label.astype(str)
    else:
        strata = subj_label.astype(str)
    rng = random.Random(seed)
    folds: list[list] = [[] for _ in range(k)]
    for stratum in sorted(strata.unique()):
        subjects = sorted(strata[strata == stratum].index)
        rng.shuffle(subjects)
        for i, s in enumerate(subjects):
            folds[i % k].append(s)
    return [set(f) for f in folds]


def inner_split(train_frame: pd.DataFrame, frac: float, seed: int,
                source_aware: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out `frac` of the TRAINING subjects (stratified) as an inner
    validation set. Model selection on it never touches the reported fold."""
    k = max(2, int(round(1.0 / frac)))
    parts = subject_folds(train_frame, k, seed=seed, source_aware=source_aware)
    held = parts[0]
    return (train_frame[~train_frame.subject_id.isin(held)],
            train_frame[train_frame.subject_id.isin(held)])


class HFClassifier(nn.Module):
    """A Hugging Face vision backbone + linear head on the CLS token."""

    def __init__(self, hf_name: str, n_classes: int) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(hf_name)
        self.head = nn.Linear(self.backbone.config.hidden_size, n_classes)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)


RESNET_STAGES = ["layer1", "layer2", "layer3", "layer4"]


def make_model(arch: str, n_classes: int, finetune: str) -> nn.Module:
    """finetune: layer4 | layer3 | all. layer3 = layer3+layer4+fc trainable."""
    if arch == "resnet18":
        # MRI_NO_PRETRAINED=1 skips the ImageNet download (offline smoke tests).
        weights = None if os.getenv("MRI_NO_PRETRAINED") else models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        if finetune != "all":
            keep = {"fc"} | set(RESNET_STAGES[RESNET_STAGES.index(finetune):])
            for name, p in model.named_parameters():
                if name.split(".")[0] not in keep:
                    p.requires_grad = False
        return model

    model = HFClassifier(ARCHS[arch]["hf"], n_classes)
    for p in model.backbone.parameters():
        p.requires_grad = False
    # Partial-unfreeze recipe mirroring the resnet layer4-only default:
    # vit_b16 always fine-tunes its last 2 encoder blocks + final norm;
    # dinov2_s stays fully frozen (linear probe) unless --finetune all.
    if arch == "vit_b16" or finetune == "all":
        # transformers v5: ViTModel keeps blocks at .layers, Dinov2Model at
        # .encoder.layer (older versions used .encoder.layer for both).
        enc = getattr(model.backbone, "encoder", None)
        blocks = (getattr(enc, "layer", None) or getattr(enc, "layers", None)
                  if enc is not None else model.backbone.layers)
        for blk in blocks[-2:]:
            for p in blk.parameters():
                p.requires_grad = True
        if hasattr(model.backbone, "layernorm"):
            for p in model.backbone.layernorm.parameters():
                p.requires_grad = True
    return model


def param_groups(model: nn.Module, arch: str, lr: float, mult: float) -> list[dict]:
    """Discriminative learning rates: the head at lr, each earlier trainable
    ResNet stage at lr*mult^depth. Other archs get a single group."""
    if arch != "resnet18" or mult >= 1.0:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    groups: dict[str, list] = defaultdict(list)
    for name, p in model.named_parameters():
        if p.requires_grad:
            groups[name.split(".")[0]].append(p)
    out = []
    for prefix, params in groups.items():
        if prefix == "fc":
            depth = 0
        elif prefix in RESNET_STAGES:
            depth = len(RESNET_STAGES) - RESNET_STAGES.index(prefix)
        else:
            depth = len(RESNET_STAGES) + 1  # stem
        out.append({"params": params, "lr": lr * (mult ** depth)})
    return out


def freeze_bn_stats(model: nn.Module) -> None:
    """Keep BatchNorm running stats of FROZEN stages fixed. model.train() would
    otherwise keep updating them on ~2k MRI slices, which is an uncontrolled
    variable between runs."""
    for name, m in model.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters(recurse=False)):
                m.eval()


def cosine_lambda(total_steps: int, warmup_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return f


def plain_state_dict(model: nn.Module) -> dict:
    """CPU state_dict with any AveragedModel 'module.' prefix stripped and the
    SWA bookkeeping buffer removed, so it loads into a plain torchvision net."""
    sd = {}
    for k, v in model.state_dict().items():
        if k == "n_averaged":
            continue
        sd[k[len("module."):] if k.startswith("module.") else k] = v.detach().cpu().clone()
    return sd


@torch.no_grad()
def slice_probs(model, loader, frame, device, n_classes: int) -> np.ndarray:
    """Per-slice probabilities with horizontal-flip TTA (average of softmaxes)."""
    model.eval()
    probs = np.zeros((len(frame), n_classes), dtype=np.float64)
    for x, _, idx in loader:
        x = x.to(device)
        p = torch.softmax(model(x), dim=1)
        p = p + torch.softmax(model(torch.flip(x, dims=[3])), dim=1)
        probs[idx.numpy()] = (p / 2).cpu().numpy()
    return probs


def tune_threshold(y_true: np.ndarray, p1: np.ndarray) -> float:
    """Balanced-accuracy-maximising cutoff on p(impaired), chosen on the inner
    validation split only (EVAL-1's honest alternative to argmax at 0.5)."""
    grid = np.linspace(0.2, 0.8, 61)
    scores = [np.mean([((p1 >= t) == (y_true == c))[y_true == c].mean() for c in (0, 1)])
              for t in grid]
    return float(grid[int(np.argmax(scores))])


def session_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int,
                    threshold: float | None = None) -> dict:
    if threshold is not None and n_classes == 2:
        y_pred = (y_prob[:, 1] >= threshold).astype(int)
    else:
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
        "balanced_accuracy": float(np.mean(per_class_recall)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1": float(f1_score(y_true, y_pred, average=avg)),
        "auc": float(auc),
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
    }


def session_eval(model, loader, frame, device, n_classes: int,
                 keep_slices: bool = False, threshold: float | None = None):
    """Session-level metrics with horizontal-flip test-time augmentation."""
    probs = slice_probs(model, loader, frame, device, n_classes)
    frame = frame.reset_index(drop=True)

    by_session = defaultdict(list)
    for i, row in frame.iterrows():
        by_session[row["session"]].append(i)

    y_true, y_prob, sessions, sources = [], [], [], []
    for session, rows in sorted(by_session.items()):
        sessions.append(session)
        y_true.append(int(frame.iloc[rows[0]]["label"]))
        sources.append(str(frame.iloc[rows[0]]["source"]))
        y_prob.append(probs[rows].mean(axis=0))
    y_true = np.array(y_true)
    y_prob = np.vstack(y_prob)

    out = session_metrics(y_true, y_prob, n_classes, threshold)
    out["sessions_detail"] = [
        {"session": s, "label": int(t), "source": src, "probs": p.tolist()}
        for s, t, src, p in zip(sessions, y_true, sources, y_prob)
    ]
    if keep_slices:
        out["slices_detail"] = [
            {"session": frame.iloc[i]["session"], "slice_idx": j,
             "label": int(frame.iloc[i]["label"]), "probs": probs[i].tolist()}
            for s, rows in sorted(by_session.items()) for j, i in enumerate(rows)
        ]
    # Per-source breakdown (only meaningful with mixed-cohort manifests).
    srcs = sorted(set(sources))
    if len(srcs) > 1:
        out["per_source"] = {}
        src_arr = np.array(sources)
        for s in srcs:
            m = src_arr == s
            if m.sum() and len(set(y_true[m])) > 1:
                out["per_source"][s] = {k: v for k, v in session_metrics(
                    y_true[m], y_prob[m], n_classes).items() if k != "confusion_matrix"}
    return out


def train_one(train_frame, val_frame, data_dir, device, args, n_classes, tag="",
              cache: dict | None = None, seed: int = SEED):
    """Train one model on train_frame.

    Selection rules (in priority order):
      * --inner-val F  : early-stop / pick the epoch on an inner split of the
                         training subjects; the outer val_frame is only reported.
      * --no-early-stop: train the full --epochs, report the LAST epoch
                         (implied by --sched cosine unless --early-stop given).
      * legacy         : early-stop on val_frame itself (optimistic).
    Returns (state_dict, val_metrics or None, history).
    """
    set_seed(seed)
    norm = ARCHS[args.arch]["norm"]
    pin = device.type == "cuda"

    inner_frame = None
    if args.inner_val > 0:
        train_frame, inner_frame = inner_split(train_frame, args.inner_val, seed,
                                               args.source_aware)

    def loader(frame, is_train):
        ds = SliceDataset(frame, data_dir, train=is_train, norm=norm, aug=args.aug, cache=cache)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=is_train,
                          num_workers=args.workers, pin_memory=pin,
                          drop_last=is_train and len(ds) > args.batch_size)

    train_loader = loader(train_frame, True)
    val_loader = loader(val_frame, False) if val_frame is not None else None
    inner_loader = loader(inner_frame, False) if inner_frame is not None else None

    model = make_model(args.arch, n_classes, args.finetune).to(device)

    counts = train_frame.groupby("label").size().reindex(range(n_classes), fill_value=1)
    weights = torch.tensor((counts.sum() / (n_classes * counts)).values, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    groups = param_groups(model, args.arch, args.lr, args.lr_mult_backbone)
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    sched = None
    if args.sched == "cosine":
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, cosine_lambda(args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch))

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Weight averaging: SWA over the tail of training (epoch-level) or EMA
    # (step-level). Both yield ONE plain network, so CAM export is unchanged.
    swa_model = ema_model = None
    swa_start = int(round(args.epochs * args.swa_start_frac)) if args.swa_start_frac > 0 else None
    if swa_start is not None:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
    if args.ema > 0:
        ema_model = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema))

    select_loader, select_frame = (inner_loader, inner_frame) if inner_loader else (val_loader, val_frame)
    selecting = (not args.no_early_stop) and select_loader is not None
    if selecting and inner_loader is None and val_loader is not None and args.experiment:
        print(f"{tag}WARNING: early-stopping on the reported fold (optimistic). "
              "Use --no-early-stop or --inner-val for honest numbers.")

    best_score, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_bn:
            freeze_bn_stats(model)
        running, t0 = 0.0, time.time()
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            if ema_model is not None:
                ema_model.update_parameters(model)
            running += loss.item() * len(y)
        train_loss = running / max(1, len(train_loader.dataset))
        if swa_model is not None and epoch > swa_start:
            swa_model.update_parameters(model)

        rec = {"epoch": epoch, "train_loss": round(train_loss, 4),
               "lr": round(max(g["lr"] for g in opt.param_groups), 8), "secs": round(time.time() - t0, 1)}
        if inner_loader is not None and (selecting or args.eval_every and epoch % args.eval_every == 0):
            inner = session_eval(model, inner_loader, inner_frame, device, n_classes)
            rec.update({"inner_accuracy": round(inner["accuracy"], 4), "inner_auc": round(inner["auc"], 4)})
        if val_loader is not None and (selecting or (args.eval_every and epoch % args.eval_every == 0)):
            val = session_eval(model, val_loader, val_frame, device, n_classes)
            rec.update({"val_accuracy": round(val["accuracy"], 4), "val_auc": round(val["auc"], 4)})
        history.append(rec)
        print(f"{tag}epoch {epoch:02d}  loss {train_loss:.4f}  " +
              "  ".join(f"{k} {v}" for k, v in rec.items() if k.endswith(("accuracy", "auc"))))

        if selecting:
            m = inner if inner_loader is not None else val
            score = m["auc"] if not np.isnan(m["auc"]) else m["macro_f1"]
            if score > best_score:
                best_score, bad, best_epoch = score, 0, epoch
                best_state = plain_state_dict(model)
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"{tag}early stop at epoch {epoch} (best epoch {best_epoch})")
                    break

    # Pick the final weights.
    if swa_model is not None and swa_model.n_averaged.item() > 0:
        torch.optim.swa_utils.update_bn(
            ((x.to(device), ) for x, _, _ in loader(train_frame, True)), swa_model, device=device)
        final_state = plain_state_dict(swa_model)
    elif ema_model is not None:
        final_state = plain_state_dict(ema_model)
    elif selecting and best_state is not None:
        final_state = best_state
    else:
        final_state = plain_state_dict(model)
        best_epoch = args.epochs

    val = None
    if val_loader is not None:
        model.load_state_dict(final_state)
        # EVAL-1: with an inner split, the decision threshold is tuned there
        # and the outer fold is scored exactly once with it. Probabilities
        # (and so AUC / saved preds) are unaffected by the threshold.
        thr = None
        if inner_loader is not None and n_classes == 2:
            inner = session_eval(model, inner_loader, inner_frame, device, n_classes)
            y_in = np.array([d["label"] for d in inner["sessions_detail"]])
            p_in = np.array([d["probs"][1] for d in inner["sessions_detail"]])
            thr = tune_threshold(y_in, p_in)
        val = session_eval(model, val_loader, val_frame, device, n_classes,
                           keep_slices=args.save_slice_preds, threshold=thr)
        val["best_epoch"] = best_epoch
        val["threshold"] = thr
    return final_state, val, history


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def agg_stat(results: list[dict], key: str) -> dict:
    vals = [f[key] for f in results if not (isinstance(f[key], float) and np.isnan(f[key]))]
    return {"mean": round(float(np.mean(vals)), 4) if vals else float("nan"),
            "std": round(float(np.std(vals)), 4) if vals else float("nan")}


def build_metrics(args, df, classes, n_slices, cv, deployed: dict, extra: dict | None = None) -> dict:
    n_classes = len(classes)
    sess_labels = df.groupby("session")["label"].first()
    model_desc = {
        "resnet18": f"ResNet-18 (ImageNet transfer learning), {n_classes}-class head, finetune={args.finetune}",
        "vit_b16": f"ViT-B/16 (google/vit-base-patch16-224-in21k, HF transformers), last 2 blocks + head fine-tuned, {n_classes}-class head",
        "dinov2_s": f"DINOv2-small (facebook/dinov2-small, HF transformers), "
                    f"{'last 2 blocks + head fine-tuned' if args.finetune == 'all' else 'frozen backbone + linear head'}, {n_classes}-class head",
    }[args.arch]
    sources = sorted(df["source"].unique())
    ds_name = "OASIS-3 (T1w MRI), labels = CDR nearest the scan date"
    if "oasis1" in sources:
        ds_name = "OASIS-3 + OASIS-1 (T1w MRI), labels = CDR nearest the scan date"
    recipe = (f"{args.optimizer} lr={args.lr} wd={args.weight_decay} sched={args.sched} "
              f"epochs={args.epochs} ls={args.label_smoothing} aug={args.aug} "
              f"finetune={args.finetune} lr_mult={args.lr_mult_backbone} "
              f"swa={args.swa_start_frac} ema={args.ema}")
    metrics = {
        "model": model_desc,
        "task": f"{args.label_mode} classification of T1w axial MRI slices, aggregated per session",
        "label_mode": args.label_mode,
        "classes": classes,
        "n_slices": n_slices,
        "dataset": {
            "name": ds_name,
            "subjects": int(df.subject_id.nunique()),
            "sessions": int(df.session.nunique()),
            "slices": int(len(df)),
            "session_class_counts": {str(k): int(v) for k, v in sess_labels.value_counts().sort_index().items()},
            "sources": {s: int(df[df.source == s].session.nunique()) for s in sources},
            "split": (f"subject-stratified {args.folds}-fold cross-validation"
                      + (f", repeated {args.repeats}x" if args.repeats > 1 else "")),
        },
        "aggregation": f"session probability = mean of its {n_slices} slice probabilities, with hflip TTA",
        "cv": cv,
        "deployed": deployed,
        "training": {"optimizer": recipe, "class_weighted_loss": True,
                     "augmentation": args.aug, "seed": SEED,
                     "selection": ("inner validation split" if args.inner_val > 0 else
                                   "fixed epoch budget, last epoch" if args.no_early_stop else
                                   "early stop on the reported fold (optimistic)")},
        "caveats": [
            "Labels come from CDR staging, not biopsy-confirmed diagnosis.",
            "Slices are 2D projections of a 3D volume; a 3D CNN could use more context.",
            "OASIS-3 is a largely North-American research cohort; other scanners/populations unverified.",
            "Out-of-distribution inputs (non-brain, CT, non-T1w) are not rejected.",
            "Reported separately from the behavioural risk score: no dataset pairs MRI with the app's six tests.",
        ],
        "version": "2.1.0",
    }
    if extra:
        metrics.update(extra)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dir containing manifest(_v2).csv and slices/")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "artifacts"))
    ap.add_argument("--experiment", default=None,
                    help="write to ml/experiments/<NAME>/ instead of --out-dir; never touches artifacts")
    ap.add_argument("--experiments-root", default=str(EXPERIMENTS_ROOT))
    ap.add_argument("--label-mode", choices=["binary", "three"], default="binary")
    ap.add_argument("--arch", choices=list(ARCHS), default="resnet18",
                    help="resnet18 = deployed CNN; vit_b16 / dinov2_s = HF transformers")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default=None,
                    help="comma-separated fold-assignment seeds, e.g. 42,43,44 = repeated CV (EVAL-1). "
                         "Default: --fold-seed .. --fold-seed+repeats-1")
    ap.add_argument("--repeats", type=int, default=1, help="shorthand for --seeds fold_seed..fold_seed+repeats-1")
    ap.add_argument("--fold-seed", type=int, default=SEED)
    ap.add_argument("--model-seeds", type=int, default=1,
                    help="models per fold (different init/aug seeds); fold headline = their probability average")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: per-arch (resnet18 3e-4, vit_b16 1e-4, dinov2_s 1e-3)")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--finetune", choices=["all", "layer4", "layer3"], default="layer4",
                    help="resnet: trainable stages (layer4 | layer3+layer4 | all); HF archs: layer4=partial, all=more")
    ap.add_argument("--lr-mult-backbone", type=float, default=1.0,
                    help="<1 gives earlier resnet stages a smaller lr (lr*mult^depth)")
    ap.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--aug", choices=["v1", "v2"], default="v1")
    ap.add_argument("--sched", choices=["none", "cosine"], default="none")
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--no-early-stop", action="store_true",
                    help="train the full --epochs and report the last epoch (implied by --sched cosine)")
    ap.add_argument("--early-stop", action="store_true", help="force early stopping even with --sched cosine")
    ap.add_argument("--inner-val", type=float, default=0.0,
                    help="fraction of TRAINING subjects held out for model selection (e.g. 0.15)")
    ap.add_argument("--eval-every", type=int, default=1, help="log val metrics every N epochs (0 = only at the end)")
    ap.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    ap.add_argument("--freeze-bn", action="store_true", help="freeze BatchNorm stats of frozen stages")
    ap.add_argument("--cache", action="store_true", help="decode all slices into RAM once")
    ap.add_argument("--swa-start-frac", type=float, default=0.0,
                    help="average weights over the last (1-frac) of epochs (e.g. 0.75); 0 = off")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of weights (e.g. 0.999); 0 = off")
    ap.add_argument("--source-aware", action="store_true",
                    help="stratify folds by (dataset source, label) and report per-source metrics")
    ap.add_argument("--save-preds", action="store_true",
                    help="write per-session fold probabilities to preds_<arch>.csv (for ensemble_eval.py)")
    ap.add_argument("--save-slice-preds", action="store_true",
                    help="experiment mode: also write per-slice OOF probabilities (preds_slices.csv)")
    ap.add_argument("--final-fit", action="store_true",
                    help="experiment mode: after CV, retrain on ALL subjects and export final/model.onnx")
    args = ap.parse_args()
    if args.lr is None:
        args.lr = ARCHS[args.arch]["lr"]
    if args.seeds:
        args.fold_seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    else:
        args.fold_seeds = [args.fold_seed + r for r in range(args.repeats)]
    args.repeats = len(args.fold_seeds)
    if args.sched == "cosine" and not args.early_stop:
        args.no_early_stop = True
    if args.arch != "resnet18" and args.finetune == "layer3":
        args.finetune = "layer4"

    set_seed()
    device = pick_device(args.device)
    data_dir = Path(args.data_dir)
    experiment_mode = args.experiment is not None
    if experiment_mode:
        out_dir = Path(args.experiments_root) / args.experiment
        if out_dir.resolve() == (REPO_ROOT / "ml" / "artifacts").resolve():
            raise SystemExit("refusing to write an experiment into ml/artifacts")
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = CLASS_NAMES[args.label_mode]
    n_classes = len(classes)
    df = load_manifest(data_dir, args.label_mode)
    n_slices = int(df.groupby("session").size().mode()[0])
    sess_labels = df.groupby("session")["label"].first()
    print(f"device={device}  label_mode={args.label_mode}  slices/session={n_slices}  mode={'experiment' if experiment_mode else 'legacy'}")
    print(f"{df.subject_id.nunique()} subjects, {df.session.nunique()} sessions, {len(df)} slices")
    print("session class counts:", sess_labels.value_counts().sort_index().to_dict())
    if df["source"].nunique() > 1:
        print("sources:", df.groupby("source").session.nunique().to_dict())

    if experiment_mode:
        cfg = dict(vars(args))
        cfg.update({"git_sha": git_sha(), "manifest_hash": manifest_hash(data_dir),
                    "device": str(device), "argv": sys.argv[1:],
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    cache = preload_cache(df, data_dir) if args.cache else None

    fold_results, deployed_state, deployed_val = [], None, None
    session_rows, slice_rows, histories = [], [], {}
    t_start = time.time()
    for r, fold_seed in enumerate(args.fold_seeds):
        folds = subject_folds(df, args.folds, seed=fold_seed, source_aware=args.source_aware)
        for i, fold_subjects in enumerate(folds):
            val_frame = df[df.subject_id.isin(fold_subjects)]
            train_frame = df[~df.subject_id.isin(fold_subjects)]
            seed_vals = []
            for s in range(args.model_seeds):
                seed = SEED + i + 100 * r + 10_000 * s
                tag = f"[rep {r} fold {i}" + (f" seed {s}" if args.model_seeds > 1 else "") + "] "
                state, val, hist = train_one(train_frame, val_frame, data_dir, device, args,
                                             n_classes, tag=tag, cache=cache, seed=seed)
                histories[tag.strip("[] ")] = hist
                seed_vals.append(val)
                print(f"{tag}sessions={val['sessions']} acc={val['accuracy']:.3f} "
                      f"bal_acc={val['balanced_accuracy']:.3f} f1={val['f1']:.3f} auc={val['auc']:.3f} "
                      f"recall={['%.2f' % x for x in val['per_class_recall']]}")
                for d in val["sessions_detail"]:
                    row = {"repeat": r, "fold": i, "seed": s, "session": d["session"],
                           "source": d["source"], "label": d["label"]}
                    row.update({f"p_{c}": round(p, 6) for c, p in enumerate(d["probs"])})
                    session_rows.append(row)
                for d in val.get("slices_detail", []):
                    row = {"repeat": r, "fold": i, "seed": s, "session": d["session"],
                           "slice_idx": d["slice_idx"], "label": d["label"]}
                    row.update({f"p_{c}": round(p, 6) for c, p in enumerate(d["probs"])})
                    slice_rows.append(row)
                if r == 0 and i == 0 and s == 0:
                    deployed_state, deployed_val = state, val
            # With several seeds per fold the fold's headline is the
            # probability-averaged (seed-ensemble) prediction.
            if args.model_seeds > 1:
                y = np.array([d["label"] for d in seed_vals[0]["sessions_detail"]])
                p = np.mean([[d["probs"] for d in v["sessions_detail"]] for v in seed_vals], axis=0)
                ens = session_metrics(y, p, n_classes)
                ens["seed_members"] = [{k: v[k] for k in ("accuracy", "auc")} for v in seed_vals]
                fold_results.append({**ens, "repeat": r, "fold": i})
                print(f"[rep {r} fold {i}] seed-ensemble acc={ens['accuracy']:.3f} auc={ens['auc']:.3f}")
            else:
                fold_results.append({**seed_vals[0], "repeat": r, "fold": i})

    cv = {
        "folds": args.folds,
        "repeats": args.repeats,
        "seeds": args.fold_seeds,
        "seeds_per_fold": args.model_seeds,
        "accuracy": agg_stat(fold_results, "accuracy"),
        "balanced_accuracy": agg_stat(fold_results, "balanced_accuracy"),
        "f1": agg_stat(fold_results, "f1"),
        "macro_f1": agg_stat(fold_results, "macro_f1"),
        "auc": agg_stat(fold_results, "auc"),
        "per_fold": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()
             if k not in ("confusion_matrix", "sessions_detail", "slices_detail", "per_source")}
            for f in fold_results
        ],
    }
    # Pooled out-of-fold metrics per repeat (each session predicted exactly
    # once per repeat; lower variance than the mean of 36-session folds).
    if session_rows:
        sdf = pd.DataFrame(session_rows)
        prob_cols = [c for c in sdf.columns if c.startswith("p_")]
        pooled = []
        for r, g in sdf.groupby("repeat"):
            g = g.groupby(["session", "label"], as_index=False)[prob_cols].mean()  # over seeds
            pooled.append(session_metrics(g["label"].to_numpy(), g[prob_cols].to_numpy(), n_classes))
        cv["pooled_oof"] = {k: agg_stat(pooled, k) for k in
                            ("accuracy", "balanced_accuracy", "f1", "macro_f1", "auc")}
        cv["pooled_oof"]["per_class_recall"] = np.mean(
            [p["per_class_recall"] for p in pooled], axis=0).round(4).tolist()
        if sdf["source"].nunique() > 1:
            cv["per_source"] = {}
            for src, gs in sdf.groupby("source"):
                per_rep = []
                for r, g in gs.groupby("repeat"):
                    g = g.groupby(["session", "label"], as_index=False)[prob_cols].mean()
                    if g["label"].nunique() > 1:
                        per_rep.append(session_metrics(g["label"].to_numpy(), g[prob_cols].to_numpy(), n_classes))
                if per_rep:
                    cv["per_source"][src] = {k: agg_stat(per_rep, k) for k in ("accuracy", "balanced_accuracy", "auc")}
                    cv["per_source"][src]["sessions"] = int(gs.session.nunique())
    cv["train_minutes"] = round((time.time() - t_start) / 60, 1)
    print("\nCV summary:", json.dumps({k: cv[k] for k in ("accuracy", "balanced_accuracy", "f1", "auc")}, indent=2))
    if "pooled_oof" in cv:
        print("pooled OOF:", json.dumps(cv["pooled_oof"], indent=2))

    # ---- artifacts -------------------------------------------------------
    if experiment_mode:
        write_experiment(args, df, classes, n_slices, cv, out_dir, session_rows, slice_rows,
                         histories, deployed_val, data_dir, device, cache, n_classes)
        return

    # Legacy mode: transformer archs write arch-suffixed files and never touch
    # the deployed mri_model.pt / mri_metrics.json / ONNX export.
    is_deployed_arch = args.arch == "resnet18"
    suffix = "" if is_deployed_arch else f"_{args.arch}"

    if args.save_preds:
        rows = []
        for f in fold_results:
            # multi-seed runs need a unique per-run fold key for ensemble joins
            fold_key = f["fold"] if args.repeats == 1 else f"{args.fold_seeds[f['repeat']]}-{f['fold']}"
            for d in f.get("sessions_detail", []):
                row = {"fold": fold_key, "session": d["session"], "label": d["label"]}
                row.update({f"p_{c}": round(p, 6) for c, p in enumerate(d["probs"])})
                rows.append(row)
        preds_path = out_dir / f"preds_{args.arch}.csv"
        pd.DataFrame(rows).to_csv(preds_path, index=False)
        print(f"saved per-session predictions: {preds_path}")

    torch.save({"state_dict": deployed_state, "img_size": IMG_SIZE, "classes": classes,
                "arch": args.arch},
               out_dir / f"mri_model{suffix}.pt")

    deployed = {
        "which": "fold-0 model (trained on the other folds, early-stopped on fold 0)",
        "fold0_validation": {k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in deployed_val.items()
                             if k not in ("sessions_detail", "slices_detail", "per_source")},
    }
    metrics = build_metrics(args, df, classes, n_slices, cv, deployed)
    (out_dir / f"mri_metrics{suffix}.json").write_text(json.dumps(metrics, indent=2))

    if is_deployed_arch:
        from export_onnx import export as export_onnx

        export_onnx(out_dir)
        print(f"saved: {out_dir}/mri_model.pt, mri_model.onnx, mri_metrics.json")
    else:
        print(f"saved: {out_dir}/mri_model{suffix}.pt, mri_metrics{suffix}.json "
              f"(deployed ResNet artifacts untouched; no ONNX export for {args.arch})")


def write_experiment(args, df, classes, n_slices, cv, out_dir: Path, session_rows, slice_rows,
                     histories, deployed_val, data_dir, device, cache, n_classes) -> None:
    sdf = pd.DataFrame(session_rows)
    sdf.to_csv(out_dir / "preds_sessions.csv", index=False)
    # Same rows in the preds_<arch>.csv format ensemble_eval.py / source_audit.py
    # read (seed-averaged per fold; fold key unique across repeats).
    prob_cols = [c for c in sdf.columns if c.startswith("p_")]
    legacy = sdf.groupby(["repeat", "fold", "session", "label"], as_index=False)[prob_cols].mean()
    legacy["fold"] = (legacy["fold"].astype(str) if args.repeats == 1 else
                      legacy["repeat"].map(lambda r: str(args.fold_seeds[r])) + "-" + legacy["fold"].astype(str))
    legacy[["fold", "session", "label"] + prob_cols].to_csv(out_dir / f"preds_{args.arch}.csv", index=False)
    if slice_rows:
        pd.DataFrame(slice_rows).to_csv(out_dir / "preds_slices.csv", index=False)
    (out_dir / "history.json").write_text(json.dumps(histories, indent=1))

    summary = {
        "experiment": args.experiment,
        "arch": args.arch,
        "classes": classes,
        "n_slices": n_slices,
        "subjects": int(df.subject_id.nunique()),
        "sessions": int(df.session.nunique()),
        "cv": cv,
        "fold0_best_epoch": deployed_val.get("best_epoch") if deployed_val else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"experiment written: {out_dir} (summary.json, preds_sessions.csv"
          f"{', preds_slices.csv' if slice_rows else ''}, history.json)")

    if not args.final_fit:
        return

    # ---- final fit on ALL subjects --------------------------------------
    # Same recipe and epoch budget as the CV runs, so the cv block above is an
    # honest estimate of this model. Requires a selection rule that does not
    # need a held-out fold: fixed budget (--no-early-stop) or --inner-val.
    if not (args.no_early_stop or args.inner_val > 0):
        raise SystemExit("--final-fit needs --no-early-stop (or --sched cosine) or --inner-val: "
                         "there is no held-out fold to early-stop on when training on all data")
    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True)
    states = []
    for s in range(args.model_seeds):
        tag = f"[final seed {s}] " if args.model_seeds > 1 else "[final] "
        state, _, hist = train_one(df, None, data_dir, device, args, n_classes, tag=tag,
                                   cache=cache, seed=SEED + 777 + 10_000 * s)
        histories[tag.strip("[] ")] = hist
        states.append(state)
    (out_dir / "history.json").write_text(json.dumps(histories, indent=1))

    if len(states) > 1:
        # Weight soup: average the seeds' weights into one deployable network.
        # Only sensible for fine-tunes from one shared init (same pretrained
        # backbone), which is the case here; compare_experiments reports
        # whether the probability ensemble beat single models.
        soup = {k: torch.stack([sd[k].float() for sd in states]).mean(0).to(states[0][k].dtype)
                for k in states[0]}
        state = soup
        which = f"retrained on all {df.subject_id.nunique()} subjects, weight-averaged over {len(states)} seeds"
    else:
        state = states[0]
        which = f"retrained on all {df.subject_id.nunique()} subjects with the CV recipe (fixed {args.epochs} epochs)"

    # final/ is laid out like ml/artifacts so promote.py can copy it and
    # external_validate.py --model-dir can evaluate it as-is.
    suffix = "" if args.arch == "resnet18" else f"_{args.arch}"
    ckpt = final_dir / f"mri_model{suffix}.pt"
    torch.save({"state_dict": state, "img_size": IMG_SIZE, "classes": classes, "arch": args.arch}, ckpt)
    deployed = {"which": which, "experiment": args.experiment,
                "cv_estimate": {k: cv[k] for k in ("accuracy", "balanced_accuracy", "f1", "auc")}}
    metrics = build_metrics(args, df, classes, n_slices, cv, deployed)
    (final_dir / f"mri_metrics{suffix}.json").write_text(json.dumps(metrics, indent=2))

    from export_onnx import export as export_resnet, export_transformer

    if args.arch == "resnet18":
        onnx_path = export_resnet(final_dir, ckpt_path=ckpt, out_path=final_dir / "mri_model.onnx")
    else:
        onnx_path = export_transformer(args.arch, final_dir, ckpt_path=ckpt,
                                       out_path=final_dir / f"mri_model_{args.arch}.onnx")
    print(f"final model: {ckpt}, {onnx_path}, {final_dir / f'mri_metrics{suffix}.json'}")
    print(f"deploy with: python ml/mri/promote.py {args.experiment} [other member experiments]")


if __name__ == "__main__":
    main()
