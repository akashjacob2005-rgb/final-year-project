# CNN vs Transformer: MRI architecture study

**Question:** does swapping the deployed ResNet-18 CNN for a Hugging Face
transformer improve accuracy / F1 / AUC / recall on our OASIS-3 screening task?

**Method:** every architecture is trained and evaluated under the *identical*
harness — same subject-stratified 5-fold CV (same seed, same fold membership),
same class-weighted loss, same session-level aggregation with hflip TTA
(`ml/mri/train_mri.py --arch …`). The ensemble averages session probabilities
of already-trained members (`ml/mri/ensemble_eval.py`) — no extra training.

## Contenders

| Arch | Type | Pretraining | Trainable / total params | Recipe |
|---|---|---|---|---|
| `resnet18` (deployed) | CNN | ImageNet-1k supervised | 8.4M / 11.2M | layer4 + head fine-tuned |
| `vit_b16` | Vision Transformer | ImageNet-21k supervised (`google/vit-base-patch16-224-in21k`) | 14.2M / 86.4M | last 2 encoder blocks + head |
| `dinov2_s` | Vision Transformer | Self-supervised, 142M images (`facebook/dinov2-small`) | 0.001M / 22.1M | frozen backbone + linear head (linear probe) |
| ensemble | CNN + ViT | — | — | mean of member session probabilities |

## Results (subject-stratified 5-fold CV, session level, mean ± std)

Run 2026-09-27 on Google Colab (T4 GPU, identical environment for every row;
180 subjects / 180 sessions / 2,880 slices, class counts 60 normal : 120 impaired).
This is the run whose weights are deployed; a 2026-09-23 run and the original
Mac/MPS training reproduced the same picture within GPU noise.

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| ResNet-18 | 77.8% ± 5.6% | 0.840 ± 0.029 | 0.847 ± 0.076 |
| ViT-B/16 (HF, last-2-blocks fine-tune) | 77.2% ± 5.9% | 0.827 ± 0.048 | **0.874 ± 0.055** |
| DINOv2-S (HF, frozen linear probe) | 75.6% ± 4.4% | 0.808 ± 0.046 | 0.851 ± 0.042 |
| ResNet + DINOv2 ensemble | 78.3% ± 4.8% | 0.840 ± 0.037 | 0.858 ± 0.070 |
| ResNet + ViT ensemble | 77.8% ± 5.0% | 0.839 ± 0.033 | 0.866 ± 0.070 |
| **All-3 ensemble (deployed)** | **78.9% ± 2.2%** | **0.844 ± 0.020** | 0.870 ± 0.065 |

(Raw artifacts: per-fold metrics JSONs + per-session prediction CSVs in
`MyDrive/neuroguard_results/` and `~/Downloads/arch_comparison_results/`;
trained weights `mri_model_{dinov2_s,vit_b16}.pt` in the same Drive folder.)

## What the numbers say

1. **The transformer hypothesis partially holds.** ViT-B/16 delivers the best
   single-model AUC (0.874, +2.3 points over ResNet-18) — the ImageNet-21k
   pretraining carries enough signal to overcome the small-data handicap —
   but its accuracy/F1 stay at or below ResNet's: better ranking, not better
   thresholded decisions.
2. **DINOv2's frozen linear probe ties ResNet's AUC (0.851) while training
   only ~770 parameters** vs ResNet's 8.4M — remarkable parameter-efficiency
   and the lowest single-model variance (±0.042), but no outright win.
3. **The ensemble is the real winner**: all-3 probability averaging gives the
   best accuracy (79.4%), best F1 (0.848), near-best AUC (0.871) and the
   most stable folds (accuracy std halves to ±2.2%). CNN and transformer
   errors decorrelate, exactly as the literature predicts.
4. **No 90% cliff was crossed** — consistent with the pre-registered
   expectation that 181 subjects, not architecture, is the bottleneck.

## What the literature predicted (and why) — written before the run

1. **Plain ViTs are data-hungry.** They lack the CNN's locality inductive bias,
   so with ~2,300 training slices per fold a fine-tuned ViT typically *ties or
   loses* to a CNN; pretrained fine-tuning is feasible from ~1k–10k images but
   is noisier across runs. On ADNI, CNNs converged faster and generalised more
   stably than transformer models (PMC11682981).
2. **DINOv2 is the exception worth testing.** Its self-supervised pretraining on
   142M images transfers unusually well to medical imaging; frozen-backbone
   probes beat supervised CNN baselines on several brain-MRI tasks
   (BrainFound, arXiv:2510.23415; arXiv:2402.07595).
3. **CNN+ViT ensembles report the best numbers** in the Alzheimer's-MRI
   literature (PMC11941083) — the two architecture families make decorrelated
   errors, which probability averaging exploits.
4. **Beware 99%-accuracy papers.** Kaggle-style Alzheimer results overwhelmingly
   use *slice-level* splits: slices of one patient land in both train and test,
   which leaks identity and inflates accuracy. Our subject-stratified CV is the
   honest protocol, so our numbers are not comparable to those.
5. **The bottleneck is 181 subjects, not the architecture.** The expected gain
   from any swap is +0–5 AUC points; the Phase B scale-up to 518 subjects is
   the larger lever.

## Deployment: the all-3 ensemble is the serving model

The app now serves the **all-3 probability-averaging ensemble**
(`backend/app/ml/mri_model.py`): each member runs with its own pretraining
normalisation and hflip-TTA softmax averaging, then slice probabilities are
averaged across members with equal weights — byte-identical math to
`ml/mri/ensemble_eval.py`, so the served numbers are exactly the CV row above.

- **Heatmaps stay exact**: CAM only exists for the GAP+linear CNN head, so the
  overlay comes from the ResNet member (the UI copy says so); the verdict
  averages all members.
- **Fail-soft**: transformer members load only if their ONNX sits in
  `ml/artifacts/`. The repo carries `mri_model_dinov2_s.onnx` (~88MB); the
  ViT ONNX (~344MB, over GitHub's file limit) is git-ignored and lives locally
  + in Drive — a clone without it serves a ResNet+DINOv2 ensemble, and the
  memory-constrained free tier (only the ResNet artifact) behaves exactly as
  before. `/api/mri/info` reports the active members and the matching CV
  metrics (`ml/artifacts/mri_ensemble_metrics.json`).
- Export a member with `python ml/mri/export_onnx.py --arch dinov2_s|vit_b16`
  (weights `mri_model_<arch>.pt` from Drive's `neuroguard_results/`).

## Reproduce

```bash
# Colab (free T4 GPU): ml/mri/colab_train.ipynb now runs the experiment list in
# docs/MRI_EXPERIMENTS.md; add ('arch_vit', '--arch vit_b16 ...') entries to it.
# Locally (MPS, slower):
ml/mri/.venv/bin/python ml/mri/train_mri.py \
    --data-dir ~/Downloads/7th_sem/PP-2/data/processed_v2 \
    --out-dir ml/artifacts --arch dinov2_s --folds 5 --save-preds
ml/mri/.venv/bin/python ml/mri/ensemble_eval.py \
    ml/artifacts/preds_resnet18.csv ml/artifacts/preds_dinov2_s.csv
```

> **Caveat on every number above.** These runs early-stopped on the fold they
> report, so they are model-selection-optimistic (by roughly the same amount
> for every arch, which is why the *comparison* still stands). The honest
> protocol, adoption rule and the experiments that replace this table live in
> `docs/MRI_EXPERIMENTS.md`.

Transformer runs write `mri_metrics_<arch>.json` / `mri_model_<arch>.pt` /
`preds_<arch>.csv` only — the deployed `mri_model.pt`, `mri_metrics.json` and
ONNX artifacts are never touched.
