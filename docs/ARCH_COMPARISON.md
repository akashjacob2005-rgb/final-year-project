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

| Model | Accuracy | F1 | AUC | Recall (normal / impaired) |
|---|---|---|---|---|
| ResNet-18 (deployed) | **75.6% ± 4.4%** | **0.816** | **0.844 ± 0.07** | ~0.58 / ~0.83 (fold 0) |
| ViT-B/16 | *pending — run `ml/mri/colab_train.ipynb`* | | | |
| DINOv2-S (linear probe) | *pending* | | | |
| ResNet + DINOv2 ensemble | *pending* | | | |

> Fill this table from the `mri_metrics_<arch>.json` / `ensemble_*.json` files
> the Colab notebook downloads. A 1-epoch local smoke run already showed the
> pipeline working (DINOv2 linear probe AUC 0.73 after one epoch; the
> ResNet+DINOv2 ensemble beat both members) — full numbers need the real runs.

## What the literature predicts (and why)

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

## Why deployment keeps ResNet-18 regardless

- The app's explainability feature is Grad-CAM/CAM heatmaps, which come free
  from the ResNet's ONNX export. Transformers would need attention-rollout
  maps — a different, weaker-validated technique plus backend/frontend rework.
- ResNet-18 is the smallest, fastest model on the Render CPU tier.
- This study is a report/viva contribution: an evidence-backed architecture
  decision, not a deployment change.

## Reproduce

```bash
# Colab (free T4 GPU, ~1 h total): run ml/mri/colab_train.ipynb top to bottom.
# Locally (MPS, slower):
ml/mri/.venv/bin/python ml/mri/train_mri.py \
    --data-dir ~/Downloads/7th_sem/PP-2/data/processed_v2 \
    --out-dir ml/artifacts --arch dinov2_s --folds 5 --save-preds
ml/mri/.venv/bin/python ml/mri/ensemble_eval.py \
    ml/artifacts/preds_resnet18.csv ml/artifacts/preds_dinov2_s.csv
```

Transformer runs write `mri_metrics_<arch>.json` / `mri_model_<arch>.pt` /
`preds_<arch>.csv` only — the deployed `mri_model.pt`, `mri_metrics.json` and
ONNX artifacts are never touched.
