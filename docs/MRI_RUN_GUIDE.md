# MRI accuracy: what to run, in order

Everything below runs on **your Mac** or **Colab**. Paths assume the layout
from earlier sessions: data in `~/Downloads/7th_sem/PP-2/data/`, OASIS-1 in
`~/Downloads/7th_sem/PP-2/external/oasis1/`, the Sep-27 ensemble predictions
in `~/Downloads/arch_comparison_results/`. Adjust if yours differ.
Run every command from the repo root.

The rules (honest CV, adoption bar) are in `docs/MRI_EXPERIMENTS.md`.

---

## 0 · Get the branch onto your Mac and GitHub (5 min)

You received `neuroguard-mri.bundle` (a git bundle of branch
`feat/mri-honest-eval`: your Mac work + `main` + the new harness, already merged).

```bash
git fetch origin                                     # needs main + feat/mri-accuracy
git fetch ~/Downloads/neuroguard-mri.bundle feat/mri-honest-eval:feat/mri-honest-eval
git checkout feat/mri-honest-eval
cd backend && .venv/bin/python -m pytest tests/ -q && cd ..   # should be all green
git push -u origin feat/mri-honest-eval              # a branch, not main: no Render deploy
```

Colab clones this branch, so the push is needed before step 3.

---

## 1 · Quick win: calibrate the ensemble you already have (10 min, no training)

Your deployed 3-model ensemble flags too many healthy scans (normal recall
≈ 0.5) because it decides at 0.5 on a model trained 2:1 impaired. Fit an
honest threshold + temperature from its out-of-fold predictions:

```bash
R=~/Downloads/arch_comparison_results
ml/mri/.venv/bin/python ml/mri/ensemble_eval.py \
    $R/preds_resnet18.csv $R/preds_dinov2_s.csv $R/preds_vit_b16.csv \
    --write-preds ml/experiments/sep27_ensemble
ml/mri/.venv/bin/python ml/mri/compare_experiments.py sep27_ensemble --calibrate sep27_ensemble
backend/.venv/bin/python ml/mri/promote.py --calibrate-deployed sep27_ensemble          # dry run
backend/.venv/bin/python ml/mri/promote.py --calibrate-deployed sep27_ensemble --yes    # install
```

Read the `cal bal acc` and `rec N/I (cal)` columns: that is the honest
(cross-fitted) estimate of the calibrated model. Restart the backend; the
results page now uses the calibrated threshold and the calibrated caveat text.

> These Sep-27 numbers are still early-stopped on the reported fold, so they
> are optimistic. Steps 2–3 replace them with honest ones.

---

## 2 · Register every scan to a standard brain (≈ 2 h on the Mac, unattended)

Aligns each brain to the MNI152 template and cuts slices at the same
anatomical positions for everyone — axial, plus coronal slices through the
hippocampus (the region that shrinks first in Alzheimer's).

```bash
ml/mri/.venv/bin/pip install antspyx nilearn deepbet     # once (Docker+SynthStrip also works)

D=~/Downloads/7th_sem/PP-2
ml/mri/.venv/bin/python ml/mri/preprocess_registered.py --export \
    --manifest-from $D/data/processed_v3 \
    --data-root $D/data --oasis1-dir $D/external/oasis1 \
    --out-dir $D/data/processed_v4 --workers 4
```

* Resumable: if it stops, run the same command again — finished scans are skipped.
* Try `--limit 3` first and open a few PNGs in `processed_v4/slices/` to check
  they look like brains (axial `_axNN.png`, coronal `_coNN.png`).
* Then zip and upload **both** exports to Google Drive (`MyDrive/` root or `MyDrive/neuroguard/`):

```bash
cd $D/data && zip -qr processed_v4.zip processed_v4 -x 'processed_v4/registered/*'
```

(`processed_v3.zip` is already on your Drive from the October runs.)

---

## 3 · Colab: find the best input, then the best models (2 sessions)

Open `ml/mri/colab_train.ipynb` in Colab (File → Open → GitHub → this repo,
branch `feat/mri-honest-eval`), set a T4 GPU, run top to bottom.

* **Block A (~2–2.5 h):** ResNet-18 on v3 (honest baseline) vs registered
  axial / coronal / both. The comparison table prints a verdict per row.
* Set `BEST_INPUT` in section 3 to the winner, then
* **Block B (~3–4 h):** improved ResNet recipe + SWA, ViT-B/16, DINOv2 on that
  input, then their ensemble, compared and calibrated.

Everything is saved to `MyDrive/neuroguard/experiments/` after each run, so a
disconnect costs at most one experiment. **Send me that folder (or
`comparison.json`)** and I'll read the results with you.

---

## 4 · Deploy the winner locally (30 min)

The last notebook cell retrains the chosen members on all subjects and
downloads `neuroguard_deploy_bundle.zip`. On the Mac:

```bash
unzip ~/Downloads/neuroguard_deploy_bundle.zip -d ml/experiments/
P="B_resnet_v2_swa B_vit_b16 B_dinov2_s --calibration-from B_ensemble"   # or just B_resnet_v2_swa
backend/.venv/bin/python ml/mri/promote.py $P --stage-only
backend/.venv/bin/python ml/mri/external_validate.py \
    --oasis1-dir ~/Downloads/7th_sem/PP-2/external/oasis1/disc1 --model-dir ml/experiments/_staged
backend/.venv/bin/python ml/mri/promote.py $P --yes
cd backend && .venv/bin/python -m pytest tests/ -q
```

If the winner uses registered input, the backend needs the same packages:
`backend/.venv/bin/pip install antspyx nilearn deepbet`. Uploads then take
~20–40 s (registration) instead of ~2 s. That is fine for a local demo; it
will not fit Render's free tier, which keeps serving the current model.

The external OASIS-1 test is run **once**, after choosing — never to choose.
