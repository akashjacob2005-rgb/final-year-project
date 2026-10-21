# MRI model: experiment protocol and adoption rule

How a candidate MRI model earns its way into `ml/artifacts/`. Written before
the experiments were run, so the bar cannot move after seeing the numbers.

## Why the old 75.6 % is not the baseline

`train_mri.py` used to early-stop on the same fold it then reported, and the
deployed artifact was the fold-0 model (trained on 4/5 of the data). Both are
standard in student projects and both are optimistic: the epoch was chosen by
peeking at the test fold, and nothing was ever trained on all subjects.
Expect the honest re-run (`e01_baseline_honest`) to land **below** 75.6 %.
That drop is the correct starting point, not a regression.

## The harness (`ml/mri/train_mri.py --experiment NAME`)

| Rule | Flag | Why |
|---|---|---|
| Fixed epoch budget, last epoch reported | `--sched cosine --no-early-stop --epochs E` | the reported fold is never used for selection; the same recipe can be refit on all data |
| `E` chosen on an inner split, once | `--inner-val 0.15 --early-stop` (`e00_innerval`) | model selection on training subjects only |
| Repeated CV | `--repeats 3` | with 36 sessions per fold, one session = 2.8 pts; single runs cannot separate recipes |
| Identical folds across candidates | same `--folds --fold-seed` and manifest | enables the paired test |
| Per-slice OOF predictions | `--save-slice-preds` | aggregation variants evaluated offline, no retraining |
| Mixed cohorts stratified by source | `--source-aware` | OASIS-1 1.5 T vs OASIS-3 3 T; reports per-source metrics |
| Never writes `ml/artifacts/` | `--experiment` | only `promote.py` does, after checks |
| Final model = all subjects, same recipe | `--final-fit` | the `cv` block is then an honest estimate of the deployed model |

Outputs land in `ml/experiments/<name>/`: `config.json` (args, git sha,
manifest hash), `summary.json`, `preds_sessions.csv`, `preds_slices.csv`,
`history.json`, and `final/` (`model.pt`, `model.onnx(.data)`, `mri_metrics.json`).
Weights and ONNX files are git-ignored; the CSV/JSON summaries are committed
so every number in the report is reproducible.

## Analysis (`ml/mri/compare_experiments.py`)

* **Pooled out-of-fold** accuracy / balanced accuracy / AUC / per-class recall
  (every session predicted once per repeat) with bootstrap 95 % CIs, next to
  the mean ± std over folds that `mri_metrics.json` reports.
* **Cross-fitted calibration**: for each fold, a temperature and a decision
  threshold are fitted on the *other* folds' OOF predictions and applied to
  that fold. The resulting "calibrated balanced accuracy" is therefore itself
  honest. This is the fix for normal-class recall 0.58: the argmax at 0.5
  inherits the 2:1 impaired training prior.
* **Paired comparison** vs `--baseline` on identical folds: per-fold AUC
  deltas, paired bootstrap CI of the pooled AUC delta, McNemar on per-session
  correctness at the cross-fitted threshold.
* **Aggregation variants** from slice predictions: mean, median, trimmed
  mean, top-4, mean-logit.

## Adoption rule

A candidate replaces the deployed artifact only if, against
`e01_baseline_honest` on identical folds with 3 repeats:

1. pooled OOF AUC improves by **≥ 0.02** and the paired bootstrap 95 % CI of
   the delta excludes 0 (`compare_experiments.py` prints `ADOPT` when 1–2 hold);
2. balanced accuracy at the cross-fitted threshold improves by **≥ 2 pts** and
   is not worse in any repeat by more than 1 pt;
3. normal-class recall at the chosen threshold is **≥ 0.70**;
4. it is a single ResNet-18 (or SWA / weight-soup of ResNet-18s) ONNX ≤ 50 MB
   with the `cam` output, backend tests green, and Render memory measured once
   with `MRI_ENABLED=true` (TODOS #4);
5. the OASIS-1 discs 1–2 external test is run **once, after** adoption and
   reported (`external_validate.py --model`) — never to choose between candidates.

Anything under ~3 pts on a single run is noise at this sample size.

## Experiment sequence

| Session | Experiments | Question |
|---|---|---|
| A (~2.5 h T4) | `e00_innerval` (pick E), `e01_baseline_honest`, `e02_recipe_v2` (AdamW, label smoothing, aug v2, frozen BN), `e03_recipe_v2_layer3` (+ layer3, discriminative LR) | honest reference; cheap recipe wins |
| B (~2.5 h) | `e04_swa` (weight averaging over the cosine tail), `e05_seeds3` (3-seed probability ensemble = upper bound; weight soup deployable) | averaging; if ensemble − SWA < 1 pt, deploy SWA |
| C (when `processed_v3` is on Drive) | `e06_v3_pooled --source-aware` (OASIS-1 discs 3–12 with a recorded CDR only — young no-CDR controls are an age shortcut) | more data; OASIS-3 subset is the apples-to-apples number |
| D (optional, new export) | 24 axial + 24 coronal slices at 30–70 % | input; only on adoption does `preprocess.py` change (training, backend and tests together) |

Deferred: attention-MIL pooling, age as auxiliary input (report an age-only
logistic baseline regardless), registered/skull-stripped input (cannot run on
the Render free tier → research ablation only).

## Deploying a winner

```bash
# Colab: --final-fit on the winner, then compare_experiments.py --calibrate WINNER;
# download the bundle, unzip into ml/experiments/ on your Mac, then:
backend/.venv/bin/python ml/mri/promote.py WINNER          # dry run: all checks
backend/.venv/bin/python ml/mri/promote.py WINNER --yes    # writes ml/artifacts/
cd backend && .venv/bin/python -m pytest tests/ -q
backend/.venv/bin/python ml/mri/external_validate.py --oasis1-dir <discs 1-2>   # once
# update docs/MODEL_CARD.md numbers, commit, push only with approval (auto-deploys)
```

`promote.py` merges `calibration.json` into `mri_metrics.json`; the backend
(`mri_model.calibration()`) then rescales the session probability and decides
at the fitted threshold, and the UI swaps its "uncalibrated" caveat.
