# External Validation — OASIS-1 (never-seen cohort)

**Date:** 2026-09-07 · **Model:** deployed binary MRI classifier (ResNet-18/ONNX, trained on OASIS-3 3T)
**Script:** `ml/mri/external_validate.py` · **Per-scan results:** `docs/external_validation_oasis1.csv`

## Why this experiment
Our headline numbers (75.6% ± 4.4% accuracy, AUC 0.84) come from subject-stratified
cross-validation *within* OASIS-3 — same scanners, same cohort, same era. External
validation asks the harder question: does the model carry real signal to data it has
never seen, from a **different study, different scanners (1.5T vs 3T), a different
decade, and a different file format**? For a screening claim, this is the test that
matters.

## Data
OASIS-1 cross-sectional, disc 1 (downloaded directly from the Wash U NRG server):
39 sessions. Volumes: `PROCESSED/MPRAGE/SUBJ_111` (N4-corrected, 1mm, native
anatomy — the closest analogue to a raw upload). Labels: per-session CDR from the
bundled metadata.

- **Labeled elderly subjects: 25** — 13 normal (CDR 0), 12 impaired (9 × CDR 0.5, 3 × CDR 1)
- **Young controls (no CDR assessed, presumed healthy): 14**

Methodological note: OASIS-1 ANALYZE headers misreport orientation; the true axial
plane was identified visually (axis 1, anterior-up as stored) before inference —
see the docstring in `external_validate.py`. Inference is byte-identical to the
deployed API: same slice fractions, normalisation, flip-TTA and aggregation.

## Results

| Metric | In-distribution (OASIS-3 5-fold CV) | **External (OASIS-1 disc1)** |
|---|---|---|
| Accuracy | 75.6% ± 4.4% | **64.0%** (n=25) |
| AUC | 0.84 ± 0.07 | **0.72** |
| Sensitivity | ~0.83 (fold-0) | **0.50** |
| Specificity | ~0.58 (fold-0) | **0.77** |
| Confusion [[tn,fp],[fn,tp]] | — | [[10,3],[6,6]] |
| Young healthy flagged impaired | — | **1 / 14 (7%)** |

## Interpretation (the honest version, for the report and viva)
1. **The model carries genuine signal to external data.** AUC 0.72 on a different
   study/scanner generation is far above chance (0.50) — the learned atrophy
   patterns are not an OASIS-3 artifact.
2. **Performance degrades under domain shift, as expected and now quantified:**
   AUC 0.84 → 0.72, accuracy 75.6% → 64%. Field strength (1.5T vs 3T), acquisition
   protocol and preprocessing lineage all differ. Quantifying this drop is the
   contribution; most student projects never measure it.
3. **Misses concentrate in CDR 0.5** — the earliest, subtlest stage (sensitivity
   0.50 overall; the CDR 1 cases score high). Consistent with the in-distribution
   finding that very-mild impairment is the hard boundary.
4. **False alarms stay controlled externally:** specificity 0.77 on labeled normals
   and only 1 of 14 young healthy controls flagged — reassuring for the
   false-positive caveat disclosed in the UI.
5. **Path to closing the gap:** train on the full 518-subject OASIS-3 list
   (Phase B), add probability calibration, and consider light intensity-domain
   augmentation to bridge scanner shift.

*Limitations: n=25 labeled subjects (one disc); OASIS-1 shares institutional
lineage with OASIS-3 (same research centre), so this is cross-study/cross-scanner
validation rather than fully independent multi-site validation.*
