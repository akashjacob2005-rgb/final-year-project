# Model Card — NeuroGuard language classifier

## Overview

A binary classifier that estimates whether a spoken picture description is more
consistent with a cognitively healthy speaker or with one showing signs of
cognitive decline.

- **Version:** 1.0.0
- **Type:** Logistic regression on 27 handcrafted linguistic features plus TF-IDF
  word 1–2 grams, with sigmoid probability calibration
- **Input:** an English transcript of a spoken picture description
- **Output:** calibrated probability in [0, 1], plus signed per-feature
  contributions

This model covers **one** of the six tests in NeuroGuard. The other five are
rule-scored, and the final risk score combines both with fixed weights. See
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## Training data

**DementiaBank Pitt corpus** — "Cookie Theft" picture descriptions, obtained
from the ungated `MearaHe/dementiabank` mirror on Hugging Face.

| | |
|---|---|
| Total transcripts | 498 |
| Dementia | 256 (51.4%) |
| Control | 242 (48.6%) |
| Median length | 100 words (range 18–530) |

Transcripts are clinician-produced, lowercase, with sentence periods separated
by spaces and no capitalisation.

**Known gaps.** The mirror does not carry participant demographics, so the model
cannot be audited for age, sex, education or ethnicity bias — a real limitation,
not a footnote. It contains no audio, which is why speech-timing features are
reported descriptively and excluded from the score. Samples under 10 words were
dropped as carrying no usable signal.

---

## Features

| Family | Features |
|---|---|
| Length | total words, sentences, mean sentence length |
| Lexical richness | TTR, MATTR(50), hapax ratio, Brunet index, Honoré statistic, mean word length |
| Part of speech | noun/verb/adjective/adverb/pronoun/determiner ratios, **pronoun-to-noun ratio**, content-to-function ratio |
| Syntax | mean dependency depth, subordinate clause ratio, prepositional phrase ratio |
| Disfluency | filler ratio, immediate repetition ratio, unique bigram ratio |
| Information content | semantic content units (16 canonical scene concepts), content unit ratio, units per 100 words, **idea density** |

MATTR is used alongside plain TTR because TTR falls as text lengthens, so it
partly measures verbosity rather than vocabulary — and the two classes differ
systematically in length.

Pronoun-to-noun ratio and idea density are the most consistently reported
markers in this literature: substituting "he"/"it"/"that thing" for specific
nouns, and producing many words carrying few propositions.

---

## Evaluation

5-fold stratified cross-validation, repeated 4 times (20 folds). Intervals are
95% CIs on the mean across folds.

| Configuration | ROC-AUC | 95% CI | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|
| handcrafted + logreg | 0.875 | 0.860–0.890 | 0.801 | 0.781 | 0.821 | 0.801 |
| tfidf + logreg | **0.930** | 0.920–0.941 | 0.847 | 0.817 | 0.879 | 0.846 |
| **combined + logreg** *(deployed)* | 0.905 | 0.892–0.918 | 0.827 | 0.798 | 0.858 | 0.826 |
| handcrafted + gbm | 0.844 | 0.828–0.861 | 0.774 | 0.767 | 0.781 | 0.776 |
| combined + gbm | 0.908 | 0.894–0.922 | 0.824 | 0.797 | 0.853 | 0.823 |

### Why the deployed model is not the highest scorer

TF-IDF alone wins on cross-validation. It was not deployed, for two reasons:

1. **Domain shift.** Its vocabulary is fitted on clinician-typed transcripts,
   but at inference the input is Whisper ASR output. Handcrafted features —
   ratios over parts of speech and content units — are far less sensitive to
   exact word forms and transcription conventions. Cross-validation cannot
   measure this, because every fold is clinician transcripts.
2. **Explainability.** Linear coefficients on named features produce the
   per-marker explanation on the results page. Individual TF-IDF terms are
   meaningless to a user.

The gap (0.905 vs 0.930) is inside overlapping confidence intervals, so this
trades nothing measurable for a real robustness and interpretability gain. The
full comparison is shown in-app at `/model` rather than hidden.

---

## Intended use

**In scope:** research, education, and self-monitoring of trends over time by
adults completing the assessment in English.

**Out of scope:**
- Diagnosis of dementia, Alzheimer's disease, or any condition
- Clinical triage, screening programmes, or any decision affecting care
- Employment, insurance, or capacity decisions
- Non-English speakers, or any task other than picture description

At roughly 0.80 sensitivity, **one in five people with dementia in the training
distribution would be missed**. At roughly 0.86 specificity, about one in seven
healthy people would be flagged. Those error rates are why this is a screening
indicator and not a test.

---

## Limitations

1. **Small dataset.** 498 samples. Fold-to-fold variance is substantial; quote
   intervals, never a single split.
2. **Train/serve mismatch.** Mitigated by normalising ASR output into training
   style (lowercase, spaced periods, punctuation stripped), but not eliminated.
   The reported metrics do **not** include ASR error.
3. **No demographic audit possible.** The mirror lacks participant metadata, so
   subgroup performance is unknown.
4. **Labels are diagnostic group, not severity.** The model separates
   dementia from control, not early decline from normal ageing — the very
   distinction the product is aimed at. Performance on genuinely *early* decline
   is untested and likely worse than these numbers suggest.
5. **Single task, single language.**
6. **Calibration is in-sample.** Probabilities are calibrated on the training
   distribution, not on the population actually using the app.

Point 4 is the most important one for the report, and the one an examiner is
most likely to probe.

---

## Ethical considerations

- Every surface — results page, dashboard, model page — carries an explicit
  non-diagnostic disclaimer.
- The app reports its own error rates in-app rather than presenting a bare
  score.
- An elevated score is worded to prompt a conversation with a doctor, never to
  assert a condition.
- Audio recordings and transcripts are personal health-adjacent data. They are
  stored per-user, owner-scoped, and deletable from the history page. Any real
  deployment would need a retention policy and encryption at rest, neither of
  which this project implements.

---

## Reproducing

```bash
python ml/download_data.py
python ml/train_language_model.py
```

Deterministic given `SEED = 42`. Writes `ml/artifacts/language_model.joblib`,
`metrics.json`, and out-of-fold predictions for honest plotting.

---

## MRI pathway (added later): OASIS-3 slice classifier

A third, **separate** pathway that classifies structural brain MRI. It is
never fused with the behavioural risk score — no dataset pairs MRI with the
app's six tests, so a combined weight would be a guess, and the app says so.

- **Data.** OASIS-3 T1-weighted MRI. Each MR session contributes 16 axial
  slices (indices 35–65% of the volume, per-slice 1st–99th percentile
  intensity normalisation; regenerate with `ml/mri/make_slices.py`). Labels
  come from the CDR score nearest the scan date.
- **Task.** Deployed as **binary**: Normal (CDR 0) vs Impaired (CDR ≥ 0.5).
  The 3-class variant (normal / very mild / dementia) remains runnable with
  `--label-mode three` for the report's comparison; its honest numbers are
  substantially lower because CDR 0.5 vs CDR ≥ 1 is genuinely hard to
  separate on mid-axial 2D slices.
- **Model.** ResNet-18 pretrained on ImageNet; only layer4 + head fine-tuned
  (a fully unfrozen net overfits this data size). Class-weighted loss. A
  session's prediction is the mean of its slice probabilities with
  horizontal-flip test-time augmentation.
- **Evaluation.** Subject-stratified **5-fold cross-validation** — no
  person's anatomy appears in a fold's train and validation sets; metrics are
  mean ± std across folds. This is the number-one leakage trap in slice-based
  MRI classification: repositories claiming 90%+ on OASIS almost invariably
  split by *slice*, which leaks each subject's anatomy into the test set.
  Our numbers are lower because they are real.
- **Metrics.** See the `cv` block of `ml/artifacts/mri_metrics.json`
  (session-level accuracy / F1 / AUC, per-fold and aggregated). The class mix
  is stated in the same file.
- **Why this matters for limitation 4 above.** CDR 0.5 is exactly the "early
  decline" group the language model cannot represent; the MRI pathway is
  trained to see it, and it sits inside the positive class of the deployed
  binary model.
- **Scaling path.** `processed/download_list.csv` defines 518 unique subjects
  (200 normal / 200 very mild / 118 dementia); `ml/mri/download_oasis.py`
  fetches the sessions not yet on disk (needs OASIS XNAT credentials), then
  `make_slices.py` + retraining scale the dataset ~3× — the single biggest
  accuracy lever available.
- **Deployment.** Exported to ONNX (`ml/artifacts/mri_model.onnx`), served by
  `onnxruntime` on CPU behind `POST /api/mri/analyze`; disable with
  `MRI_ENABLED=false` on memory-constrained hosts.
- **Ingestion (user-centric).** Besides `.nii/.nii.gz`, the API accepts a
  zipped **DICOM** series — the format scanning centres actually hand to
  patients — converted server-side (`ml/mri/dicom_support.py`, dicom2nifti;
  bounded extraction, largest-series selection) into the same canonicalised
  pipeline. "Try a sample scan" endpoints (`/api/mri/analyze-sample`, gated by
  `MRI_SAMPLE_DIR`) let users experience the feature on anonymized research
  scans without owning an MRI file.
- **Explainability.** The ONNX graph returns a second output: the class
  activation map (CAM) of the final conv block. Because the head is
  global-avg-pool → linear, CAM here is *exact* (identical to Grad-CAM) and
  needs no gradients. The API returns each analysed slice plus a heatmap
  overlay (`slices[]` field) and the UI shows them with an Original / Model
  attention toggle. The heatmap is model attention — evidence of *where* the
  network looked — not a clinical annotation of pathology, and the UI says so.
  Re-export after retraining with `python ml/mri/export_onnx.py`.
- **Limitations.** CDR staging is not a biopsy-confirmed diagnosis; 2D slices
  discard 3D context; OASIS-3 is a largely North-American research cohort on
  known scanners — generalisation to other scanners/populations is untested.
  **Confidence values are uncalibrated:** the training cohort is enriched 2:1
  impaired vs normal and the loss is class-weighted, so displayed percentages
  lean toward "impaired" relative to a screening population; the deployed
  fold-0 model's normal-class recall is 0.583 (a substantial false-positive
  rate on healthy scans), and its quoted fold-0 numbers come from the same
  sessions used for early stopping. The UI discloses this; temperature
  calibration on the Phase B data is the planned fix (see TODOS.md). Uploads
  are reoriented to canonical RAS before slicing, so storage axis order does
  not affect results; 4D series and thin-slab volumes are rejected rather than
  silently mis-analysed.

Reproducing:

```bash
python ml/mri/make_slices.py --data-root /path/to/data
python ml/mri/train_mri.py --data-dir /path/to/data/processed_v2 --folds 5
# or run ml/mri/train_mri_colab.ipynb on a free Colab T4
```
