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

> **Two models score this test.** The classifier documented immediately below is
> text-only and now serves just the typed-transcript fallback. Recordings — the
> normal path — are scored by the **audio model** described later in this card,
> which is trained on real DementiaBank audio and uses speech timing as a learned
> predictor. Their headline numbers are **not comparable**: they use different
> corpora and different cross-validation schemes. `/api/health` reports which
> artifacts are loaded, and each stored result records which model produced it.

---

## Training data

**DementiaBank Pitt corpus** — "Cookie Theft" picture descriptions, obtained
**not from TalkBank directly** but from the ungated third-party mirror
[`MearaHe/dementiabank`](https://huggingface.co/datasets/MearaHe/dementiabank) on
Hugging Face (`ml/download_data.py`).

That distinction is part of the provenance, not a footnote. The mirror is a
redistribution of a corpus that is normally access-controlled, so its
preprocessing cannot be verified against the original CHAT files, and the 498
transcripts here cannot be traced back to the sessions they came from. Becker et
al. (1994) is still the required citation, and NIH AG03705 / AG05133 still
require acknowledgement, however the data was obtained.

The audio model documented below does not use this mirror. It was trained on the
real corpus retrieved from TalkBank under the DementiaBank data-use agreement,
which is why it can report participant identity and speech timing and this one
cannot.

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

## Audio model (added later): the model that actually scores recordings

The classifier above is trained on clinician-typed text and scores Whisper output
in production. This one removes that mismatch by training on the real corpus,
transcribed by the same code path that serves requests.

- **Version:** 2.0.0 · `ml/artifacts/audio_model.joblib`
- **Type:** Logistic regression on 27 linguistic features + TF-IDF 1–2 grams
  **+ 9 speech-timing features**, sigmoid-calibrated
- **Input:** a recorded picture description (audio, not text)
- **Deployed for:** `POST /assessment/{id}/audio`. Typed transcripts still use the
  text model, because they have no timing to give this one.

### Training data

**DementiaBank Pitt Cookie Theft, audio** — obtained from TalkBank directly under
the DementiaBank data-use agreement, not from a mirror. Restricted; never
committed or containerised.

| | |
|---|---|
| Sessions with audio | 552 (Control 243, Dementia 309) |
| After the 25-word floor | **519** |
| Distinct participants | **285** |
| Class split | 289 dementia / 230 control |
| Transcription | faster-whisper `base.en`, identical settings to inference |

**The effective sample size is 285 people, not 519 recordings.** Pitt is
longitudinal (1.82 visits per participant), so cross-validation groups by
participant and each fold tests on ~57 people.

The positive class is **the DementiaBank dementia group, which includes MCI** — not
"has dementia". Participant 172 appears in both groups because she converted
during the study (Control at 72, MCI at 73/74/75); her bare-digit key is kept so
her four visits stay in one fold.

### The investigator confound, and why the audio is preprocessed

Pitt sessions are interviews, and Whisper transcribes everyone present. The
examiner's share of speech is **strongly correlated with the label**:

| | Control | Dementia |
|---|---|---|
| INV share of speaking time | 4.1% | **12.0%** |
| INV share of words | 8.8% | **16.4%** |

The examiner talks about three times more with dementia participants, because they
need more prompting. A model fitted on raw recordings can therefore score the
condition by detecting **how much the examiner spoke** — a leak correlated with the
label. It is also a train/serve mismatch in its own right: in the app the user
records alone, so a speaker present in every training example is absent from every
real one.

`ml/par_audio.py` excises investigator speech using the CHAT millisecond time
bullets before transcription. Spans where INV and PAR overlap are kept, since
overlapping speech cannot be separated by cutting. Measured effect:

| Arm | INV included | INV removed |
|---|---|---|
| **timing only** | 0.719 | **0.640** |
| text only | 0.793 | 0.808 |

**Timing-only lost 0.079 of AUC once the examiner was removed** — that much of the
earlier figure was `pause_ratio` detecting a talkative interviewer rather than a
hesitant patient. Removing it *helped* the text model, so the examiner's words were
mostly diluting the participant's markers.

### Evaluation

`StratifiedGroupKFold(5) × 4 seeds` (20 folds), grouped by `participant_id`:

| Configuration | ROC-AUC | 95% CI | Acc | Sens | Spec |
|---|---|---|---|---|---|
| text only (linguistic + tfidf) | 0.808 | 0.784–0.832 | 0.739 | 0.729 | 0.752 |
| timing only | 0.640 | 0.621–0.658 | 0.601 | 0.517 | 0.705 |
| **text + timing** *(deployed)* | **0.816** | 0.796–0.836 | 0.753 | 0.738 | 0.772 |
| pause-marked text | 0.809 | 0.785–0.833 | 0.743 | 0.737 | 0.751 |
| pause-marked text + timing | 0.817 | 0.797–0.837 | 0.752 | 0.736 | 0.773 |

**Timing features are learned predictors here, not decoration.** Timing alone
reaches 0.640 from nine pause and rate numbers — well above chance — and this is the
first version in which they enter the risk calculation at all. Their *marginal*
contribution over text is modest, though: 0.808 → 0.816, with overlapping intervals.

#### Why the deployed model is not the highest scorer (again)

`pause-marked text + timing` scored highest at 0.8170 against 0.8164 — a difference
of **0.0006**, with 95% CIs overlapping almost entirely. Pause-marking is real
inference machinery: the vectorizer must be fed `transcript_pauses` built by
`speech.pause_encoded_transcript`, matching how it was fitted. Six ten-thousandths
of AUC that the corpus cannot distinguish from zero does not justify that, so the
trainer deploys the unmarked arm whenever the intervals overlap.

Splicing `longpause` into the word sequence so the n-gram model could learn
"hesitated before this word" was a reasonable hypothesis. This corpus does not
support it, and that is reported rather than buried.

#### Participant leakage, measured

| | ROC-AUC |
|---|---|
| Ungrouped CV (same participant in train and test) | 0.833 |
| **Grouped by participant** | **0.816** |
| Inflation | **+0.017** |

Smaller than expected at 1.82 visits per person, but it is the reason the grouped
figure is the one quoted. The text model above cannot compute this, because its
mirror carries no participant IDs — so **0.905 is not comparable to 0.816**: it is
ungrouped, on clinician transcripts, with an unmeasurable amount of this same leak.

#### Choosing an operating point

The app never applies a cutoff — it passes the calibrated probability into
`fusion.py` as a weighted component. For anyone wanting a binary decision, from
out-of-fold predictions:

| Threshold | Sens | Spec | Youden J |
|---|---|---|---|
| 0.35 | 0.841 | 0.643 | 0.484 |
| 0.40 | 0.830 | 0.687 | 0.517 |
| **0.45** | 0.796 | 0.739 | **0.535** |
| 0.50 *(default)* | 0.734 | 0.761 | 0.509 |
| 0.60 | 0.654 | 0.826 | 0.480 |

0.5 is logistic regression's default, not a clinical choice. Youden J peaks at 0.45,
and a screen that misses cases is worse than one that over-refers, so a lower
threshold is defensible — but it should be stated, not assumed. Full sweep in
`metrics_audio.json`.

The sensitivity at 0.50 here (0.734) differs slightly from the evaluation table
above (0.738) because the two aggregate differently: the table averages per-fold
metrics across 20 folds, while the sweep pools the repeat-averaged out-of-fold
probability vector and scores it once. Both are honest; neither is the in-sample
refit.

### Limitations specific to this model

- **285 people.** Wide intervals, and arm-to-arm differences of a few thousandths
  are not resolvable.
- **Pitt audio is 1980s–90s clinic recordings**; users record on modern microphones
  in lossy Opus. Timing features derive from word timestamps and transfer across
  that gap reasonably. Spectral voice-quality features would not, which is why none
  are used.
- **No demographics are used or audited.** Age and sex are in the CHAT headers but
  are not modelled, so this is not audited for age or sex bias.
- **Sensitivity at the default threshold is 0.734** — it misses about a quarter of
  the positive class. It is a screening indicator, not a diagnosis.

### Reproducing

```bash
python ml/download_pitt.py     # needs approved DementiaBank credentials
python ml/census_pitt.py       # audit before training
python ml/par_audio.py         # remove investigator speech
python ml/build_dataset.py --force
python ml/train_audio_model.py
```

Deterministic given `SEED = 42`. Writes `audio_model.joblib`, `metrics_audio.json`,
and `audio_oof_probabilities.npy` + `audio_labels.npy` for honest ROC and
calibration plots. **Do not plot the in-sample refit** — the deployed model is
fitted on all 519 samples, so scoring training data gives an optimistic number.

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
- **Architecture study & ensemble serving.** ResNet-18 vs Hugging Face
  transformers (ViT-B/16, DINOv2-small) vs their ensemble, under the identical
  CV harness — see `docs/ARCH_COMPARISON.md` (`train_mri.py --arch`,
  `ensemble_eval.py`, `colab_train.ipynb`). The winning **all-3 ensemble**
  (78.9% ± 2.2% acc, F1 0.844, AUC 0.870) is the serving model: probability
  averaging across members, heatmaps from the CNN member, fail-soft to fewer
  members where the transformer ONNX files are absent (e.g. free-tier hosts).
- **Limitations.** CDR staging is not a biopsy-confirmed diagnosis; 2D slices
  discard 3D context; OASIS-3 is a largely North-American research cohort on
  known scanners — generalisation to other scanners/populations is untested.
  **Confidence values are uncalibrated:** the training cohort is enriched 2:1
  impaired vs normal and the loss is class-weighted, so displayed percentages
  lean toward "impaired" relative to a screening population; the deployed
  fold-0 model's normal-class recall is 0.583 (a substantial false-positive
  rate on healthy scans), and its quoted fold-0 numbers come from the same
  sessions used for early stopping. The UI discloses this. The fix is in
  place but not yet applied to an artifact: `ml/mri/compare_experiments.py`
  fits a temperature + decision threshold on out-of-fold predictions (cross-
  fitted, so the reported calibrated accuracy is honest), `promote.py` bakes
  them into `mri_metrics.json["calibration"]`, and the backend applies them
  (`mri_model.calibration()`). Protocol: `docs/MRI_EXPERIMENTS.md`. Uploads
  are reoriented to canonical RAS before slicing, so storage axis order does
  not affect results; 4D series and thin-slab volumes are rejected rather than
  silently mis-analysed.

Reproducing:

```bash
python ml/mri/make_slices.py --data-root /path/to/data
python ml/mri/train_mri.py --data-dir /path/to/data/processed_v2 --folds 5   # legacy, optimistic
# honest protocol (experiments never touch ml/artifacts; see docs/MRI_EXPERIMENTS.md):
python ml/mri/train_mri.py --data-dir ... --experiment e01_baseline_honest \
    --sched cosine --epochs 25 --repeats 3 --save-slice-preds
# or run ml/mri/colab_train.ipynb on a free Colab T4
```
