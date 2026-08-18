# NeuroGuard — build status and report notes

## What is built

| Area | Status |
|---|---|
| Data pipeline | **Done.** 498 labelled DementiaBank transcripts, downloaded and normalised. |
| Feature extraction | **Done.** 27 linguistic features, shared between training and inference. |
| Model training | **Done.** 5 configurations, 20-fold CV, calibrated, artifact saved. ROC-AUC 0.905 deployed. |
| Structured scoring | **Done.** MoCA rules for 5 tests plus age/education normative z-scores. |
| Fusion | **Done.** Transparent weighted combination with documented constants. |
| Backend API | **Done.** Auth, assessment, history, model transparency. 13 tests passing. |
| Database | **Done.** 5 tables. Postgres in Docker, SQLite locally. |
| Speech-to-text | **Done.** faster-whisper with timing feature extraction. |
| React frontend | **Done.** 9 pages, 6 interactive test components. Builds clean. |
| Docker | **Done.** Multi-stage image, compose with Postgres. |
| Documentation | **Done.** README, architecture, model card. |

## What is not built

1. **Cloud deployment — deliberately deferred.** The Docker image builds and the
   full stack runs locally against Postgres. Publishing to Render/Railway/Fly is
   configuration rather than code, and is being held until the UI has been
   exercised by hand.
2. **Alembic migrations.** The app currently calls `create_all` at startup. Fine
   for this project; a production system needs versioned migrations.
3. **PDF report export.** Results are on-screen only.
4. **Browser-verified interaction.** The API and the audio pipeline are both
   verified end to end, but the six interactive components — timers,
   MediaRecorder, Trail Making click tracking — have been exercised only through
   the API layer, not by clicking through them. **Do this first.**

---

## Viva preparation

The questions most likely to be asked, and the honest answers.

### "What exactly did you train, and on what?"

A binary classifier on 498 labelled DementiaBank Pitt Cookie Theft transcripts
(256 dementia, 242 control). 27 handcrafted linguistic features plus TF-IDF,
logistic regression with sigmoid calibration. Evaluated with 5-fold stratified
CV repeated 4 times. ROC-AUC 0.905 (95% CI 0.892–0.918).

### "Is the whole system machine-learned?"

No, and the app says so on its `/model` page. One of six tests is a trained
model. The other five are scored by published MoCA rules and normative z-scores.
The final combination uses fixed, documented weights because no public dataset
exists of people taking these six tests with diagnostic labels — so there is
nothing to fit fusion weights against.

Claiming otherwise would be the easiest way to lose marks. Owning it is the
strongest answer available.

### "Why isn't your best model deployed?"

TF-IDF alone scored 0.930 versus 0.905 for the deployed combined model, but the
intervals overlap. TF-IDF's vocabulary is fitted on clinician-typed transcripts
while the app feeds it Whisper ASR output; handcrafted features are far less
sensitive to that shift, and being linear they give interpretable per-marker
explanations. Cross-validation cannot measure the domain shift because every
fold is clinician transcripts.

### "What is the biggest weakness?"

The labels are diagnostic group — dementia versus control — not severity. So the
model separates established dementia from healthy controls, **not** early decline
from normal ageing, which is what the product is aimed at. Performance on
genuinely early decline is untested and probably worse than the headline number.

Second weakness: the normative tables in `ml/scoring.py` are approximations from
published ranges, not validated norms.

### "How do you know training and inference agree?"

`ml/features.py` is imported by both, never copied. The saved artifact contains
only stock scikit-learn objects, so no custom class can fail to unpickle after a
refactor.

### "What stops someone cheating?"

Stimuli are generated and stored server-side and scoring compares against the
stored copy, never against what the client sends. The Trail Making expected path
is stripped before the stimuli reach the browser. Randomising per session also
means repeat testing measures memory rather than familiarity with the app.

---

## Figures worth putting in the report

All are derivable from committed artifacts:

1. **Model comparison table** — `ml/artifacts/metrics.json`, all five
   configurations with confidence intervals.
2. **ROC curve** — from `artifacts/oof_probabilities.npy` and `labels.npy`. Use
   the out-of-fold predictions, never the in-sample refit.
3. **Calibration curve** — same arrays.
4. **Feature contributions** — a screenshot of the results page markers panel.
5. **Architecture diagram** — `docs/ARCHITECTURE.md`.
6. **Longitudinal trend** — screenshot the monitoring page after two or three
   assessments.

## Suggested report structure

1. Introduction and motivation — early detection matters, existing tools require
   a clinician
2. Literature review — linguistic markers of cognitive decline; MoCA
3. Data — DementiaBank, class balance, what is missing (demographics, audio)
4. Methodology — features, models compared, evaluation protocol
5. System design — architecture, database, security
6. Results — the comparison table and curves
7. Discussion — honest limitations, especially the group-versus-severity gap
8. Conclusion and future work — validated norms, prospective data collection,
   demographic audit
