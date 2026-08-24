# ml/ — training pipeline and shared scoring code

Everything here is imported by the backend at inference time as well as used for
training. **Do not duplicate this code into `backend/`** — a single definition
imported twice is what stops training and serving from drifting apart.

## Files

| File | Role | Used by |
|---|---|---|
| `download_data.py` | Fetch and normalise the DementiaBank transcripts | training |
| `features.py` | 27 linguistic features. **Single source of truth.** | training **and** inference |
| `scoring.py` | MoCA scoring rules and normative z-score tables | inference |
| `fusion.py` | Combine language probability with structured z-scores | inference |
| `train_language_model.py` | Cross-validate 5 configurations, pick, calibrate, save | training |
| `artifacts/` | Trained model + real metrics. Committed (~123KB). | inference |
| `download_pitt.py` | Fetch the **real** Pitt corpus, with audio (credentials required) | training |
| `build_dataset.py` | Transcribe that audio via the backend's own Whisper path | training |
| `train_audio_model.py` | Train on words **and** speech timing, grouped by participant | training |

## Training on real audio (optional, separate from the above)

`download_data.py` fetches an ungated HuggingFace mirror that is **text only**.
That is why `speech_rate_wpm`, `pause_count` and the rest are computed at
inference and then discarded — there was no audio to fit a weight against.

The three scripts at the bottom of the table fix that using the real DementiaBank
Pitt corpus:

```bash
python ml/download_pitt.py --limit 5   # smoke test first
python ml/download_pitt.py             # ~550 recordings
python ml/build_dataset.py             # Whisper over all of them
python ml/train_audio_model.py
```

Three things to know before running it:

- **Credentials.** DementiaBank is password-protected. Put `TALKBANK_EMAIL` and
  `TALKBANK_PASSWORD` in the repo-root `.env`. Request access at
  <https://talkbank.org/dementia/access/>.
- **`ml/data/pitt/` must never be committed.** The corpus is under the TalkBank
  data-use agreement: no redistribution, and it must stay out of the Docker image
  (`Dockerfile` does `COPY ml/ ./ml/`). It is gitignored and dockerignored. Cite
  Becker et al. (1994) and acknowledge NIH AG03705 / AG05133.
- **Nothing is wired into the app.** `train_audio_model.py` writes to
  `audio_model.joblib`, a new filename. The backend still loads
  `language_model.joblib` and behaves exactly as before. Deploying the new model
  is a deliberate separate step.

Expect a **lower** ROC-AUC than the current 0.905. Two reasons, both good ones:
the text is now real ASR output rather than clinician transcripts, and
cross-validation is grouped by participant. Pitt is longitudinal — the same
person recurs across yearly visits — so the old ungrouped scheme let the model
recognise the speaker rather than the condition. `metrics_audio.json` reports
both numbers under `leakage_demonstration`.

## Retraining

```bash
python ml/download_data.py          # writes data/transcripts.csv (498 rows)
python ml/train_language_model.py   # writes artifacts/language_model.joblib
```

Takes about a minute. Feature extraction is cached to `data/features.csv`;
delete that file if you change `features.py`, or the cache will be stale.

Deterministic given `SEED = 42`.

## Output

`train_language_model.py` prints the full comparison table and writes:

- `artifacts/language_model.joblib` — vectorizer, scaler, calibrated classifier.
  Only stock scikit-learn objects, so it cannot fail to unpickle after a
  refactor.
- `artifacts/metrics.json` — every model's scores, the deployment rationale, and
  the caveats. Served to the app at `/api/history/model-info` and rendered on
  the `/model` page.
- `artifacts/oof_probabilities.npy` + `labels.npy` — out-of-fold predictions, for
  honest ROC and calibration plots. Do **not** plot the in-sample refit.

## A warning about the demo numbers

The final model is refit on all 498 samples, so scoring a training transcript
gives an optimistic probability. The honest numbers are the cross-validated ones
in `metrics.json`. Use the out-of-fold arrays for any plot that goes in the
report.

## Changing the normative tables

The means and SDs in `scoring.py` are approximations from published literature
ranges, not validated norms. They are the weakest link in the pipeline.

Every test's raw response is stored in the database, so replacing these tables
and rescoring historical sessions is possible without asking anyone to retake
an assessment.
