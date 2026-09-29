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
| `census_pitt.py` | Read-only audit of what downloaded: audio coverage, class balance, participant count | training |
| `build_dataset.py` | Transcribe that audio via the backend's own Whisper path | training |
| `train_audio_model.py` | Train on words **and** speech timing, grouped by participant | training |
| `lexicon.py` / `build_proper_nouns.py` | Proper-noun list used by the feature extractor | training **and** inference |

## Training on real audio (optional, separate from the above)

`download_data.py` fetches an ungated HuggingFace mirror that is **text only**.
That is why `speech_rate_wpm`, `pause_count` and the rest are computed at
inference and then discarded — there was no audio to fit a weight against.

The three scripts at the bottom of the table fix that using the real DementiaBank
Pitt corpus:

```bash
python ml/download_pitt.py --limit 5   # smoke test first
python ml/download_pitt.py             # 552 cookie sessions, ~3.9GB, several hours
python ml/census_pitt.py               # check what actually arrived BEFORE training
python ml/build_dataset.py             # Whisper over all of them
python ml/train_audio_model.py
```

Use the backend venv — it is the only one with `requests`, `faster-whisper` and
spaCy: `backend/.venv/bin/python ml/download_pitt.py`.

### TalkBank download: four ways it silently "succeeds"

All four were hit for real. Each now has an explicit check, because every one of
them produces a corpus that looks complete and cannot be trained on.

1. **Placeholder credentials.** `TALKBANK_EMAIL=<you@example.com>` with the angle
   brackets kept from the template fails as `NOT_MATCHED`, which reads like a
   rejected account rather than a typo. Strip the brackets.
2. **The extension is `.wav`, not `.mp3`.** Every `.mp3` URL returns a genuine 404,
   which the downloader classifies as `missing` — the same status as a session
   whose audio was never released. So a wrong guess looks like "this corpus has no
   audio." See `MEDIA_EXTS`.
3. **A GET with no `Range` header returns ELEVEN BYTES.** `media.talkbank.org`
   answers `206` with `Content-Range: bytes 0-10/6814562`. The body begins with a
   valid `RIFF` header, so magic-byte sniffing passes and an 11-byte file is
   written as `ok`. See `FULL_RANGE`, and the `MIN_AUDIO_BYTES` floor that
   `census_pitt.py` / `build_dataset.py` apply when counting "has audio".
4. **Throttling looks like permanent absence.** The account is capped at
   ~200-270 KB/s *in aggregate*, not per connection — a fresh connection bursts
   near 900 KB/s for a few MB, which makes it look parallelisable. It is not.
   Sustained 4-way concurrency got the account cut off after ~69 files, and
   because nothing retried, 483 available sessions were recorded as
   `error:ConnectionError`. Hence `DEFAULT_WORKERS = 2`, `MAX_ATTEMPTS`,
   `CONSECUTIVE_FAILURE_LIMIT`, and a **non-zero exit code on an incomplete run**.
   Do not raise the worker count.

Because of (4), always re-run `download_pitt.py` until it exits 0. Completed files
are cached, so a re-run only fetches what is still absent. `census_pitt.py` is the
gate: it reports audio-backed sessions, class balance, and — the number that
actually governs the error bars — **distinct participants**. Sessions are processed
in sorted group order, so a partial run holds Control only and no positive class
at all.

Three more things to know before running it:

- **Credentials.** DementiaBank is password-protected. Put `TALKBANK_EMAIL` and
  `TALKBANK_PASSWORD` in the repo-root `.env`. Request access at
  <https://talkbank.org/dementia/access/>.
- **`ml/data/pitt/` must never be committed.** The corpus is under the TalkBank
  data-use agreement: no redistribution, and it must stay out of the Docker image
  (`Dockerfile` does `COPY ml/ ./ml/`). It is gitignored and dockerignored. Cite
  Becker et al. (1994) and acknowledge NIH AG03705 / AG05133.
- **Both models are served, each on the path whose inputs it has.**
  `train_audio_model.py` writes `audio_model.joblib`, a separate filename from
  `language_model.joblib`, and the backend loads both:

  | Request | Model | Why |
  |---|---|---|
  | `POST /{id}/audio` (a recording) | `audio_model.joblib` | has timing; trained on ASR text, so no domain shift |
  | `POST /{id}/transcript` (typed) | `language_model.joblib` | no timing to give the audio model |

  Timing is never zero-filled for the typed path: `StandardScaler` maps zeros to
  extreme negative z-scores meaning "spoke 0 words in 0 seconds", which would
  distort the probability rather than merely weaken it. A deployment without
  `audio_model.joblib` falls back to the text model instead of failing, so the
  artifact is optional. `/api/health` reports `audio_model_loaded`, and each
  stored result records which model produced it under `detail_json.model`.

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
