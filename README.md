# NeuroGuard

A web platform for early detection of cognitive decline. Users create an
account, complete a six-test cognitive battery in the browser, and receive a
screening score backed by a model trained on real clinical data — plus a
longitudinal view of how their scores change over time.

> **This is a research and educational tool. It does not diagnose dementia,
> Alzheimer's disease, or any other condition.** Sensitivity and specificity are
> both around 0.80–0.86 (see below), which is nowhere near diagnostic accuracy.

---

## What it does

Six tests, five of them drawn from the Montreal Cognitive Assessment (MoCA):

| # | Test | Domain | How it is scored |
|---|------|--------|------------------|
| 1 | **Memory recognition** | Memory | Five objects studied, filled 10s delay, then picked out of a 15-item grid. Hits minus false alarms. |
| 2 | **Picture description** | Language | Spoken description of a kitchen scene, transcribed by Whisper and scored by a **trained classifier**. |
| 3 | **Trail Making B** | Executive | Click 1→A→2→B→3→C→4→D→5→E. Completion time and errors. |
| 4 | **Serial sevens** | Attention | Subtract 7 from 100, five times. MoCA scoring, including the chained-error rule. |
| 5 | **Digit span backward** | Working memory | Digits shown one at a time, typed back in reverse. Adaptive span. |
| 6 | **Verbal fluency** | Language | As many words as possible from one letter in 60 seconds. |

Test 2 is the machine-learning core. Tests 1 and 3–6 are scored by published
MoCA rules and converted to age- and education-adjusted z-scores.

---

## Model performance

Trained on **498 labelled DementiaBank Pitt "Cookie Theft" transcripts**
(256 dementia / 242 control), evaluated with 5-fold stratified cross-validation
repeated 4 times (20 folds total).

| Configuration | ROC-AUC | 95% CI | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|
| handcrafted + logreg | 0.875 | 0.860–0.890 | 0.801 | 0.781 | 0.821 |
| tfidf + logreg | **0.930** | 0.920–0.941 | 0.847 | 0.817 | 0.879 |
| **combined + logreg** *(deployed)* | 0.905 | 0.892–0.918 | 0.827 | 0.798 | 0.858 |
| handcrafted + gbm | 0.844 | 0.828–0.861 | 0.774 | 0.767 | 0.781 |
| combined + gbm | 0.908 | 0.894–0.922 | 0.824 | 0.797 | 0.853 |

**Why the deployed model is not the top scorer.** TF-IDF wins on cross-validation,
but its vocabulary is learned from clinician-typed transcripts while at
inference the input is Whisper ASR output. The combined model is statistically
indistinguishable (0.905 vs 0.930, overlapping intervals) and includes
handcrafted features — pronoun-to-noun ratio, idea density, content units — that
depend far less on exact word forms, so it should degrade less under that domain
shift. Being linear, it also yields directly interpretable per-feature
contributions for the results page.

Reproduce with `python ml/train_language_model.py`.

### What is and is not machine-learned

Worth being precise about, because it is the first thing an examiner should ask:

| Component | Status |
|---|---|
| Picture description | **Trained model.** Real data, cross-validated, metrics above. |
| The five other tests | **Rule-scored.** MoCA rules plus normative z-scores. No model. |
| Final combination | **Fixed weights.** No dataset exists of people taking *these six tests* with diagnostic labels, so there is nothing to fit against. The weighting is a documented judgement. |
| Speech timing (pauses, rate) | **Descriptive only.** The training corpus is text with no audio, so these could not be validated. Displayed, but excluded from the score. |

---

## Running it

### With Docker (recommended)

```bash
cp .env.example .env
# Set SECRET_KEY — generate one with: openssl rand -base64 48
docker compose up --build
```

Open <http://localhost:8000>. FastAPI serves the API and the built React app
from a single origin, so the auth cookie is first-party and there is no CORS to
configure.

### Locally, for development

Two terminals.

**Backend**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r ../ml/requirements.txt
python -m spacy download en_core_web_sm

# Fetch data and train the model (~1 minute; artifacts are committed, so this
# is only needed if you want to retrain)
python ../ml/download_data.py
python ../ml/train_language_model.py

uvicorn app.main:app --reload
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` to the backend.

Without `DATABASE_URL` set, the backend uses a local SQLite file — no database
setup needed to get started.

### Tests

```bash
cd backend && ./.venv/bin/python -m pytest tests/ -v
```

13 tests covering the auth flow, a full six-test assessment, cross-user access
isolation, and the history endpoints.

---

## Architecture

```
React SPA (Vite)  ──JWT in httpOnly cookie──>  FastAPI
  Login · Dashboard · Assessment · Results        │
  · Monitoring · Model & Method                   │
                                                  ├── speech.py     Whisper STT + timing
                                                  ├── features.py   27 linguistic features
                                                  ├── language_model.py  trained classifier
                                                  ├── scoring.py    MoCA rules + norms
                                                  └── fusion.py     weighted combination
                                                  │
                                            SQLAlchemy 2.0
                                                  │
                                            PostgreSQL
                                       (SQLite for local dev)
```

`ml/features.py` is imported by both the training script and the backend rather
than copied. If training and inference computed features differently the
deployed model would be silently wrong, and that class of bug is invisible until
someone checks the numbers by hand.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design and
[docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the model's intended use and
limitations.

---

## Project layout

```
backend/
  app/
    main.py          FastAPI entrypoint, also serves the built SPA
    config.py        Settings from environment / .env
    models.py        SQLAlchemy ORM
    schemas.py       Pydantic request/response models
    core/            security (argon2, JWT), dependencies
    api/routes/      auth, assessment, history
    ml/              inference: speech, language model, scoring dispatch, stimuli
  tests/
ml/
  download_data.py       fetch the DementiaBank transcripts
  features.py            SHARED feature extraction (training + inference)
  scoring.py             MoCA scoring rules and normative tables
  fusion.py              final risk combination
  train_language_model.py
  artifacts/             trained model + real metrics (committed, ~123KB)
frontend/
  src/pages/             Login, Signup, Dashboard, Assessment, Results, History, ModelInfo, Profile
  src/tests/             the six test components
  public/assets/         kitchen-scene.svg — original artwork for test 2
frontend-legacy/         the earlier vanilla-JS prototype, kept for reference
```

---

## Security

- Passwords hashed with **argon2id**.
- JWTs in **httpOnly, SameSite cookies** — unreadable by JavaScript, so XSS
  cannot exfiltrate a session.
- Refresh tokens are **stored hashed and rotated on use**, so a stolen token
  works at most once.
- Login errors are identical for unknown-email and wrong-password, so the
  endpoint cannot be used to enumerate accounts.
- Test stimuli are generated and stored **server-side**; scoring never trusts
  what the client claims the correct answer was.
- Sessions are owner-scoped — requesting another user's result returns 404.

Set `COOKIE_SECURE=true` for any HTTPS deployment.

---

## Known limitations

1. **498 training samples is small.** Fold-to-fold variance is substantial;
   always quote the interval, never a single split.
2. **Trained on clinician transcripts, deployed on ASR.** ASR output is
   normalised to match the training style, but the domain shift is real and is
   not captured by the reported metrics.
3. **Normative tables are provisional.** The age/education means and SDs in
   `ml/scoring.py` are approximations from published literature ranges, not
   institution-validated norms. Raw responses are stored for every test, so
   sessions can be rescored once better norms are available.
4. **Trained on English picture description only.** No validity claim for other
   languages or tasks.
5. **The fusion weights are a judgement, not a fitted parameter.**
