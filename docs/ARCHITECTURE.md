# NeuroGuard — Architecture

## 1. Problem

Detect early cognitive decline from a battery of browser-administered tests, and
distinguish a cognitively healthy person from one showing signs of decline.

Three constraints shaped every decision below:

1. **No public dataset exists of people taking these six specific tests with
   diagnostic labels.** Anything claiming to be a learned end-to-end model would
   be fabricated. What *does* exist is labelled clinical speech data, so the
   language pathway is genuinely trained and the rest is honestly rule-scored.
2. **Inference input differs from training input.** The model learns from
   clinician-typed transcripts but receives Whisper ASR output at runtime.
3. **It has to actually deploy** — one container, one URL, free-tier memory.

---

## 2. System overview

```
┌──────────────────── React SPA (Vite + React Router) ────────────────────┐
│  /login  /signup  /dashboard  /assessment  /results/:id  /history       │
│  /model  /profile                                                        │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │  fetch(credentials: 'include')
                                 │  JWT access + refresh in httpOnly cookies
┌────────────────────────────────▼─────────────────────────────────────────┐
│                            FastAPI                                        │
│  /api/auth/*         signup, login, refresh, logout, profile              │
│  /api/assessment/*   start, submit test, upload audio, complete           │
│  /api/history/*      sessions, trend, summary, model-info                 │
│                                                                           │
│  ┌───────────────────── inference layer ────────────────────────┐        │
│  │ stimuli.py       randomised, server-side test content         │        │
│  │ speech.py        faster-whisper → transcript + timing         │        │
│  │ features.py *    27 linguistic features                       │        │
│  │ language_model.py  trained classifier + explanation           │        │
│  │ scoring.py *     MoCA rules → raw → age/education z-scores    │        │
│  │ fusion.py *      weighted combination → risk + band           │        │
│  └───────────────────────────────────────────────────────────────┘        │
│                    * shared with the training pipeline                     │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │  SQLAlchemy 2.0
                          ┌──────▼──────┐
                          │ PostgreSQL  │   SQLite for local dev
                          └─────────────┘
```

---

## 3. The shared-code decision

`ml/features.py`, `ml/scoring.py` and `ml/fusion.py` are imported by **both**
the training scripts and the backend. `backend/app/ml/paths.py` puts `ml/` on
`sys.path` (overridable with `NEUROGUARD_ML_DIR` for Docker).

The alternative — copying feature code into the backend — invites the worst
class of ML bug: training and inference drifting apart. A model that was fitted
on one definition of "pronoun-to-noun ratio" and served another produces
confident, plausible, wrong answers, and nothing fails loudly. One definition,
imported twice, makes that impossible.

For the same reason the saved artifact contains **only stock scikit-learn
objects** (vectorizer, scaler, classifier). No custom class is pickled, so
unpickling cannot break when a module moves.

---

## 4. Data flow for one assessment

1. `POST /api/assessment/start` — the server generates randomised stimuli (which
   five objects, which digit sequences, node layout, fluency letter), persists
   them on the session, and returns a **client-safe** copy with the Trail Making
   answer path stripped out.
2. The user works through six tests. Each structured test posts its raw response
   to `POST /api/assessment/{id}/test`.
3. Scoring uses the **stored** stimuli, never values echoed by the client, so a
   tampered browser cannot redefine what counts as correct.
4. The picture description uploads audio to `POST /api/assessment/{id}/audio`.
   Whisper transcribes it, the transcript is normalised into training style, 27
   features are extracted, and the trained classifier returns a calibrated
   probability plus per-feature contributions.
5. `POST /api/assessment/{id}/complete` averages z-scores per domain, fuses them
   with the language probability, stores the result, and locks the session.

Randomising stimuli per session is what makes **repeat testing** meaningful.
Fixed stimuli would mean the second assessment measures recall of the app rather
than memory.

---

## 5. Scoring

### Structured tests
Each is scored by its MoCA rule, then converted to a z-score against age band
and education level. Negative z always means "worse than expected", regardless
of whether the raw measure is a count (higher is better) or a time (lower is).

The MoCA rule most often implemented incorrectly is **serial sevens**: each
subtraction is judged against the number the participant actually said, not the
ideal sequence. One arithmetic slip should cost one mark, not invalidate every
answer after it. `score_serial_sevens` implements the chained rule and is
covered by a test.

### Language test
The trained classifier. See [MODEL_CARD.md](MODEL_CARD.md).

### Fusion
```
p_structured = sigmoid(-(composite_z - (-1.0)) * 1.5)
risk         = 0.5 * p_language + 0.5 * p_structured
```
Centring the logistic at z = −1.0 follows the neuropsychological convention that
z ≤ −1.0 is borderline and z ≤ −1.5 indicates impairment: a typical healthy
performer (z = 0) lands at low risk, a clearly impaired one (z = −2) at high.

Memory is weighted 1.3× the other domains, being the earliest and most sensitive
domain in decline.

If the language component is unavailable — no recording, too few words, a
transcription failure — the structured component carries the whole score, the
result says so, and confidence drops. The failure is surfaced, never hidden
behind a number that looks equally authoritative.

---

## 6. Authentication

- **argon2id** password hashing.
- **JWT access (30 min) + refresh (7 days)** in httpOnly SameSite cookies. The
  browser cannot read them, so XSS cannot steal a session. A bearer-header
  fallback exists purely so curl, tests and Swagger still work.
- **Refresh rotation**: presenting a refresh token revokes it and issues a new
  pair, so a stolen token is usable at most once. Only the SHA-256 hash is
  stored, so a database leak does not hand over live sessions.
- **Uniform login errors** for unknown email and wrong password, preventing
  account enumeration.
- **In-process login throttle**, 8 attempts per 5 minutes per IP+email. A
  multi-instance deployment would need Redis; this is honest about its scope.

---

## 7. Database

| Table | Purpose |
|---|---|
| `users` | Credentials plus age and education, needed for norming |
| `refresh_tokens` | Hashed, revocable, expiring |
| `assessment_sessions` | One per attempt; snapshots age/education and stores the stimuli and final result |
| `test_results` | One row per test, with the **raw response** as JSON |
| `audio_recordings` | File path, transcript, timing and extracted features |

Two choices worth defending:

**Raw responses are stored, not just scores.** The normative tables are
explicitly provisional. Keeping exactly what the user did means every historical
session can be rescored when better norms arrive, instead of being frozen at a
number computed from admittedly rough constants.

**Age and education are snapshotted onto the session.** Editing your profile
must not silently rewrite past results, or the longitudinal trend — the entire
point of repeat testing — becomes meaningless.

---

## 8. Deployment

A single multi-stage image: Node builds the React app, Python serves both the
API and the built assets. One origin means the auth cookie is first-party and
there is no CORS to misconfigure.

Baked into the image: Python dependencies, the spaCy model, the Whisper weights,
and the trained artifact (~123KB). Nothing is downloaded at runtime, so the
first request is fast and the container works without outbound network access.

Runs as a non-root user. `ffmpeg` is included as a decoding safety net for
browser audio.

**Memory is the binding constraint.** `base.en` needs roughly 1GB at inference.
On a tighter host, set `WHISPER_MODEL=tiny.en`.
