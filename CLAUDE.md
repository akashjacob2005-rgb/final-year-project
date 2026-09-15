# NeuroGuard — project brief for AI sessions

Cognitive-decline screening platform (final-year project). Three analysis
pathways: six behavioural tests (MoCA-style, rule-scored to z-scores), a
trained DementiaBank language model (TF-IDF + linguistic features, calibrated
logistic regression), and an OASIS-3 MRI CNN (ResNet-18 → ONNX, Grad-CAM
explainability). MRI is reported separately from the behavioural risk score
by design — no dataset pairs the two.

## Stack & layout
- `backend/` — FastAPI + SQLAlchemy (Supabase Postgres via DATABASE_URL; SQLite fallback), JWT cookies, Whisper (faster-whisper) + spaCy, onnxruntime.
- `frontend/` — Vite + React (JSX, **no TypeScript, no Tailwind**), one stylesheet `frontend/src/index.css` (token-driven; dark "Dimension" theme + light greige theme via `[data-theme="light"]`).
- `ml/` — shared feature/scoring code, imported by the backend via `backend/app/ml/paths.py` sys.path bridge. `ml/mri/` = MRI training (own venv, torch/MPS), preprocessing shared with inference (`ml/mri/preprocess.py`).
- Artifacts committed in `ml/artifacts/` (language_model.joblib, mri_model.onnx + .onnx.data + metrics JSONs).

## Commands
- Run app: `./run.sh` (backend :8000 + vite :5173), or `backend/.venv/bin/uvicorn app.main:app --app-dir backend --port 8000` + `cd frontend && npx vite`.
- Backend tests: `cd backend && .venv/bin/python -m pytest tests/ -q` (must stay green).
- Frontend build check: `cd frontend && npx vite build`.
- MRI retrain: `ml/mri/.venv/bin/python ml/mri/train_mri.py --data-dir <data>/processed_v2 --folds 5`.

## Gotchas
- **Never `git push` without the user's explicit go-ahead** — push auto-deploys the backend to Render.
- Render free tier (512MB): MRI is fail-closed in production (`MRI_ENABLED` must be set explicitly; see `backend/app/ml/mri_model.py::enabled`).
- Render cold-starts after ~15min idle — transient fetch failures are routine; frontend distinguishes retryable errors from disabled features.
- `.env` at repo root holds secrets (git-ignored). `MRI_SAMPLE_DIR` powers the "try a sample scan" buttons.
- Local data (OASIS-3) lives OUTSIDE the repo at `~/Downloads/7th_sem/PP-2/data/`.
- Deployed inference must match evaluated math (TTA = average of softmaxes; see learnings in docs/MODEL_CARD.md).

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec
