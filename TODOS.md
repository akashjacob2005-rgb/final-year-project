# TODOS

Captured by /plan-eng-review on 2026-09-01 (MRI pathway review).

## 1. Persist MRI analysis results
- **What:** Store each MRI analysis (user_id, timestamp, filename, probabilities, predicted class — NOT the images) in a new `mri_analyses` table; list past runs on the History page.
- **Why:** Results currently vanish on page refresh; History shows only the six cognitive tests.
- **Pros:** Longitudinal MRI view; stronger demo; parity with how assessments are stored.
- **Cons:** Requires a migration and a retention/privacy decision (MODEL_CARD already flags retention as unimplemented).
- **Context:** Decided in eng review Issue 2 (option 2B, defer). The analyze route is `backend/app/api/routes/mri.py`; follow the AssessmentSession/TestResult storage pattern in `assessment.py`.
- **Depends on:** A retention policy decision (how long to keep, deletable from UI like audio recordings).

## 2. Frontend test infrastructure
- **What:** Add vitest + React Testing Library; first targets: MRI upload flow (file validation, error states, busy guard), slice-viewer toggle and modal, PlainExplanation wording bands.
- **Why:** The frontend has zero automated tests; the MRI page is now the demo centerpiece.
- **Pros:** Guards the demo UI against refactors; establishes a pattern for the other pages.
- **Cons:** New framework + config in the repo; CI story needed to make it pay off.
- **Context:** Deferred in eng review Issue 4 (option 4A — backend suite landed first). Backend testing conventions live in `backend/tests/test_api.py` / `test_mri.py`.
- **Depends on:** Nothing.

## 3. Calibrate MRI probabilities after Phase B
- **What:** Fit temperature/Platt scaling on pooled out-of-fold predictions from the 518-subject retrain; re-check normal-class recall; store the calibrator with the artifact and apply it in `backend/app/ml/mri_model.py`.
- **Why:** Current confidences are uncalibrated: 2:1 impaired training prior + class-weighted loss means displayed probabilities lean impaired (deployed fold-0 normal recall: 0.583). The UI now discloses this (Issue 8, option 8A); calibration is the real fix.
- **Pros:** Displayed percentages become trustworthy; removes the strongest examiner attack on the demo.
- **Cons:** Not worthwhile at 180 subjects; must redo after any retrain.
- **Context:** Eng review Issue 8 + outside-voice findings #2/#4. Training code: `ml/mri/train_mri.py` (already computes per-fold session probabilities in `session_eval`).
- **Blocked by:** Phase B OASIS download (task: 518-subject retrain, needs XNAT credentials).

## 4. Measure, then enable MRI on Render
- **What:** After deploying with `MRI_ENABLED=false`, temporarily enable on Render, run one `/api/mri/analyze` upload while watching Render memory/latency metrics; enable permanently if it fits, otherwise document MRI as local-only (or move to a paid instance).
- **Why:** The 512MB free instance already runs Whisper tiny + spaCy; onnxruntime + per-request volume decompression is an untested addition. Sync route latency (2 ResNet passes × 16 slices on shared CPU) is also unmeasured.
- **Pros:** Feature reaches production safely, or the limitation gets documented honestly.
- **Cons:** May conclude it needs a paid plan.
- **Context:** Eng review Issue 5 (option 5A) + outside-voice #11. Gate lives in `backend/app/ml/mri_model.py::enabled()`; production now requires the env var explicitly.
- **Depends on:** Pushing the MRI diff (auto-deploys via Render).

## 5. Create DESIGN.md via /design-consultation
- **What:** A one-page design system doc: typography, spacing scale, color semantics (including the darkened badge text colors), component vocabulary (card/badge/alert/dropzone/prob-row idioms).
- **Why:** The 2026-09-01 design review had to infer the system from index.css; a written system prevents drift, especially with a Claude-design visual redesign incoming.
- **Pros:** Future pages stay consistent; design reviews calibrate against it; reads well in the report.
- **Cons:** ~15 min; may be partially superseded by the redesign's own system.
- **Context:** Captured by /plan-design-review. The de-facto system lives in frontend/src/index.css (tokens at top).
- **Depends on:** Nothing; ideally done before or with the redesign merge.
