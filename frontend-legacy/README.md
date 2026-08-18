# NeuroGuard UI v2

Replace the `frontend` folder in your existing project with this folder's `frontend`.

Backend remains compatible with the existing starter.

Run backend:
cd backend
source .venv/bin/activate
uvicorn app:app --reload

Run frontend in another VS Code terminal:
cd frontend
python3 -m http.server 5500

Open http://127.0.0.1:5500

This version adds a polished dashboard, multi-step cognitive assessment, result screen, multimodal upload cards, and monitoring page. The prediction remains a DEMO until trained models are connected.
