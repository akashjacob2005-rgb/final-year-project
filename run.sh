#!/bin/bash
# Starts NeuroGuard for local development: backend on :8000, frontend on :5173.
# Ctrl+C stops both.
cd "$(dirname "$0")"

./backend/.venv/bin/uvicorn app.main:app --reload --app-dir backend &
BACKEND_PID=$!

(cd frontend && npx vite --port 5173) &
FRONTEND_PID=$!

trap 'kill $BACKEND_PID $FRONTEND_PID 2>/dev/null' EXIT
echo ""
echo "  App: http://localhost:5173"
echo ""
wait
