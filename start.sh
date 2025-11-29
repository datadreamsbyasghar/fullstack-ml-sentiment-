#!/bin/bash

# Exit on error
set -e

# ✅ Start FastAPI backend
echo "🚀 Starting FastAPI backend..."
cd backend
uvicorn main:app --reload &
BACKEND_PID=$!
cd ..

# ✅ Start React frontend
echo "🎨 Starting React frontend..."
cd sentiment-frontend
npm start &
FRONTEND_PID=$!
cd ..

# ✅ Trap CTRL+C to stop both
trap "echo '🛑 Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID" EXIT

# ✅ Keep script alive until both exit
wait