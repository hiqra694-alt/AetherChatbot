#!/bin/bash
# Start the LiveKit voice agent worker in the background
python -m voice.agent start &

# Start the FastAPI backend
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
