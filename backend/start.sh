#!/bin/bash
# FastAPI backend only. The voice agent worker (voice/agent.py) no longer
# starts from here -- it deploys and runs independently on LiveKit Cloud
# (see requirements-worker.txt), so this process never needs to spawn it.
# For local dev, run the worker yourself in a second terminal:
#   python -m voice.agent dev
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
