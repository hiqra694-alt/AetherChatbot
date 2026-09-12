# Voice agent worker (LiveKit Cloud deployment target). Builds and runs
# voice/agent.py standalone -- never main.py/FastAPI, which deploys
# separately to Render. Build context must be this
# directory (backend/): voice/agent.py's imports are all absolute from this
# package root (e.g. `from core.config import get_settings`), so it must
# also be WORKDIR/CMD's cwd here.
FROM python:3.11-slim

WORKDIR /app

# requirements-worker.txt layers the LiveKit-specific packages on top of
# requirements.txt via `-r requirements.txt` -- see that file's header
# comment -- so both must be present for pip to resolve it.
COPY requirements.txt requirements-worker.txt ./
RUN pip install --no-cache-dir -r requirements-worker.txt

COPY . .

CMD ["python", "-m", "voice.agent", "start"]
