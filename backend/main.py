import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.chat.routers import router as chat_router

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AetherChat API Portal")

# Restrict CORS to production domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aetherchat.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(chat_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
