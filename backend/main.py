import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.chat.routers import router as chat_router
from api.documents.routers import router as documents_router
from api.memory.routers import router as memory_router

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AetherChat API Portal")

# Restrict CORS to production domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(memory_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
