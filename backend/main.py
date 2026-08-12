import asyncio
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.chat.routers import router as chat_router
from api.chat.tools import ALL_TOOLS
from api.connectors.routers import router as connectors_router
from api.documents.routers import router as documents_router
from api.memory.routers import router as memory_router
from api.tasks.routers import router as tasks_router
from core.scheduler import run_scheduler_loop
from mcp_integration.mcp_manager import mcp_manager, get_merged_tool_schemas
from mcp_integration.gmail_mcp_router import router as gmail_mcp_router

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # MCP Phase 1 (Foundation): connect to any configured external MCP
    # servers and cache their tool schemas. Purely additive -- does not
    # touch Groq routing, Supabase, or any existing native tool. If no
    # server URLs are configured yet, this is a no-op.
    await mcp_manager.initialize()

    # Startup diagnostic: log the schema list as of connect-time. As of
    # Phase 2, ChatService.stream_chat computes this same merge fresh on
    # every turn (see api/chat/services.py) and passes it to the provider --
    # this log is just a one-time snapshot for verifying what connected.
    merged_schemas = get_merged_tool_schemas(ALL_TOOLS, mcp_manager)
    logger.info(
        "MCP: merged tool schema list at startup (native + MCP):\n"
        f"{json.dumps(merged_schemas, indent=2)}"
    )

    # Phase 5: background task/reminder scheduler. Runs as a plain
    # asyncio.Task alongside request handling (never blocking it) and is
    # stopped via an asyncio.Event rather than task.cancel(), so a poll that's
    # mid-flight when shutdown begins finishes cleanly instead of being cut
    # off partway through a DB call.
    scheduler_stop_event = asyncio.Event()
    scheduler_task = asyncio.create_task(run_scheduler_loop(scheduler_stop_event))

    yield

    scheduler_stop_event.set()
    await scheduler_task

    await mcp_manager.shutdown()


app = FastAPI(title="AetherChat API Portal", lifespan=lifespan)

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
app.include_router(connectors_router)
app.include_router(documents_router)
app.include_router(memory_router)
app.include_router(tasks_router)
# Gmail MCP connector (Phase 1) -- fully isolated OAuth routes, see
# mcp_integration/gmail_mcp_router.py. Never touches connectors_router or
# the Supabase identity-linking flow it backs.
app.include_router(gmail_mcp_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
