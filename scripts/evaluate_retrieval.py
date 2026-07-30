"""
Standalone retrieval-only evaluation harness.

Loads scripts/eval_dataset.json and, for each query, calls the app's real
`get_relevant_context` (backend/api/documents/services.py) against the live
Supabase + Voyage AI backend. The LLM is never invoked - this measures the
retriever in isolation.

Run from anywhere with:
    python scripts/evaluate_retrieval.py

Requires backend/.env to define SUPABASE_URL, VOYAGE_API_KEY, and
SUPABASE_SERVICE_ROLE_KEY. The service-role key is required (not the anon
key) because get_relevant_context's `match_document_chunks` RPC is scoped by
an explicit filter_user_id argument rather than the caller's JWT, and the
document_chunks table itself denies SELECT to unauthenticated/anon roles.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPTS_DIR.parent / "backend"
EVAL_DATASET_PATH = SCRIPTS_DIR / "eval_dataset.json"

# eval_dataset.json was built from the real document_chunks rows already
# uploaded under this Supabase user; override with EVAL_USER_ID to evaluate
# against a different account's documents.
DEFAULT_EVAL_USER_ID = "de162a33-ebe8-4a7b-b508-f6f22b6d2a40"

TOP_K = 5  # fetch 5 once; Hit Rate@3 is derived by slicing the same ranked list


def _bootstrap_backend_imports():
    """
    Adds backend/ to sys.path (mirroring backend/tests/*.py's own path setup)
    so this script can import api.documents.services exactly as the running
    app does, and loads backend/.env so core.config.get_settings() and the
    service-role key are both available.
    """
    sys.path.insert(0, str(BACKEND_DIR))
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")


_bootstrap_backend_imports()

from supabase import create_client  # noqa: E402
from core.config import get_settings  # noqa: E402
from api.documents.services import get_relevant_context  # noqa: E402


def _build_supabase_client():
    settings = get_settings()
    service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not settings.supabase_url or not service_role_key:
        raise RuntimeError(
            "SUPABASE_URL and/or SUPABASE_SERVICE_ROLE_KEY are missing from "
            "backend/.env. Add SUPABASE_SERVICE_ROLE_KEY (Supabase project "
            "settings -> API -> service_role key) to run this script."
        )
    return create_client(settings.supabase_url, service_role_key)


def _is_hit(entry: dict, retrieved_chunks: list) -> bool:
    return any(
        chunk.document_name == entry["expected_document"]
        and entry["expected_chunk_text"] in chunk.chunk_text
        for chunk in retrieved_chunks
    )


def _reciprocal_rank(entry: dict, retrieved_chunks: list) -> float:
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if (
            chunk.document_name == entry["expected_document"]
            and entry["expected_chunk_text"] in chunk.chunk_text
        ):
            return 1.0 / rank
    return 0.0


async def evaluate() -> None:
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    user_id = os.environ.get("EVAL_USER_ID", DEFAULT_EVAL_USER_ID)
    supabase = _build_supabase_client()

    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    rows = []

    for entry in eval_set:
        chunks = await get_relevant_context(supabase, entry["query"], user_id, top_k=TOP_K)

        hit3 = _is_hit(entry, chunks[:3])
        hit5 = _is_hit(entry, chunks[:5])
        rr = _reciprocal_rank(entry, chunks[:5])

        hits_at_3 += int(hit3)
        hits_at_5 += int(hit5)
        reciprocal_ranks.append(rr)
        rows.append((entry["id"], hit3, hit5, rr))

    n = len(eval_set)
    hit_rate_at_3 = hits_at_3 / n if n else 0.0
    hit_rate_at_5 = hits_at_5 / n if n else 0.0
    mrr_at_5 = sum(reciprocal_ranks) / n if n else 0.0

    print(f"{'ID':<10} {'Hit@3':<7} {'Hit@5':<7} {'RR@5':<6}")
    print("-" * 32)
    for qid, hit3, hit5, rr in rows:
        print(f"{qid:<10} {'Y' if hit3 else 'N':<7} {'Y' if hit5 else 'N':<7} {rr:.3f}")

    print()
    print(f"Evaluated {n} queries (user_id={user_id})")
    print(f"Hit Rate @ 3: {hit_rate_at_3:.2%} ({hits_at_3}/{n})")
    print(f"Hit Rate @ 5: {hit_rate_at_5:.2%} ({hits_at_5}/{n})")
    print(f"MRR @ 5:      {mrr_at_5:.4f}")


if __name__ == "__main__":
    asyncio.run(evaluate())
