import asyncio
import logging
import re
import time
from functools import lru_cache
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import voyageai
from fastembed.rerank.cross_encoder import TextCrossEncoder
from rank_bm25 import BM25Okapi
from supabase import Client

from core.config import get_settings
from api.documents.schemas import DocumentMetadata, RetrievedChunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "voyage-3-lite"
MAX_WORDS_PER_CHUNK = 200
OVERLAP_WORDS = 40
MAX_WORDS_PER_EMBED_BATCH = 4000

# match_document_chunks accepts a threshold, but the required service
# signature (query, user_id, top_k) doesn't expose one, so results are
# bounded by top_k alone and this stays a permissive floor.
DEFAULT_MATCH_THRESHOLD = 0.0

# ==================================================
# Hybrid search + reranking tuning
# ==================================================
# Vector search alone conflates documents that are topically similar but
# distinct (e.g. two versions of the same CV) because cosine similarity is a
# coarse semantic signal. The fix is a two-stage refinement on top of the
# existing single vector RPC call:
#   1. Over-fetch a candidate pool via the same match_document_chunks RPC,
#      then fuse the vector ranking with a local BM25 keyword ranking
#      (Reciprocal Rank Fusion) so exact-term matches (names, dates, version
#      numbers) can out-rank a merely-similar chunk.
#   2. Re-score the fused shortlist with a local cross-encoder, which reads
#      the query and each chunk together (unlike bi-encoder cosine
#      similarity) and is far better at telling near-duplicate documents
#      apart.
# Both stages run in-process on CPU with no paid API calls.
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
CANDIDATE_MULTIPLIER = 5  # over-fetch factor applied to the requested top_k
CANDIDATE_FLOOR = 10      # minimum candidate pool, even when top_k is tiny
CANDIDATE_CEILING = 25    # cap so the local reranker stage stays cheap
RERANK_SHORTLIST_SIZE = 25  # how many RRF-fused candidates get cross-encoder scored (matches CANDIDATE_CEILING so nothing vector search already found gets excluded before the cross-encoder judges it)
RRF_K = 60  # standard smoothing constant for reciprocal rank fusion

_TOKEN_PATTERN = re.compile(r"\w+")


@lru_cache
def _get_voyage_client() -> voyageai.Client:
    """
    Lazily builds a cached Voyage AI client from settings, mirroring the
    get_settings() caching pattern so the API key is only read once.
    """
    settings = get_settings()
    return voyageai.Client(api_key=settings.voyage_api_key)


@lru_cache
def _get_reranker() -> TextCrossEncoder:
    """
    Lazily builds a cached local cross-encoder (ONNX via onnxruntime, no
    PyTorch). The first call downloads and caches the model weights from
    Hugging Face; every call after that is pure CPU inference with no
    network round trip.

    threads=1 is explicit and deliberate: fastembed only constrains
    onnxruntime's intra_op/inter_op thread counts when a `threads` value is
    passed (see fastembed.common.onnx_model.OnnxModel.load_onnx_model) --
    left at its default (None), onnxruntime falls back to its own default
    of one intra-op thread per physical core, WITH busy-spin waiting
    enabled, for low single-inference latency. That default is fine for a
    one-off batch job, but here it runs inline whenever a voice or chat
    turn calls search_knowledge_base -- an uncapped, spinning thread pool
    contending for every CPU core is exactly what starved the voice
    worker's Silero VAD (a separate, already-capped onnxruntime session --
    see voice/agent.py's module docstring) of CPU time and drove system-wide
    load to ~100%. One thread keeps this rerank's own CPU footprint small
    and non-spinning, matching Silero's own session config.
    """
    return TextCrossEncoder(model_name=RERANK_MODEL, threads=1)


# ==================================================
# Recursive Merge Chunker (ported from reference/rag_pipeline.py)
# ==================================================

def _split_oversized_paragraph(paragraph: str, max_words: int) -> List[str]:
    """
    Recursively breaks a single paragraph that exceeds max_words into smaller
    units, trying progressively finer boundaries: single newlines, then
    sentence boundaries, then a hard word-count slice as the last resort.
    """
    if len(paragraph.split()) <= max_words:
        return [paragraph]

    lines = [l.strip() for l in paragraph.split("\n") if l.strip()]
    if len(lines) > 1:
        pieces = []
        for line in lines:
            pieces.extend(_split_oversized_paragraph(line, max_words))
        return pieces

    sentences = [s.strip() for s in paragraph.split(". ") if s.strip()]
    if len(sentences) > 1:
        pieces = []
        for sentence in sentences:
            if not sentence.endswith((".", "!", "?")):
                sentence += "."
            pieces.extend(_split_oversized_paragraph(sentence, max_words))
        return pieces

    # Last resort: no natural boundary found, hard-slice by word count.
    words = paragraph.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def recursive_merge_chunk(text: str, max_words: int = MAX_WORDS_PER_CHUNK, overlap_words: int = OVERLAP_WORDS) -> List[str]:
    """
    Recursive-merge chunker: splits text into paragraphs on double newlines,
    then sequentially merges those paragraphs into chunks of up to max_words,
    seeding each new chunk with the last overlap_words words of the previous
    one to preserve context continuity across chunk boundaries. Paragraphs
    that alone exceed max_words are recursively split first so no oversized
    fragment breaks the merge loop.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    units = []
    for paragraph in paragraphs:
        units.extend(_split_oversized_paragraph(paragraph, max_words))

    chunks = []
    current_words: List[str] = []

    for unit in units:
        unit_words = unit.split()
        if not unit_words:
            continue

        if current_words and len(current_words) + len(unit_words) > max_words:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap_words:] if overlap_words > 0 else []

        current_words.extend(unit_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# ==================================================
# PDF parsing
# ==================================================

def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from an in-memory PDF using PyMuPDF, joining per-page text
    blocks so multi-column layouts don't get interleaved.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        page_texts = []
        for page in doc:
            blocks = page.get_text("blocks")
            block_strings = [b[4].strip() for b in blocks if len(b) >= 5 and b[4].strip()]
            if block_strings:
                page_texts.append("\n\n".join(block_strings))
        return "\n\n".join(page_texts)
    finally:
        doc.close()


# ==================================================
# Voyage AI embeddings
# ==================================================

def _create_batches(chunks: List[str], max_words_per_batch: int = MAX_WORDS_PER_EMBED_BATCH) -> List[List[str]]:
    """
    Groups chunks into batches so cumulative words per batch stay under
    max_words_per_batch, keeping requests within Voyage API TPM limits.
    """
    batches = []
    current_batch: List[str] = []
    current_words = 0

    for chunk in chunks:
        chunk_words = len(chunk.split())
        if current_batch and (current_words + chunk_words > max_words_per_batch):
            batches.append(current_batch)
            current_batch = [chunk]
            current_words = chunk_words
        else:
            current_batch.append(chunk)
            current_words += chunk_words

    if current_batch:
        batches.append(current_batch)

    return batches


def _embed_with_retry(texts: List[str], input_type: str, max_retries: int = 5) -> List[List[float]]:
    """
    Calls the Voyage AI embed API with exponential backoff on rate limits.
    """
    client = _get_voyage_client()
    for attempt in range(max_retries):
        try:
            res = client.embed(texts, model=EMBEDDING_MODEL, input_type=input_type)
            return res.embeddings
        except Exception as embed_err:
            err_msg = str(embed_err).lower()
            if "rate" in err_msg or "429" in err_msg:
                wait_time = 22 * (attempt + 1)
                logger.warning(f"Voyage AI rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError("Max retries exceeded for Voyage AI embedding request.")


def _embed_documents(chunks: List[str]) -> List[List[float]]:
    """
    Embeds document chunks in TPM-safe batches, preserving input order.
    """
    embeddings: List[List[float]] = []
    for batch in _create_batches(chunks):
        embeddings.extend(_embed_with_retry(batch, input_type="document"))
    return embeddings


# ==================================================
# Hybrid search (BM25 + vector RRF) and local reranking
# ==================================================

def _tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _reciprocal_rank_fusion(rankings: List[List[int]], k: int = RRF_K) -> Dict[int, float]:
    """
    Fuses multiple rankings (each a list of candidate indices, best first)
    into a single score per index using Reciprocal Rank Fusion. RRF works on
    rank position rather than raw score, so it needs no normalization between
    incomparable scales like cosine similarity and BM25.
    """
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _hybrid_rerank(query: str, rows: List[dict], top_k: int) -> List[dict]:
    """
    Refines the vector-search candidate pool (`rows`, already ordered by
    cosine similarity descending) in two stages:

    1. BM25 keyword ranking over the same candidate pool, fused with the
       vector ranking via Reciprocal Rank Fusion, so exact keyword overlap
       (e.g. a version number or name that appears in the query) can
       outweigh a merely-similar embedding.
    2. A local cross-encoder rerank of the fused shortlist, which scores
       (query, chunk_text) pairs jointly and is much better than cosine
       similarity at separating near-duplicate documents.

    Falls back to the RRF-fused order if the cross-encoder can't be loaded
    or fails at inference time (e.g. no network on first run to fetch its
    ONNX weights), so reranking is strictly additive and never a point of
    failure for retrieval — mirroring how retrieval-failure fallback already
    works elsewhere in get_relevant_context's callers.
    """
    if len(rows) <= 1:
        return rows[:top_k]

    texts = [row["chunk_text"] for row in rows]

    # rows arrive pre-sorted by vector similarity descending.
    vector_rank = list(range(len(rows)))

    bm25 = BM25Okapi([_tokenize(t) for t in texts])
    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_rank = sorted(range(len(rows)), key=lambda i: bm25_scores[i], reverse=True)

    fused_scores = _reciprocal_rank_fusion([vector_rank, bm25_rank])
    fused_order = sorted(range(len(rows)), key=lambda i: fused_scores[i], reverse=True)

    shortlist = fused_order[:RERANK_SHORTLIST_SIZE]

    try:
        reranker = _get_reranker()
        cross_scores = list(reranker.rerank(query, [texts[i] for i in shortlist]))
        final_order = [
            idx for idx, _ in sorted(zip(shortlist, cross_scores), key=lambda pair: pair[1], reverse=True)
        ]
    except Exception as rerank_err:
        logger.warning(f"Local cross-encoder rerank failed, falling back to hybrid RRF order: {rerank_err}")
        final_order = shortlist

    return [rows[i] for i in final_order[:top_k]]


# ==================================================
# Public service functions
# ==================================================

async def process_and_store_pdf(supabase: Client, file_bytes: bytes, filename: str, user_id: str, session_id: str) -> int:
    """
    Parses an in-memory PDF, chunks it with the recursive-merge strategy,
    embeds the chunks with Voyage AI (voyage-3-lite), and inserts them into
    `document_chunks` scoped to `user_id` AND `session_id`.

    `supabase` must be a client authenticated with the user's JWT — the
    document_chunks RLS INSERT policy requires auth.uid() = user_id, so an
    unauthenticated/anon client will be rejected by Postgres.

    `session_id` ties every stored chunk to the chat session it was uploaded
    in (Context Isolation): a document uploaded in one chat is never visible
    to retrieval in another, even for the same user.

    Returns the number of chunks stored.
    """
    text = _extract_text_from_pdf(file_bytes)
    if not text.strip():
        logger.warning(f"No extractable text found in '{filename}' for user {user_id}.")
        return 0

    chunks = recursive_merge_chunk(text)
    if not chunks:
        return 0

    embeddings = _embed_documents(chunks)

    rows = [
        {
            "user_id": user_id,
            "session_id": session_id,
            "document_name": filename,
            "chunk_text": chunk_text,
            "embedding": embedding,
        }
        for chunk_text, embedding in zip(chunks, embeddings)
    ]

    supabase.table("document_chunks").insert(rows).execute()

    logger.info(f"Stored {len(rows)} chunks for '{filename}' (user {user_id}, session {session_id}).")
    return len(rows)


def _get_relevant_context_sync(
    supabase: Client,
    query: str,
    user_id: str,
    session_id: str,
    top_k: int,
    document_name: Optional[str],
) -> List[RetrievedChunk]:
    """
    Synchronous body of get_relevant_context -- see that function for the
    full contract. Split out so it can run entirely off the event loop (see
    get_relevant_context): every step here is a blocking call (a Voyage AI
    HTTP round trip that can itself sleep up to ~110s across retries on a
    rate limit -- see _embed_with_retry -- a synchronous Supabase RPC round
    trip, and the local BM25 + cross-encoder rerank), and none of it
    awaits, so run directly on an event loop thread it would stall that
    loop for the full duration of all three calls, sequentially.
    """
    query_embedding = _embed_with_retry([query], input_type="query")[0]

    candidate_count = min(max(top_k * CANDIDATE_MULTIPLIER, CANDIDATE_FLOOR), CANDIDATE_CEILING)

    res = supabase.rpc(
        "match_document_chunks",
        {
            "query_embedding": query_embedding,
            "match_threshold": DEFAULT_MATCH_THRESHOLD,
            "match_count": max(candidate_count, top_k),
            "filter_user_id": user_id,
            "filter_session_id": session_id,
            "filter_document_name": document_name,
        },
    ).execute()

    rows = _hybrid_rerank(query, res.data or [], top_k)

    return [
        RetrievedChunk(
            document_name=row["document_name"],
            chunk_text=row["chunk_text"],
            similarity=row["similarity"],
        )
        for row in rows
    ]


async def get_relevant_context(
    supabase: Client,
    query: str,
    user_id: str,
    session_id: str,
    top_k: int = 3,
    document_name: Optional[str] = None,
) -> List[RetrievedChunk]:
    """
    Embeds `query` and retrieves the top_k most similar chunks belonging to
    `user_id` AND scoped to `session_id` via the `match_document_chunks`
    Supabase RPC function. `session_id` is the Context Isolation boundary:
    a document uploaded in a different chat session, even by the same user,
    is never returned here.

    When `document_name` is given, it's passed through as `filter_document_name`
    and applied inside the RPC's SQL WHERE clause — used right after a file is
    attached in the same chat request so a generic prompt like "summarize the
    pdf" is grounded in the document just uploaded, not whichever older
    document happens to rank highest by embedding similarity. This MUST stay
    a SQL-level filter rather than a post-hoc Python filter on the RPC's
    results: match_count caps the similarity search at the top-N chunks
    *before* any filtering, so a large document with many chunks (e.g.
    attention_paper.pdf) can occupy every slot in that top-N window and leave
    zero chunks for the newly attached document if filtering happens
    afterward in Python. Filtering inside the WHERE clause means the LIMIT
    is applied only after rows are already restricted to the requested
    document, so that failure mode can't happen. When `document_name` is
    None, `filter_document_name` is NULL and the RPC searches all of the
    user's documents as normal.

    Exported so backend/api/chat/services.py can call this directly to
    ground chat responses in the user's uploaded documents, without either
    module importing internals from the other. voice/tools_adapter.py's
    search_knowledge_base tool calls it too -- from a live LiveKit voice
    session, where blocking the event loop doesn't just add HTTP latency,
    it freezes the audio pipeline outright (see voice/agent.py's module
    docstring for the incident this traces back to).

    Retrieval is hybrid: this RPC call over-fetches a candidate pool (rather
    than just top_k) so a downstream local BM25 + cross-encoder reranking
    stage (_hybrid_rerank) has enough headroom to re-rank exact keyword
    matches and semantically near-duplicate chunks (e.g. two versions of the
    same CV) ahead of a chunk that merely scored well on cosine similarity.
    This still issues exactly one Supabase RPC call — only match_count
    changes — so no new round trip or schema migration is required.

    The actual work (_get_relevant_context_sync) is entirely synchronous --
    a Voyage AI embed call, a Supabase RPC call, then local BM25/cross-
    encoder reranking -- with no internal `await` at all, so it's run via
    the default executor rather than inline: every caller here already
    awaits this function, so offloading it is a pure latency/isolation
    improvement with no change to the return value or error behavior for
    any existing caller.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _get_relevant_context_sync, supabase, query, user_id, session_id, top_k, document_name
    )


async def list_user_documents(supabase: Client, user_id: str, session_id: str) -> List[DocumentMetadata]:
    """
    Lists the distinct documents a user has stored in `session_id`,
    aggregated from document_chunks (chunk_count + earliest created_at per
    document_name). Shared by the deterministic GET /api/documents endpoint
    and the chat agent's list_documents tool, so both surfaces answer "what
    documents do you have" from the same real, RLS-scoped, session-scoped
    rows instead of the chat path inferring an answer from a lossy top-k
    semantic search over search_knowledge_base.
    """
    res = (
        supabase.table("document_chunks")
        .select("document_name, created_at")
        .eq("user_id", user_id)
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )

    documents: Dict[str, DocumentMetadata] = {}
    for row in res.data or []:
        name = row["document_name"]
        if name not in documents:
            documents[name] = DocumentMetadata(document_name=name, chunk_count=0, created_at=row["created_at"])
        documents[name].chunk_count += 1

    return list(documents.values())


def format_retrieved_chunks(chunks: List[RetrievedChunk]) -> str:
    """
    Renders retrieved chunks for injection into LLM context (system prompt or
    tool result). Uses XML-style tags rather than a '---'-delimited wrapper
    because the SYSTEM_PROMPT separately instructs the model to end
    user-facing answers with a '---'-delimited source footer — sharing that
    delimiter between the internal context wrapper and the instructed
    user-facing format let a weak model conflate the two and echo the raw
    wrapper (with all retrieved document names) straight into its reply
    instead of synthesizing a clean answer.
    """
    return "\n\n".join(
        f'<retrieved_context source="{chunk.document_name}">\n'
        f"{chunk.chunk_text}\n"
        "</retrieved_context>"
        for chunk in chunks
    )
