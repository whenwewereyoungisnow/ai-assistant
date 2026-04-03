import asyncio
import json
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import database
import documents
import models
import router
import writer


# Lifespan handler — FastAPI runs the code before "yield" on startup, and
# after "yield" on shutdown. This is the standard pattern for managing
# resources (HTTP clients, DB connections) that need cleanup.
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    database.init_db()
    models.init_client()
    yield
    await models.close_client()


app = FastAPI(title="AI Assistant", lifespan=lifespan)

# Path to the templates directory. Using Path(__file__).parent so it
# resolves correctly regardless of where you run the app from.
TEMPLATES_DIR = Path(__file__).parent / "templates"


# Global exception handlers — catch httpx errors in one place so every
# endpoint gets a clean error message instead of a raw traceback.
@app.exception_handler(httpx.ConnectError)
async def ollama_connection_error(
    request: Request, exc: httpx.ConnectError
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": "Cannot connect to Ollama. Is it running on localhost:11434?"
        },
    )


@app.exception_handler(httpx.HTTPStatusError)
async def ollama_http_error(
    request: Request, exc: httpx.HTTPStatusError
) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"error": f"Ollama returned an error: {exc.response.status_code}"},
    )


# Serve the frontend. FileResponse sends a static file directly — no
# template engine needed since the frontend is pure HTML + JS.
# We set Cache-Control: no-store so the browser always fetches the latest
# version during development. Without this, you can end up debugging a
# cached old page that doesn't have your latest JS changes.
@app.get("/")
async def root() -> FileResponse:
    return FileResponse(
        TEMPLATES_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
async def get_models() -> list[dict[str, Any]]:
    return await models.list_models()


@app.get("/status")
async def get_status() -> dict[str, Any]:
    """System status: loaded models, document counts, conversations, GPU memory.

    This powers the frontend status bar. It combines data from three sources:
    - Ollama's /api/ps (running models, VRAM usage)
    - In-memory document store (document and chunk counts)
    - SQLite database (conversation count)
    """
    loop = asyncio.get_running_loop()

    # Running models from Ollama (includes size_vram per model).
    # Ollama may be down — don't let that break the whole status response.
    # The document and conversation counts are still useful without Ollama.
    try:
        running = await models.running_models()
    except Exception:
        running = []

    # Conversation count (sync SQLite call, offloaded to thread pool)
    conversations = await loop.run_in_executor(None, database.list_conversations)

    # GPU VRAM: sum of size_vram across all loaded models.
    # Ollama reports size_vram in bytes — convert to GB for display.
    total_vram_bytes = sum(m.get("size_vram", 0) for m in running)
    total_vram_gb = round(total_vram_bytes / (1024**3), 1)

    return {
        "models": [m.get("name", "unknown") for m in running],
        "documents": documents.document_count(),
        "chunks": documents.chunk_count(),
        "conversations": len(conversations),
        "gpu_vram_gb": total_vram_gb,
    }


# --- Conversation endpoints ---


# Pydantic model for the POST /conversations request body.
# BaseModel validates incoming JSON automatically — if "mode" is missing,
# FastAPI returns a 422 error with a clear message instead of crashing.
# Literal restricts mode to valid values, so invalid modes like "banana"
# get a clean 422 error instead of hitting the database and causing a 500.
class CreateConversationRequest(BaseModel):
    mode: Literal["chat", "documents", "writing", "vision"]
    title: str | None = None


class ChatRequest(BaseModel):
    message: str


class SearchRequest(BaseModel):
    query: str
    method: Literal["semantic", "keyword"] = "semantic"
    top_k: int = 5


# These conversation endpoints are plain `def` (not `async def`) because they
# call synchronous sqlite3 functions. FastAPI automatically runs plain `def`
# endpoints in a thread pool, so they don't block the async event loop.
# If these were `async def`, the blocking sqlite3 calls would freeze the
# entire server until each database operation finishes.


@app.get("/conversations")
def list_conversations() -> list[dict[str, Any]]:
    """List all conversations, newest first, with message counts."""
    return database.list_conversations()


@app.post("/conversations", status_code=201)
def create_conversation(body: CreateConversationRequest) -> dict[str, str]:
    """Create a new conversation. Returns its ID."""
    conversation_id = database.create_conversation(mode=body.mode, title=body.title)
    return {"id": conversation_id}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict[str, Any]:
    """Return a single conversation with all its messages."""
    conversation = database.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str) -> None:
    """Delete a conversation and all its messages."""
    deleted = database.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")


# --- Chat endpoint (SSE streaming) ---
#
# How the same /chat endpoint handles both modes:
# The endpoint checks the conversation's mode from the database and branches:
# - "chat" mode: classify the question → pick the best model → stream response
# - "documents" mode: search uploaded documents → build RAG prompt → stream response
#
# Both modes share the same SSE plumbing (token/done events) and conversation
# storage. The frontend always POSTs to the same URL regardless of mode —
# the backend decides what to do based on the conversation's mode field.
#
# Flow:
# 1. Frontend sends POST /chat/{id} with {"message": "user's question"}
# 2. We save the user message to the database immediately
# 3. We load conversation history from the database
# 4. Branch by mode:
#    - Chat: router.route_and_respond() classifies and streams
#    - Documents: search chunks → yield sources → stream RAG response
# 5. Save the assistant message with mode-specific metadata
# 6. If first message, auto-generate a title


# The RAG system prompt is intentionally different from the chat system prompt.
# Chat mode says "be helpful, use markdown" — the model can use general knowledge.
# Documents mode is constrained: only answer from the provided excerpts, cite
# sources, and say "I don't know" if the answer isn't in the excerpts. This
# distinction matters because users expect document Q&A to be grounded in their
# actual documents, not hallucinated from the model's training data.
RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the "
    "provided document excerpts. Only use information from the excerpts. "
    "If the answer isn't in the excerpts, say so clearly. "
    "Always cite the source document and page number when referencing "
    'information (e.g., "According to report.pdf, page 3..."). '
    "Use markdown formatting when it helps readability."
)

# Documents mode always uses the general model — no classification needed
# because we're just answering questions about document content.
RAG_MODEL = "qwen3.5:35b-a3b-coding-nvfp4"
RAG_MODEL_OPTIONS: dict[str, Any] = {"think": False}

# --- Cross-mode intelligence ---
#
# How auto-detection works:
# When a user asks a question in chat mode, we check if any uploaded documents
# contain relevant information by running a quick semantic search. If the top
# results exceed a score threshold, we augment the chat message with those
# excerpts — the model sees them as additional context and can reference them.
#
# Why cross-mode features make the assistant feel more intelligent:
# Without cross-mode, the user must manually switch to "Documents" mode to
# ask about their PDFs. With it, the assistant automatically detects when
# uploaded documents are relevant and incorporates them — like a colleague
# who remembers "oh, I read something about that in the report you shared."
# This makes three separate tools feel like one unified, context-aware assistant.
#
# The threshold is intentionally conservative (0.55). Cosine similarity from
# qwen3-embedding:4b ranges roughly 0.3 (unrelated) to 0.9 (very similar).
# 0.55 catches genuinely relevant content without false positives from
# vaguely related chunks. Easy to tune — it's a single constant.
DOC_RELEVANCE_THRESHOLD = 0.55
DOC_CONSULT_TOP_K = 3


@app.post("/chat/{conversation_id}")
async def chat_endpoint(conversation_id: str, body: ChatRequest) -> EventSourceResponse:
    """Send a message and stream the AI response via Server-Sent Events.

    The response is a stream of SSE events:
    - event: routing  → (chat mode) which model was chosen and why
    - event: sources  → (documents mode) relevant document chunks found
    - event: token    → one token of the response
    - event: done     → timing stats (response is complete)
    - event: error    → something went wrong
    """
    # Verify the conversation exists.
    # We use run_in_executor to offload blocking SQLite calls to a thread
    # pool. Without this, synchronous database I/O would freeze the async
    # event loop and block all other requests until the DB call finishes.
    # FastAPI auto-threads plain `def` endpoints (see the CRUD endpoints
    # above), but since this endpoint is `async def` (required for SSE),
    # we need to handle it manually.
    loop = asyncio.get_running_loop()
    conversation = await loop.run_in_executor(
        None, database.get_conversation, conversation_id
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Save the user's message to the database right away.
    # We do this before streaming so the message is persisted even if
    # the stream fails partway through.
    await loop.run_in_executor(
        None, database.add_message, conversation_id, "user", body.message
    )

    # Build conversation history for the model.
    # We load previous messages from the database and format them as the
    # OpenAI-style [{"role": "user", "content": "..."}, ...] array that
    # Ollama expects. The model uses this history to understand context
    # for follow-up questions like "what about the second one?" or
    # "can you explain that differently?"
    history: list[dict[str, str]] = []
    for msg in conversation.get("messages", []):
        if msg["role"] in ("user", "assistant"):
            history.append({"role": msg["role"], "content": msg["content"]})

    # Check if this is the first message (for auto-titling later)
    is_first_message = len(history) == 0
    mode = conversation.get("mode", "chat")

    if mode == "documents":
        return EventSourceResponse(
            _documents_event_stream(
                conversation_id, body.message, history, is_first_message
            )
        )
    elif mode == "writing":
        return EventSourceResponse(
            _writing_event_stream(
                conversation_id, body.message, history, is_first_message
            )
        )
    else:
        return EventSourceResponse(
            _chat_event_stream(conversation_id, body.message, history, is_first_message)
        )


async def _chat_event_stream(
    conversation_id: str,
    message: str,
    history: list[dict[str, str]],
    is_first_message: bool,
) -> AsyncGenerator[dict[str, str], None]:
    """SSE event stream for chat mode (smart routing).

    EventSourceResponse expects dicts with "event" and "data" keys.
    Each dict becomes one SSE event sent to the browser.
    """
    full_content = ""
    routing_metadata: dict[str, Any] = {}
    doc_sources: list[dict[str, Any]] = []

    try:
        # --- Cross-mode: auto-consult documents if relevant ---
        # If the user has uploaded documents, we check whether their chat
        # question relates to the document content. If so, we inject the
        # relevant excerpts into the message so the routed model can
        # reference them. The original message is already saved to the DB
        # (line above), so the augmented version is ephemeral.
        # Document consultation is best-effort — if the embedding model
        # fails to load or Ollama is down, we skip it silently and proceed
        # with the normal chat flow. The user shouldn't lose chat because
        # a secondary feature had an error.
        effective_message = message
        if documents.has_documents():
            try:
                results = await documents.search_chunks(
                    message, method="semantic", top_k=DOC_CONSULT_TOP_K
                )
                relevant = [r for r in results if r["score"] > DOC_RELEVANCE_THRESHOLD]
            except Exception:
                relevant = []
            if relevant:
                doc_sources = relevant
                excerpts = "\n\n".join(
                    f"[{s['filename']}, page {s['page']}]: {s['text']}"
                    for s in relevant
                )
                effective_message = (
                    f"{message}\n\n"
                    f"[Relevant context from uploaded documents — "
                    f"use if helpful, cite source when referencing]\n"
                    f"{excerpts}"
                )
                yield {
                    "event": "doc_consulted",
                    "data": json.dumps(
                        {
                            "sources": relevant,
                            "message": "Answer informed by uploaded documents",
                        }
                    ),
                }

        async for event in router.route_and_respond(effective_message, history):
            if event["type"] == "routing":
                routing_metadata = event
                yield {
                    "event": "routing",
                    "data": json.dumps(event),
                }

            elif event["type"] == "token":
                full_content += event["content"]
                yield {
                    "event": "token",
                    "data": json.dumps(event),
                }

            elif event["type"] == "done":
                # Save the complete assistant response to the database.
                # We store routing metadata alongside the message so it
                # can be displayed when the conversation is reloaded.
                metadata = {
                    "model": routing_metadata.get("model", ""),
                    "route": routing_metadata.get("route", ""),
                    "reason": routing_metadata.get("reason", ""),
                    "classify_ms": routing_metadata.get("classify_ms", 0),
                    "stream_ms": event.get("stream_ms", 0),
                    "total_ms": event.get("total_ms", 0),
                }
                # Save which documents were auto-consulted so the UI can
                # re-display the indicator when reloading the conversation.
                if doc_sources:
                    metadata["doc_consulted"] = [
                        {
                            "filename": s["filename"],
                            "page": s["page"],
                            "score": s["score"],
                        }
                        for s in doc_sources
                    ]
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    database.add_message,
                    conversation_id,
                    "assistant",
                    full_content,
                    metadata,
                )

                yield {
                    "event": "done",
                    "data": json.dumps(event),
                }

                # Auto-title: after the first exchange, generate a short
                # title so the sidebar shows something meaningful instead
                # of "New conversation" for every chat.
                if is_first_message:
                    asyncio.create_task(_auto_title(conversation_id, message))

    except Exception as e:
        # Save whatever was generated so far so the conversation doesn't
        # have an orphaned user message with no assistant response.
        if full_content:
            await asyncio.get_running_loop().run_in_executor(
                None,
                database.add_message,
                conversation_id,
                "assistant",
                full_content,
                {"error": str(e), "partial": True},
            )
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)}),
        }


async def _documents_event_stream(
    conversation_id: str,
    message: str,
    history: list[dict[str, str]],
    is_first_message: bool,
) -> AsyncGenerator[dict[str, str], None]:
    """SSE event stream for documents mode (RAG).

    Documents mode works differently from chat mode:
    1. Search uploaded documents for relevant chunks (no model classification)
    2. Yield a "sources" event so the frontend can display them immediately
    3. Build a RAG prompt that includes the source excerpts as context
    4. Stream the model's response (always uses the general model)

    Why include source metadata in saved messages?
    When you reload a conversation, the original documents may have been
    deleted or the server restarted (in-memory storage). By saving the
    source excerpts in the message metadata, the UI can always show what
    sources were used, even if the documents are gone.
    """
    full_content = ""
    total_start = time.monotonic()

    try:
        # Step 1: Search for relevant document chunks.
        # We use semantic search (embedding cosine similarity) to find chunks
        # whose meaning matches the question, even if different words are used.
        sources = await documents.search_chunks(message, method="semantic", top_k=5)

        # Yield sources so the frontend can display them immediately,
        # before the model starts generating tokens.
        yield {
            "event": "sources",
            "data": json.dumps({"sources": sources, "model": RAG_MODEL}),
        }

        if not sources:
            # No documents uploaded or no relevant chunks found.
            # Tell the user instead of sending an empty context to the model.
            full_content = (
                "No relevant documents found. Please upload some PDFs first, "
                "then ask your question again."
            )
            yield {
                "event": "token",
                "data": json.dumps({"type": "token", "content": full_content}),
            }
            total_ms = round((time.monotonic() - total_start) * 1000)
            metadata: dict[str, Any] = {
                "model": RAG_MODEL,
                "sources": [],
                "total_ms": total_ms,
                "stream_ms": 0,
            }
            await asyncio.get_running_loop().run_in_executor(
                None,
                database.add_message,
                conversation_id,
                "assistant",
                full_content,
                metadata,
            )
            yield {
                "event": "done",
                "data": json.dumps({"total_ms": total_ms, "stream_ms": 0}),
            }
            if is_first_message:
                asyncio.create_task(_auto_title(conversation_id, message))
            return

        # Step 2: Build the RAG prompt.
        # We inject the source excerpts directly into the user message so the
        # model sees them as context. Each excerpt is labeled with its filename
        # and page number so the model can cite them in its response.
        # This is simpler than tool-use/function-calling and works with any model.
        excerpts = "\n\n".join(
            f"[{s['filename']}, page {s['page']}]:\n{s['text']}" for s in sources
        )
        augmented_message = (
            f"Based on the following document excerpts, answer my question.\n\n"
            f"--- DOCUMENT EXCERPTS ---\n{excerpts}\n--- END EXCERPTS ---\n\n"
            f"Question: {message}"
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT}
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": augmented_message})

        # Step 3: Stream the response from the general model.
        stream_start = time.monotonic()
        async for token in models.stream_chat(RAG_MODEL, messages, RAG_MODEL_OPTIONS):
            full_content += token
            yield {
                "event": "token",
                "data": json.dumps({"type": "token", "content": token}),
            }

        stream_ms = round((time.monotonic() - stream_start) * 1000)
        total_ms = round((time.monotonic() - total_start) * 1000)

        # Save with source metadata so the frontend can re-display sources
        # when the conversation is reloaded from the database.
        metadata = {
            "model": RAG_MODEL,
            "sources": sources,
            "total_ms": total_ms,
            "stream_ms": stream_ms,
        }
        await asyncio.get_running_loop().run_in_executor(
            None,
            database.add_message,
            conversation_id,
            "assistant",
            full_content,
            metadata,
        )

        yield {
            "event": "done",
            "data": json.dumps({"total_ms": total_ms, "stream_ms": stream_ms}),
        }

        if is_first_message:
            asyncio.create_task(_auto_title(conversation_id, message))

    except Exception as e:
        if full_content:
            await asyncio.get_running_loop().run_in_executor(
                None,
                database.add_message,
                conversation_id,
                "assistant",
                full_content,
                {"error": str(e), "partial": True},
            )
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)}),
        }


async def _writing_event_stream(
    conversation_id: str,
    message: str,
    history: list[dict[str, str]],
    is_first_message: bool,
) -> AsyncGenerator[dict[str, str], None]:
    """SSE event stream for writing mode (multi-agent pipeline).

    Writing mode is fundamentally different from chat and documents modes:
    instead of one model responding to one question, two models collaborate
    across multiple rounds. The Writer drafts, the Editor critiques, and the
    Writer revises — up to 3 rounds or until the Editor approves.

    How pipeline events map to SSE events:
    - phase_start → "writing" event (tells frontend to create a new section)
    - token       → "token" event (same as chat/documents, plus a "phase" field)
    - phase_end   → "writing" event (tells frontend to finalize the section)
    - complete    → "done" event (same as chat/documents)

    Why save each phase as a separate assistant message?
    When the user reloads the conversation, they see the full creative process:
    the original draft, the editor's critique, and each revision. This is more
    useful than just seeing the final result, because it lets you understand
    *why* the final version looks the way it does and learn from the feedback.

    How the Writer/Editor model swap affects timing:
    The Writer (~21GB) and Editor (~20GB) can't fit in GPU memory together.
    Ollama swaps them automatically, costing ~20 seconds per swap. In a
    3-round pipeline, that's up to 5 swaps (draft → critique → revision →
    critique → revision → critique). Both models stream their output so the
    user sees activity during swaps instead of a blank screen.
    """
    try:
        # --- Cross-mode: check if documents can inform the writing ---
        # Same approach as chat mode: semantic search for relevant chunks,
        # filter by threshold, and pass as research context to the writer.
        # Best-effort — if the embedding model fails, skip silently.
        doc_context = None
        doc_sources: list[dict[str, Any]] = []
        if documents.has_documents():
            try:
                results = await documents.search_chunks(
                    message, method="semantic", top_k=DOC_CONSULT_TOP_K
                )
                relevant = [r for r in results if r["score"] > DOC_RELEVANCE_THRESHOLD]
            except Exception:
                relevant = []
            if relevant:
                doc_sources = relevant
                doc_context = "\n\n".join(
                    f"[{s['filename']}, page {s['page']}]: {s['text']}"
                    for s in relevant
                )
                yield {
                    "event": "doc_consulted",
                    "data": json.dumps(
                        {
                            "sources": [
                                {
                                    "filename": s["filename"],
                                    "page": s["page"],
                                    "score": s["score"],
                                }
                                for s in relevant
                            ],
                            "message": "Writing informed by uploaded documents",
                        }
                    ),
                }

        async for event in writer.run_pipeline(
            message, history, doc_context=doc_context
        ):
            if event["type"] == "phase_start":
                yield {
                    "event": "writing",
                    "data": json.dumps(event),
                }

            elif event["type"] == "token":
                yield {
                    "event": "token",
                    "data": json.dumps(
                        {
                            "type": "token",
                            "content": event["content"],
                            "phase": event["phase"],
                        }
                    ),
                }

            elif event["type"] == "phase_end":
                # Save each phase as a separate assistant message so the
                # full creative process is preserved in the database.
                phase_model = (
                    writer.EDITOR_MODEL
                    if event["phase"] == "critique"
                    else writer.WRITER_MODEL
                )
                metadata: dict[str, Any] = {
                    "phase": event["phase"],
                    "round": event["round"],
                    "model": phase_model,
                    "duration_ms": event["duration_ms"],
                    "pipeline": True,
                }
                # Persist doc consultation info on the first phase so the
                # banner re-appears when reloading the conversation.
                if doc_sources and event["phase"] == "draft" and event["round"] == 1:
                    metadata["doc_consulted"] = [
                        {
                            "filename": s["filename"],
                            "page": s["page"],
                            "score": s["score"],
                        }
                        for s in doc_sources
                    ]
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    database.add_message,
                    conversation_id,
                    "assistant",
                    event["content"],
                    metadata,
                )
                yield {
                    "event": "writing",
                    "data": json.dumps(event),
                }

            elif event["type"] == "complete":
                yield {
                    "event": "done",
                    "data": json.dumps(event),
                }
                if is_first_message:
                    asyncio.create_task(_auto_title(conversation_id, message))

            elif event["type"] == "error":
                # Pipeline-level error (e.g. Ollama crashed mid-stream).
                # Forward it as an SSE error event so the frontend shows
                # a clean message instead of silently dying.
                yield {
                    "event": "error",
                    "data": json.dumps({"error": event["error"]}),
                }

    except Exception as e:
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)}),
        }


async def _auto_title(conversation_id: str, first_message: str) -> None:
    """Generate and save a conversation title from the first message.

    Runs as a background task (asyncio.create_task) so it doesn't block
    the chat response. Uses the tiny llama3.2:3b model that's already
    resident in memory, so there's no model-loading delay.
    """
    try:
        title = await router.generate_title(first_message)
        await asyncio.get_running_loop().run_in_executor(
            None, database.update_conversation_title, conversation_id, title
        )
    except Exception:
        # Title generation is non-critical — if it fails, the conversation
        # just keeps its "New conversation" default title. No need to crash.
        pass


# --- Document endpoints ---
#
# These endpoints handle the "Documents" mode: uploading PDFs, listing
# what's been uploaded, searching across document chunks, and deleting
# documents. The actual processing (text extraction, chunking, embedding)
# lives in documents.py — these endpoints just wire HTTP to those functions.


@app.post("/documents/upload")
async def upload_document(file: UploadFile) -> dict[str, Any]:
    """Upload a PDF and process it for search.

    Accepts a multipart file upload, saves it to a temp file, extracts
    text, chunks it, generates embeddings, and stores everything in memory.
    Returns a summary with filename, page count, and chunk count.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Sanitize the filename to prevent path traversal attacks.
    # Path.name strips directory components, so "../../evil.pdf" becomes "evil.pdf".
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Save the uploaded file to a temp location for PyMuPDF to read.
    # We use the data/ directory since it already exists for the database.
    upload_dir = Path(__file__).parent / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / safe_name

    try:
        content = await file.read()

        # Limit uploads to 50 MB — large enough for any reasonable document,
        # small enough to prevent accidental memory exhaustion.
        max_upload_size = 50 * 1024 * 1024
        if len(content) > max_upload_size:
            raise HTTPException(status_code=400, detail="File too large (max 50 MB)")

        temp_path.write_bytes(content)

        result = await documents.process_pdf(temp_path, safe_name)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        # Catch corrupt/encrypted PDFs, Ollama failures, etc. so the user
        # gets a clear 400 instead of an opaque 500 error.
        raise HTTPException(
            status_code=400,
            detail=f"Could not process PDF: {e}",
        )
    finally:
        # Clean up the temp file — we've already extracted the text and
        # embeddings, so we don't need the PDF on disk anymore.
        if temp_path.exists():
            temp_path.unlink()


@app.get("/documents")
def list_documents() -> list[dict[str, Any]]:
    """List all uploaded documents with their stats."""
    return documents.list_documents()


@app.delete("/documents/{filename}", status_code=204)
def delete_document(filename: str) -> None:
    """Remove a document and all its chunks."""
    deleted = documents.delete_document(filename)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")


@app.post("/documents/search")
async def search_documents(body: SearchRequest) -> list[dict[str, Any]]:
    """Search across all uploaded documents.

    Accepts a query string and search method (semantic or keyword).
    Returns the top_k most relevant chunks with scores and metadata.
    """
    return await documents.search_chunks(
        query=body.query, method=body.method, top_k=body.top_k
    )
