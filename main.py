import asyncio
import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import database
import documents
import models
import settings
import streams
from logging_middleware import LoggingMiddleware, init_logs_table


# Lifespan handler — FastAPI runs the code before "yield" on startup, and
# after "yield" on shutdown. This is the standard pattern for managing
# resources (HTTP clients, DB connections) that need cleanup.
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    database.init_db()
    settings.init_settings()
    init_logs_table()
    models.init_client()
    yield
    await models.close_client()


app = FastAPI(title="AI Assistant", lifespan=lifespan)

# Request logging middleware — records endpoint, method, status, and timing
# for every API call (except static files and health checks) to the
# request_logs table. See logging_middleware.py for details.
app.add_middleware(LoggingMiddleware)

TEMPLATES_DIR = Path(__file__).parent / "templates"


# --- Global exception handlers ---
# Catch common errors in one place so every endpoint gets clean messages.


@app.exception_handler(httpx.ConnectError)
async def ollama_connection_error(
    request: Request, exc: httpx.ConnectError
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": "Cannot connect to Ollama. Is it running? Start it with: ollama serve"
        },
    )


@app.exception_handler(httpx.HTTPStatusError)
async def ollama_http_error(
    request: Request, exc: httpx.HTTPStatusError
) -> JSONResponse:
    # Check if Ollama returned a "model not found" error — suggest pulling it
    body = exc.response.text
    if exc.response.status_code == 404 and "not found" in body.lower():
        # Extract the model name from the error message if possible
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Model not found. Try: ollama pull <model_name>. "
                f"Ollama says: {body}"
            },
        )
    return JSONResponse(
        status_code=502,
        content={"error": f"Ollama returned an error: {exc.response.status_code}"},
    )


@app.exception_handler(sqlite3.Error)
async def database_error(request: Request, exc: sqlite3.Error) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": f"Database error: {exc}"},
    )


# --- Static files ---


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


# --- System status ---


@app.get("/status")
async def get_status() -> dict[str, Any]:
    """System status: loaded models, document counts, conversations, GPU memory."""
    loop = asyncio.get_running_loop()

    try:
        running = await models.running_models()
    except Exception:
        running = []

    conversation_count = await loop.run_in_executor(None, database.count_conversations)

    total_vram_bytes = sum(m.get("size_vram", 0) for m in running)
    total_vram_gb = round(total_vram_bytes / (1024**3), 1)

    return {
        "models": [m.get("name", "unknown") for m in running],
        "documents": documents.document_count(),
        "chunks": documents.chunk_count(),
        "conversations": conversation_count,
        "gpu_vram_gb": total_vram_gb,
    }


# --- Settings endpoints ---


class SettingsUpdateRequest(BaseModel):
    """Partial settings update. Only keys that are present will be changed."""

    general_model: str | None = None
    code_model: str | None = None
    reasoning_model: str | None = None
    rag_model: str | None = None
    writer_model: str | None = None
    editor_model: str | None = None
    max_writing_rounds: int | None = None
    search_method: str | None = None
    search_results_count: int | None = None
    theme: str | None = None


@app.get("/settings")
def get_settings() -> dict[str, Any]:
    """Return all current settings."""
    return settings.get_all_settings()


@app.put("/settings")
def update_settings(body: SettingsUpdateRequest) -> dict[str, Any]:
    """Update one or more settings. Returns all settings after the update.

    Validates ranges: max_writing_rounds must be 1-5, search_results_count
    must be 1-10, search_method must be one of the allowed values, theme
    must be dark or light.
    """
    updates: dict[str, Any] = {}

    # Only include fields that were actually provided (not None)
    for key, value in body.model_dump().items():
        if value is not None:
            updates[key] = value

    # Validate constraints
    if "max_writing_rounds" in updates:
        if not 1 <= updates["max_writing_rounds"] <= 5:
            raise HTTPException(
                status_code=400, detail="max_writing_rounds must be 1-5"
            )
    if "search_results_count" in updates:
        if not 1 <= updates["search_results_count"] <= 10:
            raise HTTPException(
                status_code=400, detail="search_results_count must be 1-10"
            )
    if "search_method" in updates:
        if updates["search_method"] not in ("semantic", "keyword", "hybrid"):
            raise HTTPException(
                status_code=400,
                detail="search_method must be semantic, keyword, or hybrid",
            )
    if "theme" in updates:
        if updates["theme"] not in ("dark", "light", "system"):
            raise HTTPException(
                status_code=400, detail="theme must be dark, light, or system"
            )

    if updates:
        settings.set_many(updates)

    return settings.get_all_settings()


# --- Conversation endpoints ---


class CreateConversationRequest(BaseModel):
    mode: Literal["chat", "documents", "writing", "vision"]
    title: str | None = None


class ChatRequest(BaseModel):
    message: str


class SearchRequest(BaseModel):
    query: str
    method: Literal["semantic", "keyword", "hybrid"] = "semantic"
    top_k: int = 5


@app.get("/conversations")
def list_conversations() -> list[dict[str, Any]]:
    return database.list_conversations()


@app.post("/conversations", status_code=201)
def create_conversation(body: CreateConversationRequest) -> dict[str, str]:
    conversation_id = database.create_conversation(mode=body.mode, title=body.title)
    return {"id": conversation_id}


@app.get("/conversations/search")
def search_conversations(q: str = Query(..., min_length=1)) -> list[dict[str, Any]]:
    """Search across all conversation messages.

    Uses substring matching on message content. Returns matching conversations
    with a content snippet showing where the match was found.
    """
    return database.search_messages(q)


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict[str, Any]:
    conversation = database.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str) -> None:
    deleted = database.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")


@app.post("/conversations/{conversation_id}/export")
def export_conversation(conversation_id: str) -> PlainTextResponse:
    """Export a conversation as a clean markdown file.

    Formats the conversation with headers, role labels, and metadata
    (model used, timing info) into a readable markdown document.
    The browser downloads it as a .md file.
    """
    conversation = database.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Build the markdown document
    title = conversation.get("title", "Conversation")
    mode = conversation.get("mode", "chat")
    created = conversation.get("created_at", "")

    lines = [
        f"# {title}",
        f"Mode: {mode} | Created: {created}",
        "",
        "---",
        "",
    ]

    for msg in conversation.get("messages", []):
        role = msg["role"].capitalize()
        content = msg["content"]
        metadata = msg.get("metadata") or {}

        lines.append(f"**{role}:**\n")
        lines.append(content)
        lines.append("")

        # Add metadata line for assistant messages
        if msg["role"] == "assistant":
            meta_parts = []
            if metadata.get("model"):
                meta_parts.append(f"Model: {metadata['model']}")
            if metadata.get("total_ms"):
                meta_parts.append(f"{metadata['total_ms']}ms")
            if metadata.get("route"):
                meta_parts.append(f"Route: {metadata['route']}")
            if meta_parts:
                lines.append(f"*{' | '.join(meta_parts)}*")
                lines.append("")

        lines.append("---")
        lines.append("")

    markdown = "\n".join(lines)
    safe_title = title.replace("/", "-").replace("\\", "-").replace('"', "'")[:50]

    return PlainTextResponse(
        content=markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{safe_title}.md"'},
    )


# --- Chat endpoint (SSE streaming) ---
#
# The same /chat endpoint handles all modes. It checks the conversation's mode
# from the database and dispatches to the appropriate stream generator in
# streams.py. All modes share the same SSE plumbing and conversation storage.


@app.post("/chat/{conversation_id}")
async def chat_endpoint(conversation_id: str, body: ChatRequest) -> EventSourceResponse:
    """Send a message and stream the AI response via Server-Sent Events."""
    loop = asyncio.get_running_loop()
    conversation = await loop.run_in_executor(
        None, database.get_conversation, conversation_id
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await loop.run_in_executor(
        None, database.add_message, conversation_id, "user", body.message
    )

    history: list[dict[str, str]] = []
    for msg in conversation.get("messages", []):
        if msg["role"] in ("user", "assistant"):
            history.append({"role": msg["role"], "content": msg["content"]})

    is_first_message = len(history) == 0
    mode = conversation.get("mode", "chat")

    if mode == "documents":
        return EventSourceResponse(
            streams.documents_event_stream(
                conversation_id, body.message, history, is_first_message
            )
        )
    elif mode == "writing":
        return EventSourceResponse(
            streams.writing_event_stream(
                conversation_id, body.message, history, is_first_message
            )
        )
    else:
        return EventSourceResponse(
            streams.chat_event_stream(
                conversation_id, body.message, history, is_first_message
            )
        )


# --- Document endpoints ---


@app.post("/documents/upload")
async def upload_document(file: UploadFile) -> dict[str, Any]:
    """Upload a PDF and process it for search."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    upload_dir = Path(__file__).parent / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / safe_name

    try:
        content = await file.read()

        max_upload_size = 50 * 1024 * 1024
        if len(content) > max_upload_size:
            raise HTTPException(
                status_code=413,
                detail="File too large — maximum upload size is 50 MB",
            )

        temp_path.write_bytes(content)
        result = await documents.process_pdf(temp_path, safe_name)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        # Provide specific error messages for common PDF failures
        if "encrypted" in error_msg.lower() or "password" in error_msg.lower():
            detail = "This PDF is password-protected. Please remove the password and try again."
        elif "corrupt" in error_msg.lower() or "invalid" in error_msg.lower():
            detail = "This PDF appears to be corrupted and could not be read."
        else:
            detail = f"Could not process PDF: {error_msg}"
        raise HTTPException(status_code=400, detail=detail)
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.get("/documents")
def list_documents_endpoint() -> list[dict[str, Any]]:
    return documents.list_documents()


@app.delete("/documents/{filename}", status_code=204)
def delete_document(filename: str) -> None:
    deleted = documents.delete_document(filename)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")


@app.post("/documents/search")
async def search_documents(body: SearchRequest) -> list[dict[str, Any]]:
    return await documents.search_chunks(
        query=body.query, method=body.method, top_k=body.top_k
    )
