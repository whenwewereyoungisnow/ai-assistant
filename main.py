import asyncio
import json
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
@app.get("/")
async def root() -> FileResponse:
    return FileResponse(TEMPLATES_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
async def get_models() -> list[dict[str, Any]]:
    return await models.list_models()


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
# How this works end-to-end:
# 1. Frontend sends POST /chat/{id} with {"message": "user's question"}
# 2. We save the user message to the database immediately
# 3. We load conversation history from the database so the model has context
# 4. router.route_and_respond() classifies the question, picks a model,
#    and streams the response token by token
# 5. We wrap that stream in an EventSourceResponse (SSE) so the frontend
#    receives events in real time
# 6. After streaming completes, we save the full assistant response to the DB
# 7. If this is the first message, we auto-generate a title in the background


@app.post("/chat/{conversation_id}")
async def chat_endpoint(conversation_id: str, body: ChatRequest) -> EventSourceResponse:
    """Send a message and stream the AI response via Server-Sent Events.

    The response is a stream of SSE events:
    - event: routing  → which model was chosen and why
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
    loop = asyncio.get_event_loop()
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

    async def event_stream() -> AsyncGenerator[dict[str, str], None]:
        """Inner generator that yields SSE events.

        EventSourceResponse expects dicts with "event" and "data" keys.
        Each dict becomes one SSE event sent to the browser.
        """
        full_content = ""
        routing_metadata: dict[str, Any] = {}

        try:
            async for event in router.route_and_respond(body.message, history):
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
                    await asyncio.get_event_loop().run_in_executor(
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
                        asyncio.create_task(_auto_title(conversation_id, body.message))

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(event_stream())


async def _auto_title(conversation_id: str, first_message: str) -> None:
    """Generate and save a conversation title from the first message.

    Runs as a background task (asyncio.create_task) so it doesn't block
    the chat response. Uses the tiny llama3.2:3b model that's already
    resident in memory, so there's no model-loading delay.
    """
    try:
        title = await router.generate_title(first_message)
        await asyncio.get_event_loop().run_in_executor(
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
