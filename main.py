from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import database
import models


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


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "AI Assistant - coming soon"}


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
