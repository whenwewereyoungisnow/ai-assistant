from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import models


# Lifespan handler — FastAPI runs the code before "yield" on startup, and
# after "yield" on shutdown. This is the standard pattern for managing
# resources (HTTP clients, DB connections) that need cleanup.
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
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
