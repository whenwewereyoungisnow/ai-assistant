# models.py — Shared Ollama communication layer
#
# Why centralize Ollama calls here?
# Every feature in this app (chat, documents, writing, vision) needs to talk
# to Ollama. If each feature made its own HTTP calls, we'd have duplicated
# connection logic, inconsistent error handling, and no single place to tune
# timeouts or swap the base URL. By routing everything through models.py:
#   1. One httpx client with shared timeout/connection settings
#   2. One place to handle Ollama errors and retries
#   3. Features import clean functions (stream_chat, embed, classify)
#      instead of dealing with raw HTTP
#   4. Easy to test — mock this one module to test any feature

import json
import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from dotenv import load_dotenv

# load_dotenv() reads key=value pairs from a .env file in the project root
# and sets them as environment variables. This keeps secrets (API keys, DB paths)
# out of source code. If .env doesn't exist yet, this is a harmless no-op.
load_dotenv()

# All Ollama API calls go to this base URL.
# Defaults to localhost but can be overridden via environment variable —
# useful when deploying to Railway where Ollama runs at a different address.
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# httpx client shared across all calls. Initialized via init_client() and
# closed via close_client() — FastAPI's lifespan handler manages this so
# connections are properly cleaned up on shutdown.
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    """Return the shared httpx client. Raises if not initialized."""
    if _client is None:
        raise RuntimeError("httpx client not initialized — call init_client() first")
    return _client


def init_client() -> None:
    """Create the shared httpx client. Call once at app startup."""
    global _client
    _client = httpx.AsyncClient(
        base_url=OLLAMA_BASE,
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
    )


async def close_client() -> None:
    """Close the shared httpx client. Call once at app shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _apply_no_think(
    model: str, messages: list[dict[str, Any]], think: bool | None
) -> list[dict[str, Any]]:
    """Append /no_think to the last user message for qwen3.5 models.

    Even with the API-level think=false parameter, qwen3.5 sometimes still
    enters thinking mode during streaming. Appending /no_think to the user
    message content is the most reliable prompt-level suppression method.
    We make a shallow copy of the messages list and only copy the last user
    message dict, so we don't mutate the caller's data.
    """
    if think is not False or "qwen3.5" not in model:
        return messages

    # Find the last user message and append /no_think
    messages = list(messages)  # shallow copy of the list
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            messages[i] = {
                **messages[i],
                "content": messages[i]["content"] + " /no_think",
            }
            break
    return messages


async def stream_chat(
    model: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any] | None = None,
) -> AsyncGenerator[str, None]:
    """Stream a chat response token-by-token from Ollama.

    Use this for user-facing responses where you want text to appear as it's
    generated (via SSE). Each yielded string is one token fragment.

    Args:
        model: Ollama model name (e.g. "qwen3.5:35b-a3b-coding-nvfp4")
        messages: Chat history in OpenAI-style format:
                  [{"role": "user", "content": "hello"}]
        options: Optional Ollama parameters like {"temperature": 0.7,
                 "num_ctx": 4096, "think": False}
    """
    # Default to 8192 token context window — enough for multi-turn conversations
    # and RAG chunks without eating too much VRAM on a single request.
    merged_options: dict[str, Any] = {"num_ctx": 8192}
    if options:
        merged_options.update(options)

    # "think" controls whether the model runs a chain-of-thought reasoning
    # step before responding. Ollama expects "think" as a TOP-LEVEL parameter,
    # not inside "options". If it's inside options, Ollama silently ignores it
    # during streaming — the model still thinks (putting content in the
    # "message.thinking" field instead of "message.content"), so stream_chat
    # yields nothing until thinking finishes. Extracting it here fixes that.
    think = merged_options.pop("think", None)
    messages = _apply_no_think(model, messages, think)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": merged_options,
    }
    if think is not None:
        payload["think"] = think

    # We use stream() to read the response line-by-line as Ollama sends it.
    # Each line is a JSON object with a "message.content" field containing
    # one token. The last line has "done": true.
    async with get_client().stream("POST", "/api/chat", json=payload) as response:
        if response.status_code != 200:
            await response.aread()
            raise RuntimeError(
                f"Ollama returned {response.status_code}: {response.text}"
            )
        async for line in response.aiter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            # Ollama can send errors mid-stream as {"error": "message"}
            if "error" in chunk:
                raise RuntimeError(f"Ollama error: {chunk['error']}")
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token


async def chat(
    model: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any] | None = None,
) -> str:
    """Send a chat request and return the complete response as a string.

    Use this for internal calls where you need the full answer at once —
    classification, summarization, editing passes. Not for user-facing
    streaming.

    Args:
        model: Ollama model name
        messages: Chat history in OpenAI-style format
        options: Optional Ollama parameters
    """
    # Default to 8192 token context window — enough for multi-turn conversations
    # and RAG chunks without eating too much VRAM on a single request.
    merged_options: dict[str, Any] = {"num_ctx": 8192}
    if options:
        merged_options.update(options)

    # Extract "think" to top level — same reason as in stream_chat above.
    think = merged_options.pop("think", None)
    messages = _apply_no_think(model, messages, think)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": merged_options,
    }
    if think is not None:
        payload["think"] = think

    response = await get_client().post("/api/chat", json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama returned {response.status_code}: {response.text}")
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"Ollama error: {data['error']}")
    message = data.get("message")
    if message is None:
        raise RuntimeError(f"Ollama returned no message: {data}")
    return message.get("content", "")


async def classify(question: str, routes: list[dict[str, str]]) -> dict[str, str]:
    """Classify a question into one of the given routes using a small fast model.

    This powers the smart router. It sends the question to llama3.2:3b (tiny,
    stays resident in memory) and asks it to pick the best route. Returns a
    dict like {"route": "code", "reason": "user is asking about Python syntax"}.

    Falls back to "general" if the model returns unparseable JSON — this is
    intentional so the app never crashes on a bad classification.

    Args:
        question: The user's message to classify
        routes: List of route options, each with "name" and "description" keys.
                Example: [{"name": "code", "description": "Programming questions"}]
    """
    # Build a description of available routes for the prompt
    route_descriptions = "\n".join(
        f'- "{r["name"]}": {r["description"]}' for r in routes
    )

    system_prompt = (
        "You are a question classifier. Given a user question and a list of "
        "routes, pick the single best route. Respond with ONLY a JSON object "
        'like {"route": "name", "reason": "brief reason"}. No other text.\n\n'
        f"Available routes:\n{route_descriptions}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    options = {
        "temperature": 0,
        "num_ctx": 1024,
    }

    try:
        raw = await chat("llama3.2:3b", messages, options)
        # Small models often wrap JSON in markdown code fences like ```json...```
        # Strip those so json.loads() can parse the actual JSON inside.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Drop only the opening fence (first line) and closing fence (last
            # line) rather than filtering every line that contains ```.  The old
            # approach could accidentally strip content if the JSON value itself
            # contained backticks.
            lines = cleaned.split("\n")
            if lines[-1].strip().startswith("```"):
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            cleaned = "\n".join(lines)
        result = json.loads(cleaned)
        # Validate the response has the expected keys
        if "route" not in result:
            raise ValueError("Missing 'route' key")
        # Make sure the route is one we actually offered
        valid_names = {r["name"] for r in routes}
        if result["route"] not in valid_names:
            raise ValueError(f"Unknown route: {result['route']}")
        return result
    except (json.JSONDecodeError, ValueError, KeyError):
        return {"route": "general", "reason": "classification failed, using fallback"}


async def embed(texts: str | list[str]) -> list[list[float]]:
    """Generate embedding vectors for one or more texts.

    Uses qwen3-embedding:4b for semantic search in the RAG pipeline. The model
    loads briefly when needed and doesn't need to stay resident.

    Args:
        texts: A single string or list of strings to embed

    Returns:
        List of embedding vectors (one per input text). Each vector is a list
        of floats.
    """
    # Ollama's /api/embed accepts "input" as string or list of strings
    if isinstance(texts, str):
        texts = [texts]

    response = await get_client().post(
        "/api/embed",
        json={"model": "qwen3-embedding:4b", "input": texts},
    )
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"Ollama embedding error: {data['error']}")
    embeddings = data.get("embeddings")
    if embeddings is None:
        raise RuntimeError(f"Unexpected Ollama response — no 'embeddings' key: {data}")
    return embeddings


async def list_models() -> list[dict[str, Any]]:
    """List all models downloaded in Ollama.

    Returns model names and sizes. Useful for the /models endpoint and for
    checking if required models are available before trying to use them.
    """
    response = await get_client().get("/api/tags")
    response.raise_for_status()
    data = response.json()
    return data.get("models", [])


async def running_models() -> list[dict[str, Any]]:
    """List models currently loaded in Ollama's memory.

    Returns which models are active right now. Useful for understanding memory
    usage — e.g., checking if we need to wait for a model swap.
    """
    response = await get_client().get("/api/ps")
    response.raise_for_status()
    data = response.json()
    return data.get("models", [])
