# router.py — Smart question routing for chat mode
#
# Why separate router.py from models.py?
# models.py is a generic Ollama client — it knows how to send messages and
# stream tokens, but it doesn't know *which* model to use for *which* question.
# That's the router's job. router.py contains chat-specific logic: classifying
# questions, picking the right model, and orchestrating the response stream.
# Other modes (writing, documents) will have their own orchestration files
# that also import from models.py but route differently.

import time
from collections.abc import AsyncGenerator
from typing import Any

import models

# Each route maps a question category to a specific model and its options.
# The classifier (llama3.2:3b) reads the "description" field to decide which
# route fits the user's question. The "model" and "options" fields tell us
# what to actually run once a route is chosen.
#
# Why think: False for general and code?
# These models are fast MoE/dense models optimized for direct answers.
# Enabling "think" mode would add a chain-of-thought step that slows them
# down without improving quality for straightforward questions.
#
# Why think: True for reasoning?
# deepseek-r1 is specifically designed for chain-of-thought reasoning.
# It produces a <think>...</think> block with its reasoning process,
# then gives the final answer. This is slower but much better for math,
# logic, and multi-step problems.
ROUTES: list[dict[str, Any]] = [
    {
        "name": "general",
        "description": "General knowledge, casual conversation, explanations, summaries, creative writing, and anything not specifically about code or complex reasoning",
        "model": "qwen3.5:35b-a3b-coding-nvfp4",
        "options": {"think": False},
    },
    {
        "name": "code",
        "description": "Programming questions, code writing, debugging, code review, technical implementation, APIs, databases, and software engineering",
        "model": "qwen3.5:27b",
        "options": {"think": False},
    },
    {
        "name": "reasoning",
        "description": "Math problems, logic puzzles, multi-step analysis, complex comparisons, scientific reasoning, and questions requiring careful step-by-step thinking",
        "model": "deepseek-r1:32b",
        "options": {"think": True},
    },
]


# The system prompt sets the assistant's baseline behavior across all routes.
# It's intentionally short — we don't want to waste context window on a long
# prompt when the user's conversation history is more valuable.
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Be concise and clear. "
    "Use markdown formatting when it helps readability. "
    "If you're unsure about something, say so."
)


async def route_and_respond(
    question: str,
    conversation_history: list[dict[str, str]],
) -> AsyncGenerator[dict[str, Any], None]:
    """Classify a question, pick the right model, and stream the response.

    This is the main entry point for chat mode. It yields a sequence of events
    that the SSE endpoint forwards to the frontend:

    1. A "routing" event with which model was chosen and why
    2. Multiple "token" events as the response streams in
    3. A "done" event with timing statistics

    Args:
        question: The user's latest message
        conversation_history: Previous messages in OpenAI format:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            This is loaded from the database and passed in so the model has context.

    Why pass conversation_history as a parameter instead of loading it here?
    Because the caller (main.py) already has the conversation_id and handles
    database access. Keeping database logic out of the router makes it easier
    to test — you can call route_and_respond() with fake history without
    needing a real database.
    """
    total_start = time.monotonic()

    # Step 1: Classify the question to pick a route.
    # This calls llama3.2:3b (tiny, stays resident) so it's fast — typically
    # under 500ms. We time it so the frontend can show routing latency.
    classify_start = time.monotonic()
    classification = await models.classify(
        question, [{"name": r["name"], "description": r["description"]} for r in ROUTES]
    )
    classify_ms = round((time.monotonic() - classify_start) * 1000)

    # Find the matching route config. We use a default fallback so that if
    # the classifier returns something unexpected, we get a safe "general"
    # response instead of a confusing StopIteration crash.
    route_name = classification["route"]
    route = next((r for r in ROUTES if r["name"] == route_name), None)
    if route is None:
        route = ROUTES[0]
        route_name = route["name"]

    yield {
        "type": "routing",
        "route": route_name,
        "model": route["model"],
        "reason": classification.get("reason", ""),
        "classify_ms": classify_ms,
    }

    # Step 2: Build the messages array for the model.
    # The pattern is: system prompt → conversation history → new user message.
    # The system prompt goes first so the model knows its role before seeing
    # any conversation. History provides context for follow-up questions.
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})

    # Step 3: Stream the response token by token.
    stream_start = time.monotonic()
    async for token in models.stream_chat(route["model"], messages, route["options"]):
        yield {"type": "token", "content": token}

    stream_ms = round((time.monotonic() - stream_start) * 1000)
    total_ms = round((time.monotonic() - total_start) * 1000)

    yield {
        "type": "done",
        "stream_ms": stream_ms,
        "total_ms": total_ms,
    }


async def generate_title(question: str) -> str:
    """Generate a short conversation title from the first user message.

    Why auto-title?
    When you have 20+ conversations in the sidebar, they all say "New
    conversation" and you can't tell them apart. Auto-titling uses the
    tiny classifier model (already resident in memory, so no loading delay)
    to generate a 3-5 word summary after the first message. This runs
    in the background and doesn't block the chat response.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Generate a short title (3-6 words) for a conversation that "
                "starts with the following message. Respond with ONLY the title, "
                "no quotes, no punctuation at the end."
            ),
        },
        {"role": "user", "content": question},
    ]

    title = await models.chat(
        "llama3.2:3b",
        messages,
        options={"temperature": 0.3, "num_ctx": 512},
    )

    # Clean up: strip whitespace, remove quotes the model might add,
    # and truncate if the model got verbose
    title = title.strip().strip('"').strip("'")
    if len(title) > 60:
        title = title[:57] + "..."
    return title or "New conversation"
