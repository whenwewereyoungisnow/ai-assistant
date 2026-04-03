# streams.py — SSE event stream generators for each conversation mode
#
# Why a separate streams.py?
# main.py was at 850 lines with three large async generators (chat, documents,
# writing) plus all the route handlers. Extracting the generators brings main.py
# under 300 lines and makes each file focused on one concern:
#   - main.py: app setup, HTTP routes, request/response handling
#   - streams.py: SSE streaming logic for each mode
#   - router.py: question classification and model selection
#   - writer.py: multi-agent writing pipeline
#
# Each generator reads model names and search settings from the settings module
# at the start of each request, so changes in the settings panel take effect
# immediately without a server restart.

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import database
import documents
import models
import router
import settings
import writer

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
DOC_RELEVANCE_THRESHOLD = 0.55
DOC_CONSULT_TOP_K = 3


# How often to auto-save partial responses during streaming.
# Every PARTIAL_SAVE_TOKENS tokens OR PARTIAL_SAVE_SECONDS seconds (whichever
# comes first), the current content is written to SQLite. This balances
# durability against write overhead. At typical rates (~30-80 tok/s), this
# means a save every 0.6-1.7 seconds — good enough that a crash loses at
# most a sentence or two.
PARTIAL_SAVE_TOKENS = 50
PARTIAL_SAVE_SECONDS = 5.0


async def _save_partial(
    message_id: str, content: str, loop: asyncio.AbstractEventLoop
) -> None:
    """Save partial streaming content to the database (runs in thread pool)."""
    await loop.run_in_executor(
        None,
        database.update_message_content,
        message_id,
        content,
        {"partial": True},
    )


async def chat_event_stream(
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
    loop = asyncio.get_running_loop()

    # Read settings at request time so changes take effect immediately
    active_routes = settings.get_routes()

    # Create a placeholder assistant message before streaming starts.
    # This gets updated incrementally during streaming so partial responses
    # survive connection drops.
    message_id = await loop.run_in_executor(
        None, database.add_message, conversation_id, "assistant", "", {"partial": True}
    )

    # Tracking for periodic partial saves
    token_count = 0
    last_save_time = time.monotonic()

    try:
        # --- Cross-mode: auto-consult documents if relevant ---
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

        async for event in router.route_and_respond(
            effective_message, history, routes=active_routes
        ):
            if event["type"] == "routing":
                routing_metadata = event
                yield {
                    "event": "routing",
                    "data": json.dumps(event),
                }

            elif event["type"] == "token":
                full_content += event["content"]
                token_count += 1
                yield {
                    "event": "token",
                    "data": json.dumps(event),
                }

                # Periodic partial save — every N tokens or N seconds
                now = time.monotonic()
                if (
                    token_count >= PARTIAL_SAVE_TOKENS
                    or now - last_save_time >= PARTIAL_SAVE_SECONDS
                ):
                    token_count = 0
                    last_save_time = now
                    await _save_partial(message_id, full_content, loop)

            elif event["type"] == "done":
                metadata = {
                    "model": routing_metadata.get("model", ""),
                    "route": routing_metadata.get("route", ""),
                    "reason": routing_metadata.get("reason", ""),
                    "classify_ms": routing_metadata.get("classify_ms", 0),
                    "stream_ms": event.get("stream_ms", 0),
                    "total_ms": event.get("total_ms", 0),
                }
                if doc_sources:
                    metadata["doc_consulted"] = [
                        {
                            "filename": s["filename"],
                            "page": s["page"],
                            "score": s["score"],
                        }
                        for s in doc_sources
                    ]
                # Final save — replaces the partial placeholder with complete
                # content and full metadata
                await loop.run_in_executor(
                    None,
                    database.update_message_content,
                    message_id,
                    full_content,
                    metadata,
                )

                yield {
                    "event": "done",
                    "data": json.dumps(event),
                }

                if is_first_message:
                    asyncio.create_task(auto_title(conversation_id, message))

    except Exception as e:
        # Save whatever we have so the partial response isn't lost
        if full_content:
            await loop.run_in_executor(
                None,
                database.update_message_content,
                message_id,
                full_content,
                {"error": str(e), "partial": True},
            )
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)}),
        }


async def documents_event_stream(
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
    4. Stream the model's response (uses the RAG model from settings)
    """
    full_content = ""
    total_start = time.monotonic()
    loop = asyncio.get_running_loop()

    # Read settings at request time
    rag_model = settings.get_setting("rag_model")
    search_method = settings.get_setting("search_method")
    search_count = settings.get_setting("search_results_count")

    # Create placeholder for partial saves
    message_id = await loop.run_in_executor(
        None, database.add_message, conversation_id, "assistant", "", {"partial": True}
    )

    token_count = 0
    last_save_time = time.monotonic()

    try:
        sources = await documents.search_chunks(
            message, method=search_method, top_k=search_count
        )

        yield {
            "event": "sources",
            "data": json.dumps({"sources": sources, "model": rag_model}),
        }

        if not sources:
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
                "model": rag_model,
                "sources": [],
                "total_ms": total_ms,
                "stream_ms": 0,
            }
            await loop.run_in_executor(
                None,
                database.update_message_content,
                message_id,
                full_content,
                metadata,
            )
            yield {
                "event": "done",
                "data": json.dumps({"total_ms": total_ms, "stream_ms": 0}),
            }
            if is_first_message:
                asyncio.create_task(auto_title(conversation_id, message))
            return

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

        stream_start = time.monotonic()
        async for token in models.stream_chat(
            rag_model, messages, RAG_MODEL_OPTIONS, keep_alive="10m"
        ):
            full_content += token
            token_count += 1
            yield {
                "event": "token",
                "data": json.dumps({"type": "token", "content": token}),
            }

            # Periodic partial save
            now = time.monotonic()
            if (
                token_count >= PARTIAL_SAVE_TOKENS
                or now - last_save_time >= PARTIAL_SAVE_SECONDS
            ):
                token_count = 0
                last_save_time = now
                await _save_partial(message_id, full_content, loop)

        stream_ms = round((time.monotonic() - stream_start) * 1000)
        total_ms = round((time.monotonic() - total_start) * 1000)

        metadata = {
            "model": rag_model,
            "sources": sources,
            "total_ms": total_ms,
            "stream_ms": stream_ms,
        }
        # Final save with complete metadata
        await loop.run_in_executor(
            None,
            database.update_message_content,
            message_id,
            full_content,
            metadata,
        )

        yield {
            "event": "done",
            "data": json.dumps({"total_ms": total_ms, "stream_ms": stream_ms}),
        }

        if is_first_message:
            asyncio.create_task(auto_title(conversation_id, message))

    except Exception as e:
        if full_content:
            await loop.run_in_executor(
                None,
                database.update_message_content,
                message_id,
                full_content,
                {"error": str(e), "partial": True},
            )
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)}),
        }


async def writing_event_stream(
    conversation_id: str,
    message: str,
    history: list[dict[str, str]],
    is_first_message: bool,
) -> AsyncGenerator[dict[str, str], None]:
    """SSE event stream for writing mode (multi-agent pipeline).

    Writing mode uses two models collaborating across multiple rounds.
    The Writer drafts, the Editor critiques, and the Writer revises.
    """
    # Read settings at request time
    writer_model = settings.get_setting("writer_model")
    editor_model = settings.get_setting("editor_model")
    max_rounds = settings.get_setting("max_writing_rounds")

    try:
        # --- Cross-mode: check if documents can inform the writing ---
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
            message,
            history,
            max_rounds=max_rounds,
            doc_context=doc_context,
            writer_model=writer_model,
            editor_model=editor_model,
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
                phase_model = (
                    editor_model if event["phase"] == "critique" else writer_model
                )
                metadata: dict[str, Any] = {
                    "phase": event["phase"],
                    "round": event["round"],
                    "model": phase_model,
                    "duration_ms": event["duration_ms"],
                    "pipeline": True,
                }
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
                    asyncio.create_task(auto_title(conversation_id, message))

            elif event["type"] == "error":
                yield {
                    "event": "error",
                    "data": json.dumps({"error": event["error"]}),
                }

    except Exception as e:
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)}),
        }


async def auto_title(conversation_id: str, first_message: str) -> None:
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
        # just keeps its "New conversation" default title.
        pass
