# writer.py — Multi-agent writing pipeline
#
# Why a separate writer.py?
# Same reason router.py exists for chat mode: each mode has its own
# orchestration logic. writer.py knows how to run the draft-critique-revise
# loop, while models.py handles the raw Ollama communication. This keeps
# each file focused on one concern.
#
# How the pipeline works:
# 1. The Writer (qwen3.5, fast MoE model) drafts a response to the user's prompt
# 2. The Editor (deepseek-r1, thinking model) critiques the draft
# 3. The Writer revises based on the critique
# 4. Repeat steps 2-3 up to max_rounds, or until the Editor approves
#
# The Writer and Editor are both large models (~20GB each) that can't fit
# in GPU memory simultaneously. Ollama automatically swaps them, but each
# swap takes ~20 seconds. That's why both phases stream their output — so
# the user sees activity during the swap instead of staring at a blank screen.

import re
import time
from collections.abc import AsyncGenerator
from typing import Any

import models

# --- Model configuration ---

WRITER_MODEL = "qwen3.5:35b-a3b-coding-nvfp4"
WRITER_OPTIONS: dict[str, Any] = {"think": False}

EDITOR_MODEL = "deepseek-r1:32b"
EDITOR_OPTIONS: dict[str, Any] = {"think": True}

# --- System prompts ---
#
# The Writer prompt emphasizes creativity and structure. It tells the model
# to use markdown and to incorporate feedback when revising.
# The Editor prompt emphasizes critical analysis. The key instruction is the
# APPROVED convention: if the draft is good enough, the editor starts its
# response with "APPROVED" so the pipeline knows to stop early.

WRITER_SYSTEM_PROMPT = (
    "You are a skilled writer. Produce clear, well-structured, engaging prose. "
    "Use markdown formatting when it helps readability. "
    "When revising, carefully address every point of the editor's feedback "
    "while maintaining your creative voice. Do not mention the editor or "
    "the revision process in your output — just produce the improved text."
)

EDITOR_SYSTEM_PROMPT = (
    "You are a sharp literary editor. Review the draft for clarity, structure, "
    "tone, factual accuracy, grammar, and completeness. Provide specific, "
    "actionable feedback organized by priority.\n\n"
    "If the draft is excellent and needs no meaningful changes, start your "
    'response with the word "APPROVED" followed by a brief note on why '
    "it's ready. Only approve if the writing truly meets a high standard."
)


async def run_pipeline(
    prompt: str,
    history: list[dict[str, str]],
    max_rounds: int = 3,
) -> AsyncGenerator[dict[str, Any], None]:
    """Run the write-critique-revise pipeline, yielding events for each phase.

    This is the main entry point for writing mode. It yields a sequence of
    events that the SSE endpoint in main.py forwards to the frontend:

    1. phase_start — a new phase is beginning (draft, critique, or revision)
    2. token      — one token from the currently-active model
    3. phase_end  — a phase completed, with full content and timing
    4. complete   — the pipeline is done

    Args:
        prompt: The user's writing request (e.g. "Write a blog post about...")
        history: Previous messages in the conversation (currently unused by the
                 pipeline, but passed for consistency with other mode orchestrators
                 and potential future use like "revise what we worked on earlier")
        max_rounds: Maximum draft-critique-revise cycles before stopping.
                    Default 3. The Editor can approve early to stop sooner.

    Why yield events instead of returning a final result?
    Because the pipeline takes minutes (model swaps + generation). Yielding
    events lets the frontend show real-time progress: the draft streaming in,
    then the critique, then the revision. Without events, the user would
    stare at a blank screen for several minutes.
    """
    total_start = time.monotonic()

    # Track all drafts and critiques so each revision round has full context.
    # The Writer sees: original prompt + all prior drafts + all editor feedback.
    # This lets it learn from the critique trajectory, not just the latest one.
    drafts: list[str] = []
    critiques: list[str] = []

    for round_num in range(1, max_rounds + 1):
        # --- DRAFT / REVISION PHASE ---
        # Round 1 is the initial draft. Rounds 2+ are revisions informed by
        # the editor's critique. We use different phase names so the frontend
        # can label them differently.
        phase = "draft" if round_num == 1 else "revision"

        yield {
            "type": "phase_start",
            "phase": phase,
            "round": round_num,
            "model": WRITER_MODEL,
        }

        # Build the Writer's message context.
        # The pattern is: system prompt, then the user's original request,
        # then alternating draft/critique pairs from prior rounds, then
        # (for revisions) a final instruction to revise.
        writer_messages: list[dict[str, str]] = [
            {"role": "system", "content": WRITER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Add prior rounds as context so the writer can see its trajectory
        for i, (draft_text, critique_text) in enumerate(zip(drafts, critiques)):
            writer_messages.append(
                {"role": "assistant", "content": f"[Draft {i + 1}]\n{draft_text}"}
            )
            writer_messages.append(
                {"role": "user", "content": f"[Editor feedback]\n{critique_text}"}
            )

        if round_num > 1:
            writer_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Please revise your draft, carefully addressing all of "
                        "the editor's feedback above."
                    ),
                }
            )

        # Writing tasks produce long-form content, so use a larger context
        # window than the default 8192 even for the first draft. Revisions
        # need more still since they include all prior drafts + critiques.
        writer_options = dict(WRITER_OPTIONS)
        writer_options["num_ctx"] = 16384 if round_num > 1 else 12288

        phase_start = time.monotonic()
        draft_content = ""
        try:
            async for token in models.stream_chat(
                WRITER_MODEL, writer_messages, writer_options
            ):
                draft_content += token
                yield {"type": "token", "content": token, "phase": phase}
        except Exception as e:
            # If Ollama crashes or times out mid-stream, yield an error event
            # so the frontend gets a clean notification instead of an abrupt
            # stream termination with no explanation.
            yield {"type": "error", "error": str(e), "phase": phase}
            return

        phase_ms = round((time.monotonic() - phase_start) * 1000)
        drafts.append(draft_content)

        yield {
            "type": "phase_end",
            "phase": phase,
            "round": round_num,
            "content": draft_content,
            "duration_ms": phase_ms,
        }

        # --- CRITIQUE PHASE ---
        # The Editor reviews the latest draft against the original prompt.
        # It gets a fresh context (no prior critique history) so it evaluates
        # the current draft on its own merits rather than anchoring on previous
        # feedback.

        yield {
            "type": "phase_start",
            "phase": "critique",
            "round": round_num,
            "model": EDITOR_MODEL,
        }

        editor_messages: list[dict[str, str]] = [
            {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original writing request:\n{prompt}\n\n"
                    f"Draft to review:\n{draft_content}"
                ),
            },
        ]

        # Editor needs a large context to read the full draft + produce
        # critique. deepseek-r1's thinking adds overhead on top of the
        # draft length, so 16384 gives plenty of room.
        editor_options = dict(EDITOR_OPTIONS)
        editor_options["num_ctx"] = 16384

        critique_start = time.monotonic()
        critique_content = ""
        try:
            async for token in models.stream_chat(
                EDITOR_MODEL, editor_messages, editor_options
            ):
                critique_content += token
                yield {"type": "token", "content": token, "phase": "critique"}
        except Exception as e:
            yield {"type": "error", "error": str(e), "phase": "critique"}
            return

        critique_ms = round((time.monotonic() - critique_start) * 1000)
        critiques.append(critique_content)

        yield {
            "type": "phase_end",
            "phase": "critique",
            "round": round_num,
            "content": critique_content,
            "duration_ms": critique_ms,
        }

        # Check if the Editor approved the draft.
        # deepseek-r1 wraps its reasoning in <think>...</think> tags before
        # the actual response. We strip those before checking for "APPROVED"
        # so the thinking process doesn't accidentally trigger approval.
        clean_critique = re.sub(
            r"<think>[\s\S]*?</think>", "", critique_content
        ).strip()
        if clean_critique.upper().startswith("APPROVED"):
            total_ms = round((time.monotonic() - total_start) * 1000)
            yield {
                "type": "complete",
                "total_ms": total_ms,
                "rounds": round_num,
                "approved": True,
            }
            return

    # If we exhausted all rounds without approval, present the final draft anyway.
    total_ms = round((time.monotonic() - total_start) * 1000)
    yield {
        "type": "complete",
        "total_ms": total_ms,
        "rounds": max_rounds,
        "approved": False,
    }
