# CLAUDE.md — AI Assistant Platform

## What this project is

A personal AI assistant that combines smart model routing, document
chat (RAG), multi-agent writing, image understanding, and custom
personas into one unified web application. It runs entirely on local
Ollama models with persistent conversation storage via SQLite.

This is a learning project. I'm a beginner developer building this to
understand how to combine multiple AI features into a single product.
Explain key decisions and new concepts when you introduce them — not
every line, but the "why" behind architectural choices.

## Architecture (locked in — don't change these)

- **Backend:** FastAPI + Uvicorn (Python)
- **Frontend:** Plain HTML + vanilla JavaScript in templates/index.html (no React, no build step)
- **Streaming:** Server-Sent Events (SSE) via sse-starlette
- **HTTP client:** httpx (for calling Ollama)
- **Database:** SQLite via Python's built-in sqlite3 (no ORM, no SQLAlchemy)
- **Document extraction:** PyMuPDF (PDF), python-docx (DOCX), built-in csv/open (TXT, MD, CSV)
- **Keyword search:** rank-bm25
- **Embedding search:** qwen3-embedding:4b + numpy cosine similarity
- **Default search:** Hybrid (BM25 + semantic combined, weighted 0.4/0.6)
- **Markdown rendering:** marked.js + highlight.js (loaded from CDN)
- **Dependency management:** uv (never pip)

## File structure

This is a multi-file project. Each feature gets its own file:

- main.py — FastAPI app, routes, startup
- models.py — All Ollama API calls (shared by every feature)
- router.py — Question classifier + model routing logic
- documents.py — Document upload, chunking, search (RAG)
- writer.py — Multi-agent writing pipeline
- personas.py — Custom persona CRUD and built-in personas
- database.py — SQLite conversation storage
- templates/index.html — Full frontend (HTML + JS + CSS)

Do NOT put everything in main.py. Do NOT create unnecessary files.
If a new function clearly belongs in an existing file, put it there.
Only create a new file if it represents a genuinely separate concern.

## Models

- **Classifier:** `llama3.2:3b` — tiny, stays resident, routes questions
- **General route:** `qwen3.5:35b-a3b` — fast MoE (think: false)
- **Code route:** `qwen3.5:27b` — dense, strong at programming (think: false)
- **Reasoning route:** `deepseek-r1:32b` — chain-of-thought (thinking ON)
- **Writer:** `qwen3.5:35b-a3b` — creative drafting (think: false)
- **Editor:** `deepseek-r1:32b` — detailed critique (thinking ON)
- **Embeddings:** `qwen3-embedding:4b` — semantic search
- **Vision:** `llama3.2-vision` — image understanding (11B, ~7GB)
- **Ollama API:** `http://localhost:11434`

## Memory constraints

- Writer (35b-a3b, ~26GB) and Editor (deepseek-r1, ~20GB) cannot fit
  simultaneously. Ollama swaps them on each handoff (~20s per swap).
- Classifier (llama3.2:3b, ~2.3GB) stays resident alongside any large model.
- Embedding model loads briefly for upload/search, doesn't need to stay loaded.
- Vision model (llama3.2-vision, ~7GB) loads on demand for image queries.
- OLLAMA_MAX_LOADED_MODELS=2 is set so classifier + one large model co-reside.

## Four modes

1. **Chat** — Smart-routed conversation. Classifier picks model.
2. **Documents** — RAG over uploaded files (PDF, TXT, MD, DOCX, CSV). Hybrid search.
3. **Writing** — Multi-agent pipeline. Writer drafts, Editor critiques, iterate.
4. **Vision** — Image understanding. Drag in screenshots, photos, diagrams.

All four share the same conversation storage, Ollama layer, persona system, and UI framework.

## Custom personas

Personas are reusable system prompts stored in SQLite. They stack with
mode-specific prompts: persona defines "who you are," mode prompt defines
"what to do right now." Built-in personas are seeded on first run. Users
can create, edit, and delete custom personas.

## Conversation branching

Conversations can be branched from any message. A branch creates a new
conversation copying history up to that point. Users can retry any
response with a different model or persona. Branching is stored in
SQLite via parent_message_id and branch_from_message_id fields.

## Rules

- Keep it simple. Don't over-engineer.
- Don't add dependencies I haven't asked for.
- Don't refactor working code unless I ask you to.
- When something is new to me (SQLite, file structure patterns, cross-module
  imports, multimodal APIs, markdown rendering), add a short comment
  block explaining it.
- If you think my approach is wrong, say so and explain why — but don't
  silently change the plan.
- Test each feature after implementing it. Don't say "done" without
  running the code.
- When creating the frontend, keep it in a single index.html file.
  Use CSS variables for theming. No external CSS frameworks.
- CDN libraries allowed: marked.js, highlight.js, DOMPurify. No others
  without asking.
- All model responses must be rendered as markdown with syntax highlighting.

## Tech conventions

- Python 3.13 (managed by uv)
- Type hints on all functions
- Ruff for formatting and linting
- f-strings, not .format()
- async functions for all endpoints and Ollama calls
- Use uvicorn: `uv run python -m uvicorn main:app --reload`

## Build sequence

This project follows a twelve-step build plan (see ai-assistant-project.md).
Each step has a specific scope. Don't build ahead — only implement what
the current step asks for.
