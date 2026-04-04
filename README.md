# AI Assistant

A personal AI assistant running entirely on local Ollama models. Combines smart model routing, document chat (RAG), multi-agent writing, image understanding, and custom personas in one web app.

## Features

- **Chat** — Smart-routed conversation. A classifier picks the best model for each question.
- **Documents** — RAG over uploaded files (PDF, TXT, MD, DOCX, CSV) with hybrid BM25 + semantic search.
- **Writing** — Multi-agent pipeline. Writer drafts, Editor critiques, iterate.
- **Vision** — Image understanding via drag-and-drop.
- **Personas** — Reusable system prompts that stack with mode-specific behavior.
- **Conversation branching** — Branch from any message to explore different directions.

## Stack

- **Backend:** Python 3.13, FastAPI, SSE streaming
- **Frontend:** Plain HTML + vanilla JS (single file)
- **Models:** Ollama (local) — qwen3.5, deepseek-r1, llama3.2, qwen3-embedding
- **Database:** SQLite
- **Package manager:** uv

## Setup

```bash
# Install dependencies
uv sync

# Make sure Ollama is running with the required models
ollama pull qwen3.5:35b-a3b-coding-nvfp4
ollama pull qwen3.5:27b
ollama pull deepseek-r1:32b
ollama pull llama3.2:3b
ollama pull qwen3-embedding:4b
ollama pull llama3.2-vision

# Start the server
uv run python -m uvicorn main:app --reload
```

Then open http://localhost:8000 in your browser.
