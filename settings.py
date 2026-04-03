# settings.py — Persistent settings system backed by SQLite
#
# Why a separate settings module?
# Settings need to be read on every request (which model to use, how many
# search results, etc.) but written rarely (only when the user changes them
# in the settings panel). An in-memory cache makes reads instant while
# SQLite ensures persistence across server restarts.
#
# How the cache works:
# On startup, init_settings() loads all rows from the `settings` table into
# a Python dict (_cache). get_setting() reads from the dict (zero I/O).
# set_setting() writes to SQLite AND updates the dict, so they're always
# in sync. For a single-process, single-user app, there's no cache
# invalidation problem — the only writer is this process.

import json
from typing import Any

from database import connect

# ---------------------------------------------------------------------------
# Default settings — these define what's available and their initial values.
# When init_settings() runs, any key not already in the database gets seeded
# with its default value. This means adding a new setting is just adding
# an entry here — existing databases get the default on next startup.
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    # Model assignments for each route / feature
    "general_model": "qwen3.5:35b-a3b-coding-nvfp4",
    "code_model": "qwen3.5:27b",
    "reasoning_model": "deepseek-r1:32b",
    "rag_model": "qwen3.5:35b-a3b-coding-nvfp4",
    "writer_model": "qwen3.5:35b-a3b-coding-nvfp4",
    "editor_model": "deepseek-r1:32b",
    # Vision
    "vision_model": "gemma4:31b",
    # Writing pipeline
    "max_writing_rounds": 3,
    # Document search
    "search_method": "hybrid",
    "search_results_count": 5,
    # UI — "system" syncs with the OS dark/light mode preference
    "theme": "system",
}

# In-memory cache. Loaded on startup, updated on every write.
_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def init_settings() -> None:
    """Create the settings table and seed defaults for any missing keys.

    Safe to call on every startup — existing settings are not overwritten.
    New settings (added to DEFAULTS in a code update) get their default
    values inserted automatically, so you never need a migration.
    """
    conn = connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Seed defaults for keys that don't exist yet.
        # INSERT OR IGNORE skips keys that already have a row.
        for key, default_value in DEFAULTS.items():
            conn.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, json.dumps(default_value)),
            )

        conn.commit()

        # Load all settings into the in-memory cache
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
        for row in rows:
            _cache[row["key"]] = json.loads(row["value"])

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read settings (from cache — no I/O)
# ---------------------------------------------------------------------------


def get_setting(key: str) -> Any:
    """Read a single setting. Returns the default if the key isn't cached.

    This is called on every request (to get model names, search config, etc.)
    so it must be fast. Reading from a Python dict is ~50ns — effectively free.
    """
    if key in _cache:
        return _cache[key]
    return DEFAULTS.get(key)


def get_all_settings() -> dict[str, Any]:
    """Return a copy of all settings. Used by GET /settings."""
    # Start with defaults so newly added keys appear even if not yet in DB
    result = dict(DEFAULTS)
    result.update(_cache)
    return result


# ---------------------------------------------------------------------------
# Write settings (to SQLite + cache)
# ---------------------------------------------------------------------------


def set_setting(key: str, value: Any) -> None:
    """Update a single setting. Writes to SQLite and updates the cache."""
    conn = connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        conn.commit()
    finally:
        conn.close()
    _cache[key] = value


def set_many(updates: dict[str, Any]) -> None:
    """Update multiple settings in one transaction.

    Used by PUT /settings when the user saves the settings panel.
    Batching into one transaction is both faster and atomic — if the
    server crashes mid-save, either all changes persist or none do.
    """
    conn = connect()
    try:
        for key, value in updates.items():
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
        conn.commit()
    finally:
        conn.close()
    _cache.update(updates)


# ---------------------------------------------------------------------------
# Route builder — constructs the ROUTES list from current settings
# ---------------------------------------------------------------------------


def get_routes() -> list[dict[str, Any]]:
    """Build the smart-routing ROUTES list from current settings.

    This replaces the hardcoded ROUTES constant in router.py. The route
    descriptions stay the same (they're used by the classifier to decide
    which route to pick), but the model names come from settings so the
    user can swap models without editing code.

    Why not just modify router.ROUTES directly?
    Because router.py defines the default structure (names, descriptions,
    options). Settings only override the model names. If a model setting
    is missing or invalid, the original defaults from router.py still work.
    """
    return [
        {
            "name": "general",
            "description": (
                "General knowledge, casual conversation, explanations, "
                "summaries, creative writing, and anything not specifically "
                "about code or complex reasoning"
            ),
            "model": get_setting("general_model"),
            "options": {"think": False},
        },
        {
            "name": "code",
            "description": (
                "Programming questions, code writing, debugging, code review, "
                "technical implementation, APIs, databases, and software "
                "engineering"
            ),
            "model": get_setting("code_model"),
            "options": {"think": False},
        },
        {
            "name": "reasoning",
            "description": (
                "Math problems, logic puzzles, multi-step analysis, complex "
                "comparisons, scientific reasoning, and questions requiring "
                "careful step-by-step thinking"
            ),
            "model": get_setting("reasoning_model"),
            "options": {"think": True},
        },
    ]
