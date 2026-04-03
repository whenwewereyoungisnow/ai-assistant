# database.py — SQLite conversation storage
#
# Why SQLite?
# SQLite is a file-based database that's built into Python's standard library.
# Unlike PostgreSQL or MySQL, there's no server to install or configure — it
# just reads and writes a single file (data/assistant.db). That makes it
# perfect for a local-first app like this: zero setup, zero dependencies,
# and the data is just a file you can back up by copying it. SQLite handles
# thousands of concurrent reads and plenty of writes for a single-user app.
#
# Why no ORM (like SQLAlchemy)?
# ORMs add complexity and hide what's actually happening. Since we only have
# two tables and simple queries, writing SQL directly is clearer and easier
# to debug. For a learning project, seeing the actual SQL helps you understand
# how databases work.

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np

# Type aliases for the allowed values. Literal means "only these exact strings
# are valid" — your editor and type checker will catch mistakes like
# add_message(..., role="bot") before you even run the code.
Mode = Literal["chat", "documents", "writing", "vision"]
Role = Literal["user", "assistant", "system"]

# Database file lives in data/ so it's separate from code.
# Path(__file__).parent resolves to the directory containing database.py,
# so this always points to the right place regardless of where you run
# the app from (e.g. `cd / && uv run python -m uvicorn main:app` still works).
DB_PATH = Path(__file__).parent / "data" / "assistant.db"


def connect() -> sqlite3.Connection:
    """Create a connection to the SQLite database.

    sqlite3.Row makes rows behave like dicts — you can access columns by name
    (row["title"]) instead of by index (row[0]). Much more readable.
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Enable foreign keys. SQLite has foreign key support but it's OFF by
    # default for backwards compatibility. We need it ON so that deleting a
    # conversation automatically deletes its messages (CASCADE).
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create the database tables if they don't exist.

    Call this once at app startup. If the tables already exist, CREATE TABLE
    IF NOT EXISTS is a no-op — safe to call every time.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = connect()
    try:
        # The conversations table tracks each chat session.
        # - id: A UUID string (e.g. "a1b2c3d4-..."). We use UUIDs instead of
        #   auto-incrementing integers because they're globally unique — useful
        #   if we ever sync or merge databases.
        # - mode: Which feature this conversation uses ("chat", "documents",
        #   "writing", "vision").
        # - created_at / updated_at: Timestamps for sorting and display.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT 'New conversation',
                mode        TEXT NOT NULL CHECK (mode IN ('chat', 'documents', 'writing', 'vision')),
                created_at  TIMESTAMP NOT NULL,
                updated_at  TIMESTAMP NOT NULL
            )
        """)

        # The messages table stores every message in every conversation.
        #
        # What's a foreign key?
        # The "REFERENCES conversations(id)" line creates a link between the two
        # tables. It means every message MUST belong to an existing conversation —
        # you can't insert a message with a fake conversation_id. The database
        # enforces this automatically, so you never end up with orphaned messages.
        #
        # ON DELETE CASCADE means: when you delete a conversation, SQLite
        # automatically deletes all its messages too. No manual cleanup needed.
        #
        # Why store metadata as JSON text?
        # Different message types carry different extra data: routing decisions
        # for chat messages, source chunks for RAG, timing info for writing.
        # Instead of adding a column for every possible field (which would mean
        # constant schema changes), we store a flexible JSON blob. SQLite even
        # has JSON functions to query inside it if needed later.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id                TEXT PRIMARY KEY,
                conversation_id   TEXT NOT NULL
                                  REFERENCES conversations(id) ON DELETE CASCADE,
                role              TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                content           TEXT NOT NULL,
                metadata          TEXT,
                created_at        TIMESTAMP NOT NULL
            )
        """)

        # Index on conversation_id so fetching all messages for a conversation
        # is fast. Without this, SQLite would scan every row in the table.
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id)
        """)

        # Document chunks table — persists PDF chunks and their embeddings
        # so they survive server restarts without re-uploading and re-embedding.
        #
        # How embedding caching reduces startup time with many documents:
        # Without caching, every restart means re-uploading PDFs and calling
        # the embedding model for every chunk (~0.5s per batch of 20). A 100-page
        # PDF with 500 chunks would take ~12 seconds just for embeddings. With
        # SQLite caching, those same chunks load in milliseconds from disk —
        # no Ollama needed at all during startup.
        #
        # Embeddings are stored as BLOBs (raw float64 bytes). A 1024-dimension
        # embedding is 8KB as a BLOB — compact and fast to read/write. We
        # reconstruct the numpy array with np.frombuffer() on load.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                page        INTEGER NOT NULL,
                text        TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                created_at  TIMESTAMP NOT NULL,
                UNIQUE(filename, chunk_index)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_filename
            ON document_chunks(filename)
        """)

        conn.commit()
    finally:
        conn.close()


def create_conversation(mode: Mode, title: str | None = None) -> str:
    """Create a new conversation and return its ID.

    Args:
        mode: One of "chat", "documents", "writing", "vision"
        title: Display title. Defaults to "New conversation" — the frontend
               can auto-update this from the first user message later.
    """
    conversation_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()

    conn = connect()
    try:
        conn.execute(
            "INSERT INTO conversations (id, title, mode, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, title or "New conversation", mode, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return conversation_id


def list_conversations() -> list[dict[str, Any]]:
    """Return all conversations, newest first, with message counts.

    The COUNT subquery avoids loading all messages just to count them.
    This is a common SQL pattern: use a correlated subquery to add
    computed columns without a separate query.
    """
    conn = connect()
    try:
        rows = conn.execute("""
            SELECT c.*,
                   (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
            FROM conversations c
            ORDER BY c.updated_at DESC
        """).fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    """Return a conversation with all its messages, or None if not found.

    Messages are ordered by created_at so they appear in chronological order.
    Metadata is parsed from JSON text back into a Python dict.
    """
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            return None

        conversation = dict(row)

        message_rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conversation_id,),
        ).fetchall()
    finally:
        conn.close()

    messages = []
    for msg in message_rows:
        msg_dict = dict(msg)
        # Parse the JSON metadata back into a Python dict so the API
        # returns structured data, not a raw JSON string.
        if msg_dict["metadata"] is not None:
            msg_dict["metadata"] = json.loads(msg_dict["metadata"])
        messages.append(msg_dict)

    conversation["messages"] = messages
    return conversation


def add_message(
    conversation_id: str,
    role: Role,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Add a message to a conversation and return the message ID.

    Also updates the conversation's updated_at timestamp so it floats to
    the top of the list — just like how chat apps show the most recently
    active conversation first.

    Args:
        conversation_id: Which conversation this message belongs to
        role: "user", "assistant", or "system"
        content: The message text
        metadata: Optional dict of extra info (routing decisions, sources, etc.)
                  Stored as a JSON string in the database.
    """
    message_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    metadata_json = json.dumps(metadata) if metadata is not None else None

    conn = connect()
    try:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, conversation_id, role, content, metadata_json, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        conn.commit()
    finally:
        conn.close()
    return message_id


def update_conversation_title(conversation_id: str, title: str) -> None:
    """Update a conversation's title.

    Typically called after the first user message to set a meaningful title
    instead of the default "New conversation".
    """
    conn = connect()
    try:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title, conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation and all its messages. Returns True if it existed.

    Thanks to ON DELETE CASCADE on the messages foreign key, deleting the
    conversation row automatically deletes all associated messages — no
    need for a separate DELETE FROM messages query.
    """
    conn = connect()
    try:
        cursor = conn.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def count_conversations() -> int:
    """Return total number of conversations. Faster than list_conversations()."""
    conn = connect()
    try:
        row = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
        return row[0]
    finally:
        conn.close()


def search_messages(query: str, limit: int = 50) -> list[dict[str, Any]]:
    """Search message content across all conversations.

    Uses SQLite's LIKE for simple substring matching — fast enough for a
    personal app with thousands of messages. Returns conversations that
    contain matching messages, with a content snippet from the first match.

    Why not full-text search (FTS5)?
    FTS5 would be faster for large datasets, but requires creating a
    virtual table and keeping it in sync. For a single-user app, LIKE
    is simpler and works fine up to ~100k messages.

    Args:
        query: Text to search for (case-insensitive substring match)
        limit: Maximum number of results to return

    Returns:
        List of dicts with conversation info + matching snippet.
    """
    # Escape LIKE wildcards so % and _ in the query are treated as literal
    # characters. Without this, searching for "100%" would match "1000" etc.
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    conn = connect()
    try:
        # Search messages and group by conversation so each conversation
        # appears at most once. We grab the first matching message as a
        # snippet so the user can see why the conversation matched.
        rows = conn.execute(
            """
            SELECT c.id, c.title, c.mode, c.updated_at,
                   m.content AS snippet, m.role
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE m.content LIKE ? ESCAPE '\'
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (f"%{escaped}%", limit),
        ).fetchall()
    finally:
        conn.close()

    results: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(row)
        # Trim the snippet to ~150 chars around the match for display
        content = row_dict["snippet"]
        lower_content = content.lower()
        idx = lower_content.find(query.lower())
        if idx != -1:
            start = max(0, idx - 60)
            end = min(len(content), idx + len(query) + 60)
            snippet = (
                ("..." if start > 0 else "")
                + content[start:end]
                + ("..." if end < len(content) else "")
            )
        else:
            snippet = content[:150] + ("..." if len(content) > 150 else "")
        row_dict["snippet"] = snippet
        results.append(row_dict)

    return results


# ---------------------------------------------------------------------------
# Document chunk persistence
# ---------------------------------------------------------------------------


def save_chunks(filename: str, chunks: list[dict[str, Any]]) -> None:
    """Persist document chunks and embeddings to SQLite.

    Replaces any existing chunks for the same filename (handles re-uploads).
    Embeddings are stored as raw float64 bytes in a BLOB column.

    Args:
        filename: The document filename (used as the grouping key)
        chunks: List of chunk dicts, each with "text", "embedding" (numpy array),
                "page", and "chunk_index" keys.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = connect()
    try:
        # Delete old chunks for this file (handles re-uploads)
        conn.execute("DELETE FROM document_chunks WHERE filename = ?", (filename,))

        for chunk in chunks:
            embedding_blob = chunk["embedding"].astype(np.float64).tobytes()
            conn.execute(
                "INSERT INTO document_chunks "
                "(filename, chunk_index, page, text, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    filename,
                    chunk["chunk_index"],
                    chunk["page"],
                    chunk["text"],
                    embedding_blob,
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def load_all_chunks() -> list[dict[str, Any]]:
    """Load all cached document chunks from SQLite.

    Returns chunk dicts with numpy array embeddings reconstructed from BLOBs.
    Called at startup to restore documents without needing Ollama.
    """
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT filename, chunk_index, page, text, embedding "
            "FROM document_chunks ORDER BY filename, chunk_index"
        ).fetchall()
    finally:
        conn.close()

    chunks: list[dict[str, Any]] = []
    for row in rows:
        embedding = np.frombuffer(row["embedding"], dtype=np.float64)
        chunks.append(
            {
                "text": row["text"],
                "embedding": embedding,
                "filename": row["filename"],
                "page": row["page"],
                "chunk_index": row["chunk_index"],
            }
        )
    return chunks


def delete_chunks(filename: str) -> None:
    """Delete all cached chunks for a document."""
    conn = connect()
    try:
        conn.execute("DELETE FROM document_chunks WHERE filename = ?", (filename,))
        conn.commit()
    finally:
        conn.close()


def get_cached_filenames() -> list[str]:
    """Return distinct filenames that have cached chunks."""
    conn = connect()
    try:
        rows = conn.execute("SELECT DISTINCT filename FROM document_chunks").fetchall()
    finally:
        conn.close()
    return [row["filename"] for row in rows]


def update_message_content(
    message_id: str, content: str, metadata: dict[str, Any] | None = None
) -> None:
    """Update the content and metadata of an existing message.

    Used during streaming to periodically save partial responses so they
    survive connection drops. The message row must already exist (created
    as a placeholder before streaming starts).
    """
    metadata_json = json.dumps(metadata) if metadata is not None else None
    conn = connect()
    try:
        conn.execute(
            "UPDATE messages SET content = ?, metadata = ? WHERE id = ?",
            (content, metadata_json, message_id),
        )
        conn.commit()
    finally:
        conn.close()
