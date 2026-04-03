# logging_middleware.py — Request logging via Starlette middleware
#
# Why middleware instead of per-endpoint logging?
# Middleware intercepts every request in one place. Without it, you'd need
# to add timing + logging code to every single endpoint. Middleware also
# catches errors — if an endpoint throws an unhandled exception, the
# middleware still logs the failed request with its status code.
#
# What gets logged:
# Every API call (except static files and health checks) is recorded with
# its endpoint path, HTTP method, status code, and response time in
# milliseconds. This creates an audit trail for debugging ("why was that
# response slow?") and usage analytics ("which features are used most?").

import asyncio
import time
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from database import connect

# Endpoints to skip logging — these are high-frequency or non-interesting.
# /status is polled every 12 seconds and would flood the log table.
_SKIP_PATHS = {"/", "/health", "/status", "/favicon.ico"}


def init_logs_table() -> None:
    """Create the request_logs table if it doesn't exist.

    Uses INTEGER PRIMARY KEY for the id column, which SQLite auto-increments
    without needing AUTOINCREMENT. The AUTOINCREMENT keyword adds overhead
    (it maintains a separate sequence table) and is only needed if you must
    guarantee IDs are never reused after deletion — we don't need that.
    """
    conn = connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS request_logs (
                id          INTEGER PRIMARY KEY,
                endpoint    TEXT NOT NULL,
                method      TEXT NOT NULL,
                status_code INTEGER,
                duration_ms INTEGER,
                timestamp   TIMESTAMP NOT NULL
            )
        """)
        # Index on timestamp for efficient "recent requests" queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_logs_timestamp
            ON request_logs(timestamp)
        """)
        conn.commit()
    finally:
        conn.close()


def _log_request(
    endpoint: str, method: str, status_code: int, duration_ms: int
) -> None:
    """Insert a log entry into the database. Runs synchronously.

    This is called from the middleware's finally block. Since middleware
    runs in the async event loop, we keep the INSERT fast (single row,
    indexed table). For a personal app doing ~1 request/second, this
    adds negligible overhead.
    """
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO request_logs (endpoint, method, status_code, duration_ms, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                endpoint,
                method,
                status_code,
                duration_ms,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that logs request timing and status to SQLite.

    How Starlette middleware works:
    For each incoming HTTP request, Starlette calls dispatch() with the
    request and a call_next function. We wrap call_next in timing code
    to measure how long the endpoint took, then log the result. The
    response passes through unchanged — we're observing, not modifying.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip logging for static/frequent endpoints
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        start = time.monotonic()
        status_code = 500  # Default in case call_next throws

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_ms = round((time.monotonic() - start) * 1000)
            try:
                # Fire-and-forget: offload the synchronous SQLite write to a
                # thread pool so it doesn't block the async event loop. We
                # intentionally don't await the result — logging should never
                # slow down the response.
                asyncio.get_running_loop().run_in_executor(
                    None,
                    _log_request,
                    request.url.path,
                    request.method,
                    status_code,
                    duration_ms,
                )
            except Exception:
                # Logging should never break the actual request.
                pass
