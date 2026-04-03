# documents.py — Document processing, storage, and search (RAG pipeline)
#
# This module handles the "Documents" mode of the assistant: upload documents
# (PDF, TXT, MD, DOCX, CSV), extract text, split into searchable chunks,
# generate embedding vectors, and search those chunks by meaning or keywords.
#
# Each file type has its own processor function, but they all share the same
# pipeline: extract text -> clean -> chunk -> embed -> store in memory + SQLite.

import csv
import io
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

import database
import models

# Supported file types for upload. Auto-detected from extension.
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv"}

# ---------------------------------------------------------------------------
# In-memory document store
# ---------------------------------------------------------------------------
# _documents stores metadata per file: {"filename": {..., "chunks": [...]}}
# _all_chunks is a flat list of every chunk across all documents, used for search.
# _bm25 is the keyword search index, rebuilt whenever documents change.
#
# These are module-level globals. In a multi-user app you'd use a proper database,
# but for a single-user local app, module globals work fine and are much simpler.

_documents: dict[str, dict[str, Any]] = {}
_all_chunks: list[dict[str, Any]] = []
_bm25: BM25Okapi | None = None


def _rebuild_search_index() -> None:
    """Rebuild the flat chunk list and BM25 index from all stored documents.

    Called after any document is added or removed. BM25 needs all documents
    at construction time (it computes IDF — inverse document frequency —
    across the full corpus), so we rebuild the whole index rather than
    trying to update it incrementally.
    """
    global _all_chunks, _bm25

    _all_chunks = []
    for doc in _documents.values():
        _all_chunks.extend(doc["chunks"])

    if _all_chunks:
        # BM25 expects a list of tokenized documents (list of word lists).
        # Simple whitespace tokenization works well enough for search —
        # the BM25 algorithm handles term frequency and document length
        # normalization automatically.
        tokenized = [chunk["text"].lower().split() for chunk in _all_chunks]
        _bm25 = BM25Okapi(tokenized)
    else:
        _bm25 = None


def load_cached_documents() -> None:
    """Restore documents from SQLite on startup.

    This is the key to surviving restarts without re-uploading. On startup
    we read all chunks and embeddings from the database, group them by
    filename, and rebuild the in-memory search index. No Ollama needed —
    the embeddings are already computed and stored as BLOBs.

    Call this once during app startup, after database.init_db().
    """
    cached_chunks = database.load_all_chunks()
    if not cached_chunks:
        return

    # Group chunks by filename to reconstruct _documents
    docs: dict[str, list[dict[str, Any]]] = {}
    for chunk in cached_chunks:
        filename = chunk["filename"]
        if filename not in docs:
            docs[filename] = []
        docs[filename].append(chunk)

    for filename, chunks in docs.items():
        # Estimate page count from the max page number in the chunks
        max_page = max(c["page"] for c in chunks)
        _documents[filename] = {
            "filename": filename,
            "pages": max_page,
            "chunks": chunks,
        }

    _rebuild_search_index()


# ---------------------------------------------------------------------------
# Text extraction and cleaning
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Clean raw text extracted from a PDF.

    PDF extraction often produces artifacts: excessive whitespace, page
    numbers, headers/footers repeated on every page, and ligature
    characters. This function normalizes the text for chunking.
    """
    # Replace common ligatures that PDF extractors sometimes produce
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")

    # Collapse runs of whitespace (tabs, multiple spaces) into single spaces,
    # but preserve paragraph breaks (double newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines into 2 (paragraph separator)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _chunk_text(
    text: str, max_chars: int = 500, min_chars: int = 100, overlap: int = 50
) -> list[str]:
    """Split text into overlapping chunks for embedding and search.

    Strategy:
    1. Split on paragraph breaks (double newline) — these are natural
       semantic boundaries in most documents.
    2. Merge small consecutive paragraphs into chunks up to max_chars.
       This avoids creating tiny chunks that lack enough context for
       good embeddings.
    3. If a single paragraph exceeds max_chars, split it on sentence
       boundaries.
    4. Add overlap between chunks so information at chunk boundaries
       isn't lost. If a sentence spans two chunks, the overlap ensures
       it appears in at least one chunk fully.

    Args:
        max_chars: Target maximum chunk size (soft limit — won't split mid-sentence)
        min_chars: Minimum chunk size. Chunks below this get merged with the next one.
        overlap: Characters of overlap between consecutive chunks
    """
    # Split on paragraph breaks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Merge short paragraphs and split long ones into candidate segments
    segments: list[str] = []
    for para in paragraphs:
        if len(para) > max_chars:
            # Split long paragraphs on sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                if sentence.strip():
                    segments.append(sentence.strip())
        else:
            segments.append(para)

    # Build chunks by merging segments up to max_chars
    chunks: list[str] = []
    current = ""

    for segment in segments:
        if current and len(current) + len(segment) + 1 > max_chars:
            chunks.append(current)
            # Start next chunk with overlap from the end of the current one.
            # This ensures information near chunk boundaries appears in
            # both the previous and next chunk.
            if overlap > 0 and len(current) > overlap:
                # Snap to the nearest word boundary so chunks don't start
                # with broken word fragments like "tion of the..."
                overlap_text = current[-overlap:]
                space_idx = overlap_text.find(" ")
                if space_idx != -1:
                    overlap_text = overlap_text[space_idx + 1 :]
                current = overlap_text + " " + segment
            else:
                current = segment
        else:
            current = (current + " " + segment).strip() if current else segment

    # Don't forget the last accumulated chunk
    if current:
        # If the last chunk is too small, merge it with the previous one
        if len(current) < min_chars and chunks:
            chunks[-1] = chunks[-1] + " " + current
        else:
            chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# Markdown stripping (for .md files)
# ---------------------------------------------------------------------------


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting before chunking .md files.

    We strip formatting because the embedding model should match on the
    actual content ("install Python"), not the formatting syntax
    ("## Install Python"). Paragraph structure (double newlines) is
    preserved since _chunk_text() uses it as a split boundary.
    """
    # Remove code fences but keep the code content
    text = re.sub(r"```[\w]*\n?", "", text)
    # Remove images: ![alt](url)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # Convert links to just text: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove heading markers (# ## ### etc.) but keep the text
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Remove blockquote markers
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    return text


# ---------------------------------------------------------------------------
# Shared embedding + storage pipeline
# ---------------------------------------------------------------------------


async def _embed_and_store(
    filename: str,
    full_text: str,
    page_count: int,
    page_texts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Chunk text, generate embeddings, and store in memory + SQLite.

    This is the shared pipeline that all file-type processors call after
    extracting their text. It handles:
    1. Splitting text into overlapping chunks
    2. Assigning each chunk to a page (if page_texts provided)
    3. Batch-embedding all chunks via Ollama
    4. Storing in the in-memory index and persisting to SQLite

    Args:
        filename: Document identifier (original filename)
        full_text: The extracted, cleaned text content
        page_count: Number of pages (1 for non-paginated formats)
        page_texts: Optional per-page text for accurate page assignment.
                    Only PDFs provide this; other formats set page=1.

    Returns:
        Summary dict: {"filename", "pages", "chunks", "replaced"}
    """
    chunk_texts = _chunk_text(full_text)

    if not chunk_texts:
        raise ValueError(f"No text could be extracted from {filename}")

    # Page assignment: for PDFs we match chunks to pages using text overlap.
    # For other formats (txt, md, docx, csv) everything maps to page 1.
    def _find_page(chunk_text: str) -> int:
        if not page_texts:
            return 1
        sample_start = min(60, len(chunk_text) // 3)
        sample = chunk_text[sample_start : sample_start + 80]
        for page_info in page_texts:
            if sample in page_info["text"]:
                return page_info["page"]
        for page_info in page_texts:
            if chunk_text[:80] in page_info["text"]:
                return page_info["page"]
        return 1

    # Generate embeddings in batches of 20
    batch_size = 20
    all_embeddings: list[list[float]] = []
    try:
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i : i + batch_size]
            embeddings = await models.embed(batch)
            all_embeddings.extend(embeddings)
    except Exception as e:
        raise ValueError(f"Failed to generate embeddings for {filename}: {e}") from e

    # Build chunk dicts
    chunks: list[dict[str, Any]] = []
    for i, (text, embedding) in enumerate(zip(chunk_texts, all_embeddings)):
        chunks.append(
            {
                "text": text,
                "embedding": np.array(embedding),
                "filename": filename,
                "page": _find_page(text),
                "chunk_index": i,
            }
        )

    # Store in memory and persist to SQLite
    was_replaced = filename in _documents
    _documents[filename] = {
        "filename": filename,
        "pages": page_count,
        "chunks": chunks,
    }
    _rebuild_search_index()
    database.save_chunks(filename, chunks)

    return {
        "filename": filename,
        "pages": page_count,
        "chunks": len(chunks),
        "replaced": was_replaced,
    }


# ---------------------------------------------------------------------------
# File-type processors
# ---------------------------------------------------------------------------


async def process_pdf(file_path: Path, filename: str) -> dict[str, Any]:
    """Extract text from a PDF, chunk it, embed, and store."""
    import pymupdf

    doc = pymupdf.open(file_path)
    pages: list[dict[str, Any]] = []
    full_text_parts: list[str] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        cleaned = _clean_text(text)
        if cleaned:
            pages.append({"page": page_num + 1, "text": cleaned})
            full_text_parts.append(cleaned)

    doc.close()

    if not full_text_parts:
        raise ValueError(f"No text could be extracted from {filename}")

    full_text = "\n\n".join(full_text_parts)
    return await _embed_and_store(filename, full_text, len(pages), page_texts=pages)


async def process_txt(file_path: Path, filename: str) -> dict[str, Any]:
    """Process a plain text file (.txt).

    Reads with utf-8 encoding, falling back to latin-1 if the file contains
    non-UTF-8 bytes (common in older documents). latin-1 never fails because
    every byte value 0-255 maps to a valid character.
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="latin-1")

    cleaned = _clean_text(text)
    if not cleaned:
        raise ValueError(f"No text found in {filename}")

    return await _embed_and_store(filename, cleaned, page_count=1)


async def process_md(file_path: Path, filename: str) -> dict[str, Any]:
    """Process a Markdown file (.md).

    Strips markdown formatting (headers, bold, links, code fences) before
    chunking so the embedding model matches on content, not syntax.
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="latin-1")

    stripped = _strip_markdown(text)
    cleaned = _clean_text(stripped)
    if not cleaned:
        raise ValueError(f"No text found in {filename}")

    return await _embed_and_store(filename, cleaned, page_count=1)


async def process_docx(file_path: Path, filename: str) -> dict[str, Any]:
    """Process a Word document (.docx).

    Uses python-docx to extract paragraph text. DOCX files don't have
    reliable page numbers without rendering (page breaks depend on fonts
    and margins), so we estimate page count from text length.
    """
    from docx import Document

    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    if not paragraphs:
        raise ValueError(f"No text could be extracted from {filename}")

    full_text = "\n\n".join(paragraphs)
    cleaned = _clean_text(full_text)
    # Rough page estimate: ~3000 chars per page
    page_count = max(1, len(cleaned) // 3000 + 1)

    return await _embed_and_store(filename, cleaned, page_count)


async def process_csv(file_path: Path, filename: str) -> dict[str, Any]:
    """Process a CSV file.

    How CSV chunking works:
    Unlike documents that flow as prose, CSVs are structured data with rows
    and columns. Simply joining all cells into a wall of text would lose the
    structure. Instead, we convert each row (or group of rows) into a
    readable sentence-like format: "Row 1: name=Alice, age=30, city=Berlin".

    Why we include column headers with each chunk:
    If a chunk just says "Alice, 30, Berlin", the embedding model has no idea
    what those values mean. Including headers as context ("name=Alice") lets
    the model understand that "Alice" is a name and "Berlin" is a city, so
    a search for "people in Berlin" can match correctly.
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="latin-1")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError(f"No columns found in {filename}")

    # Convert rows to readable text with column headers as context
    row_texts: list[str] = []
    for i, row in enumerate(reader, 1):
        parts = [
            f"{col}={val}"
            for col, val in row.items()
            if col is not None and val and val.strip()
        ]
        if parts:
            row_texts.append(f"Row {i}: {', '.join(parts)}")

    if not row_texts:
        raise ValueError(f"No data found in {filename}")

    # Group rows into chunks of ~10 rows each for better context
    chunk_size = 10
    chunks: list[str] = []
    header_line = f"Columns: {', '.join(reader.fieldnames)}"
    for i in range(0, len(row_texts), chunk_size):
        group = row_texts[i : i + chunk_size]
        chunks.append(header_line + "\n" + "\n".join(group))

    full_text = "\n\n".join(chunks)
    return await _embed_and_store(filename, full_text, page_count=1)


async def process_document(file_path: Path, filename: str) -> dict[str, Any]:
    """Auto-detect file type and process accordingly.

    This is the main entry point for document upload. It dispatches to the
    appropriate processor based on the file extension.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return await process_pdf(file_path, filename)
    elif ext == ".txt":
        return await process_txt(file_path, filename)
    elif ext == ".md":
        return await process_md(file_path, filename)
    elif ext == ".docx":
        return await process_docx(file_path, filename)
    elif ext == ".csv":
        return await process_csv(file_path, filename)
    else:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {supported}")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _cosine_similarity(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Compute cosine similarity between two vectors.

    Cosine similarity measures the angle between two vectors, ignoring their
    magnitude (length). The result ranges from -1 to 1:
      1.0 = identical direction (same meaning)
      0.0 = perpendicular (unrelated)
     -1.0 = opposite direction (opposite meaning)

    The formula is: dot(a, b) / (||a|| * ||b||)
    - dot(a, b) is the dot product: sum of element-wise multiplication
    - ||a|| is the magnitude: sqrt(sum of squares)

    This is the standard way to compare embedding vectors because it focuses
    on the direction (meaning) rather than the magnitude (which can vary
    based on text length).
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


async def search_chunks(
    query: str,
    method: str = "semantic",
    top_k: int = 5,
    embedding_model: str | None = None,
) -> list[dict[str, Any]]:
    """Search stored document chunks by meaning or keywords.

    Two search methods:

    "semantic" (default):
      Embeds the query into a vector and compares it against every chunk's
      embedding using cosine similarity. Finds chunks with similar *meaning*
      even if they use different words. For example, searching "automobile"
      would find chunks about "cars" and "vehicles."

    "keyword" (BM25):
      Classic information retrieval using term frequency and inverse document
      frequency. Finds chunks that contain the exact words in the query.
      Better for specific terms, names, or jargon that the embedding model
      might not understand well.

    "hybrid" (default recommended):
      Combines both methods using Reciprocal Rank Fusion (RRF). Each method
      produces a ranking; RRF scores each result as 1/(rank + k) and sums
      the scores across methods. This avoids the problem of BM25 and cosine
      similarity being on completely different scales — RRF works on ranks,
      not raw scores. Hybrid consistently outperforms either method alone.

    Args:
        query: The search text
        method: "semantic", "keyword", or "hybrid"
        top_k: Number of results to return
        embedding_model: Optional override for the embedding model name

    Returns:
        List of dicts with text, score, filename, and page number,
        sorted by relevance (highest score first).
    """
    if not _all_chunks:
        return []

    if method == "semantic":
        return await _semantic_search(query, top_k, embedding_model)

    elif method == "keyword":
        return _keyword_search(query, top_k)

    elif method == "hybrid":
        # Reciprocal Rank Fusion: combine rankings from both methods.
        # Fetch more candidates than needed (2x) so the fusion has
        # enough overlap to produce good combined results.
        semantic_results = await _semantic_search(query, top_k * 2, embedding_model)
        keyword_results = _keyword_search(query, top_k * 2)
        return _fuse_results(semantic_results, keyword_results, top_k)

    else:
        raise ValueError(
            f"Unknown search method: {method}. Use 'semantic', 'keyword', or 'hybrid'."
        )


async def _semantic_search(
    query: str, top_k: int, embedding_model: str | None = None
) -> list[dict[str, Any]]:
    """Run semantic (embedding-based) search over all chunks."""
    embed_kwargs: dict[str, Any] = {}
    if embedding_model:
        embed_kwargs["model"] = embedding_model

    query_embeddings = await models.embed(query, **embed_kwargs)
    query_vec = np.array(query_embeddings[0])

    scores: list[tuple[float, int]] = []
    for i, chunk in enumerate(_all_chunks):
        chunk_vec = chunk["embedding"]
        score = _cosine_similarity(query_vec, chunk_vec)
        scores.append((score, i))

    scores.sort(key=lambda x: x[0], reverse=True)
    return _build_results(scores[:top_k])


def _keyword_search(query: str, top_k: int) -> list[dict[str, Any]]:
    """Run BM25 keyword search over all chunks."""
    if _bm25 is None:
        return []

    tokenized_query = query.lower().split()
    bm25_scores = _bm25.get_scores(tokenized_query)
    indexed_scores = [(float(score), i) for i, score in enumerate(bm25_scores)]
    indexed_scores.sort(key=lambda x: x[0], reverse=True)
    return _build_results(indexed_scores[:top_k])


def _build_results(scored_indices: list[tuple[float, int]]) -> list[dict[str, Any]]:
    """Convert (score, chunk_index) pairs into result dicts."""
    results: list[dict[str, Any]] = []
    for score, idx in scored_indices:
        chunk = _all_chunks[idx]
        results.append(
            {
                "text": chunk["text"],
                "score": round(score, 4),
                "filename": chunk["filename"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
            }
        )
    return results


def _fuse_results(
    semantic: list[dict[str, Any]],
    keyword: list[dict[str, Any]],
    top_k: int,
    k: int = 60,
) -> list[dict[str, Any]]:
    """Combine two ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF assigns each result a score of 1/(rank + k) where k is a constant
    (typically 60). Results that appear in both lists get their scores summed.
    This produces a combined ranking that's better than either individual
    method because semantic search catches meaning while keyword search
    catches exact terms.

    Why RRF instead of score averaging?
    BM25 scores range 0-30+ while cosine similarity ranges 0-1. You can't
    meaningfully average them without normalization, and normalization is
    fragile (depends on the score distribution). RRF sidesteps this entirely
    by working with ranks, not scores.
    """
    # Build a map of chunk_index -> fused score
    fused: dict[int, float] = {}
    chunk_data: dict[int, dict[str, Any]] = {}

    for rank, result in enumerate(semantic):
        ci = result["chunk_index"]
        fused[ci] = fused.get(ci, 0) + 1.0 / (rank + k)
        chunk_data[ci] = result

    for rank, result in enumerate(keyword):
        ci = result["chunk_index"]
        fused[ci] = fused.get(ci, 0) + 1.0 / (rank + k)
        chunk_data[ci] = result

    # Sort by fused score descending
    sorted_chunks = sorted(fused.items(), key=lambda x: x[1], reverse=True)

    results: list[dict[str, Any]] = []
    for ci, score in sorted_chunks[:top_k]:
        entry = dict(chunk_data[ci])
        entry["score"] = round(score, 4)
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------


def has_documents() -> bool:
    """Return True if any documents are loaded in memory."""
    return bool(_all_chunks)


def document_count() -> int:
    """Return the number of uploaded documents."""
    return len(_documents)


def chunk_count() -> int:
    """Return the total number of chunks across all documents."""
    return len(_all_chunks)


def list_documents() -> list[dict[str, Any]]:
    """Return metadata for all uploaded documents.

    Returns a list of dicts, each with filename, page count, and chunk count.
    Useful for the sidebar document list in the UI.
    """
    return [
        {
            "filename": doc["filename"],
            "pages": doc["pages"],
            "chunks": len(doc["chunks"]),
        }
        for doc in _documents.values()
    ]


def delete_document(filename: str) -> bool:
    """Remove a document and all its chunks. Returns True if it existed.

    After deletion, the BM25 index is rebuilt without the removed chunks.
    The embedding vectors are freed when the chunk dicts are garbage collected.
    Also removes from SQLite so the document doesn't reappear on restart.
    """
    if filename not in _documents:
        return False

    del _documents[filename]
    _rebuild_search_index()
    database.delete_chunks(filename)
    return True
