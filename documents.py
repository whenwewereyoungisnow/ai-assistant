# documents.py — Document processing, storage, and search (RAG pipeline)
#
# This module handles the "Documents" mode of the assistant: upload a PDF,
# extract its text, split it into searchable chunks, generate embedding
# vectors, and search those chunks by meaning (semantic) or keywords (BM25).
#
# Why store everything in memory instead of a database?
# For a personal assistant with dozens of documents, in-memory storage is:
#   1. Fast — cosine similarity on 10k chunks takes < 1ms with numpy
#   2. Simple — no extra infrastructure to install (no Pinecone, ChromaDB, pgvector)
#   3. Good enough — a single user won't have thousands of documents
#
# When would you switch to a vector database?
# When you have thousands of documents, need persistence across server restarts,
# want filtered queries (e.g. "search only in documents from 2024"), or need
# multi-user support. For now, re-uploading after restart is fine — the tradeoff
# is simplicity over durability.

import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

import models

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
# Document processing
# ---------------------------------------------------------------------------


async def process_pdf(file_path: Path, filename: str) -> dict[str, Any]:
    """Extract text from a PDF, chunk it, embed the chunks, and store them.

    This is the main entry point for document upload. It:
    1. Opens the PDF with PyMuPDF and extracts text page by page
    2. Cleans artifacts from the extracted text
    3. Splits into overlapping chunks (~500 chars each)
    4. Generates embedding vectors for each chunk via Ollama
    5. Stores everything in memory for search

    Args:
        file_path: Path to the uploaded PDF file on disk
        filename: Original filename (used as the document identifier)

    Returns:
        Summary dict: {"filename": str, "pages": int, "chunks": int}
    """
    import pymupdf

    # Extract text from each page. PyMuPDF is fast and handles most PDF
    # layouts well. Each page's text is tracked separately so we can
    # record which page each chunk came from.
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
    page_count = len(pages)

    if not full_text_parts:
        raise ValueError(f"No text could be extracted from {filename}")

    # Combine all pages and chunk the full text
    full_text = "\n\n".join(full_text_parts)
    chunk_texts = _chunk_text(full_text)

    # Figure out which page each chunk belongs to by checking which page's
    # text contains the start of the chunk. This is a heuristic — chunks
    # that span page boundaries get assigned to the page where they start.
    def _find_page(chunk_text: str) -> int:
        # Sample from the middle of the chunk to avoid the overlap region
        # at the start, which may come from a different page.
        sample_start = min(60, len(chunk_text) // 3)
        sample = chunk_text[sample_start : sample_start + 80]
        for page_info in pages:
            if sample in page_info["text"]:
                return page_info["page"]
        # Fallback: try the original start
        for page_info in pages:
            if chunk_text[:80] in page_info["text"]:
                return page_info["page"]
        return 1

    # Generate embeddings for all chunks.
    # How embedding works:
    # An embedding model converts text into a dense vector of numbers
    # (e.g., 1024 floats). Texts with similar meaning end up close together
    # in this vector space — "happy" and "joyful" would have similar vectors,
    # while "happy" and "database" would be far apart. This lets us find
    # relevant chunks by comparing vectors instead of matching exact words.
    #
    # We batch chunks (20 at a time) to avoid sending huge payloads to Ollama.
    # Each API call embeds multiple texts in one pass, which is much faster
    # than embedding one at a time.
    batch_size = 20
    all_embeddings: list[list[float]] = []
    try:
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i : i + batch_size]
            embeddings = await models.embed(batch)
            all_embeddings.extend(embeddings)
    except Exception as e:
        raise ValueError(f"Failed to generate embeddings for {filename}: {e}") from e

    # Build chunk dicts with text, embedding, and metadata
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

    # Store and rebuild search index. If a document with the same filename
    # was already uploaded, this replaces it — we track that so the caller
    # knows whether it was a fresh upload or a replacement.
    was_replaced = filename in _documents
    _documents[filename] = {
        "filename": filename,
        "pages": page_count,
        "chunks": chunks,
    }
    _rebuild_search_index()

    return {
        "filename": filename,
        "pages": page_count,
        "chunks": len(chunks),
        "replaced": was_replaced,
    }


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

    Args:
        query: The search text
        method: "semantic" or "keyword"
        top_k: Number of results to return

    Returns:
        List of dicts with text, score, filename, and page number,
        sorted by relevance (highest score first).
    """
    if not _all_chunks:
        return []

    if method == "semantic":
        # Embed the query and compare against all chunk embeddings
        query_embeddings = await models.embed(query)
        query_vec = np.array(query_embeddings[0])

        scores: list[tuple[float, int]] = []
        for i, chunk in enumerate(_all_chunks):
            chunk_vec = chunk["embedding"]
            score = _cosine_similarity(query_vec, chunk_vec)
            scores.append((score, i))

        # Sort by score descending, take top_k
        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:top_k]

    elif method == "keyword":
        if _bm25 is None:
            return []

        # BM25 scores the query against the tokenized corpus.
        # Higher score = more relevant (based on term frequency,
        # inverse document frequency, and document length normalization).
        tokenized_query = query.lower().split()
        bm25_scores = _bm25.get_scores(tokenized_query)
        # Pair scores with indices and sort
        indexed_scores = [(float(score), i) for i, score in enumerate(bm25_scores)]
        indexed_scores.sort(key=lambda x: x[0], reverse=True)
        top = indexed_scores[:top_k]

    else:
        raise ValueError(
            f"Unknown search method: {method}. Use 'semantic' or 'keyword'."
        )

    # Build result dicts (without the raw embedding — too large to return)
    results: list[dict[str, Any]] = []
    for score, idx in top:
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
    """
    if filename not in _documents:
        return False

    del _documents[filename]
    _rebuild_search_index()
    return True
