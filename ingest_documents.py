"""Download, parse, chunk, embed and index source documents for MindSpace.

Run once after installing requirements-rag-embeddings.txt:
    python ingest_documents.py

Supported local files: PDF, HTML/HTM, TXT and Markdown. Seed documents are
downloaded from the official source URLs in knowledge/sources.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
DOCUMENT_DIR = KNOWLEDGE_DIR / "documents"
SOURCES_FILE = KNOWLEDGE_DIR / "sources.json"
VECTOR_STORE_DIR = KNOWLEDGE_DIR / "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".md"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def read_source_manifest() -> list[dict]:
    if not SOURCES_FILE.is_file():
        raise FileNotFoundError(f"Source manifest not found: {SOURCES_FILE}")
    payload = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("knowledge/sources.json must contain a 'sources' list.")
    return sources


def download_seed_documents(force: bool = False) -> None:
    """Download original seed documents into knowledge/documents/."""
    import requests

    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    for source in read_source_manifest():
        filename = source["file_name"]
        download_url = source["download_url"]
        expected_format = source["format"].lower()
        target = DOCUMENT_DIR / filename

        if target.exists() and not force:
            LOGGER.info("Using existing document: %s", filename)
            continue

        try:
            response = requests.get(
                download_url,
                headers={"User-Agent": "MindSpaceStudentRAG/1.0 (document ingestion)"},
                timeout=(15, 120),
            )
            response.raise_for_status()
            content = response.content

            if expected_format == "pdf" and not content.startswith(b"%PDF-"):
                raise ValueError("The downloaded response is not a PDF file.")
            if expected_format == "html":
                content_type = response.headers.get("content-type", "").lower()
                looks_like_html = (
                    b"<html" in content[:5000].lower()
                    or b"<!doctype" in content[:5000].lower()
                )
                if "html" not in content_type and not looks_like_html:
                    raise ValueError("The downloaded response does not look like HTML.")

            target.write_bytes(content)
            LOGGER.info("Downloaded %s (%s bytes)", filename, len(content))
        except Exception as exc:
            LOGGER.warning("Could not download %s: %s", filename, exc)


def _metadata_for_file(path: Path, source_lookup: dict) -> dict:
    source = source_lookup.get(path.name, {})
    return {
        "title": source.get("title") or path.stem.replace("_", " ").replace("-", " "),
        "url": source.get("source_url") or "",
        "publisher": source.get("publisher") or "User-provided document",
        "source_file": path.name,
        "file_type": path.suffix.lower().lstrip("."),
        "license": source.get("license") or "User-provided; verify reuse rights",
    }


def load_documents():
    """Parse local source files into LangChain Document objects with metadata."""
    from langchain_core.documents import Document

    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    source_lookup = {item["file_name"]: item for item in read_source_manifest()}
    documents = []

    for path in sorted(DOCUMENT_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        metadata = _metadata_for_file(path, source_lookup)
        extension = path.suffix.lower()

        if extension == ".pdf":
            import fitz

            with fitz.open(path) as pdf:
                pdf_metadata = pdf.metadata or {}
                for page_number, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()
                    if not text:
                        continue
                    page_metadata = {
                        **metadata,
                        "page": page_number,
                        "total_pages": len(pdf),
                    }
                    if pdf_metadata.get("author"):
                        page_metadata["author"] = str(pdf_metadata["author"])
                    documents.append(Document(
                        page_content=text,
                        metadata=page_metadata,
                    ))

        elif extension in {".html", ".htm"}:
            from bs4 import BeautifulSoup

            html = path.read_text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
            for tag in soup.find_all([
                "script", "style", "noscript", "nav", "footer",
                "header", "aside", "form", "button",
            ]):
                tag.decompose()
            content_root = soup.find("main") or soup.find("article") or soup.body or soup
            text = content_root.get_text("\n", strip=True)
            if len(text) < 100:
                LOGGER.warning("Skipping HTML with too little extractable text: %s", path.name)
                continue
            html_metadata = dict(metadata)
            if page_title and not source_lookup.get(path.name, {}).get("title"):
                html_metadata["title"] = page_title
            documents.append(Document(page_content=text, metadata=html_metadata))

        else:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                documents.append(Document(page_content=text, metadata=metadata))

    if not documents:
        raise RuntimeError(
            "No readable source documents were found. Run without --local-only "
            "to download the seed sources, or add PDF/HTML/TXT/MD files to "
            "knowledge/documents/."
        )
    LOGGER.info("Parsed %d document/page record(s).", len(documents))
    return documents


def split_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split loaded pages/documents while retaining their source metadata."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for chunk_number, chunk in enumerate(chunks, start=1):
        page = chunk.metadata.get("page", "doc")
        start = chunk.metadata.get("start_index", 0)
        chunk.metadata["chunk_index"] = chunk_number
        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata.get('source_file', 'document')}:{page}:{start}"
        )
    LOGGER.info(
        "Split %d document/page record(s) into %d chunk(s) "
        "(chunk_size=%d, overlap=%d).",
        len(documents), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def build_vector_store(chunks: list) -> int:
    """Embed chunks with MiniLM and persist a FAISS inner-product index.

    Embeddings are normalized before storage, so inner product is cosine
    similarity. Chunk text and source metadata are stored separately as JSON;
    no pickle-based document store is used.
    """
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if not chunks:
        raise ValueError("Cannot build a vector store without chunks.")

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("The embedding model returned an unexpected vector shape.")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    index_tmp = VECTOR_STORE_DIR / "index.faiss.tmp"
    chunks_tmp = VECTOR_STORE_DIR / "chunks.json.tmp"
    manifest_tmp = VECTOR_STORE_DIR / "manifest.json.tmp"

    faiss.write_index(index, str(index_tmp))
    chunk_records = [
        {"page_content": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]
    chunks_tmp.write_text(
        json.dumps(chunk_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "embedding_model": EMBEDDING_MODEL,
        "vector_store": "FAISS IndexFlatIP with normalized MiniLM embeddings",
        "similarity": "cosine similarity (inner product over normalized vectors)",
        "chunk_count": len(chunks),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(index_tmp, VECTOR_STORE_DIR / "index.faiss")
    os.replace(chunks_tmp, VECTOR_STORE_DIR / "chunks.json")
    os.replace(manifest_tmp, VECTOR_STORE_DIR / "manifest.json")
    return len(chunks)


def build_pipeline(
    local_only: bool = False,
    force_download: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> int:
    if not local_only:
        download_seed_documents(force=force_download)
    documents = load_documents()
    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    count = build_vector_store(chunks)
    LOGGER.info(
        "RAG ingestion completed: %d chunk(s) indexed in %s",
        count, VECTOR_STORE_DIR,
    )
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download, parse, chunk and index MindSpace source documents."
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Skip downloads and ingest files already present in knowledge/documents/.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download seed documents even when local copies already exist.",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    build_pipeline(
        local_only=args.local_only,
        force_download=args.force_download,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
