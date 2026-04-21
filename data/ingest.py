"""
ingest.py — Document Ingestion Pipeline
========================================
Đọc tài liệu từ data/docs/, chunk theo section (===),
embed bằng OpenAI text-embedding-3-small, lưu vào ChromaDB.

Tái sử dụng từ Day08/lab/index.py với điều chỉnh paths và collection.

Usage:
    python data/ingest.py
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

DOCS_DIR = Path(__file__).parent / "docs"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "eval_docs"

# Chunk size và overlap cho tài liệu tiếng Việt
CHUNK_SIZE = 400       # tokens (ước lượng bằng số ký tự / 4)
CHUNK_OVERLAP = 80     # tokens overlap giữa các chunk


# =============================================================================
# STEP 1: PREPROCESS
# Làm sạch text trước khi chunk và embed
# =============================================================================

def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    """
    Preprocess một tài liệu: extract metadata từ header và làm sạch nội dung.

    Args:
        raw_text: Toàn bộ nội dung file text
        filepath: Đường dẫn file để làm source mặc định

    Returns:
        Dict chứa:
          - "text": nội dung đã clean
          - "metadata": dict với source, department, effective_date, access
    """
    lines = raw_text.strip().split("\n")
    metadata = {
        "source": filepath,
        "section": "",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }
    content_lines = []
    header_done = False

    # Map of known header keys to metadata field names
    header_map = {
        "Source": "source",
        "Department": "department",
        "Effective Date": "effective_date",
        "Access": "access",
    }

    for line in lines:
        if not header_done:
            matched = False
            for header_key, meta_key in header_map.items():
                if line.startswith(header_key + ":"):
                    metadata[meta_key] = line.split(":", 1)[1].strip()
                    matched = True
                    break

            if matched:
                continue
            elif line.startswith("==="):
                header_done = True
                content_lines.append(line)
            elif line.strip() == "" or line.isupper():
                continue
        else:
            content_lines.append(line)

    cleaned_text = "\n".join(content_lines)

    # Normalize text
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = cleaned_text.strip()

    return {
        "text": cleaned_text,
        "metadata": metadata,
    }


# =============================================================================
# STEP 2: CHUNK
# Chia tài liệu thành các đoạn nhỏ theo cấu trúc tự nhiên (===)
# =============================================================================

def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk một tài liệu đã preprocess thành danh sách các chunk nhỏ.

    Args:
        doc: Dict với "text" và "metadata" (output của preprocess_document)

    Returns:
        List các Dict, mỗi dict là một chunk với:
          - "text": nội dung chunk
          - "metadata": metadata gốc + "section" của chunk đó
    """
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks = []

    # Split theo heading pattern "=== ... ==="
    sections = re.split(r"(===.*?===)", text)

    current_section = "General"
    current_section_text = ""

    for part in sections:
        if re.match(r"===.*?===", part):
            if current_section_text.strip():
                section_chunks = _split_by_size(
                    current_section_text.strip(),
                    base_metadata=base_metadata,
                    section=current_section,
                )
                chunks.extend(section_chunks)
            current_section = part.strip("= ").strip()
            current_section_text = ""
        else:
            current_section_text += part

    # Lưu section cuối cùng
    if current_section_text.strip():
        section_chunks = _split_by_size(
            current_section_text.strip(),
            base_metadata=base_metadata,
            section=current_section,
        )
        chunks.extend(section_chunks)

    return chunks


def _split_by_size(
    text: str,
    base_metadata: Dict,
    section: str,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    """
    Helper: Split text dài thành chunks với overlap.
    Split theo paragraph (\\n\\n) trước, rồi ghép đến khi đủ size.
    """
    if len(text) <= chunk_chars:
        return [{
            "text": text,
            "metadata": {**base_metadata, "section": section},
        }]

    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk_parts: List[str] = []
    current_chunk_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)

        if current_chunk_len + para_len > chunk_chars and current_chunk_parts:
            chunk_text = "\n\n".join(current_chunk_parts)
            chunks.append({
                "text": chunk_text,
                "metadata": {**base_metadata, "section": section},
            })

            # Tạo overlap
            overlap_parts: List[str] = []
            overlap_len = 0
            for p in reversed(current_chunk_parts):
                if overlap_len + len(p) <= overlap_chars:
                    overlap_parts.insert(0, p)
                    overlap_len += len(p)
                else:
                    break

            current_chunk_parts = overlap_parts + [para]
            current_chunk_len = sum(len(p) for p in current_chunk_parts)
        else:
            current_chunk_parts.append(para)
            current_chunk_len += para_len

    if current_chunk_parts:
        chunk_text = "\n\n".join(current_chunk_parts)
        chunks.append({
            "text": chunk_text,
            "metadata": {**base_metadata, "section": section},
        })

    return chunks


# =============================================================================
# STEP 3: EMBED + STORE
# Embed các chunk và lưu vào ChromaDB
# =============================================================================

def get_embedding(text: str) -> List[float]:
    """
    Tạo embedding vector cho một đoạn text.
    Sử dụng OpenAI text-embedding-3-small.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return response.data[0].embedding


def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Pipeline hoàn chỉnh: đọc docs → preprocess → chunk → embed → store vào ChromaDB.
    """
    import chromadb

    print(f"📂 Đang build index từ: {docs_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    doc_files = list(docs_dir.glob("*.txt"))

    if not doc_files:
        print(f"❌ Không tìm thấy file .txt trong {docs_dir}")
        return

    for filepath in doc_files:
        print(f"  Processing: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")

        doc = preprocess_document(raw_text, str(filepath))
        chunks = chunk_document(doc)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filepath.stem}_{i}"
            embedding = get_embedding(chunk["text"])
            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[chunk["metadata"]],
            )

        total_chunks += len(chunks)
        print(f"    → {len(chunks)} chunks indexed")

    print(f"\n✅ Hoàn thành! Tổng số chunks: {total_chunks}")


# =============================================================================
# STEP 4: INSPECT / KIỂM TRA
# =============================================================================

def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 5) -> None:
    """
    In ra n chunk đầu tiên trong ChromaDB để kiểm tra chất lượng index.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection(COLLECTION_NAME)
        results = collection.get(limit=n, include=["documents", "metadatas"])

        print(f"\n=== Top {n} chunks trong index ===\n")
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            print(f"[Chunk {i+1}]")
            print(f"  Source: {meta.get('source', 'N/A')}")
            print(f"  Section: {meta.get('section', 'N/A')}")
            print(f"  Department: {meta.get('department', 'N/A')}")
            print(f"  Text preview: {doc[:200]}...")
            print()
    except Exception as e:
        print(f"Lỗi khi đọc index: {e}")
        print("Hãy chạy build_index() trước.")


def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Kiểm tra phân phối metadata trong toàn bộ index.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection(COLLECTION_NAME)
        results = collection.get(include=["metadatas"])

        total = len(results["metadatas"])
        print(f"\n📊 Tổng chunks: {total}")

        departments: Dict[str, int] = {}
        sources: Dict[str, int] = {}

        for meta in results["metadatas"]:
            dept = meta.get("department", "unknown")
            departments[dept] = departments.get(dept, 0) + 1

            src = Path(meta.get("source", "unknown")).name
            sources[src] = sources.get(src, 0) + 1

        print("\nPhân bố theo department:")
        for dept, count in sorted(departments.items()):
            print(f"  {dept}: {count} chunks")

        print("\nPhân bố theo source:")
        for src, count in sorted(sources.items()):
            print(f"  {src}: {count} chunks")

    except Exception as e:
        print(f"Lỗi: {e}. Hãy chạy build_index() trước.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 14: Build Eval Index")
    print("=" * 60)

    # Bước 1: Kiểm tra docs
    doc_files = list(DOCS_DIR.glob("*.txt"))
    print(f"\nTìm thấy {len(doc_files)} tài liệu:")
    for f in doc_files:
        print(f"  - {f.name}")

    # Bước 2: Test preprocess + chunking (không cần API key)
    print("\n--- Test preprocess + chunking ---")
    for filepath in doc_files[:1]:
        raw = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw, str(filepath))
        chunks = chunk_document(doc)
        print(f"\nFile: {filepath.name}")
        print(f"  Metadata: {doc['metadata']}")
        print(f"  Số chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n  [Chunk {i+1}] Section: {chunk['metadata']['section']}")
            print(f"  Text: {chunk['text'][:150]}...")

    # Bước 3: Build index (cần API key)
    print("\n--- Build Full Index ---")
    build_index()

    # Bước 4: Kiểm tra index
    list_chunks()
    inspect_metadata_coverage()

    print("\n✅ Ingestion hoàn thành!")
