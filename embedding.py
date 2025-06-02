from __future__ import annotations

from pathlib import Path
import os
import uuid
import argparse
from typing import Iterable, List

import pandas as pd
import tiktoken
from dotenv import load_dotenv
import openai
import qdrant_client
from qdrant_client.models import PointStruct, VectorParams, Distance

try:
    # Try the import path from the working notebook first
    from docling.datamodel.document import DoclingDocument
    from docling.chunking import HybridChunker

    print("Using docling imports (notebook style)")
except ImportError:
    # Fall back to docling_core imports
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

    print("Using docling_core imports")

# Add this after your imports, before the constants section:
parser = argparse.ArgumentParser(description="Embed documents into Qdrant")
parser.add_argument(
    "--file", "-f", help="Process single file (filename only, e.g. 'Humana EOC.json')"
)
parser.add_argument("--debug", action="store_true", help="Enable debug output")
args = parser.parse_args()

# ────────────────────────────────────────────────
# 1. ENV & CONSTANTS
# ────────────────────────────────────────────────
load_dotenv()  # .env → env vars

MAX_TOKENS = 8191
EMBEDDING_MODEL = "text-embedding-3-small"  # 1 536-d vectors
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "docs")  # fallback name

OUT_DIR = Path("./extracted_docs")  # Docling JSONs

if args.file:
    target_file = OUT_DIR / args.file
    if not target_file.exists():
        print(f"❌ File not found: {target_file}")
        exit(1)
    JSONS = [target_file]
    print(f"Processing single file: {args.file}")
else:
    JSONS = sorted(OUT_DIR.glob("*.json"))
    print(f"Found {len(JSONS)} Docling JSON files")

# ────────────────────────────────────────────────
# 2. SETUP – OpenAI, tokenizer, chunker, Qdrant
# ────────────────────────────────────────────────
# Try notebook-style initialization first
try:
    chunker = HybridChunker(tokenizer=EMBEDDING_MODEL)
    print(f"Using HybridChunker with tokenizer: {EMBEDDING_MODEL}")
except:
    # Fall back to manual tokenizer setup
    try:
        from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

        enc = tiktoken.encoding_for_model("gpt-4o")
        tokenizer = OpenAITokenizer(tokenizer=enc, max_tokens=MAX_TOKENS)
        chunker = HybridChunker(tokenizer=tokenizer)
        print("Using HybridChunker with OpenAITokenizer")
    except:
        print("ERROR: Could not initialize chunker")

openai_client = openai.Client()

qdrant = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


def ensure_collection(name: str) -> None:
    """Create the collection (1536-d cosine) if it doesn’t exist yet."""
    if name in {c.name for c in qdrant.get_collections().collections}:
        return  # already there
    qdrant.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    print(f"Created collection '{name}'")


ensure_collection(COLLECTION_NAME)


# ────────────────────────────────────────────────
# 3. HELPERS
# ────────────────────────────────────────────────
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch-embed a list of strings with OpenAI and return the vectors."""
    resp = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # The API returns results in order
    return [data.embedding for data in resp.data]


def chunk_doc(path: Path):
    doc = DoclingDocument.load_from_json(path)
    if args.debug:
        print(f"   📄 Doc origin: {doc.origin}")
        print(f"   📄 Doc structure: {type(doc)}")
        print(f"   📄 Content: {len(doc.texts)} texts, {len(doc.tables)} tables")

    if doc.origin is None:
        raise ValueError(f"{path} has no origin; cannot compute point ids.")

    binary_hash = str(doc.origin.binary_hash)
    chunk_count = 0

    # Strategy 1: Try standard chunking for text-heavy documents
    try:
        for i, chunk in enumerate(chunker.chunk(doc)):
            chunk_count += 1
            uid = uuid.uuid5(uuid.NAMESPACE_DNS, f"{binary_hash}-{i}")
            text_to_insert = chunker.contextualize(chunk=chunk)

            # Handle different chunk types
            if hasattr(chunk, "text") and hasattr(chunk, "meta"):
                meta = chunk.meta.export_json_dict()
                # Add the text content to payload for search results
                meta["text"] = text_to_insert
                yield (
                    str(uid),
                    text_to_insert,
                    meta,
                )
            else:
                meta = {"text": str(text_to_insert)}
                yield (
                    str(uid),
                    str(text_to_insert),
                    meta,
                )
    except Exception as e:
        if args.debug:
            print(f"   ❌ Standard chunking error: {e}")

    # Strategy 2: If no chunks generated and document has tables, extract table content
    if chunk_count == 0 and hasattr(doc, "tables") and len(doc.tables) > 0:
        if args.debug:
            print(f"   🔄 Fallback: Extracting content from {len(doc.tables)} tables")

        for table_idx, table in enumerate(doc.tables):
            try:
                # Try to get table as markdown or text
                table_text = ""
                if hasattr(table, "export_to_markdown"):
                    try:
                        # Try with doc parameter first to avoid deprecation warning
                        table_text = table.export_to_markdown(doc=doc)
                    except:
                        # Fallback to deprecated method
                        table_text = table.export_to_markdown()
                elif hasattr(table, "text"):
                    table_text = table.text
                elif hasattr(table, "data"):
                    # Convert table data to text representation
                    table_text = str(table.data)
                else:
                    table_text = str(table)

                if (
                    table_text and len(table_text.strip()) > 10
                ):  # Only if meaningful content
                    chunk_count += 1
                    uid = uuid.uuid5(
                        uuid.NAMESPACE_DNS, f"{binary_hash}-table-{table_idx}"
                    )

                    # Create metadata for table chunk
                    meta = {
                        "doc_hash": binary_hash,
                        "table_index": table_idx,
                        "content_type": "table",
                        "chunk_index": table_idx,
                        "text": table_text,  # Add text to payload for search results
                    }

                    # Add bounding box if available
                    if hasattr(table, "prov") and table.prov:
                        meta["bbox"] = (
                            table.prov[0].bbox.as_dict() if table.prov[0].bbox else None
                        )
                        meta["page_no"] = (
                            table.prov[0].page_no
                            if hasattr(table.prov[0], "page_no")
                            else None
                        )

                    yield (str(uid), table_text, meta)

            except Exception as e:
                if args.debug:
                    print(f"   ❌ Error processing table {table_idx}: {e}")

    # Strategy 3: If still no chunks, try to extract from markdown export
    if chunk_count == 0:
        if args.debug:
            print(f"   🔄 Last resort: Using markdown export")
        try:
            md_content = doc.export_to_markdown()
            if (
                md_content and len(md_content.strip()) > 50
            ):  # Only if substantial content
                chunk_count += 1
                uid = uuid.uuid5(uuid.NAMESPACE_DNS, f"{binary_hash}-markdown")
                meta = {
                    "doc_hash": binary_hash,
                    "content_type": "markdown_export",
                    "chunk_index": 0,
                    "text": md_content,  # Add text to payload for search results
                }
                yield (str(uid), md_content, meta)
        except Exception as e:
            if args.debug:
                print(f"   ❌ Markdown export failed: {e}")

    if args.debug:
        print(f"   🔍 Total chunks generated: {chunk_count}")


def upsert_points(
    collection: str,
    ids: Iterable[str],
    vectors: Iterable[List[float]],
    payloads: Iterable[dict],
) -> None:
    """Send the points to Qdrant."""
    points = [
        PointStruct(id=pid, vector=vec, payload=pl)
        for pid, vec, pl in zip(ids, vectors, payloads)
    ]
    qdrant.upsert(collection_name=collection, points=points)


# ────────────────────────────────────────────────
# 4. MAIN – iterate over every Docling JSON and upsert
# ────────────────────────────────────────────────
for json_path in JSONS:
    print(f"→ Processing {json_path.name}")

    # 4-a  Chunk & gather
    chunks = list(chunk_doc(json_path))
    if not chunks:
        print(f"   ⚠️  No chunks generated - skipping")
        continue

    ids, texts, payloads = zip(*chunks)
    print(f"   {len(texts)} chunks")

    # 4-b  Embed
    vectors = embed_texts(list(texts))
    print("   embeddings ok")

    # 4-c  Upsert
    upsert_points(COLLECTION_NAME, ids, vectors, payloads)
    print(f"   upserted {len(ids)} points\n")

print("🎉  All done!")
