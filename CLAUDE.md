# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medicare Policy Chatbot - A FastAPI-based neural search and visual-grounding service for Medicare Summary of Benefits (SOB) and Evidence of Coverage (EOC) PDFs. The system uses Docling for PDF extraction, Qdrant for hybrid search, and maintains bounding box information for visual grounding.

## Key Commands

### Running the Service
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server locally
uvicorn service:app --reload

# Run with Docker (after ETL)
docker build -t medicare-policy-chatbot .
docker run -d -p 8000:8000 --env-file ./.env -v $(pwd)/extracted_docs:/app/extracted_docs --name medicare-chatbot-service medicare-policy-chatbot
```

### ETL Process (One-time setup)
1. Run `notebooks/01_extract_medicare_policy_docs.ipynb` (GPU required) - Extracts PDFs to JSON + PNGs
2. Run `notebooks/02_transform_and_load_medicare_policy_docs.ipynb` - Chunks and loads to Qdrant

### Code Quality
```bash
ruff check .
pytest  # Note: No test framework is currently configured
```

## Architecture Overview

### Data Flow
1. **PDF Processing**: `docs/*.pdf` → Docling → `extracted_docs/` (JSON + PNGs with bounding boxes)
2. **Embedding & Storage**: Chunked documents → BGE embeddings + BM25 → Qdrant vector DB
3. **Search & Retrieval**: FastAPI endpoints → Hybrid search → Results with bounding boxes
4. **Visual Grounding**: Search results → Annotated PNGs with highlighted text regions

### Core Components

- **service.py**: Main FastAPI application
  - `/api/plans`: List Medicare plans
  - `/api/search`: Text search across documents
  - `/api/visual_grounding`: Search with bounding box data
  - `/api/annotate_result`: Generate annotated images

- **hybrid_search.py**: Qdrant hybrid search wrapper (dense + sparse vectors with RRF fusion)

- **plan_service.py**: Manages plan metadata and document hash mappings from `plans.json`

- **config.py**: Environment configuration using Pydantic settings

### Key Dependencies
- **docling==2.31.0**: PDF extraction with layout preservation
- **qdrant-client==1.14.2**: Vector database for hybrid search
- **fastembed==0.6.1**: Embedding generation (BAAI/bge-small-en-v1.5)
- **fastapi==0.115.12**: Web framework

### Environment Variables
Create `.env` from `env.sample`:
- `QDRANT_URL`: Qdrant instance URL (default: ":memory:")
- `QDRANT_API_KEY`: API key if required
- `EMBED_MODEL_ID`: Dense embedding model (default: "sentence-transformers/all-MiniLM-L6-v2")
- `SPARSE_MODEL_ID`: Sparse model for BM25 (default: "Qdrant/bm25")
- `COLLECTION`: Qdrant collection name (default: "medicare_policy_docs")

### Working with the Codebase

1. **Adding New Documents**: Place PDFs in `docs/`, update `plans.json`, run both ETL notebooks
2. **Modifying Search**: Changes to `hybrid_search.py` affect how results are retrieved and ranked
3. **API Changes**: Update `service.py` and ensure proper error handling for production use
4. **Visual Grounding**: Bounding boxes are normalized [0,1] - multiply by image dimensions for display

### Current State Notes
- `embedding.py` contains WIP code for testing OpenAI embeddings (not integrated)
- The system expects `extracted_docs/` to exist with processed documents
- GPU accelerates both Docling extraction (~3x) and embedding generation