# Medicare Policy Chatbot 📄🧠

A FastAPI‑based **neural search** and **visual‑grounding** service for Medicare
_Summary of Benefits_ (SOB) and _Evidence of Coverage_ (EOC) PDFs.

<table>
<tr><td>🗂️ Extraction</td><td><b>Docling</b> + SmolDocling‑256M on GPU</td></tr>
<tr><td>🔍 Search</td><td><b>Qdrant</b> hybrid (dense BGE‑small‑en + BM25)</td></tr>
<tr><td>🎯 Grounding</td><td>Bounding‑box payload returned with every hit</td></tr>
</table>

---

## Contents

| Folder / file                                                | Purpose                                                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `notebooks/01_extract_medicare_policy_docs.ipynb`            | **GPU** notebook — converts PDFs → Docling JSON + per‑page PNGs (6 min for 206 pages) |
| `notebooks/02_transform_and_load_medicare_policy_docs.ipynb` | **CPU or T4 GPU** — chunks, embeds, and bulk‑loads to local Qdrant, keeping bboxes    |
| `service.py`                                                 | FastAPI server with `/api/search`, `/api/visual_grounding`, `/api/annotate_result`    |
| `hybrid_search.py`                                           | Thin wrapper around Qdrant `client.query()`                                           |
| `plan_service.py`                                            | Maps plan IDs → SOB / EOC binary hashes                                               |
| `plans.json`                                                 | Declarative list of plans and their document hashes                                   |
| `extracted_docs/`                                            | One `<doc>.json` + `/<doc>/<page>.png` folder per PDF                                 |
| `docs/`                                                      | Contains the original source PDF documents for Medicare plans                         |
| `requirements.txt`                                           | Pinned versions (`docling 0.28.2`, `qdrant-client[fastembed-gpu]`, `fastapi`, …)      |

---

## Pipeline Overview

```mermaid
flowchart LR
    A[Source PDFs in ./docs/] -->|01 Extract| B[Docling JSON + page PNGs<br/>(+ bbox provenance)]
    B -->|02 Transform/Load| C[Qdrant local file (:memory: for dev)]
    C -->|/api/search| D[Top‑K chunks<br/>payload = {doc_name,page_no,bbox}]
    D -->|/api/visual_grounding| E[Bounding boxes drawn<br/>on cached PNG]
```

- **Docling settings**
  `generate_page_images=True`, `image_mode=REFERENCED`, and
  `res.document.export_figures()` write lean JSON + external PNGs.

- **Chunker**
  `HybridChunker` (\~200 tokens) keeps layout coherence.

- **Dense model (FastEmbed)**
  `BAAI/bge-small-en-v1.5` (384‑d) on CUDAExecutionProvider.

- **Sparse model**
  `Qdrant/bm25` fused with dense results via built‑in RRF.

- **Bounding boxes**
  Normalised `[l,t,r,b]` (0‑1) stored in payload → drawn in `/api/annotate_result`.

---

## Quickstart

### 0 · Prereqs

- Python 3.10+
- GPU optional (T4 gives 3× faster embeddings)

### 1 · Clone & env

```bash
git clone https://github.com/jharkins/medicare-policy-chatbot.git
cd medicare-policy-chatbot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2 · Run the notebooks (one‑time ETL)

| Notebook                                       | Runtime type  | What it does                                                       |
| ---------------------------------------------- | ------------- | ------------------------------------------------------------------ |
| **01_extract_medicare_policy_docs**            | GPU (T4/A100) | Processes PDFs from `docs/` ➜ converts to JSON + PNGs              |
| **02_transform_and_load_medicare_policy_docs** | CPU / GPU     | Chunks extracted data ➜ `client.add()` to Qdrant with bbox payload |

Both notebooks are re‑entrant; rerun any time new PDFs arrive.

### 3 · FastAPI server

```bash
uvicorn service:app --reload
```

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

Environment variables (defaults work for `:memory:` Qdrant):

```bash
export QDRANT_URL=":memory:" # or "http://localhost:6333" for a persistent instance
export EMBED_MODEL_ID="sentence-transformers/all-MiniLM-L6-v2" # (matches .env.sample; BAAI/bge-small-en-v1.5 also recommended)
export SPARSE_MODEL_ID=Qdrant/bm25
export COLLECTION=medicare_policy_docs
export QDRANT_API_KEY="" # (if your Qdrant instance requires an API key)
```

These can be set in your shell or by creating a `.env` file in the project root (see `.env.sample`).

---

## Core Endpoints

| Method & Path                                       | Purpose                       |
| --------------------------------------------------- | ----------------------------- |
| **GET /api/plans**                                  | list loaded plans & hashes    |
| **GET /api/search** `?q=`                           | top‑K hybrid search chunks    |
| **GET /api/visual_grounding** <br>`?q=&plan_id=&k=` | same as search + bbox payload |
| **POST /api/annotate_result**                       | JSON → PNG with rectangles    |

Bounding‑box array is always normalised; front‑end multiplies by displayed img w/h.

---

## GPU vs CPU Cheat‑sheet

| Stage                | Free CPU                 | T4 GPU                  |
| -------------------- | ------------------------ | ----------------------- |
| Docling extraction   | ❌ (slow)                | ✅ (~6 min / 200 pages) |
| FastEmbed embeddings | ✅ (~1 min / 3 k chunks) | ✅ (~20 s)              |
| Qdrant query + LLM   | ✅                       | (GPU idle)              |

---

## Contributing

1. Fork → `git checkout -b feature/...` → PR
2. Run `ruff check .` and `pytest` before pushing.
3. Large PDFs? Test in `:memory:` then point to real Qdrant.

---

## License

MIT © BitStorm Technologies
