# Stage 1: Build
FROM python:3.13-slim AS builder

WORKDIR /app

# Define cache home for sentence-transformers within the builder stage
ENV SENTENCE_TRANSFORMERS_HOME=/app/st_builder_cache

COPY requirements.txt .
# Install sentence-transformers first to ensure its cache logic is available
RUN pip install --no-cache-dir sentence-transformers && \
    pip install --no-cache-dir -r requirements.txt

# Download the model, it will be cached at $SENTENCE_TRANSFORMERS_HOME (i.e. /app/st_builder_cache)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Stage 2: Runtime
FROM python:3.13-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY service.py .
COPY hybrid_search.py .
COPY plan_service.py .
COPY plans.json .
COPY config.py .
# extracted_docs will be mounted as a volume
# docs/ are used by notebooks, not directly by the service in this model

# Environment variables for the final stage
# Defaulting to a typical local Qdrant URL
ENV QDRANT_URL="http://localhost:6333"
ENV EMBED_MODEL_ID="sentence-transformers/all-MiniLM-L6-v2"
ENV SPARSE_MODEL_ID="Qdrant/bm25"
ENV COLLECTION="medicare_policy_docs"
ENV QDRANT_API_KEY=""
# Target cache location in final image
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/torch/sentence_transformers

# Create the directory for the model cache and copy pre-downloaded models from builder
RUN mkdir -p $SENTENCE_TRANSFORMERS_HOME
# Copy from the explicit cache location in builder to the target location in final image
COPY --from=builder /app/st_builder_cache $SENTENCE_TRANSFORMERS_HOME

EXPOSE 8000

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"] 