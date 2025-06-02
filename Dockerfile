# Stage 1: Build
FROM python:3.13-slim AS builder

WORKDIR /app

COPY requirements.txt .
# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.13-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY service.py .
COPY streamlit_chat.py .
COPY hybrid_search.py .
COPY plan_service.py .
COPY plans.json .
COPY config.py .
COPY embedding.py .
COPY entrypoint.sh .

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# extracted_docs will be mounted as a volume
# docs/ are used by notebooks, not directly by the service in this model

# Environment variables for the final stage
ENV QDRANT_URL="http://localhost:6333"
ENV QDRANT_API_KEY=""
ENV EMBED_MODEL_ID="sentence-transformers/all-MiniLM-L6-v2"
ENV SPARSE_MODEL_ID="Qdrant/bm25"
ENV COLLECTION="medicare_policy_docs"
ENV OPENAI_API_KEY=""

# Expose both FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Use entrypoint script to support multiple services
# Examples:
# docker run -e SERVICE=fastapi ...        (default: FastAPI only)
# docker run -e SERVICE=streamlit ...      (Streamlit only)  
# docker run -e SERVICE=both ...           (Both services)
ENTRYPOINT ["./entrypoint.sh"] 