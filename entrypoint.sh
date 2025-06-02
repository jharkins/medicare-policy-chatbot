#!/bin/bash

# entrypoint.sh - Docker entrypoint script for Medicare Policy Chatbot

set -e

# Default service is FastAPI
SERVICE=${SERVICE:-fastapi}

case "$SERVICE" in
    "fastapi")
        echo "Starting FastAPI service on port 8000..."
        exec uvicorn service:app --host 0.0.0.0 --port 8000
        ;;
    "streamlit")
        echo "Starting Streamlit chat interface on port 8501..."
        exec streamlit run streamlit_chat.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
        ;;
    "both")
        echo "Starting both FastAPI (8000) and Streamlit (8501) services..."
        # Start FastAPI in background
        uvicorn service:app --host 0.0.0.0 --port 8000 &
        # Start Streamlit in foreground
        exec streamlit run streamlit_chat.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
        ;;
    *)
        echo "Unknown service: $SERVICE"
        echo "Valid options: fastapi, streamlit, both"
        exit 1
        ;;
esac