#!/usr/bin/env python3

"""
Streamlit Chat Interface for Medicare Policy Chatbot

This provides a user-friendly chat interface for querying Medicare documents
using the FastAPI backend service.
"""

import streamlit as st
import requests
import json
from typing import List, Dict, Any
from datetime import datetime
import base64
from io import BytesIO

# Configuration
API_BASE_URL = "http://localhost:8000"

# Set page config
st.set_page_config(
    page_title="Medicare Policy Chat",
    page_icon="🏥",
    layout="wide"
)

def get_plans() -> List[Dict]:
    """Get list of available Medicare plans"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/plans")
        if response.status_code == 200:
            return response.json()["plans"]
    except Exception as e:
        st.error(f"Failed to load plans: {e}")
    return []

def search_documents(query: str, plan_id: str = None, k: int = 5) -> List[Dict]:
    """Search documents using visual grounding endpoint"""
    try:
        params = {"q": query, "k": k}
        if plan_id and plan_id != "All Plans":
            params["plan_id"] = plan_id
        
        response = requests.get(f"{API_BASE_URL}/api/visual_grounding", params=params)
        if response.status_code == 200:
            return response.json()["result"]
    except Exception as e:
        st.error(f"Search failed: {e}")
    return []

def get_annotated_image(binary_hash: str, page: int, boxes: List[Dict]) -> bytes:
    """Get annotated image with bounding boxes"""
    try:
        payload = {
            "binary_hash": binary_hash,
            "page": page,
            "boxes": boxes
        }
        response = requests.post(f"{API_BASE_URL}/api/annotate_result", json=payload)
        if response.status_code == 200:
            return response.content
    except Exception as e:
        st.error(f"Failed to get annotated image: {e}")
    return None

def format_result_card(result: Dict, show_image: bool = False) -> None:
    """Format a search result as a card"""
    with st.container():
        st.markdown("---")
        
        # Header with plan info
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**📋 {result.get('plan_name', 'Unknown Plan')}**")
        with col2:
            st.markdown(f"*Plan ID: {result.get('plan_id', 'N/A')}*")
        with col3:
            if result.get('annotate_request_body'):
                page_num = result['annotate_request_body'].get('page', 'N/A')
                st.markdown(f"*Page: {page_num}*")
        
        # Main content
        text_content = result.get('text', '')
        if text_content:
            st.markdown("**📄 Content:**")
            # Limit text display to avoid overwhelming the UI
            if len(text_content) > 800:
                st.markdown(text_content[:800] + "...")
                with st.expander("Show full content"):
                    st.markdown(text_content)
            else:
                st.markdown(text_content)
        
        # Headings if available
        headings = result.get('headings', [])
        if headings:
            st.markdown("**📑 Section:** " + " → ".join(headings))
        
        # Show annotated image if requested
        if show_image and result.get('annotate_request_body'):
            annotate_data = result['annotate_request_body']
            if annotate_data.get('boxes'):
                with st.expander("🖼️ View highlighted document"):
                    with st.spinner("Loading annotated image..."):
                        image_data = get_annotated_image(
                            annotate_data['binary_hash'],
                            annotate_data['page'],
                            annotate_data['boxes']
                        )
                        if image_data:
                            st.image(image_data, caption=f"Page {annotate_data['page']} with highlighted content")

def main():
    # Title and description
    st.title("🏥 Medicare Policy Chatbot")
    st.markdown("Ask questions about Medicare plans and get answers from official policy documents.")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Load plans
        plans = get_plans()
        plan_options = ["All Plans"] + [f"{p['plan_id']} - {p['plan_name']}" for p in plans]
        selected_plan_option = st.selectbox("Filter by Plan:", plan_options)
        
        # Extract plan_id from selection
        selected_plan_id = None
        if selected_plan_option != "All Plans":
            selected_plan_id = selected_plan_option.split(" - ")[0]
        
        # Number of results
        num_results = st.slider("Number of results:", 1, 10, 5)
        
        # Show images toggle
        show_images = st.checkbox("Show annotated images", value=False)
        
        st.markdown("---")
        st.markdown("**💡 Example Questions:**")
        st.markdown("- What is my maximum out of pocket?")
        st.markdown("- What are the copays for specialists?")
        st.markdown("- What prescription drugs are covered?")
        st.markdown("- How do I contact customer service?")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                # Assistant message with results
                st.markdown(message["content"])
                if "results" in message:
                    for result in message["results"]:
                        format_result_card(result, show_images)

    # Chat input
    if prompt := st.chat_input("Ask a question about Medicare policies..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get search results
        with st.chat_message("assistant"):
            with st.spinner("Searching Medicare documents..."):
                results = search_documents(prompt, selected_plan_id, num_results)
            
            if results:
                # Generate response
                plan_filter_text = f" for {selected_plan_option}" if selected_plan_id else ""
                response_text = f"I found {len(results)} relevant sections{plan_filter_text} that address your question:"
                st.markdown(response_text)
                
                # Display results
                for result in results:
                    format_result_card(result, show_images)
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "results": results
                })
            else:
                no_results_msg = "I couldn't find any relevant information for your question. Try rephrasing or using different keywords."
                st.markdown(no_results_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": no_results_msg
                })

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
        "Medicare Policy Chatbot • Powered by FastAPI + Qdrant + OpenAI Embeddings"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()