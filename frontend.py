import streamlit as st
import requests
import time

st.title("RAG Document Assistant")

# File upload section
with st.expander("Upload Document", expanded=True):
    uploaded_file = st.file_uploader("Choose PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            try:
                response = requests.post(
                    "http://localhost:8000/upload/",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())}
                )
                response.raise_for_status()
                st.success(response.json().get("message"))
            except Exception as e:
                st.error(f"Upload failed: {str(e)}")

# Query section
query = st.text_input("Ask about the document:", placeholder="Enter your question...")
if query:
    with st.spinner("Searching documents..."):
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:8000/query/",
                json={"query": query},
                timeout=30
            )
            response.raise_for_status()
            
            st.markdown(f"**Answer:** {response.json().get('answer')}")
            st.caption(f"Response time: {time.time() - start_time:.2f}s")
            
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")
        except Exception as e:
            st.error(f"Processing error: {str(e)}")