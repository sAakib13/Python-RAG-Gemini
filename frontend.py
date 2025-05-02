import streamlit as st
import requests

st.title("RAG Document Assistant")

uploaded_file = st.file_uploader("Upload PDF")
if uploaded_file:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    try:
        response = requests.post("http://localhost:8000/upload/", files=files)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if "message" in data:
            st.success(data["message"])
        elif "error" in data:
            st.error(f"Error: {data['error']}")
        else:
            st.warning("Unexpected response from server")

    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
    except ValueError as e:
        st.error(f"JSON decode error: {e}")

query = st.text_input("Ask about the document:")
if query:
    response = requests.post("http://localhost:8000/query/", json={"query": query})
    st.markdown(f"**Answer:** {response.json().get('answer')}")
