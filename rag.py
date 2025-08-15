import streamlit as st
from qdrant_client import QdrantClient, models
from langchain_huggingface import HuggingFaceEmbeddings

# --- BARE-BONES DIAGNOSTIC RAG ---
COLLECTION_NAME = "educade_final_db" # The new, clean name

def get_db_info():
    """Connects to Qdrant and gets collection info."""
    try:
        qdrant_url = st.secrets["QDRANT_URL"]
        qdrant_api_key = st.secrets["QDRANT_API_KEY"]
        if not qdrant_url or not qdrant_api_key:
            return "Error", "Secrets are missing from Streamlit Cloud."
        
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        info = client.get_collection(collection_name=COLLECTION_NAME)
        return "Success", f"Collection '{COLLECTION_NAME}' found with {info.vectors_count or 0} vectors."
    except Exception as e:
        return "Error", f"Failed to connect or find collection. Details: {e}"

def get_answer(user_question, grade, subject):
    """Performs a simple search."""
    try:
        client = QdrantClient(url=st.secrets["QDRANT_URL"], api_key=st.secrets["QDRANT_API_KEY"])
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        query_vector = embeddings.embed_query(user_question)
        
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=1
        )
        
        if not search_results:
             return "I couldn't find anything specific in my books, but I'll use my general knowledge!"
        else:
            return f"Success! I found a result in my books: {search_results[0].payload['text'][:50]}..."
    
    except Exception as e:
        # This is the error you were seeing.
        return f"Oh no! My memory bank had an error. Please tell my owner this: {e}"