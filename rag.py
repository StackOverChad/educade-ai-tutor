import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from openai import OpenAI
import re

# --- HYBRID CREDENTIALS LOADER ---
try:
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

# --- INITIALIZE CLIENTS (Simpler) ---
pc = Pinecone(api_key=pinecone_api_key) if pinecone_api_key else None
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
    
# --- CONSTANTS AND CONFIGS ---
INDEX_NAME = "educade-prod-db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
LANGUAGE_CONFIGS = {
    "en": { "name": "English", "english_name": "English", "requires_translation": False, "system_prompt": "You are a playful tutor for a child named {name}..." },
    # (Your other languages go here)
}

def get_answer(messages, grade, subject, lang, child_name, app_mode):
    if not pc or not openai_client:
        return {"answer": "Error: App is not configured. Please check API Keys.", "image_url": None, "choices": None}
    
    user_message = messages[-1]["content"]
    
    try:
        index = pc.Index(INDEX_NAME)
        question_vector = embeddings.embed_query(user_message)
        
        query_response = index.query(
            vector=question_vector,
            top_k=3,
            filter={"grade": {"$eq": grade}, "subject": {"$eq": subject}},
            include_metadata=True
        )
        
        if not query_response['matches']:
            context = "No specific information found in my books for that."
        else:
            context = "\n".join([match['metadata']['text'] for match in query_response['matches']])
        
        # (Your full AI response logic would go here)
        final_answer = f"Based on my books, I found this: {context[:100]}..."
        return {"answer": final_answer, "image_url": None, "choices": None}

    except Exception as e:
        st.error(f"Oh no! Sparky had a problem searching his memory bank. Error: {e}")
        return {"answer": "I'm having a little trouble thinking right now.", "image_url": None, "choices": None}