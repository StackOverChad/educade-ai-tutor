import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from openai import OpenAI
import re

# --- HYBRID CREDENTIALS LOADER (UPDATED FOR GROQ) ---
try:
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    groq_api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

# --- INITIALIZE CLIENTS (UPDATED FOR GROQ) ---
pc = Pinecone(api_key=pinecone_api_key) if pinecone_api_key else None
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None # <-- NEW
    
# --- CONSTANTS AND CONFIGS ---
INDEX_NAME = "educade-prod-db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
LANGUAGE_CONFIGS = {
    "en": { "name": "English", "english_name": "English", "system_prompt": "You are Sparky, a playful, encouraging, and fun AI tutor for a young child named {name}. Your personality is curious and cheerful. **Your most important rule is to NEVER give the answer directly.** Instead, you must guide the child to discover the answer themselves by asking a simple, fun hint or a guiding question. When the child guesses, either confirm their answer enthusiastically or gently guide them if they are wrong. Keep your responses short, simple, and in English." },
    # (All your other language configurations go here)
}

# --- IMAGE GENERATION (DISABLED) ---
# Groq does not have an image model, so this feature is disabled.
# If you still have OpenAI credits, you could re-enable it by passing the openai_client.
def should_generate_image(text_response):
    return None
def generate_illustration(keyword):
    return None

# --- MAIN RAG FUNCTION (UPDATED TO USE GROQ) ---
def get_answer(messages, grade, subject, lang, child_name, app_mode):
    if not pc or not groq_client: # Check for the groq_client now
        return {"answer": "Error: App is not configured. Please check API Keys.", "image_url": None, "choices": None}
    
    user_message = messages[-1]["content"]
    final_answer, image_url, choices = "", None, None
    
    try:
        if app_mode == "Story Mode":
            story_prompt = f"You are a master storyteller for a child named {child_name}. Continue the story based on the child's last choice. The story should be educational and related to {subject} for {grade}. End your response with a clear choice for the child using the format [CHOICE: Option 1 | Option 2]. Keep the story engaging and magical."
            story_messages = [{"role": "system", "content": story_prompt}, *messages[1:]]
            # --- USE GROQ FOR STORYTELLING ---
            completion = groq_client.chat.completions.create(
                model="llama3-70b-8192", messages=story_messages, temperature=0.8
            )
            final_answer = completion.choices[0].message.content
        else: # Tutor Mode
            index = pc.Index(INDEX_NAME)
            question_vector = embeddings.embed_query(user_message)
            
            query_response = index.query(
                vector=question_vector, top_k=3,
                filter={"grade": {"$eq": grade}, "subject": {"$eq": subject}},
                include_metadata=True
            )
            
            context = "\n".join([match['metadata']['text'] for match in query_response['matches']]) if query_response['matches'] else "No specific information found in my books for that. I will use my general knowledge."
            
            config = LANGUAGE_CONFIGS.get(lang, LANGUAGE_CONFIGS["en"])
            system_prompt = config["system_prompt"].format(name=child_name)
            
            updated_messages = [
                {"role": "system", "content": system_prompt},
                *messages,
                {"role": "system", "content": f"Use this context from a textbook to help you form your hint, but do not mention the context directly:\n---\n{context}\n---"}
            ]

            # --- USE GROQ FOR CONVERSATION ---
            completion = groq_client.chat.completions.create(
                model="llama3-70b-8192", # Using a powerful and fast Groq model
                messages=updated_messages,
                temperature=0.7
            )
            final_answer = completion.choices[0].message.content

        # NOTE: Image generation is disabled as Groq does not support it.
        # image_keyword = should_generate_image(final_answer)
        # if image_keyword: image_url = generate_illustration(image_keyword)
        
        choices_match = re.search(r'\[CHOICE:\s*(.*?)\s*\]', final_answer)
        if choices_match:
            final_answer = final_answer.replace(choices_match.group(0), "").strip()
            choices = [c.strip() for c in choices_match.group(1).split('|')]
        
        return {"answer": final_answer, "image_url": None, "choices": choices}

    except Exception as e:
        st.error(f"Oh no! Sparky had a problem. Please tell the owner this: {e}")
        return {"answer": "I'm having a little trouble thinking right now.", "image_url": None, "choices": None}