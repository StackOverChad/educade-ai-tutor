import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from openai import OpenAI
from groq import Groq
import re

# --- HYBRID CREDENTIALS LOADER ---
try:
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    groq_api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

# --- INITIALIZE CLIENTS ---
pc = Pinecone(api_key=pinecone_api_key) if pinecone_api_key else None
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
    
# --- CONSTANTS AND CONFIGS ---
INDEX_NAME = "educade-prod-db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
LANGUAGE_CONFIGS = {
    "en": {
        "name": "English", "english_name": "English",
        "system_prompt": """
        **Your Character:** You are Sparky, a cheerful and curious robot tutor for a young child named {name}.
        **Your Personality:** You are super encouraging, a little bit silly, and you love using emojis (like 🤖, ✨, 🤔, 🚀, 🎉). You sometimes say robot sounds like "Beep boop!".
        **Your Audience:** A young child (ages 3-9). Use short, simple sentences and easy words.
        **--- YOUR GOLDEN RULES ---**
        1. **NEVER give the direct answer.** This is your most important rule. Your job is to help the child think for themselves.
        2. **ALWAYS be positive and encouraging.** Never say "no," "wrong," or "that's incorrect." Instead, say things like "That's a super guess!", "You're so close!", "Great thinking!".
        3. **ALWAYS respond with a guiding question or a fun hint.**
        4. **USE THE CHILD'S NAME** to make it personal.
        **--- YOUR CONVERSATIONAL PATTERN ---**
        1. **Child asks a question:** Read the question and the context from the textbook.
        2. **Give your first hint:** Formulate a fun, simple question that hints at the answer. Example: "Beep boop! What a fun question, {name}! 🤔 What's the part of your face that can smell yummy cookies baking? 🍪"
        3. **Child guesses:**
            * **If the guess is correct:** Celebrate! Say "YES! You got it, {name}! 🎉 You're a superstar! The nose helps us smell everything!"
            * **If the guess is incorrect:** Be encouraging! Say "That's a great guess! Your mouth helps with tasting, which is super close! The part for smelling is right above it. What do you think? 👃"
        """
    },
    # (Your other language configurations go here)
}

# --- MAIN RAG FUNCTION ---
def get_answer(messages, grade, subject, lang, child_name, app_mode):
    if not pc or not groq_client:
        return {"answer": "Error: App is not configured. Please check API Keys.", "image_url": None, "choices": None}
    
    user_message = messages[-1]["content"]
    final_answer, image_url, choices = "", None, None
    
    try:
        if app_mode == "Story Mode":
            story_prompt = f"You are a master storyteller for a child named {child_name}. Continue the story based on the child's last choice. The story should be educational and related to {subject} for {grade}. End your response with a clear choice for the child using the format [CHOICE: Option 1 | Option 2]. Keep the story engaging and magical."
            story_messages = [{"role": "system", "content": story_prompt}, *messages[1:]]
            completion = groq_client.chat.completions.create(model="llama3-70b-8192", messages=story_messages, temperature=0.8)
            final_answer = completion.choices[0].message.content
        else: # Tutor Mode
            index = pc.Index(INDEX_NAME)
            question_vector = embeddings.embed_query(user_message)
            query_response = index.query(vector=question_vector, top_k=3, filter={"grade": {"$eq": grade}, "subject": {"$eq": subject}}, include_metadata=True)
            context = "\n".join([match['metadata']['text'] for match in query_response['matches']]) if query_response['matches'] else "No specific information found in my books for that. I will use my general knowledge."
            
            config = LANGUAGE_CONFIGS.get(lang, LANGUAGE_CONFIGS["en"])
            system_prompt = config["system_prompt"].format(name=child_name)
            
            # --- THIS IS THE CRITICAL FIX ---
            # Create a "clean" version of the history for the API, containing only role and content.
            cleaned_history = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            # --------------------------------

            updated_messages = [
                {"role": "system", "content": system_prompt},
                *cleaned_history, # Use the cleaned history
                {"role": "system", "content": f"Use this context from a textbook to help you form your hint, but do not mention the context directly:\n---\n{context}\n---"}
            ]

            completion = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=updated_messages,
                temperature=0.7
            )
            final_answer = completion.choices[0].message.content

        choices_match = re.search(r'\[CHOICE:\s*(.*?)\s*\]', final_answer)
        if choices_match:
            final_answer = final_answer.replace(choices_match.group(0), "").strip()
            choices = [c.strip() for c in choices_match.group(1).split('|')]
        
        return {"answer": final_answer, "image_url": None, "choices": choices}

    except Exception as e:
        st.error(f"Oh no! Sparky had a problem. Please tell the owner this: {e}")
        return {"answer": "I'm having a little trouble thinking right now.", "image_url": None, "choices": None}