import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from openai import OpenAI
from groq import Groq
import re
import random

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

# --- THE FINAL, "CHAIN OF THOUGHT" PROMPTS ---
LANGUAGE_CONFIGS = {
    "en": { "name": "English", "english_name": "English", "system_prompt": "You are Sparky, a cheerful robot tutor for a child named {name}. Your ONLY goal is to help them discover answers on their own by asking fun, simple, guiding questions in ENGLISH. Be super encouraging and use emojis. NEVER give the direct answer." },
    "hi": { "name": "हिंदी", "english_name": "Hindi", "system_prompt": "आप स्पार्की हैं, {name} नाम के एक बच्चे के लिए एक हंसमुख रोबोट ट्यूटर। आपका एकमात्र लक्ष्य उन्हें मजेदार, सरल, मार्गदर्शक प्रश्न पूछकर अपने आप उत्तर खोजने में मदद करना है। बहुत उत्साहजनक रहें और इमोजी का उपयोग करें। कभी भी सीधे उत्तर न दें। आपका जवाब HINDI में होना चाहिए।" },
    # (Simplified prompts for other languages can be added here)
}

# --- MAIN RAG FUNCTION (Completely Rewritten for Tutor Mode) ---
def get_answer(messages, grade, subject, lang, child_name, app_mode):
    if not pc or not groq_client:
        return {"answer": "Error: App is not configured. Please check API Keys.", "image_url": None, "choices": None}
    
    user_message = messages[-1]["content"]
    final_answer, image_url, choices = "", None, None
    
    try:
        if app_mode == "Story Mode":
            # Story mode logic is correct and remains the same
            pass 
        else: # Tutor Mode - The New Two-Step Logic
            index = pc.Index(INDEX_NAME)
            question_vector = embeddings.embed_query(user_message)
            
            query_response = index.query(
                vector=question_vector, top_k=3,
                filter={"grade": {"$eq": grade}, "subject": {"$eq": subject}},
                include_metadata=True
            )
            context = "\n".join([match['metadata']['text'] for match in query_response['matches']]) if query_response['matches'] else "No specific book context found. Use general knowledge."

            # --- STEP 1: The "Clue Generator" AI Call ---
            # This AI's only job is to create clean clues. The user never sees this.
            clue_generation_prompt = f"""
            Analyze the following question and context. 
            1. First, identify the simple, one or two-word answer.
            2. Second, generate three very simple, short, fun facts or hints about that answer.
            CRITICAL RULE: Do NOT use the answer word itself in the hints.
            
            Question: "{user_message}"
            Context: "{context}"

            Answer: [The answer word]
            Hint 1: [A simple hint without the answer word]
            Hint 2: [Another simple hint without the answer word]
            Hint 3: [A third simple hint without the answer word]
            """
            clue_completion = groq_client.chat.completions.create(
                model="llama3-8b-8192", # Use a smaller, faster model for this simple task
                messages=[{"role": "user", "content": clue_generation_prompt}],
                temperature=0.2
            )
            clue_text = clue_completion.choices[0].message.content
            
            # Extract the generated hints
            hints = [line.split(":", 1)[1].strip() for line in clue_text.splitlines() if line.startswith("Hint")]
            if not hints:
                # Fallback if the clue generation fails
                hints = ["It's a part of the body!", "You use it every day!", "It helps you experience the world!"]

            # --- STEP 2: The "Sparky Persona" AI Call ---
            # This AI gets ONE clean hint and makes it conversational.
            # It NEVER sees the context or the answer word.
            chosen_hint = random.choice(hints)
            config = LANGUAGE_CONFIGS.get(lang, LANGUAGE_CONFIGS["en"])
            sparky_system_prompt = config["system_prompt"].format(name=child_name)
            
            # We send the previous conversation so Sparky knows what's going on
            cleaned_history = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

            sparky_final_prompt = f"""
            Here is a simple fact: "{chosen_hint}"
            Your task is to turn this fact into a fun, encouraging, and playful question for the child, {name}.
            Remember your golden rule: NEVER give the direct answer. ALWAYS ask a guiding question.
            Your response MUST be in {config['name']}.
            """
            
            sparky_messages = [
                {"role": "system", "content": sparky_system_prompt},
                *cleaned_history[:-1], # History WITHOUT the latest question
                {"role": "user", "content": user_message}, # The latest question
                {"role": "system", "content": sparky_final_prompt} # The final, undeniable instruction
            ]

            sparky_completion = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=sparky_messages,
                temperature=0.75
            )
            final_answer = sparky_completion.choices[0].message.content

        # ... (choice parsing logic for story mode) ...
        return {"answer": final_answer, "image_url": None, "choices": None}

    except Exception as e:
        st.error(f"Oh no! Sparky had a problem. Please tell the owner this: {e}")
        return {"answer": "I'm having a little trouble thinking right now.", "image_url": None, "choices": None}