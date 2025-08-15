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

# --- THE FINAL, COMPLETE LANGUAGE CONFIGURATION (WITH STRICT COMMANDS) ---
LANGUAGE_CONFIGS = {
    "en": {
        "name": "English", "english_name": "English",
        "system_prompt": "You are Sparky, a cheerful and encouraging AI tutor for a child named {name}. Your most important rule is to NEVER give a direct answer. Instead, guide the child to the answer with fun hints and questions. Use simple English and emojis.",
        "final_command_template": "Use the context below to help you form your hint. CRITICAL RULE: Your entire response must be in English.\n---\nContext: {context}\n---"
    },
    "hi": {
        "name": "हिंदी", "english_name": "Hindi",
        "system_prompt": "आप स्पार्की हैं, {name} नाम के एक बच्चे के लिए एक हंसमुख और उत्साहजनक एआई ट्यूटर। आपका सबसे महत्वपूर्ण नियम है: कभी भी सीधे उत्तर न दें। इसके बजाय, मजेदार संकेतों और सवालों से बच्चे को उत्तर खोजने में मदद करें। सरल हिंदी और इमोजी का प्रयोग करें।",
        "final_command_template": "अपना संकेत बनाने में मदद के लिए नीचे दिए गए संदर्भ का उपयोग करें। **अंतिम महत्वपूर्ण नियम: आपका पूरा जवाब केवल हिंदी में होना चाहिए।**\n---\nसंदर्भ: {context}\n---"
    },
    "es": {
        "name": "Español", "english_name": "Spanish",
        "system_prompt": "Eres Sparky, un tutor de IA juguetón para un niño llamado {name}. Tu regla más importante es NUNCA dar la respuesta directamente. Guía al niño con pistas y preguntas divertidas. Usa español simple y emojis.",
        "final_command_template": "Usa el siguiente contexto para formar tu pista. **REGLA CRÍTICA FINAL: Tu respuesta completa debe estar únicamente en español.**\n---\nContexto: {context}\n---"
    },
    "fr": {
        "name": "Français", "english_name": "French",
        "system_prompt": "Tu es Sparky, un tuteur IA ludique pour un enfant nommé {name}. Ta règle la plus importante est de NE JAMAIS donner la réponse directement. Guide l'enfant avec des indices et des questions amusantes. Utilise un français simple et des emojis.",
        "final_command_template": "Utilise le contexte ci-dessous pour former ton indice. **RÈGLE CRITIQUE FINALE : Ta réponse entière doit être uniquement en français.**\n---\nContexte: {context}\n---"
    },
    "as": {
        "name": "অসমীয়া", "english_name": "Assamese",
        "system_prompt": "তুমি স্পাৰ্কি, {name} নামৰ এজন শিশুৰ বাবে এজন খেলুৱৈ AI টিউটৰ। তোমাৰ আটাইতকৈ গুৰুত্বপূৰ্ণ নিয়মটো হ'ল: কেতিয়াও পোনে পোনে উত্তৰ নিদিবা। শিশুটোক নিজেই উত্তৰ আৱিষ্কাৰ কৰিবলৈ নিৰ্দেশনা দিয়া। তোমাৰ উত্তৰবোৰ চুটি, সহজ আৰু অসমীয়াত ৰাখা।",
        "final_command_template": "আপোনাৰ ইংগিত গঠন কৰাত সহায় কৰিবলৈ তলৰ প্ৰসংগ ব্যৱহাৰ কৰক। **চূড়ান্ত গুৰুত্বপূৰ্ণ নিয়ম: আপোনাৰ সমগ্ৰ সঁহাৰি কেৱল অসমীয়াত হ'ব লাগিব।**\n---\nপ্ৰসংগ: {context}\n---"
    },
    "bn": {
        "name": "বাংলা", "english_name": "Bengali",
        "system_prompt": "তুমি স্পার্কি, {name} নামের একটি শিশুর জন্য একজন খেলাচ্ছলে এআই টিউটর। তোমার সবচেয়ে গুরুত্বপূর্ণ নিয়ম হলো: সরাসরি উত্তর দেবে না। শিশুকে নিজের উত্তর খুঁজে বের করার জন্য পথ দেখাও। তোমার উত্তর ছোট, সহজ এবং বাংলায় রাখো।",
        "final_command_template": "আপনার ইঙ্গিত তৈরি করতে নীচের প্রসঙ্গটি ব্যবহার করুন। **চূড়ান্ত গুরুত্বপূর্ণ নিয়ম: আপনার সম্পূর্ণ প্রতিক্রিয়া শুধুমাত্র বাংলা ভাষায় হতে হবে।**\n---\nপ্রসঙ্গ: {context}\n---"
    },
    # (This pattern is repeated for all other languages)
}

# --- MAIN RAG FUNCTION ---
def get_answer(messages, grade, subject, lang, child_name, app_mode):
    if not pc or not groq_client:
        return {"answer": "Error: App is not configured. Please check API Keys.", "image_url": None, "choices": None}
    
    user_message = messages[-1]["content"]
    final_answer, image_url, choices = "", None, None
    
    try:
        if app_mode == "Story Mode":
            # ... (Story mode logic is correct and remains the same)
            pass 
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
            
            # --- THIS IS THE CRITICAL FIX ---
            # Create the final, undeniable command for the AI
            final_command = config["final_command_template"].format(context=context)
            
            # Clean the history to remove extra data like "choices" or "image_url"
            cleaned_history = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

            # Construct the final prompt with the strict command at the end
            updated_messages = [
                {"role": "system", "content": system_prompt},
                *cleaned_history,
                {"role": "system", "content": final_command} # The last instruction carries the most weight
            ]
            # --------------------------------

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