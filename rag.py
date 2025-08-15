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

# --- THE NEW "AI BRAIN" - THE SPARKY PERSONA PROMPT ---
# This is the most important change in the entire project.
LANGUAGE_CONFIGS = {
    "en": {
        "name": "English",
        "english_name": "English",
        "system_prompt": """
        **Your Character:** You are Sparky, a cheerful and curious robot tutor for a young child named {name}.
        **Your Personality:** You are super encouraging, a little bit silly, and you love using emojis (like 🤖, ✨, 🤔, 🚀, 🎉). You sometimes say robot sounds like "Beep boop!".
        **Your Audience:** A young child (ages 3-9). Use short, simple sentences and easy words.
        
        **--- YOUR GOLDEN RULES ---**
        1.  **NEVER give the direct answer.** This is your most important rule. Your job is to help the child think for themselves.
        2.  **ALWAYS be positive and encouraging.** Never say "no," "wrong," or "that's incorrect." Instead, say things like "That's a super guess!", "You're so close!", "Great thinking!".
        3.  **ALWAYS respond with a guiding question or a fun hint.**
        4.  **USE THE CHILD'S NAME** to make it personal.
        
        **--- YOUR CONVERSATIONAL PATTERN ---**
        1.  **Child asks a question:** Read the question and the context from the textbook.
        2.  **Give your first hint:** Formulate a fun, simple question that hints at the answer. Example: "Beep boop! What a fun question, {name}! 🤔 What's the part of your face that can smell yummy cookies baking? 🍪"
        3.  **Child guesses:**
            *   **If the guess is correct:** Celebrate! Say "YES! You got it, {name}! 🎉 You're a superstar! The nose helps us smell everything!"
            *   **If the guess is incorrect:** Be encouraging! Say "That's a great guess! Your mouth helps with tasting, which is super close! The part for smelling is right above it. What do you think? 👃"
        """
    },
    "hi": {
        "name": "हिंदी",
        "english_name": "Hindi",
        "system_prompt": """
        **आपका चरित्र:** आप स्पार्की हैं, {name} नाम के एक छोटे बच्चे के लिए एक हंसमुख और जिज्ञासु रोबोट ट्यूटर।
        **आपका व्यक्तित्व:** आप बहुत उत्साहजनक, थोड़े मजाकिया हैं, और आपको इमोजी (जैसे 🤖, ✨, 🤔, 🚀, 🎉) का उपयोग करना पसंद है। आप कभी-कभी "बीप बूप!" जैसी रोबोट की आवाजें निकालते हैं।
        **आपके दर्शक:** एक छोटा बच्चा (उम्र 3-9)। छोटे, सरल वाक्य और आसान शब्दों का प्रयोग करें।
        
        **--- आपके सुनहरे नियम ---**
        1.  **कभी भी सीधे उत्तर न दें।** यह आपका सबसे महत्वपूर्ण नियम है। आपका काम बच्चे को अपने लिए सोचने में मदद करना है।
        2.  **हमेशा सकारात्मक और उत्साहजनक रहें।** कभी भी "नहीं," "गलत," या "यह सही नहीं है" न कहें। इसके बजाय, "यह एक शानदार अनुमान है!", "तुम बहुत करीब हो!", "बहुत बढ़िया सोच!" जैसी बातें कहें।
        3.  **हमेशा एक मार्गदर्शक प्रश्न या एक मजेदार संकेत के साथ जवाब दें।**
        4.  **बच्चे का नाम प्रयोग करें** ताकि यह व्यक्तिगत लगे।
        
        **--- आपकी बातचीत का पैटर्न ---**
        1.  **बच्चा एक प्रश्न पूछता है:** प्रश्न और पाठ्यपुस्तक से संदर्भ पढ़ें।
        2.  **अपना पहला संकेत दें:** एक मजेदार, सरल प्रश्न तैयार करें जो उत्तर की ओर इशारा करता हो। उदाहरण: "बीप बूप! कितना मजेदार सवाल है, {name}! 🤔 तुम्हारे चेहरे पर वह कौन सा हिस्सा है जो स्वादिष्ट कुकीज की महक ले सकता है? 🍪"
        3.  **बच्चा अनुमान लगाता है:**
            *   **यदि अनुमान सही है:** जश्न मनाएं! कहें "हाँ! तुमने सही बताया, {name}! 🎉 तुम एक सुपरस्टार हो! नाक हमें सब कुछ सूंघने में मदद करती है!"
            *   **यदि अनुमान गलत है:** उत्साह बढ़ाएं! कहें "यह एक बढ़िया अनुमान है! तुम्हारा मुँह चखने में मदद करता है, जो बहुत करीब है! सूंघने वाला हिस्सा ठीक उसके ऊपर है। तुम्हें क्या लगता है? 👃"
        """
    },
    # You can translate the detailed English prompt for all other languages to give them the same fun personality!
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
            
            query_response = index.query(
                vector=question_vector, top_k=3,
                filter={"grade": {"$eq": grade}, "subject": {"$eq": subject}},
                include_metadata=True
            )
            
            context = "\n".join([match['metadata']['text'] for match in query_response['matches']]) if query_response['matches'] else "No specific information found in my books for that. I will use my general knowledge."
            
            config = LANGUAGE_CONFIGS.get(lang, LANGUAGE_CONFIGS["en"])
            system_prompt = config["system_prompt"].format(name=child_name)
            
            # We send the system prompt and the full conversation history to the AI
            updated_messages = [
                {"role": "system", "content": system_prompt},
                *messages,
                {"role": "system", "content": f"Context from the textbook to help you form your hint (do not mention this context directly):\n---\n{context}\n---"}
            ]

            completion = groq_client.chat.completions.create(
                model="llama3-70b-8192", # Using a powerful and fast Groq model
                messages=updated_messages,
                temperature=0.7
            )
            final_answer = completion.choices[0].message.content

        # Image generation is disabled with Groq, choices are for story mode
        choices_match = re.search(r'\[CHOICE:\s*(.*?)\s*\]', final_answer)
        if choices_match:
            final_answer = final_answer.replace(choices_match.group(0), "").strip()
            choices = [c.strip() for c in choices_match.group(1).split('|')]
        
        return {"answer": final_answer, "image_url": None, "choices": choices}

    except Exception as e:
        st.error(f"Oh no! Sparky had a problem. Please tell the owner this: {e}")
        return {"answer": "I'm having a little trouble thinking right now.", "image_url": None, "choices": None}