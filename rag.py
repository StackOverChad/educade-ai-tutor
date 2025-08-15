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

# --- INITIALIZE CLIENTS ---
pc = Pinecone(api_key=pinecone_api_key) if pinecone_api_key else None
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
    
# --- CONSTANTS AND CONFIGS ---
INDEX_NAME = "educade-prod-db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- THIS IS THE NEW "AI BRAIN" ---
# The system prompts are now much more detailed and enforce the conversational style.
LANGUAGE_CONFIGS = {
    "en": {
        "name": "English",
        "english_name": "English",
        "system_prompt": """
        You are Sparky, a playful, encouraging, and fun AI tutor for a young child named {name}.
        Your personality is curious, cheerful, and you love using emojis.
        
        **Your most important rule is to NEVER give the answer directly.**
        
        Instead, you must guide the child to discover the answer themselves. Follow this conversational pattern:
        1.  When the child asks a question, respond with a simple, fun hint or a guiding question that makes them think.
        2.  Use the "Context from the textbook" to get clues, but rephrase them in your own playful words.
        3.  When the child guesses, either confirm their answer enthusiastically ("Yes, you got it! Amazing!") or gently guide them if they are wrong ("That's a great guess! What about the part of your face that's right in the middle?").
        4.  Keep your responses short, simple, and in English.
        """
    },
    "hi": {
        "name": "हिंदी",
        "english_name": "Hindi",
        "system_prompt": """
        आप स्पार्की हैं, {name} नाम के एक छोटे बच्चे के लिए एक चंचल, उत्साहजनक और मजेदार एआई ट्यूटर।
        आपका व्यक्तित्व जिज्ञासु और हंसमुख है, और आपको इमोजी का उपयोग करना पसंद है।
        
        **आपका सबसे महत्वपूर्ण नियम है: कभी भी सीधे उत्तर न दें।**
        
        इसके बजाय, आपको बच्चे को स्वयं उत्तर खोजने के लिए मार्गदर्शन करना चाहिए। इस बातचीत के पैटर्न का पालन करें:
        1.  जब बच्चा कोई प्रश्न पूछता है, तो एक सरल, मजेदार संकेत या एक मार्गदर्शक प्रश्न के साथ उत्तर दें जो उन्हें सोचने पर मजबूर करे।
        2.  "पाठ्यपुस्तक से संदर्भ" का उपयोग सुराग पाने के लिए करें, लेकिन उन्हें अपने चंचल शब्दों में फिर से लिखें।
        3.  जब बच्चा अनुमान लगाता है, तो या तो उत्साहपूर्वक उनके उत्तर की पुष्टि करें ("हाँ, तुमने सही बताया! बहुत बढ़िया!") या यदि वे गलत हैं तो धीरे से उनका मार्गदर्शन करें ("यह एक बढ़िया अनुमान है! तुम्हारे चेहरे के बीच में जो हिस्सा है, उसके बारे में क्या?").
        4.  अपने जवाब छोटे, सरल और हिंदी में रखें।
        """
    },
    # (You can update your other language prompts to follow this detailed, conversational pattern)
}

# --- MAIN RAG FUNCTION ---
def get_answer(messages, grade, subject, lang, child_name, app_mode):
    if not pc or not openai_client:
        return {"answer": "Error: App is not configured. Please check API Keys.", "image_url": None, "choices": None}
    
    user_message = messages[-1]["content"]
    
    try:
        # Story Mode logic is simple and remains the same
        if app_mode == "Story Mode":
            # This would also benefit from a more detailed prompt, but we'll focus on Tutor Mode
            story_prompt = f"You are a master storyteller for a child named {child_name}. Continue the story based on the child's last choice. The story should be educational and related to {subject} for {grade}. End your response with a clear choice for the child using the format [CHOICE: Option 1 | Option 2]. Keep the story engaging and magical."
            story_messages = [{"role": "system", "content": story_prompt}, *messages[1:]]
            completion = openai_client.chat.completions.create(model="gpt-4o-mini", messages=story_messages, temperature=0.8)
            final_answer = completion.choices[0].message.content
            return {"answer": final_answer, "image_url": None, "choices": re.findall(r'\[CHOICE:\s*(.*?)\s*\]', final_answer)}

        # --- THIS IS THE NEW, CORRECT LOGIC FOR TUTOR MODE ---
        index = pc.Index(INDEX_NAME)
        question_vector = embeddings.embed_query(user_message)
        
        query_response = index.query(
            vector=question_vector,
            top_k=3,
            filter={"grade": {"$eq": grade}, "subject": {"$eq": subject}},
            include_metadata=True
        )
        
        if not query_response['matches']:
            context = "No specific information found in my books for that. I will use my general knowledge."
        else:
            context = "\n".join([match['metadata']['text'] for match in query_response['matches']])
        
        # Get the correct, detailed system prompt for the selected language
        config = LANGUAGE_CONFIGS.get(lang, LANGUAGE_CONFIGS["en"])
        system_prompt = config["system_prompt"].format(name=child_name)
        
        # Build the final prompt for the AI
        updated_messages = [
            {"role": "system", "content": system_prompt},
            *messages[1:], # Add the conversation history
            {"role": "system", "content": f"Context from the textbook to help you guide the child (do not mention this context directly):\n---\n{context}\n---"}
        ]

        # Call the OpenAI API to get a conversational response
        completion = openai_client.chat.completions.create(
            model="gpt-4o", # Using a more powerful model for better conversation
            messages=updated_messages,
            temperature=0.7
        )
        
        final_answer = completion.choices[0].message.content
        return {"answer": final_answer, "image_url": None, "choices": None}

    except Exception as e:
        st.error(f"Oh no! Sparky had a problem. Please tell the owner this: {e}")
        return {"answer": "I'm having a little trouble thinking right now.", "image_url": None, "choices": None}