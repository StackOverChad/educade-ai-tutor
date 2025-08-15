import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from openai import OpenAI
import re

# --- HYBRID CREDENTIALS LOADER ---
# This block of code works BOTH locally and on Streamlit Cloud.
qdrant_url = None
qdrant_api_key = None
openai_api_key = None
error_message = None

try:
    # This is the primary method for Streamlit Cloud deployment
    qdrant_url = st.secrets["QDRANT_URL"]
    qdrant_api_key = st.secrets["QDRANT_API_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    if not qdrant_url or not qdrant_api_key or not openai_api_key:
        error_message = "One or more secrets are empty in the Streamlit Cloud dashboard."
except (KeyError, FileNotFoundError):
    # This is the fallback for local development
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not qdrant_url or not qdrant_api_key or not openai_api_key:
        error_message = "Could not find credentials in the .env file for local development."

# --- INITIALIZE CLIENTS ---
# Initialize clients only if all credentials were successfully loaded
if not error_message:
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    openai_client = OpenAI(api_key=openai_api_key)
else:
    qdrant_client = None
    openai_client = None

# --- CONSTANTS AND CONFIGS ---
#
# --- THE FINAL, CLEAN COLLECTION NAME ---
COLLECTION_NAME = "educade_final_db_v2"
#
#
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

LANGUAGE_CONFIGS = {
    "en": { "name": "English", "english_name": "English", "requires_translation": False, "system_prompt": "You are a playful and encouraging tutor for a child named {name}. Always be encouraging and cheerful. When the child asks a question, you respond with a hint or a question back to make them think and guess the answer interactively. Keep the conversation friendly, simple, and fun." },
    "hi": { "name": "हिंदी", "english_name": "Hindi", "requires_translation": True, "system_prompt": "आप {name} नाम के एक बच्चे के लिए एक चंचल और उत्साहजनक हिंदी ट्यूटर हैं। आप हमेशा देवनागरी लिपि में केवल हिंदी में उत्तर देते हैं। आप कभी भी सीधा उत्तर नहीं देते, बल्कि बच्चे को सोचने में मदद करने के लिए एक संकेत या एक मजेदार प्रश्न पूछते हैं।", "few_shot_user": "आसमान नीला क्यों है?", "few_shot_assistant": "वाह, क्या बढ़िया सवाल है! हमारे आकाश में एक जादू की परत है जो सूरज की रोशनी से नीला रंग बिखेर देती है। क्या आप अनुमान लगा सकते हैं? 🤔", "final_prompt_template": "मुख्य तथ्य '{fact}' का प्रयोग करके एक संकेत बनाएं। अब, ऊपर दिए गए उदाहरण का अनुसरण करते हुए, उपयोगकर्ता के प्रश्न का उत्तर एक मजेदार संकेत या नए प्रश्न के साथ शुद्ध हिंदी में दें।\nउपयोगकर्ता का प्रश्न: \"{question}\"" },
    # (All your other language configurations go here)
}

# --- HELPER FUNCTIONS ---
def should_generate_image(text_response):
    """Decide if a response is suitable for image generation and extract the keyword."""
    prompt = f"Extract a simple, visualizable concept (like 'a happy lion', 'the planet Saturn') from this text. If none, say 'None'. Text: \"{text_response}\""
    try:
        completion = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=15)
        result = completion.choices[0].message.content.strip()
        return None if result.lower() in ['none', ''] else result
    except: return None

def generate_illustration(keyword):
    """Generates an image using DALL-E 3 and returns the URL."""
    image_prompt = f"a cute cartoon drawing of {keyword}, for a child's storybook, vibrant colors, simple and friendly style"
    try:
        response = openai_client.images.generate(model="dall-e-3", prompt=image_prompt, size="1024x1024", quality="standard", n=1)
        return response.data[0].url
    except Exception as e:
        print(f"DALL-E error: {e}"); return None

# --- MAIN RAG FUNCTION ---
def get_answer(messages, grade, subject, lang, child_name, app_mode):
    """
    Main function to get a response from the AI.
    Includes robust error handling for configuration and database issues.
    """
    # First, check if clients were initialized correctly.
    if error_message or not qdrant_client or not openai_client:
        st.error(f"Configuration Error: {error_message or 'Clients could not be initialized.'}")
        return {"answer": "I can't connect to my brain right now. Please tell my owner to check the API Keys and Secrets.", "image_url": None, "choices": None}

    user_message = messages[-1]["content"]
    final_answer, image_url, choices = "", None, None

    if app_mode == "Story Mode":
        story_prompt = f"You are a master storyteller for a child named {child_name}. Continue the story based on the child's last choice. The story should be educational and related to {subject} for {grade}. End your response with a clear choice for the child using the format [CHOICE: Option 1 | Option 2]. Keep the story engaging and magical."
        story_messages = [{"role": "system", "content": story_prompt}, *messages[1:]]
        completion = openai_client.chat.completions.create(model="gpt-4o-mini", messages=story_messages, temperature=0.8)
        final_answer = completion.choices[0].message.content
    else: # Tutor Mode
        try:
            question_vector = embeddings.embed_query(user_message)
            search_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=question_vector,
                limit=3,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="grade", match=models.MatchValue(value=grade)),
                        models.FieldCondition(key="subject", match=models.MatchValue(value=subject)),
                    ]
                )
            )
            if not search_results:
                context = "No specific information found in my books for that. I'll use my general knowledge."
            else:
                context = "\n".join([hit.payload.get("text", "") for hit in search_results])
        
        except UnexpectedResponse:
            st.error("Oh no! Sparky's memory bank (database collection) seems to be missing or empty. Please ask the website owner to re-ingest the learning materials.")
            return {"answer": "I can't seem to access my knowledge right now. Please tell my owner to check the database and re-upload the book data!", "image_url": None, "choices": None}
        except Exception as e:
            st.error(f"An unexpected database error occurred. This might be a connection issue. Please try again. Error: {e}")
            return {"answer": "I'm having a little trouble thinking right now. Please try again in a moment.", "image_url": None, "choices": None}

        config = LANGUAGE_CONFIGS.get(lang, LANGUAGE_CONFIGS["en"]).copy()
        config["system_prompt"] = config["system_prompt"].format(name=child_name)
        
        if config.get("requires_translation", False):
            extractor_prompt = f"Extract the single keyword that answers the question from the context.\nQuestion: \"{user_message}\"\nContext: \"{context}\"\nKeyword:"
            extractor_completion = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": extractor_prompt}], temperature=0.0, max_tokens=10)
            extracted_fact_en = extractor_completion.choices[0].message.content.strip() or "information"
            translator_prompt = f"Translate '{extracted_fact_en}' into {config['name']}. Output only the translation."
            translator_completion = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": translator_prompt}], temperature=0.0, max_tokens=20)
            translated_fact = translator_completion.choices[0].message.content.strip() or ""
            generator_messages = [{"role": "system", "content": config["system_prompt"]}, {"role": "user", "content": config["few_shot_user"]}, {"role": "assistant", "content": config["few_shot_assistant"]}, {"role": "user", "content": config["final_prompt_template"].format(fact=translated_fact, question=user_message)}]
            final_completion = openai_client.chat.completions.create(model="gpt-4o-mini", messages=generator_messages, temperature=0.7)
            final_answer = final_completion.choices[0].message.content
        else:
            updated_messages = [{"role": "system", "content": config["system_prompt"]}, *messages[1:], {"role": "system", "content": f"Context:\n{context}"}]
            completion = openai_client.chat.completions.create(model="gpt-4o-mini", messages=updated_messages, temperature=0.7)
            final_answer = completion.choices[0].message.content

        image_keyword = should_generate_image(final_answer)
        if image_keyword: image_url = generate_illustration(image_keyword)
    
    choice_match = re.search(r'\[CHOICE:\s*(.*?)\s*\]', final_answer)
    if choice_match:
        final_answer = final_answer.replace(choice_match.group(0), "").strip()
        choices = [choice.strip() for choice in choice_match.group(1).split('|')]

    return {"answer": final_answer, "image_url": image_url, "choices": choices}