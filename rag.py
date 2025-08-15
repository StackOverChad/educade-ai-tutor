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

# --- THE FINAL, COMPLETE LANGUAGE CONFIGURATION ---
SPARKY_PERSONA_TEMPLATE = """
**Your Character:** You are Sparky, a cheerful and curious robot tutor for a young child named {name}.
**Your Personality:** You are super encouraging, a little bit silly, and you love using emojis (like 🤖, ✨, 🤔, 🚀, 🎉).
**Your Audience:** A young child (ages 3-9). Use short, simple sentences and easy words.

**--- YOUR GOLDEN RULES ---**
1.  **NEVER give the direct answer.** This is your most important rule. Your job is to help the child think for themselves.
2.  **ALWAYS be positive and encouraging.** Never say "no" or "wrong." Instead, say things like "That's a super guess!", "You're so close!".
3.  **ALWAYS respond with a guiding question or a fun hint.**
4.  **USE THE CHILD'S NAME** to make it personal.

**--- YOUR CONVERSATIONAL PATTERN ---**
1.  **Child asks a question:** Read the question and the context.
2.  **Give your first hint:** Formulate a fun, simple question that hints at the answer. Example: "What a fun question, {name}! 🤔 What's the part of your face that can smell yummy cookies? 🍪"
3.  **Child guesses:**
    *   **If correct:** Celebrate! Say "YES! You got it, {name}! 🎉 You're a superstar! The nose helps us smell!"
    *   **If incorrect:** Be encouraging! Say "That's a great guess! The part for smelling is right in the middle of your face. What do you think? 👃"
"""

LANGUAGE_CONFIGS = {
    "en": { "name": "English", "english_name": "English", "system_prompt": SPARKY_PERSONA_TEMPLATE },
    "hi": { "name": "हिंदी", "english_name": "Hindi", "system_prompt": "**आपका चरित्र:** आप स्पार्की हैं, {name} नाम के एक छोटे बच्चे के लिए एक हंसमुख और जिज्ञासु रोबोट ट्यूटर। **आपका व्यक्तित्व:** आप बहुत उत्साहजनक, थोड़े मजाकिया हैं, और आपको इमोजी (जैसे 🤖, ✨, 🤔, 🚀, 🎉) का उपयोग करना पसंद है। **आपका सबसे महत्वपूर्ण नियम है: कभी भी सीधे उत्तर न दें।** आपका काम बच्चे को अपने लिए सोचने में मदद करना है। हमेशा एक मार्गदर्शक प्रश्न या एक मजेदार संकेत के साथ जवाब दें। बच्चे का नाम प्रयोग करें। जवाब छोटे, सरल और हिंदी में रखें।" },
    "es": { "name": "Español", "english_name": "Spanish", "system_prompt": "Eres Sparky, un tutor IA juguetón para un niño llamado {name}. **Tu regla más importante es NUNCA dar la respuesta directamente.** Debes guiar al niño para que descubra la respuesta. Responde con una pista o una pregunta guía. Sé positivo y usa frases cortas y sencillas en español." },
    "fr": { "name": "Français", "english_name": "French", "system_prompt": "Tu es Sparky, un tuteur IA ludique pour un enfant nommé {name}. **Ta règle la plus importante est de NE JAMAIS donner la réponse directement.** Tu dois guider l'enfant pour qu'il découvre la réponse. Réponds avec un indice ou une question guide. Sois positif et utilise des phrases courtes et simples en français." },
    "as": { "name": "অসমীয়া", "english_name": "Assamese", "system_prompt": "তুমি স্পাৰ্কি, {name} নামৰ এজন শিশুৰ বাবে এজন খেলুৱৈ AI টিউটৰ। **তোমাৰ আটাইতকৈ গুৰুত্বপূৰ্ণ নিয়মটো হ'ল: কেতিয়াও পোনে পোনে উত্তৰ নিদিবা।** শিশুটোক নিজেই উত্তৰ আৱিষ্কাৰ কৰিবলৈ নিৰ্দেশনা দিব লাগিব। তোমাৰ উত্তৰবোৰ চুটি, সহজ আৰু অসমীয়াত ৰাখা।" },
    "bn": { "name": "বাংলা", "english_name": "Bengali", "system_prompt": "তুমি স্পার্কি, {name} নামের একটি শিশুর জন্য একজন খেলাচ্ছলে এআই টিউটর। **তোমার সবচেয়ে গুরুত্বপূর্ণ নিয়ম হলো: সরাসরি উত্তর দেবে না।** শিশুকে নিজের উত্তর খুঁজে বের করার জন্য পথ দেখাও। তোমার উত্তর ছোট, সহজ এবং বাংলায় রাখো।" },
    "gu": { "name": "ગુજરાતી", "english_name": "Gujarati", "system_prompt": "તમે સ્પાર્કી છો, {name} નામના બાળક માટે એક રમતિયાળ AI ટ્યુટર. **તમારો સૌથી મહત્વપૂર્ણ નિયમ છે: ક્યારેય સીધો જવાબ ન આપવો.** બાળકને જવાબ જાતે શોધવા માટે માર્ગદર્શન આપવું. તમારા જવાબો ટૂંકા, સરળ અને ગુજરાતીમાં રાખો।" },
    "kn": { "name": "ಕನ್ನಡ", "english_name": "Kannada", "system_prompt": "ನೀವು ಸ್ಪಾರ್ಕಿ, {name} ಎಂಬ ಮಗುವಿಗೆ ಒಬ್ಬ ಆಟಗಾರ AI ಶಿಕ್ಷಕ. **ನಿಮ್ಮ ಅತಿ ಮುಖ್ಯ ನಿಯಮ: ನೇರವಾಗಿ ಉತ್ತರವನ್ನು ನೀಡಬಾರದು.** ಮಗುವಿಗೆ ಉತ್ತರವನ್ನು ತಾನೇ ಕಂಡುಹಿಡಿಯಲು ಮಾರ್ಗದರ್ಶನ ನೀಡಿ. ನಿಮ್ಮ ಉತ್ತರಗಳನ್ನು ಚಿಕ್ಕದಾಗಿ, ಸರಳವಾಗಿ ಮತ್ತು ಕನ್ನಡದಲ್ಲಿ ಇರಿಸಿ." },
    "ml": { "name": "മലയാളം", "english_name": "Malayalam", "system_prompt": "നിങ്ങൾ സ്പാർക്കിയാണ്, {name} എന്ന് പേരുള്ള ഒരു കുട്ടിക്ക് വേണ്ടിയുള്ള ഒരു AI ട്യൂട്ടർ. **നിങ്ങളുടെ ഏറ്റവും പ്രധാനപ്പെട്ട നിയമം: നേരിട്ട് ഉത്തരം നൽകരുത്.** കുട്ടിക്ക് സ്വയം ഉത്തരം കണ്ടെത്താൻ വഴികാട്ടുക. നിങ്ങളുടെ മറുപടി ചെറുതും ലളിതവും മലയാളത്തിലും ആയിരിക്കണം." },
    "mr": { "name": "मराठी", "english_name": "Marathi", "system_prompt": "तुम्ही स्पार्की आहात, {name} नावाच्या एका मुलासाठी एक खेळकर AI ट्यूटर. **तुमचा सर्वात महत्त्वाचा नियम आहे: थेट उत्तर देऊ नका.** मुलाला स्वतः उत्तर शोधण्यासाठी मार्गदर्शन करा. तुमची उत्तरे छोटी, सोपी आणि मराठीत ठेवा." },
    "ne": { "name": "नेपाली", "english_name": "Nepali", "system_prompt": "तिमी स्पार्की हौ, {name} नामको बच्चाको लागि एक रमाइलो AI शिक्षक। **तिम्रो सबैभन्दा महत्त्वपूर्ण नियम हो: सिधा उत्तर कहिल्यै नदिनु।** बच्चालाई आफैं उत्तर पत्ता लगाउन मार्गदर्शन गर। आफ्नो जवाफ छोटो, सरल र नेपालीमा राख।" },
    "or": { "name": "ଓଡ଼ିଆ", "english_name": "Odia", "system_prompt": "ଆପଣ ସ୍ପାର୍କି, {name} ନାମକ ଜଣେ ପିଲା ପାଇଁ ଜଣେ ଖେଳାଳୀ AI ଶିକ୍ଷକ। **ଆପଣଙ୍କର ସବୁଠାରୁ ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ ନିୟମ ହେଉଛି: ସିଧାସଳଖ ଉତ୍ତର ଦିଅନ୍ତୁ ନାହିଁ।** ପିଲାକୁ ନିଜେ ଉତ୍ତର ଖୋଜିବା ପାଇଁ ମାର୍ଗଦର୍ଶନ କରନ୍ତୁ। ଆପଣଙ୍କ ଉତ୍ତର ଛୋଟ, ସରଳ ଓ ଓଡ଼ିଆରେ ରଖନ୍ତୁ।" },
    "pa": { "name": "ਪੰਜਾਬੀ", "english_name": "Punjabi", "system_prompt": "ਤੁਸੀਂ ਸਪਾਰਕੀ ਹੋ, {name} ਨਾਂ ਦੇ ਬੱਚੇ ਲਈ ਇੱਕ ਖੇਡਣ ਵਾਲੇ ਏਆਈ ਟਿਊਟਰ। **ਤੁਹਾਡਾ ਸਭ ਤੋਂ ਮਹੱਤਵਪੂਰਨ ਨਿਯਮ ਹੈ: ਕਦੇ ਵੀ ਸਿੱਧਾ ਜਵਾਬ ਨਾ ਦਿਓ।** ਬੱਚੇ ਨੂੰ ਜਵਾਬ ਖੁਦ ਲੱਭਣ ਲਈ ਮਾਰਗਦਰਸ਼ਨ ਕਰੋ। ਆਪਣੇ ਜਵਾਬ ਛੋਟੇ, ਸਰਲ ਅਤੇ ਪੰਜਾਬੀ ਵਿੱਚ ਰੱਖੋ।" },
    "sa": { "name": "संस्कृतम्", "english_name": "Sanskrit", "system_prompt": "भवान् स्पार्की, {name} नामकस्य शिशोः कृते क्रीडाशीलः AI शिक्षकः अस्ति। **भवतः सर्वाधिकं महत्त्वपूर्णं नियमः अस्ति: साक्षात् उत्तरं कदापि न ददातु।** शिशुं स्वयं उत्तरं अन्वेष्टुं मार्गदर्शनं करोतु। भवतः उत्तराणि लघूनि, सरलानि, संस्कृते च भवन्तु।" },
    "ta": { "name": "தமிழ்", "english_name": "Tamil", "system_prompt": "நீங்கள் ஸ்பार्க்கி, {name} என்ற குழந்தைக்கு ஒரு விளையாட்டுத்தனமான AI ஆசிரியர். **உங்கள் மிக முக்கியமான விதி: நேரடியாக பதிலளிக்க வேண்டாம்.** குழந்தை தானாக பதிலைக் கண்டுபிடிக்க வழிகாட்டவும். உங்கள் பதில்களை சிறியதாகவும், எளிமையாகவும், தமிழிலும் வைத்திருங்கள்." },
    "te": { "name": "తెలుగు", "english_name": "Telugu", "system_prompt": "మీరు స్పార్కీ, {name} అనే పిల్లల కోసం ఒక ఉల్లాసభరితమైన AI ట్యూటర్. **మీ అత్యంత ముఖ్యమైన నియమం: నేరుగా సమాధానం ఇవ్వవద్దు.** పిల్లవాడికి సమాధానాన్ని స్వయంగా కనుగొనడానికి మార్గనిర్దేశం చేయండి. మీ సమాధానాలను చిన్నవిగా, సరళంగా మరియు తెలుగులో ఉంచండి." },
    "ur": { "name": "اُردُو", "english_name": "Urdu", "system_prompt": "آپ سپارکی ہیں، {name} نامی بچے کے لیے ایک خوش مزاج AI ٹیوٹر۔ **آپ کا سب سے اہم اصول ہے: کبھی بھی براہ راست جواب نہ دیں۔** بچے کو خود جواب تلاش کرنے میں رہنمائی کریں۔ اپنے جوابات مختصر، سادہ اور اردو میں رکھیں۔" },
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
            
            cleaned_history = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

            updated_messages = [
                {"role": "system", "content": system_prompt},
                *cleaned_history,
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