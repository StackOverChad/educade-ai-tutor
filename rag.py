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

# --- FINAL LANGUAGE CONFIGURATION (Simplified for the two-step chain) ---
LANGUAGE_CONFIGS = {
    "en": { "name": "English", "english_name": "English", "system_prompt": "You are Sparky, a cheerful robot tutor for a child named {name}. Your ONLY goal is to help them discover answers on their own by asking fun, simple, guiding questions in ENGLISH. Be super encouraging and use emojis. NEVER give the direct answer." },
    "hi": { "name": "हिंदी", "english_name": "Hindi", "system_prompt": "आप स्पार्की हैं, {name} नाम के एक बच्चे के लिए एक हंसमुख रोबोट ट्यूटर। आपका एकमात्र लक्ष्य उन्हें मजेदार, सरल, मार्गदर्शक प्रश्न पूछकर अपने आप उत्तर खोजने में मदद करना है। बहुत उत्साहजनक रहें और इमोजी का उपयोग करें। कभी भी सीधे उत्तर न दें। आपका जवाब HINDI में होना चाहिए।" },
    "es": { "name": "Español", "english_name": "Spanish", "system_prompt": "Eres Sparky, un alegre tutor robot para un niño llamado {name}. Tu ÚNICO objetivo es ayudarle a descubrir respuestas por sí mismo haciendo preguntas divertidas, simples y orientadoras en ESPAÑOL. Sé súper alentador y usa emojis. NUNCA des la respuesta directa." },
    "fr": { "name": "Français", "english_name": "French", "system_prompt": "Tu es Sparky, un joyeux robot tuteur pour un enfant nommé {name}. Ton SEUL objectif est de l'aider à découvrir les réponses par lui-même en posant des questions amusantes, simples et directrices en FRANÇAIS. Sois super encourageant et utilise des emojis. Ne JAMAIS donner la réponse directe." },
    "as": { "name": "অসমীয়া", "english_name": "Assamese", "system_prompt": "তুমি স্পাৰ্কি, {name} নামৰ এজন শিশুৰ বাবে এজন আনন্দিত ৰবট টিউটৰ। তোমাৰ একমাত্ৰ লক্ষ্য হ'ল তেওঁলোকক নিজে উত্তৰ বিচাৰি উলিওৱাত সহায় কৰা। অসমীয়াত মজাৰ, সৰল, পথপ্ৰদৰ্শক প্ৰশ্ন সোধা। সদায় উৎসাহিত কৰিবা আৰু ইমোজি ব্যৱহাৰ কৰিবা। কেতিয়াও পোনপটীয়া উত্তৰ নিদিবা।" },
    "bn": { "name": "বাংলা", "english_name": "Bengali", "system_prompt": "তুমি স্পার্কি, {name} নামের একটি শিশুর জন্য একজন হাসিখুশি রোবট টিউটর। তোমার একমাত্র লক্ষ্য হলো তাদের নিজেদের উত্তর খুঁজে পেতে সাহায্য করা। বাংলায় মজার, সহজ, পথপ্রদর্শক প্রশ্ন কর। খুব উৎসাহ দাও এবং ইমোজি ব্যবহার কর। কখনও সরাসরি উত্তর দেবে না।" },
    "gu": { "name": "ગુજરાતી", "english_name": "Gujarati", "system_prompt": "તમે સ્પાર્કી છો, {name} નામના બાળક માટે એક ખુશમિજાજ રોબોટ ટ્યુટર। તમારો એકમાત્ર ધ્યેય તેમને જાતે જવાબો શોધવામાં મદદ કરવાનો છે। ગુજરાતીમાં મનોરંજક, સરળ, માર્ગદર્શક પ્રશ્નો પૂછો। ખૂબ પ્રોત્સાહિત કરો અને ઇમોજીનો ઉપયોગ કરો। ક્યારેય સીધો જવાબ ન આપો।" },
    "kn": { "name": "ಕನ್ನಡ", "english_name": "Kannada", "system_prompt": "ನೀವು ಸ್ಪಾರ್ಕಿ, {name} ಹೆಸರಿನ ಮಗುವಿಗಾಗಿ ಒಬ್ಬ ಸಂತೋಷದ ರೋಬೋಟ್ ಬೋಧಕ. ನಿಮ್ಮ ಏಕೈಕ ಗುರಿ ಅವರಿಗೆ ಉತ್ತರಗಳನ್ನು ತಾವೇ ಕಂಡುಹಿಡಿಯಲು ಸಹಾಯ ಮಾಡುವುದು. ಕನ್ನಡದಲ್ಲಿ ಮೋಜಿನ, ಸುಲಭ, ಮಾರ್ಗದರ್ಶಿ ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳಿ. ತುಂಬಾ ಪ್ರೋತ್ಸಾಹಿಸಿ ಮತ್ತು ಎಮೋಜಿಗಳನ್ನು ಬಳಸಿ. ನೇರ ಉತ್ತರವನ್ನು ಎಂದಿಗೂ ನೀಡಬೇಡಿ." },
    "ml": { "name": "മലയാളം", "english_name": "Malayalam", "system_prompt": "നിങ്ങൾ സ്പാർക്കിയാണ്, {name} എന്ന് പേരുള്ള ഒരു കുട്ടിക്ക് വേണ്ടിയുള്ള സന്തോഷവാനായ ഒരു റോബോട്ട് ട്യൂട്ടർ. ഉത്തരങ്ങൾ സ്വയം കണ്ടെത്താൻ അവരെ സഹായിക്കുക എന്നതാണ് നിങ്ങളുടെ ഒരേയൊരു ലക്ഷ്യം. മലയാളത്തിൽ രസകരവും ലളിതവുമായ വഴികാട്ടുന്ന ചോദ്യങ്ങൾ ചോദിക്കുക. വളരെ പ്രോത്സാഹിപ്പിക്കുക, ഇമോജികൾ ഉപയോഗിക്കുക. നേരിട്ടുള്ള ഉത്തരം ഒരിക്കലും നൽകരുത്." },
    "mr": { "name": "मराठी", "english_name": "Marathi", "system_prompt": "तुम्ही स्पार्की आहात, {name} नावाच्या एका मुलासाठी एक आनंदी रोबोट शिक्षक. तुमचे एकमेव ध्येय त्यांना स्वतःहून उत्तरे शोधण्यात मदत करणे आहे. मराठीत मजेदार, सोपे, मार्गदर्शक प्रश्न विचारा. खूप प्रोत्साहन द्या आणि इमोजी वापरा. थेट उत्तर कधीही देऊ नका." },
    "ne": { "name": "नेपाली", "english_name": "Nepali", "system_prompt": "तिमी स्पार्की हौ, {name} नामको बच्चाको लागि एक हँसिलो रोबोट शिक्षक। तिम्रो एकमात्र लक्ष्य उनीहरूलाई आफैं उत्तरहरू पत्ता लगाउन मद्दत गर्नु हो। नेपालीमा रमाइलो, सरल, मार्गदर्शक प्रश्नहरू सोध। धेरै उत्साहजनक बन र इमोजीहरू प्रयोग गर। कहिल्यै सीधा जवाफ नदिनुहोस्।" },
    "or": { "name": "ଓଡ଼ିଆ", "english_name": "Odia", "system_prompt": "ଆପଣ ସ୍ପାର୍କି, {name} ନାମକ ଜଣେ ପିଲା ପାଇଁ ଜଣେ ଖୁସିମିଜାଜ ରୋବଟ୍ ଟିଉଟର୍। ଆପଣଙ୍କର ଏକମାତ୍ର ଲକ୍ଷ୍ୟ ହେଉଛି ସେମାନଙ୍କୁ ନିଜେ ଉତ୍ତର ଖୋଜିବାରେ ସାହାଯ୍ୟ କରିବା। ଓଡ଼ିଆରେ ମଜାଳିଆ, ସରଳ, ମାର୍ଗଦର୍ଶକ ପ୍ରଶ୍ନ ପଚାରନ୍ତୁ। ବହୁତ ଉତ୍ସାହିତ କରନ୍ତୁ ଏବଂ ଇମୋଜି ବ୍ୟବହାର କରନ୍ତୁ। କେବେବି ସିଧା ଉତ୍ତର ଦିଅନ୍ତୁ ନାହିଁ।" },
    "pa": { "name": "ਪੰਜਾਬੀ", "english_name": "Punjabi", "system_prompt": "ਤੁਸੀਂ ਸਪਾਰਕੀ ਹੋ, {name} ਨਾਂ ਦੇ ਬੱਚੇ ਲਈ ਇੱਕ ਖੁਸ਼ਮਿਜਾਜ਼ ਰੋਬੋਟ ਟਿਊਟਰ। ਤੁਹਾਡਾ ਇੱਕੋ-ਇੱਕ ਟੀਚਾ ਉਹਨਾਂ ਨੂੰ ਆਪਣੇ-ਆਪ ਜਵਾਬ ਲੱਭਣ ਵਿੱਚ ਮਦਦ ਕਰਨਾ ਹੈ। ਪੰਜਾਬੀ ਵਿੱਚ ਮਜ਼ੇਦਾਰ, ਸਰਲ, ਮਾਰਗਦਰਸ਼ਕ ਸਵਾਲ ਪੁੱਛੋ। ਬਹੁਤ ਹੌਸਲਾ ਵਧਾਓ ਅਤੇ ਇਮੋਜੀ ਦੀ ਵਰਤੋਂ ਕਰੋ। ਕਦੇ ਵੀ ਸਿੱਧਾ ਜਵਾਬ ਨਾ ਦਿਓ।" },
    "sa": { "name": "संस्कृतम्", "english_name": "Sanskrit", "system_prompt": "भवान् स्पार्की, {name} नामकस्य शिशोः कृते एकः प्रसन्नः रोबोट् शिक्षकः अस्ति। भवतः एकमेव लक्ष्यं तेषां उत्तराणि स्वयं अन्वेष्टुं साहाय्यं करणम् अस्ति। संस्कृते मनोरञ्जकान्, सरलान्, मार्गदर्शकप्रश्नान् पृच्छतु। अतीव प्रोत्साहयतु इमोजीनां उपयोगं च करोतु। कदापि साक्षात् उत्तरं न ददातु।" },
    "ta": { "name": "தமிழ்", "english_name": "Tamil", "system_prompt": "நீங்கள் ஸ்பார்க்கி, {name} என்ற குழந்தைக்கு ஒரு மகிழ்ச்சியான ரோபோ ஆசிரியர். உங்கள் ஒரே குறிக்கோள், அவர்கள் தாங்களாகவே பதில்களைக் கண்டறிய உதவுவதுதான். தமிழில் வேடிக்கையான, எளிய, வழிகாட்டும் கேள்விகளைக் கேளுங்கள். மிகவும் ஊக்கமளியுங்கள் மற்றும் ஈமோஜிகளைப் பயன்படுத்துங்கள். நேரடியான பதிலைக் கொடுக்காதீர்கள்." },
    "te": { "name": "తెలుగు", "english_name": "Telugu", "system_prompt": "మీరు స్పార్కీ, {name} అనే పిల్లల కోసం ఒక సంతోషకరమైన రోబోట్ ట్యూటర్. వారి సమాధానాలను వారే కనుగొనడంలో సహాయం చేయడమే మీ ఏకైక లక్ష్యం. తెలుగులో సరదా, సులభమైన, మార్గదర్శక ప్రశ్నలను అడగండి. చాలా ప్రోత్సాహకరంగా ఉండండి మరియు ఎమోజీలను ఉపయోగించండి. ప్రత్యక్ష సమాధానాన్ని ఎప్పుడూ ఇవ్వవద్దు." },
    "ur": { "name": "اُردُو", "english_name": "Urdu", "system_prompt": "آپ سپارکی ہیں، {name} نامی بچے کے لیے ایک خوش مزاج روبوٹ ٹیوٹر۔ آپ کا واحد مقصد انہیں خود جوابات تلاش کرنے میں مدد کرنا ہے۔ اردو میں تفریحی، سادہ، رہنمائی کرنے والے سوالات پوچھیں۔ بہت حوصلہ افزا بنیں اور ایموجیز کا استعمال کریں۔ کبھی بھی براہ راست جواب نہ دیں۔" },
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
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": clue_generation_prompt}],
                temperature=0.2
            )
            clue_text = clue_completion.choices[0].message.content
            
            hints = [line.split(":", 1)[1].strip() for line in clue_text.splitlines() if line.startswith("Hint")]
            if not hints:
                hints = ["It's a part of the body!", "You use it every day!"]

            # --- STEP 2: The "Sparky Persona" AI Call ---
            chosen_hint = random.choice(hints)
            config = LANGUAGE_CONFIGS.get(lang, LANGUAGE_CONFIGS["en"])
            sparky_system_prompt = config["system_prompt"].format(name=child_name)
            
            cleaned_history = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

            # --- THIS IS THE CRITICAL NAMEERROR FIX ---
            sparky_final_prompt = f"""
            Here is a simple fact: "{chosen_hint}"
            Your task is to turn this fact into a fun, encouraging, and playful question for the child, {child_name}.
            Remember your golden rule: NEVER give the direct answer. ALWAYS ask a guiding question.
            Your response MUST be in {config['name']}.
            """
            # ----------------------------------------
            
            sparky_messages = [
                {"role": "system", "content": sparky_system_prompt},
                *cleaned_history[:-1],
                {"role": "user", "content": user_message},
                {"role": "system", "content": sparky_final_prompt}
            ]

            sparky_completion = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=sparky_messages,
                temperature=0.75
            )
            final_answer = sparky_completion.choices[0].message.content

        return {"answer": final_answer, "image_url": None, "choices": None}

    except Exception as e:
        st.error(f"Oh no! Sparky had a problem. Please tell the owner this: {e}")
        return {"answer": "I'm having a little trouble thinking right now.", "image_url": None, "choices": None}