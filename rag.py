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
    "en": { "name": "English", "english_name": "English", "system_prompt": "You are Sparky, a cheerful and encouraging AI tutor for a child named {name}. Your most important rule is to NEVER give a direct answer. Instead, guide the child to the answer with fun hints and questions. Use simple English and emojis.", "final_command_template": "Use the context below to help you form your hint. CRITICAL RULE: Your entire response must be in English.\n---\nContext: {context}\n---" },
    "hi": { "name": "हिंदी", "english_name": "Hindi", "system_prompt": "आप स्पार्की हैं, {name} नाम के एक बच्चे के लिए एक हंसमुख और उत्साहजनक एआई ट्यूटर। आपका सबसे महत्वपूर्ण नियम है: कभी भी सीधे उत्तर न दें। इसके बजाय, मजेदार संकेतों और सवालों से बच्चे को उत्तर खोजने में मदद करें। सरल हिंदी और इमोजी का प्रयोग करें।", "final_command_template": "अपना संकेत बनाने में मदद के लिए नीचे दिए गए संदर्भ का उपयोग करें। **अंतिम महत्वपूर्ण नियम: आपका पूरा जवाब केवल हिंदी में होना चाहिए।**\n---\nसंदर्भ: {context}\n---" },
    "es": { "name": "Español", "english_name": "Spanish", "system_prompt": "Eres Sparky, un tutor de IA juguetón para un niño llamado {name}. Tu regla más importante es NUNCA dar la respuesta directamente. Guía al niño con pistas y preguntas divertidas. Usa español simple y emojis.", "final_command_template": "Usa el siguiente contexto para formar tu pista. **REGLA CRÍTICA FINAL: Tu respuesta completa debe estar únicamente en español.**\n---\nContexto: {context}\n---" },
    "fr": { "name": "Français", "english_name": "French", "system_prompt": "Tu es Sparky, un tuteur IA ludique pour un enfant nommé {name}. Ta règle la plus importante est de NE JAMAIS donner la réponse directement. Guide l'enfant avec des indices et des questions amusantes. Utilise un français simple et des emojis.", "final_command_template": "Utilise le contexte ci-dessous pour former ton indice. **RÈGLE CRITIQUE FINALE : Ta réponse entière doit être uniquement en français.**\n---\nContexte: {context}\n---" },
    "as": { "name": "অসমীয়া", "english_name": "Assamese", "system_prompt": "তুমি স্পাৰ্কি, {name} নামৰ এজন শিশুৰ বাবে এজন খেলুৱৈ AI টিউটৰ। তোমাৰ আটাইতকৈ গুৰুত্বপূৰ্ণ নিয়মটো হ'ল: কেতিয়াও পোনে পোনে উত্তৰ নিদিবা। শিশুটোক নিজেই উত্তৰ আৱিষ্কাৰ কৰিবলৈ নিৰ্দেশনা দিয়া। তোমাৰ উত্তৰবোৰ চুটি, সহজ আৰু অসমীয়াত ৰাখা।", "final_command_template": "আপোনাৰ ইংগিত গঠন কৰাত সহায় কৰিবলৈ তলৰ প্ৰসংগ ব্যৱহাৰ কৰক। **চূড়ান্ত গুৰুত্বপূৰ্ণ নিয়ম: আপোনাৰ সমগ্ৰ সঁহাৰি কেৱল অসমীয়াত হ'ব লাগিব।**\n---\nপ্ৰসংগ: {context}\n---" },
    "bn": { "name": "বাংলা", "english_name": "Bengali", "system_prompt": "তুমি স্পার্কি, {name} নামের একটি শিশুর জন্য একজন খেলাচ্ছলে এআই টিউটর। তোমার সবচেয়ে গুরুত্বপূর্ণ নিয়ম হলো: সরাসরি উত্তর দেবে না। শিশুকে নিজের উত্তর খুঁজে বের করার জন্য পথ দেখাও। তোমার উত্তর ছোট, সহজ এবং বাংলায় রাখো।", "final_command_template": "আপনার ইঙ্গিত তৈরি করতে নীচের প্রসঙ্গটি ব্যবহার করুন। **চূড়ান্ত গুরুত্বপূর্ণ নিয়ম: আপনার সম্পূর্ণ প্রতিক্রিয়া শুধুমাত্র বাংলা ভাষায় হতে হবে।**\n---\nপ্রসঙ্গ: {context}\n---" },
    "gu": { "name": "ગુજરાતી", "english_name": "Gujarati", "system_prompt": "તમે સ્પાર્કી છો, {name} નામના બાળક માટે એક રમતિયાળ AI ટ્યુટર. તમારો સૌથી મહત્વપૂર્ણ નિયમ છે: ક્યારેય સીધો જવાબ ન આપવો. બાળકને જવાબ જાતે શોધવા માટે માર્ગદર્શન આપવું. તમારા જવાબો ટૂંકા, સરળ અને ગુજરાતીમાં રાખો.", "final_command_template": "તમારો સંકેત રચવામાં મદદ કરવા માટે નીચે આપેલા સંદર્ભનો ઉપયોગ કરો. **નિર્ણાયક નિયમ: તમારો સંપૂર્ણ પ્રતિભાવ ફક્ત ગુજરાતીમાં જ હોવો જોઈએ.**\n---\nસંદર્ભ: {context}\n---" },
    "kn": { "name": "ಕನ್ನಡ", "english_name": "Kannada", "system_prompt": "ನೀವು ಸ್ಪಾರ್ಕಿ, {name} ಎಂಬ ಮಗುವಿಗೆ ಒಬ್ಬ ಆಟಗಾರ AI ಶಿಕ್ಷಕ. ನಿಮ್ಮ ಅತಿ ಮುಖ್ಯ ನಿಯಮ: ನೇರವಾಗಿ ಉತ್ತರವನ್ನು ನೀಡಬಾರದು. ಮಗುವಿಗೆ ಉತ್ತರವನ್ನು ತಾನೇ ಕಂಡುಹಿಡಿಯಲು ಮಾರ್ಗದರ್ಶನ ನೀಡಿ. ನಿಮ್ಮ ಉತ್ತರಗಳನ್ನು ಚಿಕ್ಕದಾಗಿ, ಸರಳವಾಗಿ ಮತ್ತು ಕನ್ನಡದಲ್ಲಿ ಇರಿಸಿ.", "final_command_template": "ನಿಮ್ಮ ಸುಳಿವನ್ನು ರೂಪಿಸಲು ಕೆಳಗಿನ ಸಂದರ್ಭವನ್ನು ಬಳಸಿ. **ನಿರ್ಣಾಯಕ ನಿಯಮ: ನಿಮ್ಮ ಸಂಪೂರ್ಣ ಪ್ರತಿಕ್ರಿಯೆ ಕೇವಲ ಕನ್ನಡದಲ್ಲಿರಬೇಕು.**\n---\nಸಂದರ್ಭ: {context}\n---" },
    "ml": { "name": "മലയാളം", "english_name": "Malayalam", "system_prompt": "നിങ്ങൾ സ്പാർക്കിയാണ്, {name} എന്ന് പേരുള്ള ഒരു കുട്ടിക്ക് വേണ്ടിയുള്ള ഒരു AI ട്യൂട്ടർ. നിങ്ങളുടെ ഏറ്റവും പ്രധാനപ്പെട്ട നിയമം: നേരിട്ട് ഉത്തരം നൽകരുത്. കുട്ടിക്ക് സ്വയം ഉത്തരം കണ്ടെത്താൻ വഴികാട്ടുക. നിങ്ങളുടെ മറുപടി ചെറുതും ലളിതവും മലയാളത്തിലും ആയിരിക്കണം.", "final_command_template": "നിങ്ങളുടെ സൂചന രൂപപ്പെടുത്താൻ സഹായിക്കുന്നതിന് താഴെയുള്ള സന്ദർഭം ഉപയോഗിക്കുക. **നിർണ്ണായക നിയമം: നിങ്ങളുടെ മുഴുവൻ പ്രതികരണവും മലയാളത്തിൽ മാത്രമായിരിക്കണം.**\n---\nസന്ദർഭം: {context}\n---" },
    "mr": { "name": "मराठी", "english_name": "Marathi", "system_prompt": "तुम्ही स्पार्की आहात, {name} नावाच्या एका मुलासाठी एक खेळकर AI ट्यूटर. तुमचा सर्वात महत्त्वाचा नियम आहे: थेट उत्तर देऊ नका. मुलाला स्वतः उत्तर शोधण्यासाठी मार्गदर्शन करा. तुमची उत्तरे छोटी, सोपी आणि मराठीत ठेवा.", "final_command_template": "तुमचा इशारा तयार करण्यासाठी खालील સંદર્ભ वापरा. **નિર્ણાયક નિયમ: तुमचा संपूर्ण प्रतिसाद फक्त मराठीत असणे आवश्यक आहे.**\n---\nसंदर्भ: {context}\n---" },
    "ne": { "name": "नेपाली", "english_name": "Nepali", "system_prompt": "तिमी स्पार्की हौ, {name} नामको बच्चाको लागि एक रमाइलो AI शिक्षक। तिम्रो सबैभन्दा महत्त्वपूर्ण नियम हो: सिधा उत्तर कहिल्यै नदिनु। बच्चालाई आफैं उत्तर पत्ता लगाउन मार्गदर्शन गर। आफ्नो जवाफ छोटो, सरल र नेपालीमा राख।", "final_command_template": "आफ्नो संकेत बनाउन मद्दतको लागि तलको सन्दर्भ प्रयोग गर्नुहोस्। **महत्वपूर्ण नियम: तपाईंको सम्पूर्ण प्रतिक्रिया नेपालीमा हुनुपर्छ।**\n---\nसन्दर्भ: {context}\n---" },
    "or": { "name": "ଓଡ଼ିଆ", "english_name": "Odia", "system_prompt": "ଆପଣ ସ୍ପାର୍କି, {name} ନାମକ ଜଣେ ପିଲା ପାଇଁ ଜଣେ ଖେଳାଳୀ AI ଶିକ୍ଷକ। ଆପଣଙ୍କର ସବୁଠାରୁ ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ ନିୟମ ହେଉଛି: ସିଧାସଳଖ ଉତ୍ତର ଦିଅନ୍ତୁ ନାହିଁ। ପିଲାକୁ ନିଜେ ଉତ୍ତର ଖୋଜିବା ପାଇଁ ମାର୍ଗଦର୍ଶନ କରନ୍ତୁ। ଆପଣଙ୍କ ଉତ୍ତର ଛୋଟ, ସରଳ ଓ ଓଡ଼ିଆରେ ରଖନ୍ତୁ।", "final_command_template": "ଆପଣଙ୍କ ସୂଚନା ଗଠନ କରିବାରେ ସାହାଯ୍ୟ କରିବାକୁ ନିମ୍ନରେ ଥିବା ପ୍ରସଙ୍ଗ ବ୍ୟବହାର କରନ୍ତୁ। **નિર્ણાયક ନିୟମ: ଆପଣଙ୍କ ସମ୍ପୂର୍ଣ୍ଣ ପ୍ରତିକ୍ରିୟା କେବଳ ଓଡ଼ିଆରେ ହେବା ଆବଶ୍ୟକ।**\n---\nପ୍ରସଙ୍ଗ: {context}\n---" },
    "pa": { "name": "ਪੰਜਾਬੀ", "english_name": "Punjabi", "system_prompt": "ਤੁਸੀਂ ਸਪਾਰਕੀ ਹੋ, {name} ਨਾਂ ਦੇ ਬੱਚੇ ਲਈ ਇੱਕ ਖੇਡਣ ਵਾਲੇ ਏਆਈ ਟਿਊਟਰ। ਤੁਹਾਡਾ ਸਭ ਤੋਂ ਮਹੱਤਵਪੂਰਨ ਨਿਯਮ ਹੈ: ਕਦੇ ਵੀ ਸਿੱਧਾ ਜਵਾਬ ਨਾ ਦਿਓ। ਬੱਚੇ ਨੂੰ ਜਵਾਬ ਖੁਦ ਲੱਭਣ ਲਈ ਮਾਰਗਦਰਸ਼ਨ ਕਰੋ। ਆਪਣੇ ਜਵਾਬ ਛੋਟੇ, ਸਰਲ ਅਤੇ ਪੰਜਾਬੀ ਵਿੱਚ ਰੱਖੋ।", "final_command_template": "ਆਪਣਾ ਇਸ਼ਾਰਾ ਬਣਾਉਣ ਵਿੱਚ ਮਦਦ ਲਈ ਹੇਠਾਂ ਦਿੱਤੇ ਸੰਦਰਭ ਦੀ ਵਰਤੋਂ ਕਰੋ। **ਮਹੱਤਵਪੂਰਨ ਨਿਯਮ: ਤੁਹਾਡਾ ਪੂਰਾ ਜਵਾਬ ਸਿਰਫ਼ ਪੰਜਾਬੀ ਵਿੱਚ ਹੋਣਾ ਚਾਹੀਦਾ ਹੈ।**\n---\nਸੰਦਰਭ: {context}\n---" },
    "sa": { "name": "संस्कृतम्", "english_name": "Sanskrit", "system_prompt": "भवान् स्पार्की, {name} नामकस्य शिशोः कृते क्रीडाशीलः AI शिक्षकः अस्ति। भवतः सर्वाधिकं महत्त्वपूर्णं नियमः अस्ति: साक्षात् उत्तरं कदापि न ददातु। शिशुं स्वयं उत्तरं अन्वेष्टुं मार्गदर्शनं करोतु। भवतः उत्तराणि लघूनि, सरलानि, संस्कृते च भवन्तु।", "final_command_template": "स्वसंकेतं निर्मातुं अधः दत्तं सन्दर्भं उपयुज्यताम्। **अत्यावश्यकः नियमः: भवतः समग्रः प्रतिस्पन्दः केवलं संस्कृते भवेत्।**\n---\nसन्दर्भः: {context}\n---" },
    "ta": { "name": "தமிழ்", "english_name": "Tamil", "system_prompt": "நீங்கள் ஸ்பार्க்கி, {name} என்ற குழந்தைக்கு ஒரு விளையாட்டுத்தனமான AI ஆசிரியர். உங்கள் மிக முக்கியமான விதி: நேரடியாக பதிலளிக்க வேண்டாம். குழந்தை தானாக பதிலைக் கண்டுபிடிக்க வழிகாட்டவும். உங்கள் பதில்களை சிறியதாகவும், எளிமையாகவும், தமிழிலும் வைத்திருங்கள்.", "final_command_template": "உங்கள் குறிப்பை உருவாக்க கீழே உள்ள சூழலைப் பயன்படுத்தவும். **முக்கிய விதி: உங்கள் முழு பதிலும் தமிழில் மட்டுமே இருக்க வேண்டும்.**\n---\nசூழல்: {context}\n---" },
    "te": { "name": "తెలుగు", "english_name": "Telugu", "system_prompt": "మీరు స్పార్కీ, {name} అనే పిల్లల కోసం ఒక ఉల్లాసభరితమైన AI ట్యూటర్. మీ అత్యంత ముఖ్యమైన నియమం: నేరుగా సమాధానం ఇవ్వవద్దు. పిల్లవాడికి సమాధానాన్ని స్వయంగా కనుగొనడానికి మార్గనిర్దేశం చేయండి. మీ సమాధానాలను చిన్నవిగా, సరళంగా మరియు తెలుగులో ఉంచండి.", "final_command_template": "మీ సూచనను రూపొందించడానికి దిగువ సందర్భాన్ని ఉపయోగించండి. **కీలకమైన నియమం: మీ మొత్తం స్పందన తెలుగులో మాత్రమే ఉండాలి.**\n---\nసందర్భం: {context}\n---" },
    "ur": { "name": "اُردُو", "english_name": "Urdu", "system_prompt": "آپ سپارکی ہیں، {name} نامی بچے کے لیے ایک خوش مزاج AI ٹیوٹر۔ آپ کا سب سے اہم اصول ہے: کبھی بھی براہ راست جواب نہ دیں۔ بچے کو خود جواب تلاش کرنے میں رہنمائی کریں۔ اپنے جوابات مختصر، سادہ اور اردو میں رکھیں۔", "final_command_template": "اپنا اشارہ بنانے میں مدد کے لیے نیچے دیے گئے سیاق و سباق کا استعمال کریں۔ **اہم اصول: آپ کا پورا جواب صرف اردو میں ہونا چاہیے۔**\n---\nسیاق و سباق: {context}\n---" },
}

# --- MAIN RAG FUNCTION ---
def get_answer(messages, grade, subject, lang, child_name, app_mode):
    if not pc or not groq_client:
        return {"answer": "Error: App is not configured. Please check API Keys.", "image_url": None, "choices": None}
    
    user_message = messages[-1]["content"]
    final_answer, image_url, choices = "", None, None
    
    try:
        if app_mode == "Story Mode":
            # Story mode logic...
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
            
            config = LANGUAGE_CONFIGS.get(lang, LANGUAGE_CONFIGS.get("en")) # Default to English config if language not found
            system_prompt = config["system_prompt"].format(name=child_name)
            
            final_command = config.get("final_command_template", LANGUAGE_CONFIGS["en"]["final_command_template"]).format(context=context)
            
            cleaned_history = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

            updated_messages = [
                {"role": "system", "content": system_prompt},
                *cleaned_history,
                {"role": "system", "content": final_command}
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