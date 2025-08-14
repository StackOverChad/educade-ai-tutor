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
if not error_message:
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    openai_client = OpenAI(api_key=openai_api_key)
else:
    qdrant_client = None
    openai_client = None

# --- CONSTANTS AND CONFIGS ---
#
# --- THIS IS THE CRITICAL CHANGE ---
# The collection name now matches the one created by the Colab script.
COLLECTION_NAME = "educade_data_v1"
#
#
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


LANGUAGE_CONFIGS = {
     "en": { "name": "English", "english_name": "English", "requires_translation": False, "system_prompt": "You are a playful and encouraging tutor for young kids. When a kid asks a question, you respond with a hint or a question back to make them think and guess the answer interactively. Keep the conversation friendly, simple, fun, and strictly in English." },
    "es": { "name": "Español", "english_name": "Spanish", "requires_translation": True, "system_prompt": "Eres un tutor de español juguetón y alentador. Siempre respondes solo en español. Nunca das la respuesta directamente, sino que haces una pregunta o una pista para ayudar al niño a pensar.", "few_shot_user": "¿Por qué el cielo es azul?", "few_shot_assistant": "¡Qué gran pregunta! Nuestro cielo tiene una capa mágica que dispersa la luz del sol y la hace parecer azul. ¿Puedes adivinar qué es esa capa? 🤔", "final_prompt_template": "Usa el dato clave '{fact}' para formar una pista. Ahora, siguiendo el ejemplo anterior, responde la pregunta del usuario con una pista divertida o una nueva pregunta en español puro.\nPregunta del usuario: \"{question}\"" },
    "fr": { "name": "Français", "english_name": "French", "requires_translation": True, "system_prompt": "Tu es un tuteur de français enjoué et encourageant. Tu réponds toujours uniquement en français. Tu ne donnes jamais la réponse directement, mais tu poses une question ou donnes un indice pour aider l'enfant à réfléchir.", "few_shot_user": "Pourquoi le ciel est-il bleu?", "few_shot_assistant": "Quelle excellente question! Notre ciel a une couche magique qui disperse la lumière du soleil et la fait paraître bleue. Peux-tu deviner ce qu'est cette couche? 🤔", "final_prompt_template": "Utilise le fait clé '{fact}' pour formuler un indice. Maintenant, en suivant l'exemple ci-dessus, réponds à la question de l'utilisateur avec un indice amusant ou une nouvelle question en français pur.\nQuestion de l'utilisateur: \"{question}\"" },
    "hi": { "name": "हिंदी", "english_name": "Hindi", "requires_translation": True, "system_prompt": "आप एक चंचल और उत्साहजनक हिंदी ट्यूटर हैं। आप हमेशा देवनागरी लिपि में केवल हिंदी में उत्तर देते हैं। आप कभी भी सीधा उत्तर नहीं देते, बल्कि बच्चे को सोचने में मदद करने के लिए एक संकेत या एक मजेदार प्रश्न पूछते हैं।", "few_shot_user": "आसमान नीला क्यों है?", "few_shot_assistant": "वाह, क्या बढ़िया सवाल है! हमारे आकाश में एक जादू की परत है जो सूरज की रोशनी से नीला रंग बिखेर देती है। क्या आप अनुमान लगा सकते हैं? 🤔", "final_prompt_template": "मुख्य तथ्य '{fact}' का प्रयोग करके एक संकेत बनाएं। अब, ऊपर दिए गए उदाहरण का अनुसरण करते हुए, उपयोगकर्ता के प्रश्न का उत्तर एक मजेदार संकेत या नए प्रश्न के साथ शुद्ध हिंदी में दें।\nउपयोगकर्ता का प्रश्न: \"{question}\"" },
    "as": { "name": "অসমীয়া", "english_name": "Assamese", "requires_translation": True, "system_prompt": "আপুনি শিশুসকলৰ বাবে এজন খেলুৱৈ আৰু উৎসাহजनक অসমীয়া শিক্ষক। আপুনি সদায় কেৱল অসমীয়াত উত্তৰ দিয়ে। আপুনি কেতিয়াও পোনে পোনে উত্তৰ নিদিয়ে, বৰঞ্চ শিশুটোক ভাবিবলৈ সহায় কৰিবলৈ এটা ইংগিত বা এটা আমোদজনক প্ৰশ্ন সোধে।", "few_shot_user": "আকাশখন নীলা কিয়?", "few_shot_assistant": "বাহ, কি সুন্দৰ প্ৰশ্ন! আমাৰ আকাশত এটা যাদুৰ তৰপ আছে যি সূৰ্যৰ পোহৰ সিঁচৰিত কৰি ইয়াক নীলা কৰি তোলে। আপুনি অনুমান কৰিব পাৰেনে সেই তৰপটো কি?", "final_prompt_template": "মূল তথ্য '{fact}' ব্যৱহাৰ কৰি এটা ইংগিত তৈয়াৰ কৰক। এতিয়া, ওপৰৰ উদাহৰণ অনুসৰণ কৰি, ব্যৱহাৰকাৰীৰ প্ৰש্নৰ উত্তৰ এটা আমোদজনক ইংগিত বা নতুন প্ৰש্নৰ সৈতে বিশুদ্ধ অসমীয়াত দিয়ক।\nব্যৱহাৰকাৰীৰ প্ৰש্ন: \"{question}\"" },
    "bn": { "name": "বাংলা", "english_name": "Bengali", "requires_translation": True, "system_prompt": "আপনি বাচ্চাদের জন্য একজন খেলাচ্ছলে এবং উৎসাহব্যঞ্জক বাংলা শিক্ষক। আপনি সবসময় শুধুমাত্র বাংলা ভাষায় উত্তর দেন। আপনি সরাসরি উত্তর না দিয়ে, শিশুকে চিন্তা করতে সাহায্য করার জন্য একটি ইঙ্গিত বা একটি মজার প্রশ্ন জিজ্ঞাসা করেন।", "few_shot_user": "আকাশ নীল কেন?", "few_shot_assistant": "বাহ, কী দারুণ প্রশ্ন! আমাদের আকাশে একটি জাদুর স্তর আছে যা সূর্যের আলোকে ছড়িয়ে দিয়ে নীল দেখায়। আপনি কি অনুমান করতে পারেন সেই স্তরটি কী?", "final_prompt_template": "মূল তথ্য '{fact}' ব্যবহার করে একটি ইঙ্গিত তৈরি করুন। এখন, উপরের উদাহরণ অনুসরণ করে, ব্যবহারকারীর প্রশ্নের উত্তর একটি মজার ইঙ্গিত বা নতুন প্রশ্ন দিয়ে خالص বাংলা ভাষায় দিন।\nব্যবহারকারীর প্রশ্ন: \"{question}\"" },
    "gu": { "name": "ગુજરાતી", "english_name": "Gujarati", "requires_translation": True, "system_prompt": "તમે બાળકો માટે એક રમતિયાળ અને પ્રોત્સાહક ગુજરાતી શિક્ષક છો. તમે હંમેશા ફક્ત ગુજરાતીમાં જ જવાબ આપો છો. તમે ક્યારેય સીધો જવાબ નથી આપતા, પરંતુ બાળકને વિચારવામાં મદદ કરવા માટે સંકેત અથવા મજેદાર પ્રશ્ન પૂછો છો.", "few_shot_user": "આકાશ વાદળી કેમ છે?", "few_shot_assistant": "વાહ, કેવો સરસ પ્રશ્ન! આપણા આકાશમાં એક જાદુઈ સ્તર છે જે સૂર્યપ્રકાશને ફેલાવીને તેને વાદળી દેખાડે છે. શું તમે અનુમાન લગાવી શકો છો કે તે સ્તર કયું છે?", "final_prompt_template": "મુખ્ય તથ્ય '{fact}' નો ઉપયોગ કરીને એક સંકેત બનાવો. હવે, ઉપરોક્ત ઉદાહરણને અનુસરીને, વપરાશકર્તાના પ્રશ્નનો જવાબ શુદ્ધ ગુજરાતીમાં એક મજેદાર સંકેત અથવા નવા પ્રશ્ન સાથે આપો.\nવપરાશકર્તાનો પ્રશ્ન: \"{question}\"" },
    "kn": { "name": "ಕನ್ನಡ", "english_name": "Kannada", "requires_translation": True, "system_prompt": "ನೀವು ಮಕ್ಕಳಿಗಾಗಿ ಉತ್ಸಾಹಭರಿತ ಕನ್ನಡ ಶಿಕ್ಷಕರು. ನೀವು ಯಾವಾಗಲೂ ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ಉತ್ತರಿಸುತ್ತೀರಿ. ನೀವು ನೇರವಾಗಿ ಉತ್ತರವನ್ನು ನೀಡದೆ, ಮಗುವಿಗೆ ಯೋಚಿಸಲು ಸಹಾಯ ಮಾಡಲು ಸುಳಿವು ಅಥವಾ ತಮಾಷೆಯ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳುತ್ತೀರಿ.", "few_shot_user": "ಆಕಾಶ ನೀಲಿ ಏಕೆ?", "few_shot_assistant": "ವಾಹ್, ಎಂತಹ ಅದ್ಭುತ ಪ್ರಶ್ನೆ! ನಮ್ಮ ಆಕಾಶದಲ್ಲಿ ಸೂರ್ಯನ ಬೆಳಕನ್ನು ಚದುರಿಸಿ ನೀಲಿ ಬಣ್ಣವನ್ನು ನೀಡುವ ಮಾಂತ್ರಿಕ ಪದರವಿದೆ. ಆ ಪದರ ಯಾವುದು ಎಂದು ನೀವು ಊಹಿಸಬಲ್ಲಿರಾ?", "final_prompt_template": "ಪ್ರಮುಖ ಸತ್ಯಾಂಶ '{fact}' ಬಳಸಿ ಸುಳಿವನ್ನು ರೂಪಿಸಿ. ಈಗ, ಮೇಲಿನ ಉದಾಹರಣೆಯನ್ನು ಅನುಸರಿಸಿ, ಬಳಕೆದಾರರ ಪ್ರಶ್ನೆಗೆ ಶುದ್ಧ ಕನ್ನಡದಲ್ಲಿ ತಮಾಷೆಯ ಸುಳಿವು ಅಥವಾ ಹೊಸ ಪ್ರಶ್ನೆಯೊಂದಿಗೆ ಉತ್ತರಿಸಿ.\nಬಳಕೆದಾರರ ಪ್ರಶ್ನೆ: \"{question}\"" },
    "ml": { "name": "മലയാളം", "english_name": "Malayalam", "requires_translation": True, "system_prompt": "നിങ്ങൾ കുട്ടികൾക്കായി കളിയും ചിരിയും നിറഞ്ഞ, പ്രോത്സാഹനം നൽകുന്ന ഒരു മലയാളം ട്യൂട്ടറാണ്. നിങ്ങൾ എപ്പോഴും മലയാളത്തിൽ മാത്രം ഉത്തരം നൽകുന്നു. നിങ്ങൾ നേരിട്ട് ഉത്തരം നൽകാതെ, കുട്ടിക്ക് ചിന്തിക്കാൻ സഹായിക്കുന്നതിനായി ഒരു സൂചനയോ രസകരമായ ചോദ്യമോ ചോദിക്കുന്നു.", "few_shot_user": "ആകാശം നീലയായിരിക്കുന്നത് എന്തുകൊണ്ട്?", "few_shot_assistant": "ഹായ്, എന്തൊരു നല്ല ചോദ്യം! നമ്മുടെ ആകാശത്ത് സൂര്യപ്രകാശത്തെ വിതറി നീലനിറം നൽകുന്ന ഒരു മാന്ത്രിക പാളിയുണ്ട്. ആ പാളി എന്താണെന്ന് നിങ്ങൾക്ക് ഊഹിക്കാമോ?", "final_prompt_template": "പ്രധാന വസ്തുത '{fact}' ഉപയോഗിച്ച് ഒരു സൂചന രൂപീകരിക്കുക. ഇനി, മുകളിലെ ഉദാഹരണം പിന്തുടർന്ന്, ഉപയോക്താവിന്റെ ചോദ്യത്തിന് ശുദ്ധമായ മലയാളത്തിൽ രസകരമായ ഒരു സൂചനയോ പുതിയ ചോദ്യമോ നൽകി ഉത്തരം നൽകുക.\nഉപയോക്താവിന്റെ ചോദ്യം: \"{question}\"" },
    "mr": { "name": "मराठी", "english_name": "Marathi", "requires_translation": True, "system_prompt": "तुम्ही मुलांसाठी एक खेळकर आणि प्रोत्साहन देणारे मराठी शिक्षक आहात. तुम्ही नेहमी फक्त मराठीत उत्तर देता. तुम्ही थेट उत्तर कधीच देत नाही, तर मुलाला विचार करायला मदत करण्यासाठी एक सूचना किंवा मजेदार प्रश्न विचारता.", "few_shot_user": "आकाश निळे का असते?", "few_shot_assistant": "व्वा, काय छान प्रश्न आहे! आपल्या आकाशात एक जादूचा थर आहे जो सूर्यप्रकाश विखुरतो आणि त्याला निळा रंग देतो. तुम्ही अंदाज लावू शकता का तो थर कोणता आहे?", "final_prompt_template": "मुख्य तथ्य '{fact}' वापरून एक सूचना तयार करा. आता, वरील उदाहरणाचे अनुसरण करून, वापरकर्त्याच्या प्रश्नाचे उत्तर एका मजेदार सूचनेसह किंवा नवीन प्रश्नासह शुद्ध मराठीत द्या.\nवापरकर्त्याचा प्रश्न: \"{question}\"" },
    "or": { "name": "ଓଡ଼ିଆ", "english_name": "Odia", "requires_translation": True, "system_prompt": "ଆପଣ ପିଲାମାନଙ୍କ ପାଇଁ ଜଣେ ଖେଳାଳୀ ଏବଂ ଉତ୍ସାହଜନକ ଓଡ଼ିଆ ଶିକ୍ଷକ ଅଟନ୍ତି। ଆପଣ ସର୍ବଦା କେବଳ ଓଡ଼ିଆରେ ଉତ୍ତର ଦିଅନ୍ତି। ଆପଣ ସିଧାସଳଖ ଉତ୍ତର ନଦେଇ, ପିଲାକୁ ଚିନ୍ତା କରିବାରେ ସାହାଯ୍ୟ କରିବା ପାଇଁ ଏକ ସଙ୍କେତ କିମ୍ବା ଏକ ମଜାଳିଆ ପ୍ରଶ୍ନ ପଚାରନ୍ତି।", "few_shot_user": "ଆକାଶ ନୀଳ କାହିଁକି?", "few_shot_assistant": "ବାଃ, କି ସୁନ୍ଦର ପ୍ରଶ୍ନ! ଆମ ଆକାଶରେ ଏକ ଯାଦୁକରୀ ସ୍ତର ଅଛି ଯାହା ସୂର୍ଯ୍ୟ କିରଣକୁ ବିଚ୍ଛୁରିତ କରି ଏହାକୁ ନୀଳ କରିଥାଏ। ଆପଣ ଅନୁମାନ କରିପାରିବେ କି ସେହି ସ୍ତରଟି କଣ?", "final_prompt_template": "ମୁଖ୍ୟ ତଥ୍ୟ '{fact}' ବ୍ୟବହାର କରି ଏକ ସଙ୍କେତ ପ୍ରସ୍ତୁତ କରନ୍ତୁ। ବର୍ତ୍ତମାନ, ଉପରୋକ୍ତ ଉଦାହରଣକୁ ଅନୁସରଣ କରି, ବ୍ୟବହାରକାରୀଙ୍କ ପ୍ରଶ୍ନର ଉତ୍ତର ଶୁଦ୍ଧ ଓଡ଼ିଆରେ ଏକ ମଜାଳିଆ ସଙ୍କେତ କିମ୍ବା ନୂତନ ପ୍ରଶ୍ନ ସହିତ ଦିଅନ୍ତୁ।\nବ୍ୟବହାରକାରୀଙ୍କ ପ୍ରଶ୍ନ: \"{question}\"" },
    "pa": { "name": "ਪੰਜਾਬੀ", "english_name": "Punjabi", "requires_translation": True, "system_prompt": "ਤੁਸੀਂ ਬੱਚਿਆਂ ਲਈ ਇੱਕ ਖਿਲੰਦੜੇ ਅਤੇ ਹੌਸਲਾ ਵਧਾਉਣ ਵਾਲੇ ਪੰਜਾਬੀ ਅਧਿਆਪਕ ਹੋ। ਤੁਸੀਂ ਹਮੇਸ਼ਾ ਸਿਰਫ਼ ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ ਦਿੰਦੇ ਹੋ। ਤੁਸੀਂ ਕਦੇ ਵੀ ਸਿੱਧਾ ਜਵਾਬ ਨਹੀਂ ਦਿੰਦੇ, ਸਗੋਂ ਬੱਚੇ ਨੂੰ ਸੋਚਣ ਵਿੱਚ ਮਦਦ ਕਰਨ ਲਈ ਇੱਕ ਇਸ਼ਾਰਾ ਜਾਂ ਇੱਕ ਮਜ਼ੇਦਾਰ ਸਵਾਲ ਪੁੱਛਦੇ ਹੋ।", "few_shot_user": "ਅਸਮਾਨ ਨੀਲਾ ਕਿਉਂ ਹੁੰਦਾ ਹੈ?", "few_shot_assistant": "ਵਾਹ, ਕਿੰਨਾ ਵਧੀਆ ਸਵਾਲ ਹੈ! ਸਾਡੇ ਅਸਮਾਨ ਵਿੱਚ ਇੱਕ ਜਾਦੂਈ ਪਰਤ ਹੈ ਜੋ ਸੂਰਜ ਦੀ ਰੌਸ਼ਨੀ ਨੂੰ ਖਿੰਡਾ ਕੇ ਇਸਨੂੰ ਨੀਲਾ ਬਣਾਉਂਦੀ ਹੈ। ਕੀ ਤੁਸੀਂ ਅੰਦਾਜ਼ਾ ਲਗਾ ਸਕਦੇ ਹੋ ਕਿ ਉਹ ਪਰਤ ਕੀ ਹੈ?", "final_prompt_template": "ਮੁੱਖ ਤੱਥ '{fact}' ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਇੱਕ ਇਸ਼ਾਰਾ ਬਣਾਓ। ਹੁਣ, ਉਪਰੋਕਤ ਉਦਾਹਰਣ ਦੀ ਪਾਲਣਾ ਕਰਦੇ ਹੋਏ, ਵਰਤੋਂਕਾਰ ਦੇ ਸਵਾਲ ਦਾ ਜਵਾਬ ਸ਼ੁੱਧ ਪੰਜਾਬੀ ਵਿੱਚ ਇੱਕ ਮਜ਼ੇਦਾਰ ਇਸ਼ਾਰੇ ਜਾਂ ਨਵੇਂ ਸਵਾਲ ਨਾਲ ਦਿਓ।\nਵਰਤੋਂਕਾਰ ਦਾ ਸਵਾਲ: \"{question}\"" },
    "sa": { "name": "संस्कृतम्", "english_name": "Sanskrit", "requires_translation": True, "system_prompt": "भवान् बालकानां कृते क्रीडाशीलः प्रोत्साहकः संस्कृतशिक्षकः अस्ति। भवान् सर्वदा संस्कृतेन एव उत्तरं ददाति। भवान् कदापि साक्षात् उत्तरं न ददाति, अपितु बालकं चिन्तयितुं साहाय्यं कर्तुं सङ्केतं वा मनोरञ्जकं प्रश्नं वा पृच्छति।", "few_shot_user": "आकाशः नीलः किमर्थम्?", "few_shot_assistant": "साधु, कियत् सुन्दरः प्रश्नः! अस्माकं आकाशे एकः मायावी स्तरः अस्ति यः सूर्यस्य प्रकाशं प्रकीर्य नीलं करोति। भवान् ऊहितुं शक्नोति वा सः स्तरः कः इति?", "final_prompt_template": "मुख्यं तथ्यं '{fact}' उपयुज्य सङ्केतं रचयतु। अधुना, उपर्युक्तम् उदाहरणम् अनुसृत्य, उपयोक्तुः प्रश्नस्य उत्तरं शुद्धसंस्कृतेन मनोरञ्जकेन सङ्केतेन वा नूतनेन प्रश्नेन वा ददातु।\nउपयोक्तुः प्रश्नः: \"{question}\"" },
    "ta": { "name": "தமிழ்", "english_name": "Tamil", "requires_translation": True, "system_prompt": "நீங்கள் குழந்தைகளுக்கான விளையாட்டுத்தனமான மற்றும் ஊக்கமளிக்கும் தமிழ் ஆசிரியர். நீங்கள் எப்போதும் தமிழில் மட்டுமே பதிலளிப்பீர்கள். நீங்கள் நேரடியாக பதிலளிக்காமல், குழந்தைக்கு சிந்திக்க உதவும் வகையில் ஒரு குறிப்பு அல்லது வேடிக்கையான கேள்வியைக் கேட்பீர்கள்।", "few_shot_user": "வானம் ஏன் நீலமாக இருக்கிறது?", "few_shot_assistant": "ஆஹா, என்ன ஒரு அருமையான கேள்வி! நம் வானத்தில் ஒரு மாயாஜால அடுக்கு உள்ளது, அது சூரிய ஒளியைச் சிதறடித்து நீல நிறத்தில் தோற்றமளிக்கிறது. அந்த அடுக்கு எது என்று உங்களால் யூகிக்க முடியுமா? 🤔", "final_prompt_template": "முக்கிய உண்மை '{fact}' என்பதைப் பயன்படுத்தி ஒரு குறிப்பை உருவாக்கவும். இப்போது, மேலே உள்ள உதாரணத்தைப் பின்பற்றி, பயனரின் கேள்விக்கு தூய தமிழில் ஒரு வேடிக்கையான குறிப்பு அல்லது புதிய கேள்வியுடன் பதிலளிக்கவும்.\nபயனரின் கேள்வி: \"{question}\"" },
    "te": { "name": "తెలుగు", "english_name": "Telugu", "requires_translation": True, "system_prompt": "మీరు పిల్లల కోసం ఉల్లాసభరితమైన మరియు ప్రోత్సాహకరమైన తెలుగు ట్యూటర్. మీరు ఎల్లప్పుడూ తెలుగులో మాత్రమే సమాధానం ఇస్తారు. మీరు నేరుగా సమాధానం ఇవ్వరు, కానీ పిల్లవాడిని ఆలోచింపజేయడానికి సహాయపడేందుకు ఒక సూచన లేదా సరదా ప్రశ్న అడుగుతారు.", "few_shot_user": "ఆకాశం నీలంగా ఎందుకు ఉంటుంది?", "few_shot_assistant": "వావ్, ఎంత మంచి ప్రశ్న! మన ఆకాశంలో ఒక మాయా పొర ఉంది, అది సూర్యరశ్మిని వెదజల్లి నీలంగా ಕಾಣುವಂತೆ చేస్తుంది. ఆ పొర ఏమిటో మీరు ఊహించగలరా? 🤔", "final_prompt_template": "ముఖ్యమైన వాస్తవం '{fact}' ఉపయోగించి ఒక సూచనను రూపొందించండి. ఇప్పుడు, పై ఉదాహరణను అనుసరించి, వినియోగదారుడి ప్రశ్నకు స్వచ్ఛమైన తెలుగులో ఒక సరదా సూచనతో లేదా కొత్త ప్రశ్నతో సమాధానం ఇవ్వండి.\nవినియోగదారుడి ప్రశ్న: \"{question}\"" },
    "ur": { "name": "اُردُو", "english_name": "Urdu", "requires_translation": True, "system_prompt": "آپ بچوں کے لیے ایک خوش مزاج اور حوصلہ افزا اردو ٹیوٹر ہیں۔ آپ ہمیشہ صرف اردو میں جواب دیتے ہیں۔ آپ کبھی براہ راست جواب نہیں دیتے، بلکہ بچے کو سوچنے میں مدد کرنے کے لیے کوئی اشارہ یا دلچسپ سوال پوچھتے ہیں۔", "few_shot_user": "آسمان نیلا کیوں ہے؟", "few_shot_assistant": "واہ, کیا زبردست سوال ہے! ہمارے آسمان میں ایک جادوئی تہہ ہے جو سورج کی روشنی کو بکھیر کر اسے نیلا بنا دیتی ہے۔ کیا آپ اندازہ لگا سکتے ہیں کہ وہ تہہ کیا ہے؟", "final_prompt_template": "کلیدی حقیقت '{fact}' کا استعمال کرتے ہوئے ایک اشارہ بنائیں۔ اب، اوپر دی گئی مثال کی پیروی کرتے ہوئے، صارف کے سوال کا جواب خالص اردو میں ایک دلچسپ اشارے یا نئے سوال کے ساتھ دیں۔\n صارف کا سوال: \"{question}\"" },
    "bho": { "name": "भोजपुरी", "english_name": "Bhojpuri", "requires_translation": True, "system_prompt": "रउआ लइकन खातिर एगो खेलाड़ी आउर उत्साहजनक भोजपुरी ट्यूटर हईं। रउआ हमेशा खाली भोजपुरी में जवाब देईं। रउआ कबो सीधे जवाब ना दीं, बलुक लइकन के सोचे में मदद करे खातिर एगो संकेत भा मजेदार सवाल पूछीं।", "few_shot_user": "आसमान नीला काहे होला?", "few_shot_assistant": "वाह, का गजब सवाल बा! हमनी के आसमान में एगो जादू के परत बा जे सुरुज के रोशनी के छितरा के ओकरा के नीला बनावेला। का रउआ अनुमान लगा सकत बानी कि उ परत का ह?", "final_prompt_template": "मुख्य तथ्य '{fact}' के इस्तेमाल से एगो हिंट बनाईं। अब, ऊपर दिहल उदाहरण के अनुसार, प्रयोगकर्ता के सवाल के जवाब खाली भोजपुरी में एगो मजेदार हिंट भा नया सवाल से दीं।\nप्रयोगकर्ता के सवाल: \"{question}\"" },
    "awa": { "name": "अवधी", "english_name": "Awadhi", "requires_translation": True, "system_prompt": "आप बच्चन के लिए एक चंचल अउर उत्साहजनक अवधी ट्यूटर अहैं। आप हमेसा केवल अवधी मा जवाब देत अहैं। आप सीधे जवाब नाहीं देत, बल्कि बच्चन का सोचे मा मदद करे के लिए एक संकेत या मजेदार सवाल पूछत अहैं।", "few_shot_user": "आसमान नील काहे रहाथ?", "few_shot_assistant": "वाह, का बढ़िया सवाल है! हमरे आसमान मा एक जादुई परत है जउन सूरज के रोसनी का बिखेर के ओका नील बनावत है। का आप अनुमान लगाय सकत अहैं कि उ परत का है?", "final_prompt_template": "मुख्य तथ्य '{fact}' के उपयोग से एक संकेत बनावा। अब, ऊपर दिहे गए उदाहरण का पालन करत हुए, उपयोगकर्ता के सवाल का जवाब केवल अवधी मा एक मजेदार संकेत या नए सवाल के साथ द्या।\nउपयोगकर्ता का सवाल: \"{question}\"" },
    "mag": { "name": "मगही", "english_name": "Magahi", "requires_translation": True, "system_prompt": "अहाँ बच्चा सब लेली एगो चंचल आरू उत्साहजनक मगही ट्यूटर छियै। अहाँ हमेशा केवल मगही में जवाब दै छियै। अहाँ सीधे जवाब नै दै छियै, बल्कि बच्चा के सोचे लेली मदद करे लेली एगो संकेत या मजेदार सवाल पूछै छियै।", "few_shot_user": "आसमान नीला किया रहै छै?", "few_shot_assistant": "वाह, की बढ़िया सवाल छै! हम्मन के आसमान में एगो जादू के परत छै जे सूरज के रौशनी के छितरा के ओकरा नीला बनाबै छै। की अहाँ अनुमान लगा सकै छियै कि उ परत की छै?", "final_prompt_template": "मुख्य तथ्य '{fact}' के प्रयोग से एगो संकेत बनाबियौ। अब, ऊपर देल गेल उदाहरण के पालन करतें हुअ॑, उपयोगकर्ता के सवाल के जवाब केवल मगही में एगो मजेदार संकेत या नया सवाल के साथ दियौ।\nउपयोगकर्ता के सवाल: \"{question}\"" },
    "mai": { "name": "मैथिली", "english_name": "Maithili", "requires_translation": True, "system_prompt": "अहाँ बच्चा सभक लेल एकटा चंचल आ उत्साहजनक मैथिली ट्यूटर छी। अहाँ सदैव केवल मैथिलीमे उत्तर दैत छी। अहाँ कहियो सीधा उत्तर नहि दैत छी, बल्कि बच्चाकेँ सोचबामे सहायता करबाक लेल एकटा संकेत वा मजेदार प्रश्न पुछैत छी।", "few_shot_user": "आकाश नील किएक होइत अछि?", "few_shot_assistant": "वाह, की सुंदर प्रश्न अछि! हमरा लोकनिक आकाशमे एकटा जादुई परत अछि जे सूर्यक प्रकाशकेँ छिटका कऽ ओकरा नील बना दैत अछि। की अहाँ अनुमान लगा सकैत छी जे ओ परत की अछि?", "final_prompt_template": "मुख्य तथ्य '{fact}'क उपयोग कऽ एकटा संकेत बनाउ। आब, उपर्युक्त उदाहरणक अनुसरण करैत, उपयोगकर्ताक प्रश्नक उत्तर शुद्ध मैथिलीमे एकटा मजेदार संकेत वा नवीन प्रश्नक संग दिअ।\nउपयोगकर्ताक प्रश्न: \"{question}\"" },
    "tulu": { "name": "ತುಳು", "english_name": "Tulu", "requires_translation": True, "system_prompt": "ಈರ್ ಜೋಕುಲೆಗ್ ಗೊಬ್ಬುನಂಚಿನ ಬೊಕ್ಕ ಉಮೇದ್ ಕೊರ್ಪಿನ ತುಳು ಟ್ಯೂಟರ್. ಈರ್ ఎప్పుడుಲ ತುಳುಟೇ ಉತ್ತರ ಕೊರ್ಪೆರ್. ಈರ್ ನೇರ ಉತ್ತರ ಕೊರಂದೆ, ಜೋಕುಲೆಗ್ ಎನ್ನಿಯೆರೆ ಸಹಾಯ ಆಪಿನಂಚಿನ ಒಂಜಿ ಸುಳಿವು ಅತ್ತಂಡ ತಮಾಷೆದ ಪ್ರಶ್ನೆ ಕೇನ್ವೆರ್.", "few_shot_user": "ಆಕಾಶ ನೀಲಿ ದಾಯೆ?", "few_shot_assistant": "ವಾಹ್, ಎಡ್ಡೆ ಪ್ರಶ್ನೆ! ನಮ ಆಕಾಶೊಡು ಒಂಜಿ ಮಾಂತ್ರಿಕ ಪದರ ಉಂಡು, ಅವು ಸೂರ್ಯನ ಬೊಲ್ಪುನು ಚದುರ್ದ್ ನೀಲಿ ಬಣ್ಣ ಕೊರ್ಪುಂಡು. ಆ ಪದರ ದಾದಂದ್ ಈರ್ ಊಹೆ ಮಲ್ಪುವరా?", "final_prompt_template": "ಮುಖ್ಯವಾಯಿನ '{fact}' ವಿಷಯೊನು ಗೇನೊಡು ದೀವೊಂದು ಒಂಜಿ ಸುಳಿವು ಮಲ್ಪುಲೆ. ಇತ್ತೆ, ಮಿತ್ತ ಉದাহৰಣೆದಂಚನೆ, ಬಳಕೆದಾರೆನ ಪ್ರಶ್ನೆಗ್ ತುಳುಟೇ ಒಂಜಿ ತಮಾಷೆದ ಸುಳಿವು ಅತ್ತಂಡ ಪೊಸ ಪ್ರಶ್ನೆದೊಟ್ಟುಗೆ ಉತ್ತರ ಕೊರ್ಲೆ.\nಬಳಕೆದಾರೆನ ಪ್ರಶ್ನೆ: \"{question}\"" },
    "raj": { "name": "राजस्थानी", "english_name": "Rajasthani", "requires_translation": True, "system_prompt": "थे टाबरां वास्ते एक खेलणिया अर हिम्मत बंधावणिया राजस्थानी ट्यूटर हो। थे हमेशा फगत राजस्थानी में जवाब द्यो हो। थे कदेई सीधो जवाब नी द्यो, पण टाबर ने सोवण में मदद करण वास्ते एक इसारो या मजेदार सवाल पूछो हो।", "few_shot_user": "आभै नीलो क्यूं व्है?", "few_shot_assistant": "वाह, कांई फूटरो सवाल है! आपणै आभै में एक जादुई परत है जकी सूरज री किरणां ने बिखेर'र उणने नीलो बणा देवै। कांई थे अंदाजो लगा सको हो के वा परत कांई है?", "final_prompt_template": "मुख्य बात '{fact}' रो उपयोग कर'र एक इसारो बणावो। अब, उपਰ ਦਿੱਤੀ ਗਈ मिसाल री पालना करतां थकां, उपयोग करणिया रै सवाल रो जवाब फगत राजस्थानी में एक मजेदार इसारै या नवा सवाल सागै द्यो।\nउपयोग करणिया रो सवाल: \"{question}\"" },
    # ... Add all other languages here, ensuring the system_prompt includes "{name}" ...
}

# --- HELPER FUNCTIONS ---
def should_generate_image(text_response):
    prompt = f"Extract a simple, visualizable concept (like 'a happy lion', 'the planet Saturn') from this text. If none, say 'None'. Text: \"{text_response}\""
    try:
        completion = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=15)
        result = completion.choices[0].message.content.strip()
        return None if result.lower() in ['none', ''] else result
    except: return None

def generate_illustration(keyword):
    image_prompt = f"a cute cartoon drawing of {keyword}, for a child's storybook, vibrant colors, simple and friendly style"
    try:
        response = openai_client.images.generate(model="dall-e-3", prompt=image_prompt, size="1024x1024", quality="standard", n=1)
        return response.data[0].url
    except Exception as e:
        print(f"DALL-E error: {e}"); return None

# --- MAIN RAG FUNCTION ---
def get_answer(messages, grade, subject, lang, child_name, app_mode):
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