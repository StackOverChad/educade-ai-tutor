import os
import streamlit as st
import base64
from rag import get_answer, LANGUAGE_CONFIGS
from tts import text_to_speech
from streamlit_mic_recorder import mic_recorder
import openai
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Sparky AI Tutor", page_icon="🤖", layout="centered")

# --- STYLING FUNCTIONS ---
def apply_standalone_styling(image_file):
    if not os.path.exists(image_file): return
    with open(image_file, "rb") as f: img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{ background: transparent; }}
        .stApp::before {{
            content: ""; position: fixed; left: 0; right: 0; top: 0; bottom: 0; z-index: -1;
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover; background-repeat: no-repeat; background-attachment: fixed;
            opacity: 0.4; 
        }}
        h1, h2, h3, h4, h5, h6 {{ color: black !important; }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def apply_embed_styling():
    style = """
        <style>
        .stApp { background: transparent !important; }
        .main .block-container { padding: 0.75rem 1rem 1rem 1rem !important; }
        h1 { font-size: 20px !important; font-weight: 600 !important; color: #1E293B !important; padding: 0 !important; margin-bottom: 1rem !important; }
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def list_grades():
    if not os.path.exists("books"): os.makedirs("books")
    return sorted([d for d in os.listdir("books") if os.path.isdir(os.path.join("books", d))])

def list_subjects(grade):
    if not grade: return []
    path = os.path.join("books", grade)
    if not os.path.exists(path): return []
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

def transcribe_voice(audio_bytes):
    if not audio_bytes: return ""
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "voice_question.wav"
    with st.spinner("Sparky is listening... 👂"):
        try:
            transcript = openai_client.audio.transcriptions.create(model="whisper-1", file=audio_file)
            st.toast(f"I heard: \"{transcript.text}\"")
            return transcript.text
        except Exception as e:
            st.error(f"Sorry, I had trouble understanding. Error: {e}")
            return ""

def send_message(user_text=None):
    if not st.session_state.get('selected_grade') or not st.session_state.get('selected_subject'):
        st.toast("Please select a grade and subject first! ⚙️", icon="⚠️")
        return
    text_to_send = user_text if user_text is not None else st.session_state.get('user_input', '')
    text_to_send = text_to_send.strip()
    if text_to_send:
        st.session_state.messages.append({"role": "user", "content": text_to_send})
        with st.spinner("Sparky is thinking... 🤔"):
            result = get_answer(
                messages=st.session_state.messages, grade=st.session_state.selected_grade,
                subject=st.session_state.selected_subject, lang=st.session_state.selected_lang_code,
                child_name=st.session_state.child_name, app_mode=st.session_state.app_mode
            )
        st.session_state.messages.append({ "role": "assistant", "content": result["answer"], "image_url": result["image_url"], "choices": result["choices"] })
        if result["answer"]:
            try:
                audio_file_path = text_to_speech(result["answer"], lang=st.session_state.selected_lang_code)
                with open(audio_file_path, "rb") as audio_file: st.session_state.audio_to_play = audio_file.read()
            except Exception: st.session_state.audio_to_play = None
        st.session_state.user_input = ""

def reset_conversation():
    name = st.session_state.get('child_name'); mode = st.session_state.get('app_mode'); lang_code = st.session_state.get('selected_lang_code')
    grade = st.session_state.get('selected_grade'); subject = st.session_state.get('selected_subject')
    st.session_state.clear()
    st.session_state.child_name = name; st.session_state.app_mode = mode; st.session_state.selected_lang_code = lang_code
    st.session_state.selected_grade = grade; st.session_state.selected_subject = subject
    st.session_state.messages = []
    st.rerun()

def initialize_chat_messages():
    if st.session_state.app_mode == "Tutor Mode":
        st.session_state.messages = [{"role": "assistant", "content": f"Hi {st.session_state.child_name}! I'm Sparky! 🤖 What do you want to learn about today?"}]
    else:
        st.session_state.messages = []

def display_chat_message(msg):
    is_user = msg["role"] == "user"; avatar = "🧑‍🚀" if is_user else "🤖"
    bubble_style = "background-color: {bg}; color: {txt}; padding: 8px 14px; font-size: 14px; border-radius: 18px; max-width: 90%; display: inline-block; word-wrap: break-word; margin-bottom: 4px; box-shadow: 0 1px 2px 0 rgba(0,0,0,0.1);"
    container_style = "display: flex; align-items: flex-start; justify-content: {align}; margin-bottom: 8px; gap: 8px;"
    if is_user:
        align = "flex-end"; formatted_style = bubble_style.format(bg="#006AFF", txt="white")
        html = f'<div style="{container_style.format(align=align)}"><div style="{formatted_style}">{msg["content"]}</div><div style="font-size: 1.5rem;">{avatar}</div></div>'
    else:
        align = "flex-start"; formatted_style = bubble_style.format(bg="#ECEFF1", txt="black")
        content_html = f'<div style="{formatted_style}">{msg["content"]}'
        if msg.get("image_url"): content_html += f'<br><img src="{msg["image_url"]}" style="max-width: 100%; border-radius: 15px; margin-top: 8px;">'
        content_html += '</div>'
        html = f'<div style="{container_style.format(align=align)}"><div style="font-size: 1.5rem;">{avatar}</div>{content_html}</div>'
    st.markdown(html, unsafe_allow_html=True)

# --- UI & APP LOGIC ---
is_embedded = st.query_params.get("embed") == "true"
if is_embedded: apply_embed_styling()
else: apply_standalone_styling("./assets/background.png")

openai_client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

if 'child_name' not in st.session_state:
    if not is_embedded: st.title("🚀 Welcome!")
    st.subheader("What should Sparky call you?")
    name = st.text_input("My name is...", label_visibility="collapsed")
    if name:
        st.session_state.child_name = name; st.session_state.app_mode = "Tutor Mode"; st.session_state.messages = []
        st.rerun()
else:
    if is_embedded:
        st.markdown(f"<h1>🚀 Sparky's Universe for {st.session_state.child_name}!</h1>", unsafe_allow_html=True)
    else:
        st.title(f"🚀 Sparky's Universe for {st.session_state.child_name}!")
    
    with st.sidebar:
        st.header("⚙️ Settings"); st.radio("Choose a mode:", ["Tutor Mode", "Story Mode"], key="app_mode", on_change=reset_conversation)
        language_options = { f"{config['name']} ({config['english_name']})" if code != 'en' else config['name']: code for code, config in LANGUAGE_CONFIGS.items() }
        selected_display_name = st.selectbox("Select Language", options=language_options.keys(), key="lang_select")
        st.session_state.selected_lang_code = language_options[selected_display_name]
        grades = list_grades(); st.session_state.selected_grade = st.selectbox("Select Grade", grades) if grades else None
        subjects = list_subjects(st.session_state.selected_grade); st.session_state.selected_subject = st.selectbox("Select Subject", subjects) if subjects else None
        st.button("🚀 Start New Chat!", on_click=reset_conversation, use_container_width=True)

    chat_container = st.container(height=380)
    if not st.session_state.get("messages"): initialize_chat_messages()
    if st.session_state.app_mode == "Story Mode" and not st.session_state.messages:
        if st.session_state.selected_subject:
            prompt = f"Let's start an adventure about {st.session_state.selected_subject}!"
            send_message(user_text=prompt); st.rerun()
        else: chat_container.warning("Please select a subject to start a story!")
            
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] in ["user", "assistant"]: display_chat_message(msg)

    if st.session_state.get("audio_to_play"):
        st.audio(st.session_state.audio_to_play, autoplay=True); st.session_state.audio_to_play = None

    last_message = st.session_state.messages[-1] if st.session_state.messages else {}
    if last_message.get("choices"):
        st.markdown("##### What happens next?")
        cols = st.columns(len(last_message["choices"]))
        for i, choice in enumerate(last_message["choices"]):
            cols[i].button(f"➡️ {choice}", on_click=send_message, args=(choice,), use_container_width=True, key=f"choice_{i}")
    else:
        def voice_callback():
            if st.session_state.recorder.get('bytes'): send_message(user_text=transcribe_voice(st.session_state.recorder['bytes']))
        
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.text_input("Ask Sparky a question...", key="user_input", on_change=send_message, label_visibility="collapsed")
        with col2:
            mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key='recorder', callback=voice_callback, use_container_width=True)