import streamlit as st
from rag import get_db_info, get_answer

st.set_page_config(page_title="AI Tutor Diagnostic", layout="centered")

# --- DIAGNOSTIC PANEL ---
st.header("🔬 AI Tutor - Diagnostic Panel")
st.info("This panel shows the real-time status of the database connection.")

status, message = get_db_info()

if status == "Success":
    st.success(f"**Database Status:** {message}")
else:
    st.error(f"**Database Status:** {message}")
st.divider()

# --- SIMPLE CHAT INTERFACE ---
st.subheader("Test Chat")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_question = st.chat_input("Ask a test question")
if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.chat_message("user").write(user_question)
    
    with st.spinner("Thinking..."):
        # We use hardcoded values for the test
        response = get_answer(user_question, "Grade1", "English")
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)