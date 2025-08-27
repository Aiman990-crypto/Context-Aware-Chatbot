# app.py

import streamlit as st
from chatbot import load_vectorstore, init_chatbot

# ===============================
# 📌 Setup
# ===============================
st.set_page_config(page_title="🧠 Local RAG Chatbot", layout="centered")
st.title("🧠 Context-Aware Chatbot ")

# Initialize chatbot on first run
if "qa" not in st.session_state:
    db, _ = load_vectorstore("knowledge.txt")
    st.session_state.qa = init_chatbot()

if "history" not in st.session_state:
    st.session_state.history = []


# ===============================
# 📌 Chat Input
# ===============================
user_input = st.text_input("💬 You:", "")

if st.button("Ask"):
    if user_input:
        result = st.session_state.qa({"question": user_input})
        answer = result["answer"]

        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", answer))


# ===============================
# 📌 Display Chat History
# ===============================
for role, msg in st.session_state.history:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")
