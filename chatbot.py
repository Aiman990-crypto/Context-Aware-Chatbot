# chatbot.py

import streamlit as st
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Streamlit UI
st.set_page_config(page_title="Local RAG Chatbot", page_icon="🧠")
st.title("🧠 Local RAG Chatbot")
st.write("Ask me anything and I'll try to answer using the model!")

# Load model
@st.cache_resource
def load_model():
    pipe = pipeline("text-generation", model="gpt2", max_new_tokens=100)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_model()

# Chat
user_input = st.text_input("💬 Your question:")

if st.button("Ask"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            response = llm(user_input)
        st.success("✅ Answer:")
        st.write(response)
    else:
        st.warning("⚠️ Please enter a question.")
