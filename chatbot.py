# chatbot.py

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.memory import ConversationBufferMemory


# ===============================
# 📌 Load or Create Vector Store
# ===============================
def load_vectorstore(file_path="knowledge.txt", db_path="vector_store"):
    loader = TextLoader(file_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(documents, embedding_model)
    db.save_local(db_path)
    return db, embedding_model


# ===============================
# 📌 Load Chatbot
# ===============================
def init_chatbot(db_path="vector_store"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

    generator = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
    llm = HuggingFacePipeline(pipeline=generator)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        db.as_retriever(),
        memory=memory
    )
    return qa


# ===============================
# 📌 Streamlit App
# ===============================
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 Context-Aware RAG Chatbot")

# Load or init vector store
qa = init_chatbot()

# Chat UI
user_input = st.text_input("💬 Ask me something:")

if st.button("Ask"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            result = qa({"question": user_input})
        st.write("**Answer:**", result["answer"])
    else:
        st.warning("Please type a question.")
