# chatbot.py

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ===============================
# 📌 Load or Create Vector Store
# ===============================
def load_vectorstore(file_path="knowledge.txt", db_path="vector_store"):
    # Load docs
    loader = TextLoader(file_path)
    docs = loader.load()

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = splitter.split_documents(docs)

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build FAISS
    db = FAISS.from_documents(documents, embedding_model)
    db.save_local(db_path)
    return db, embedding_model


# ===============================
# 📌 Load Chatbot
# ===============================
def init_chatbot(db_path="vector_store"):
    # Embeddings wrapper
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Reload FAISS
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

    # Local HuggingFace LLM
    generator = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512, temperature=0)
    llm = HuggingFacePipeline(pipeline=generator)

    # Memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Conversational Retrieval Chain
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        db.as_retriever(),
        memory=memory
    )
    return qa