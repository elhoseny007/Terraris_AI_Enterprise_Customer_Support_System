print("Loading...")

import os
import time
import streamlit as st

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Terraris AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================================================
# CSS
# ======================================================
st.markdown("""
<style>
/* GLOBAL */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #040816;
    color: white;
}

/* REMOVE STREAMLIT DEFAULT */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* NAVBAR */
.navbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 60px;
    background: rgba(4,8,22,0.6);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

.logo {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: 1px;
}

.logo span {
    color: #45e7ff;
}

.menu {
    font-size: 34px;
    color: white;
}

/* HERO */
.hero {
    min-height: 70vh;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    overflow: hidden;
    padding: 120px 20px 40px 20px;
    background:
    linear-gradient(
        rgba(4,8,22,0.88),
        rgba(4,8,22,0.93)
    ),
    url("https://images.unsplash.com/photo-1516321318423-f06f85e504b3?q=80&w=2070");
    background-size: cover;
    background-position: center;
}

/* GLOW EFFECT */
.glow1 {
    position: absolute;
    width: 450px;
    height: 450px;
    background: #00d9ff;
    opacity: 0.08;
    border-radius: 50%;
    filter: blur(120px);
    top: -100px;
    left: -100px;
    animation: float1 8s ease-in-out infinite;
}

.glow2 {
    position: absolute;
    width: 400px;
    height: 400px;
    background: #7b61ff;
    opacity: 0.08;
    border-radius: 50%;
    filter: blur(120px);
    bottom: -100px;
    right: -100px;
    animation: float2 8s ease-in-out infinite;
}

@keyframes float1 {
    0% {transform: translateY(0px);}
    50% {transform: translateY(40px);}
    100% {transform: translateY(0px);}
}

@keyframes float2 {
    0% {transform: translateY(0px);}
    50% {transform: translateY(-40px);}
    100% {transform: translateY(0px);}
}

/* HERO CONTENT */
.hero-content {
    position: relative;
    z-index: 5;
    text-align: center;
    max-width: 1100px;
}

/* BADGE */
.badge {
    display: inline-block;
    padding: 12px 28px;
    border-radius: 50px;
    border: 1px solid rgba(69,231,255,0.3);
    background: rgba(69,231,255,0.08);
    color: #b4f6ff;
    letter-spacing: 2px;
    margin-bottom: 30px;
    font-size: 14px;
}

/* TITLE */
.main-title {
    font-size: 82px;
    font-weight: 800;
    line-height: 1.05;
    margin-bottom: 25px;
}

.highlight {
    background: linear-gradient(
        90deg,
        #45e7ff,
        #7f8cff
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* SUBTITLE */
.subtitle {
    color: rgba(255,255,255,0.72);
    font-size: 21px;
    line-height: 1.8;
    max-width: 850px;
    margin: auto;
}

/* SEARCH AREA CONTAINER WRAPPER */
.search-container {
    max-width: 850px;
    margin: 30px auto;
    padding: 25px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border-radius: 25px;
}

/* INPUT */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    color: white;
    border-radius: 18px;
    padding: 18px;
    font-size: 18px;
}

/* BUTTON */
.stButton button {
    width: 100%;
    border: none;
    padding: 16px;
    border-radius: 18px;
    background: linear-gradient(
        90deg,
        #42e8ff,
        #637bff
    );
    color: white;
    font-size: 18px;
    font-weight: 700;
    transition: 0.3s;
}

.stButton button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(66,232,255,0.25);
}

/* RESULT */
.result-box {
    margin-top: 30px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 25px;
    padding: 30px;
    text-align: left;
}

/* RESPONSIVE */
@media (max-width: 900px) {
    .navbar { padding: 18px 25px; }
    .main-title { font-size: 48px; }
    .subtitle { font-size: 17px; }
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# NAVBAR
# ======================================================
st.markdown("""
<div class="navbar">
    <div class="logo">
        TERRARIS<span>.AI</span>
    </div>
    <div class="menu">
        ☰
    </div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# HERO
# ======================================================
st.markdown("""
<div class="hero">
    <div class="glow1"></div>
    <div class="glow2"></div>
    <div class="hero-content">
        <div class="badge">AI IT SERVICE SYSTEM</div>
        <h1 class="main-title">
            Enterprise AI <br>
            Customer Support <br>
            <span class="highlight">Powered by RAG</span>
        </h1>
        <p class="subtitle">
            Intelligent semantic retrieval, AI-generated responses,
            vector search, BM25 hybrid search, and real-time
            customer support automation engineered for enterprise scale.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# API KEY & MODEL Setup
# ======================================================

groq_api_key = "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"

generator = ChatGroq(
    groq_api_key=groq_api_key,
    temperature=0.7,
    model="llama-3.3-70b-versatile"
)

# ======================================================
# LOAD RESOURCES (Cached)
# ======================================================

@st.cache_resource
def initialize_resources():
    path = r"C:\Users\ELZAHBIA\Downloads\customer_support_tickets_200k.csv"
    
    loader = CSVLoader(
        file_path=path,
        metadata_columns=["ticket_id", "status", "priority", "product"]
    )
    documents = loader.load()[:5000]

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    split_docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_directory = r"E:\Desktop\chroma_db"

    if os.path.exists(persist_directory):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )

    bm25 = BM25Retriever.from_documents(split_docs)
    bm25.k = 5
    return vector_store, bm25

# Safely catch data loading initialization
try:
    vector_store, bm25 = initialize_resources()
except Exception as e:
    st.error(f"Resource Initialization Error: Check your file paths. Details: {e}")
    st.stop()

# ======================================================
# RETRIEVAL & GENERATION LOGIC
# ======================================================
def retrieve_documents(query):
    vector_docs = vector_store.as_retriever(search_kwargs={"k": 5}).invoke(query)
    bm25_docs = bm25.invoke(query)
    unique_docs = {doc.page_content: doc for doc in (vector_docs + bm25_docs)}
    return list(unique_docs.values())[:5]

def generate_answer(query):
    docs = retrieve_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    You are an expert customer support assistant.
    Answer ONLY using the provided context.

    Context:
    {context}

    User Question:
    {query}
    """
    response = generator.invoke(prompt)
    return response.content

# ======================================================
# SEARCH SECTION (Fixed layout container mapping)
# ======================================================
# ============================================
# AI CHAT CONTAINER - PRO UI VERSION
# ============================================

with st.container():

    st.markdown("""
    <div class="chat-wrapper">
        <div class="chat-header">
            <h1>🤖 AI Assistant</h1>
            <p>Ask anything about your customer support tickets</p>
        </div>
    """, unsafe_allow_html=True)

    # ===============================
    # CHAT HISTORY
    # ===============================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # عرض الرسائل القديمة
    for msg in st.session_state.messages:

        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <div class="message-content">
                    {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="bot-message">
                <div class="message-content">
                    {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ===============================
    # INPUT AREA
    # ===============================
    query = st.chat_input("Ask about customer tickets...")

    # ===============================
    # PROCESS MESSAGE
    # ===============================
    if query:

        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        # Show user message instantly
        st.markdown(f"""
        <div class="user-message">
            <div class="message-content">
                {query}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Generate AI response
        with st.spinner("AI is thinking..."):

            time.sleep(2)

            answer = generate_answer(query)

            # Save AI response
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            # Show AI response
            st.markdown(f"""
            <div class="bot-message">
                <div class="message-content">
                    {answer}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)