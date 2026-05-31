# 🤖 Terraris AI - Enterprise Customer Support System

Terraris AI is a next-generation **Enterprise Customer Support Platform** powered by an **Advanced Hybrid RAG Pipeline**. It combines dense vector retrieval with traditional sparse keyword search to deliver contextual, lightning-fast, and highly accurate AI-generated support responses based on enterprise ticketing data.

## 🔥 Key Architectural Features

- **Hybrid Search Engine (Vector + BM25):** Merges semantic search via **ChromaDB** (`all-MiniLM-L6-v2` embeddings) with keyword matching via **BM25Retriever** to guarantee comprehensive context retrieval.
- **Enterprise-Scale Text Splitting:** Utilizes `RecursiveCharacterTextSplitter` to optimize chunk sizes for fine-grained semantic overlapping.
- **Ultra-Fast Generation Layer:** Driven by Groq's hardware acceleration using the `llama-3.3-70b-versatile` model.
- **Premium Cyberpunk UI:** Custom custom-crafted dark/glow HTML & CSS embedded natively into **Streamlit**.

## 🚀 Quick Start

### 1. Clone the Project
```bash
