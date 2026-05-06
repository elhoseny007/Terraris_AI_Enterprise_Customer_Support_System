# Advanced Customer Support RAG System (Gemma 1.1 2B)

An advanced Retrieval-Augmented Generation (RAG) pipeline designed to automate customer support ticket resolutions with high precision and efficiency. This system leverages state-of-the-art LLM orchestration and retrieval techniques to provide accurate answers while minimizing latency and operational costs.

## 🚀 Key Features

* **LLM Orchestration:** Powered by **Gemma 1.1 2B** and orchestrated using **LangGraph** for sophisticated, agentic workflows.
* **Hybrid Search:** Combines **BM25** (keyword-based) and **ChromaDB Vector Search** (semantic-based) to ensure maximum retrieval coverage.
* **Smart Reranking:** Integrates a **Cross-Encoder Reranker** to refine and prioritize the most relevant context for the LLM.
* **Semantic Caching:** Implements a custom **Semantic Caching Layer** with TTL (Time-To-Live) logic to instantly serve recurring queries, reducing API costs and response times.
* **Efficient Indexing:** Utilizes **Recursive Character Text Splitting** and HuggingFace embeddings for optimized processing of large-scale support datasets.

## 🛠️ Tech Stack

* **Language:** Python
* **Frameworks:** LangChain, LangGraph
* **LLM:** Google Gemma 1.1 2B (via HuggingFace Transformers)
* **Vector DB:** ChromaDB
* **Retrieval:** BM25, Semantic Search, Cross-Encoder Reranking
* **Embeddings:** sentence-transformers/all-MiniLM-L6-v2

## 📋 System Architecture

1.  **Semantic Cache Check:** The system first checks if a similar query has been answered recently.
2.  **Hybrid Retrieval:** If no cache hit, it performs a parallel search using BM25 and Vector embeddings.
3.  **Reranking:** Top results are scored and reranked using a Cross-Encoder for precision.
4.  **Generation:** The LLM generates a solution based strictly on the retrieved support ticket resolutions.
5.  **Cache Update:** The new answer is stored in the semantic cache for future use.

## 💻 Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/elhoseny007/Advanced-Customer-Support-RAG-System-Gemma-1.1-2B.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install langchain langchain-community chromadb sentence-transformers transformers accelerate rank_bm25 langgraph
    ```
3.  **HuggingFace Login:** Ensure you have a HuggingFace token with access to the Gemma model.
4.  **Run the Notebook/Script:** Mount your Google Drive or provide the path to your CSV dataset to begin indexing.

---
Developed by **Elhoseny Hassan**
