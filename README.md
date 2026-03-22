[🚀 Live App Link](https://your-app-url.streamlit.app)

# AI Regulations GPT 
### *Global Compliance Intelligence for the AI Era*

AI Regulations GPT is a high-precision, stateful RAG (Retrieval-Augmented Generation) system designed to help financial institutions and auditors navigate the complex "graph" of global AI regulations including the EU AI Act, NIST (USA), and Singapore's Agentic Framework.

## 🚀 Key Features

* **Multi-Model Specialist-Auditor Pipeline:** Utilizes a dual-model architecture where **Qwen3-32B** acts as a Technical Drafter for fact extraction and **GPT-OSS 120B** acts as an editor for formal refinement and logic.
* **Hierarchical RAG (Parent-Document Retrieval):** Solves the "context vs. precision" trade-off by indexing small semantic child-chunks while retrieving full parent-sections for the LLM to ensure no regulatory nuance is lost.
* **Jurisdictional Routing & Memory:** Features an exhaustive mapping of global regions (EU, UK, USA, Singapore, Canada) and maintains "Session State," allowing users to ask follow-up questions without re-specifying the country.
* **Hybrid Search & Reranking:** Combines BM25 keyword matching with Vector similarity search, further filtered by a **FlashRank** cross-encoder reranker for 99% factual precision.
* **Official Citation Mapping:** Provides verifiable sources for all answers, including the specific regulation and article, allowing you to easily cross-check accuracy and trust the results.

## 🛠️ Technical Stack

* **LLM Orchestration:** LangChain & Groq LPU (Inference).
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (384-dimensional) for cloud-ready, high-speed performance.
* **Vector Database:** ChromaDB.
* **Frontend:** Streamlit.

## 📂 Project Structure

```text
AI-Regulations-GPT/
├── app.py                # Main Streamlit Application & RAG Logic
├── requirements.txt      # Cloud Deployment Dependencies
├── .gitignore            # Security filter for API keys and DBs
├── data/                 # Regulatory PDFs (EU, US, UK, SGP)
└── README.md             # Project Documentation
