# --- 1. ENVIRONMENT & DEPENDENCY CONFIGURATION ---
# High-compatibility fix for cloud-based SQLite versions in 2026
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from dotenv import load_dotenv

# Core AI & Retrieval Framework
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever

# 2026 Retrieval Standards (Supporting Legacy & Modern Modules)
try:
    from langchain_classic.retrievers import (
        ParentDocumentRetriever, 
        EnsembleRetriever, 
        ContextualCompressionRetriever
    )
    from langchain_classic.storage import LocalFileStore, create_kv_docstore
    from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
except ImportError:
    from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever, ContextualCompressionRetriever
    from langchain_community.storage import LocalFileStore
    from langchain.storage import create_kv_docstore
    from langchain_community.document_compressors import FlashrankRerank

# --- 2. GLOBAL SYSTEM SETTINGS ---
load_dotenv()
st.set_page_config(page_title="AI Regulations GPT", page_icon="⚖️", layout="wide")

if "last_location" not in st.session_state:
    st.session_state.last_location = None

# --- 3. ARCHITECTURAL COMPONENTS ---

class SemanticSplitterAdapter(TextSplitter):
    """
    Ensures long-form regulatory documents are split into semantically 
    meaningful chunks while respecting model context windows.
    """
    def __init__(self, semantic_chunker, max_chars=5000):
        super().__init__()
        self.semantic_chunker = semantic_chunker
        self.max_chars = max_chars
        self.safety_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=300)
        
    def split_text(self, text: str):
        safe_blocks = self.safety_splitter.split_text(text)
        final_output = []
        for block in safe_blocks:
            try:
                final_output.extend(self.semantic_chunker.split_text(block))
            except Exception:
                final_output.extend(self.safety_splitter.split_text(block))
        return final_output

@st.cache_resource
def initialize_knowledge_base():
    """
    Initializes the multi-layered retrieval engine, including 
    vector storage, docstores, and LLM reasoning models.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Persistent Vector Storage
    vector_db = Chroma(
        collection_name="global_compliance_index",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Parent Document Store
    fs = LocalFileStore("./parent_store")
    docstore = create_kv_docstore(fs)

    # Hierarchical Indexing Strategy
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=400)
    child_splitter = SemanticSplitterAdapter(
        SemanticChunker(embeddings, breakpoint_threshold_type="standard_deviation"),
        max_chars=5000
    )

    h_retriever = ParentDocumentRetriever(
        vectorstore=vector_db,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        id_key="doc_id"
    )
    
    # Advanced Reranking & Model Duo
    reranker = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2")
    drafter = ChatGroq(model="qwen/qwen3-32b", temperature=0.1)
    refiner = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    
    return h_retriever, reranker, drafter, refiner, docstore, vector_db

retriever, reranker, drafter_llm, refiner_llm, docstore, vector_db = initialize_knowledge_base()

# --- 4. REGULATORY INGESTION PIPELINE ---

# Map of raw files to official jurisdictional tags
REGULATORY_MAP = {
    "EU_AI_ACT.pdf": "EU", 
    "NIST_RAI.pdf": "USA", 
    "SR11_7.pdf": "USA",
    "a-pro-innovation-approach-to-ai-regulation-UK.pdf": "UK",
    "OSFI E23.pdf": "Canada", 
    "mgf-for-agentic-ai-sgp.pdf": "Singapore",
    "ISOIEC420012023.pdf": "Global"
}

if len(vector_db.get()['ids']) == 0:
    with st.status("Initializing Knowledge Base...", expanded=True) as status:
        data_path = "./data"
        if os.path.exists(data_path):
            for filename in os.listdir(data_path):
                if filename in REGULATORY_MAP:
                    st.write(f"Indexing: {filename}...")
                    loader = PyPDFLoader(os.path.join(data_path, filename))
                    pages = loader.load()
                    for page in pages:
                        page.metadata["location"] = REGULATORY_MAP[filename]
                        page.metadata["source_file"] = filename
                    retriever.add_documents(pages)
            status.update(label="System Ready", state="complete")

# --- 5. INTELLIGENT AUDIT ORCHESTRATION ---

def run_compliance_audit(user_query):
    """
    Orchestrates jurisdictional detection, hybrid retrieval, 
    and multi-model reasoning to provide verified guidance.
    """
    query_norm = user_query.lower()
    
    # Official Statutory Citations
    CITATIONS = {
        "EU_AI_ACT.pdf": "Regulation (EU) 2024/1689 of the European Parliament and of the Council (Artificial Intelligence Act)",
        "NIST_RAI.pdf": "NIST AI 100-1: Artificial Intelligence Risk Management Framework (AI RMF 1.0) - U.S. Department of Commerce",
        "SR11_7.pdf": "SR Letter 11-7: Guidance on Model Risk Management - Board of Governors of the Federal Reserve System",
         "a-pro-innovation-approach-to-ai-regulation-UK.pdf": "A pro-innovation approach to AI regulation - UK Department for Science, Innovation and Technology",
         "OSFI E23.pdf": "Guideline E-23: Model Risk Management - Office of the Superintendent of Financial Institutions (OSFI) Canada",
         "mgf-for-agentic-ai-sgp.pdf": "Model AI Governance Framework for Generative AI - Personal Data Protection Commission (PDPC) Singapore",
         "ISOIEC420012023.pdf": "ISO/IEC 42001:2023 Information technology — Artificial intelligence — Management system"

    }

    # Jurisdictional Intelligence Mapping
    jurisdiction_db = {
        "uk": "UK", "united kingdom": "UK", "britain": "UK", "london": "UK",
        "eu": "EU", "european union": "EU", "europe": "EU", "brussels": "EU",
        "france": "EU", "germany": "EU", "spain": "EU", "italy": "EU",
        "usa": "USA", "us": "USA", "america": "USA", "united states": "USA",
        "canada": "Canada", "osfi": "Canada", "toronto": "Canada",
        "singapore": "Singapore", "sgp": "Singapore", "mas": "Singapore",
        "global": "Global", "iso": "Global", "international": "Global"
    }
    
    matches = set([v for k, v in jurisdiction_db.items() if k in query_norm])
    if matches:
        st.session_state.last_location = list(matches)
    
    active_locs = st.session_state.last_location
    if not active_locs:
        st.warning("Please specify a jurisdiction (e.g., EU, USA, UK).")
        return

    # Hybrid Retrieval (BM25 + Vector)
    parent_keys = list(docstore.yield_keys())
    parent_docs = [docstore.mget([k])[0] for k in parent_keys]
    lexical_search = BM25Retriever.from_documents(parent_docs)
    lexical_search.k = 5

    hybrid_engine = EnsembleRetriever(retrievers=[retriever, lexical_search], weights=[0.7, 0.3])
    
    # Context Extraction & Reranking
    raw_results = hybrid_engine.invoke(
        user_query, 
        search_kwargs={"filter": {"location": {"$in": active_locs}}, "k": 8}
    )
    refined_results = reranker.compress_documents(raw_results, user_query)

    # Multi-Model Reasoning Chain
    context_payload = "\n\n".join([f"[{d.metadata['location']}]: {d.page_content}" for d in refined_results])
    
    d_prompt = ChatPromptTemplate.from_template("[Technical Specialist] Extract: {context}\nQuery: {question}")
    r_prompt = ChatPromptTemplate.from_template("[Regulatory Advisor] Finalize: {context}\nDraft: {draft}")
    
    draft_response = (d_prompt | drafter_llm | StrOutputParser()).invoke({"context": context_payload, "question": user_query})
    final_output = (r_prompt | refiner_llm | StrOutputParser()).invoke({"context": context_payload, "draft": draft_response})
    
    # UI Presentation
    st.subheader(f"Regulatory Guidance: {' & '.join(active_locs)}")
    st.markdown(final_output)
    
    with st.expander("Verified Statutory Sources"):
        for d in refined_results:
            source_name = CITATIONS.get(d.metadata['source_file'], d.metadata['source_file'])
            st.write(f"📖 {source_name} (Page {d.metadata.get('page', 0)+1})")

# --- 6. INTERFACE LAYER ---
st.title("⚖️ AI Regulations GPT")
user_input = st.text_input("Enter your regulatory query:", placeholder="e.g., What are the EU transparency requirements?")

if user_input:
    run_compliance_audit(user_input)