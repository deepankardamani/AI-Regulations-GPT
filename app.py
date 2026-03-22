__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import time
from dotenv import load_dotenv

# --- 1. THE EXACT IMPORTS (Cells 1, 2, 3, 6, 10) ---
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever

# Handling the "Classic" vs "Standard" naming for Cloud stability
try:
    from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever, ContextualCompressionRetriever
    from langchain_community.storage import LocalFileStore
    from langchain.storage import create_kv_docstore
    from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
except ImportError:
    from langchain_community.retrievers import ParentDocumentRetriever, EnsembleRetriever, ContextualCompressionRetriever
    from langchain_community.storage import LocalFileStore
    from langchain.storage import create_kv_docstore
    from langchain_community.document_compressors import FlashrankRerank

# --- 2. CONFIGURATION & STATE ---
load_dotenv()
st.set_page_config(page_title="AI Regulations GPT", page_icon="⚖️", layout="wide")

if "last_location" not in st.session_state:
    st.session_state.last_location = None

# --- 3. SEMANTIC ADAPTER (Cell 3 - Verbatim) ---
class SemanticSplitterAdapter(TextSplitter):
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
                semantic_chunks = self.semantic_chunker.split_text(block)
                final_output.extend(semantic_chunks)
            except Exception:
                final_output.extend(self.safety_splitter.split_text(block))
        return final_output

# --- 4. ENGINE INITIALIZATION (Cells 3 & 6) ---
@st.cache_resource
def init_full_engine():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Storage Layers
    vector_db = Chroma(
        collection_name="global_compliance_production",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    fs = LocalFileStore("./parent_store")
    docstore = create_kv_docstore(fs)

    # Hierarchical Splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=400)
    child_splitter = SemanticSplitterAdapter(
        SemanticChunker(embeddings, breakpoint_threshold_type="standard_deviation"),
        max_chars=5000
    )

    # Hierarchical Retriever
    h_retriever = ParentDocumentRetriever(
        vectorstore=vector_db,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        id_key="doc_id"
    )
    
    # Flashrank Reranker (Senior Auditor)
    reranker_model = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2")
    
    # LLMs (Specialist-Auditor Duo)
    drafter_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.1)
    refiner_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    
    return h_retriever, reranker_model, drafter_llm, refiner_llm, docstore, vector_db

retriever, reranker_model, drafter_llm, refiner_llm, docstore, vector_db = init_full_engine()

# --- 5. DATA INGESTION PIPELINE (Cells 4 & 5 - Verbatim) ---
LOCATION_MAP = {
    "EU_AI_ACT.pdf": "EU",
    "NIST_RAI.pdf": "USA",
    "SR11_7.pdf": "USA",
    "a-pro-innovation-approach-to-ai-regulation-UK.pdf": "UK",
    "OSFI E23.pdf": "Canada",
    "mgf-for-agentic-ai-sgp.pdf": "Singapore",
    "ISOIEC420012023.pdf": "Global"
}

# Only run ingestion if the database is empty
if len(vector_db.get()['ids']) == 0:
    with st.status("Initiating Document Ingestion Pipeline...", expanded=True) as status:
        for filename, location in LOCATION_MAP.items():
            path = f"data/{filename}"
            if os.path.exists(path):
                st.write(f"Processing: {filename} ({location})...")
                loader = PyPDFLoader(path)
                pages = loader.load()
                for page in pages:
                    page.metadata["location"] = location
                    page.metadata["source_file"] = filename
                retriever.add_documents(pages)
        status.update(label="Ingestion Complete!", state="complete")

# --- 6. MULTI-MODEL LOGIC (Cell 10 - Verbatim) ---
def get_refined_audit(context, query):
    drafter_template = """[SYSTEM: TECHNICAL SPECIALIST]
Extract and summarize core requirements: {context}
Focus on: Thresholds, actions, prohibitions.
Question: {question}
Draft:"""
    
    refiner_template = """[ROLE: SENIOR REGULATORY ADVISOR]
Provide clear, authoritative guidance. 
1. No audit report labels. 2. Instructional tone. 3. Use bullet points.
Context: {context}
Draft: {draft}
Regulatory Guidance:"""

    drafter_prompt = ChatPromptTemplate.from_template(drafter_template)
    refiner_prompt = ChatPromptTemplate.from_template(refiner_template)
    
    draft = (drafter_prompt | drafter_llm | StrOutputParser()).invoke({"context": context, "question": query})
    return (refiner_prompt | refiner_llm | StrOutputParser()).invoke({"context": context, "draft": draft})

# --- 7. THE STATEFUL ORCHESTRATOR (Cell 11 - Verbatim) ---
def run_global_audit(query):
    query_lower = query.lower()
    
    DOC_TITLES = {
        "EU_AI_ACT.pdf": "EU AI Act (Regulation (EU) 2024/1689)",
        "NIST_RAI.pdf": "NIST AI Risk Management Framework 1.0",
        "SR11_7.pdf": "US Fed/OCC SR 11-7: Guidance on Model Risk Management",
        "a-pro-innovation-approach-to-ai-regulation-UK.pdf": "UK AI Regulation: A Pro-Innovation Approach",
        "OSFI E23.pdf": "Canada OSFI Guideline E-23",
        "mgf-for-agentic-ai-sgp.pdf": "Singapore Agentic AI Framework",
        "ISOIEC420012023.pdf": "ISO/IEC 42001:2023"
    }

    # --- THE FULL JURISDICTION MAP (Verbatim from Cell 11) ---
    jurisdiction_map = {
        "uk": "UK", "united kingdom": "UK", "britain": "UK", "great britain": "UK", 
        "england": "UK", "scotland": "UK", "wales": "UK", "london": "UK",
        "eu": "EU", "european union": "EU", "europe": "EU", "brussels": "EU",
        "france": "EU", "germany": "EU", "spain": "EU", "italy": "EU", "belgium": "EU",
        "netherlands": "EU", "sweden": "EU", "norway": "EU", "denmark": "EU", 
        "finland": "EU", "ireland": "EU", "portugal": "EU", "austria": "EU", 
        "greece": "EU", "poland": "EU", "czechia": "EU", "romania": "EU", 
        "hungary": "EU", "slovakia": "EU", "bulgaria": "EU", "croatia": "EU",
        "usa": "USA", "us": "USA", "america": "USA", "united states": "USA", "nist": "USA",
        "canada": "Canada", "ontario": "Canada", "toronto": "Canada", "osfi": "Canada",
        "singapore": "Singapore", "sgp": "Singapore", "sg": "Singapore", "mas": "Singapore",
        "global": "Global", "iso": "Global", "international": "Global"
    }
    
    detected_locs = set([v for k, v in jurisdiction_map.items() if k in query_lower])
            
    if detected_locs:
        st.session_state.last_location = list(detected_locs)
    
    selected_locs = st.session_state.last_location

    if not selected_locs:
        st.warning("❌ No active jurisdiction. Please specify regions (e.g., UK, US, EU).")
        return

    # Hybrid & BM25 Retrieval logic
    all_parent_keys = list(docstore.yield_keys())
    parent_documents = [docstore.mget([k])[0] for k in all_parent_keys]
    bm25_retriever = BM25Retriever.from_documents(parent_documents)
    bm25_retriever.k = 5

    hybrid_retriever = EnsembleRetriever(retrievers=[retriever, bm25_retriever], weights=[0.7, 0.3])
    
    # Advanced Retrieval with Reranking
    search_kwargs = {"filter": {"location": {"$in": selected_locs}}, "k": 8}
    docs = hybrid_retriever.invoke(query, search_kwargs=search_kwargs)
    
    # Reranking
    compressed_docs = reranker_model.compress_documents(docs, query)

    # Synthesis & Display
    context_text = "\n\n".join([f"[{d.metadata['location']}]: {d.page_content}" for d in compressed_docs])
    final_report = get_refined_audit(context_text, query)
    
    st.markdown(f"### Audit Report: {' & '.join(selected_locs).upper()}")
    st.write(final_report)
    
    with st.expander("Evidence Sources"):
        for d in compressed_docs:
            st.write(f"- {DOC_TITLES.get(d.metadata['source_file'])} (Page {d.metadata.get('page', 0)+1})")

# --- 8. UI WRAPPER ---
st.title("⚖️ AI Regulations GPT")
query = st.text_input("Run Global Audit Query:")
if query:
    run_global_audit(query)