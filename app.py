import sys
# CRITICAL: Fix for SQLite version on Streamlit Cloud/Hugging Face
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

import streamlit as st
import os
import re
from dotenv import load_dotenv

# --- GEN AI STACK ---
# Standard Text Splitters
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

# Vector Store & Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Loaders & Models
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- RETRIEVERS (Now in langchain_classic for 2026) ---
from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# --- UPDATED STORAGE & DOCSTORE ---
from langchain_classic.storage import LocalFileStore
try:
    from langchain_classic.storage import create_kv_docstore
except ImportError:
    # Fallback for internal utility path
    from langchain_classic.storage._lc_store import create_kv_docstore

# Reranker
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

# --- 1. CONFIG & API ---
load_dotenv()
st.set_page_config(page_title="AI Regulations GPT", page_icon="⚖️", layout="wide")

# Fetch API Key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- 2. ARCHITECTURE ---

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
                final_output.extend(self.semantic_chunker.split_text(block))
            except Exception:
                final_output.extend(self.safety_splitter.split_text(block))
        return final_output

@st.cache_resource
def initialize_engine():
    if not GROQ_API_KEY:
        st.error("🔑 GROQ_API_KEY missing! Add it to your Secrets/Environment Variables.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(collection_name="audit_vault", embedding_function=embeddings, persist_directory="./chroma_db")
    
    fs = LocalFileStore("./parent_store")
    docstore = create_kv_docstore(fs)
    p_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=400)
    c_splitter = SemanticSplitterAdapter(SemanticChunker(embeddings, breakpoint_threshold_type="standard_deviation"))

    retriever = ParentDocumentRetriever(
        vectorstore=vector_db, docstore=docstore, 
        child_splitter=c_splitter, parent_splitter=p_splitter, id_key="doc_id"
    )
    
    reranker = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2")
    drafter = ChatGroq(model="qwen/qwen3-32b", temperature=0.1, api_key=GROQ_API_KEY)
    refiner = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=GROQ_API_KEY)
    
    return retriever, reranker, drafter, refiner, docstore, vector_db

engine, reranker, drafter_llm, refiner_llm, docstore, vector_db = initialize_engine()

# --- 3. JURISDICTIONAL MAPPING ---
REGULATORY_MAP = {
    "EU_AI_ACT.pdf": "EU", "NIST_RAI.pdf": "USA", "SR11_7.pdf": "USA",
    "a-pro-innovation-approach-to-ai-regulation-UK.pdf": "UK",
    "OSFI E23.pdf": "Canada", "mgf-for-agentic-ai-sgp.pdf": "Singapore",
    "ISOIEC420012023.pdf": "Global"
}

# --- 4. INGESTION LOGIC ---
if len(vector_db.get()['ids']) == 0:
    data_path = "./data"
    if os.path.exists(data_path):
        with st.status("🏗️ Building Audit Vault...", expanded=True) as status:
            for filename in os.listdir(data_path):
                if filename in REGULATORY_MAP:
                    loader = PyPDFLoader(os.path.join(data_path, filename))
                    pages = loader.load()
                    for page in pages:
                        page.metadata["location"] = REGULATORY_MAP[filename]
                        page.metadata["source_file"] = filename
                    engine.add_documents(pages)
                    st.write(f"✅ Indexed: {filename}")
            status.update(label="Vault Ready", state="complete")

# --- 5. THE AUDIT ENGINE ---
def run_audit(query):
    # Detect Regions
    regions = {
        r"\buk\b|\bunited kingdom\b": "UK", r"\beu\b|\beurope\b": "EU",
        r"\busa\b|\bus\b": "USA", r"\bsingapore\b|\bsgp\b": "Singapore",
        r"\bcanada\b": "Canada"
    }
    active = [v for k, v in regions.items() if re.search(k, query.lower())]
    if not active: return "⚠️ Please specify a jurisdiction.", []

    # Parallel Search
    all_results = []
    for loc in active:
        p_keys = list(docstore.yield_keys())
        loc_parents = [docstore.mget([k])[0] for k in p_keys if docstore.mget([k])[0].metadata.get('location') == loc]
        if loc_parents:
            bm25 = BM25Retriever.from_documents(loc_parents); bm25.k = 5
            ensemble = EnsembleRetriever(retrievers=[engine, bm25], weights=[0.7, 0.3])
            all_results.extend(ensemble.invoke(query, search_kwargs={"filter": {"location": loc}, "k": 10}))

    if not all_results: return "⚠️ No matches found.", []

    # Rerank & LLM Chain
    refined = reranker.compress_documents(all_results, query)
    context = "\n\n".join([f"[{d.metadata['location']}]: {d.page_content}" for d in refined])
    
    d_prompt = ChatPromptTemplate.from_template("Summarize {locs} requirements.\nContext: {c}\nQ: {q}")
    r_prompt = ChatPromptTemplate.from_template("Draft: {d}\nRefine into formal guidance for {locs}.\nContext: {c}")
    
    draft = (d_prompt | drafter_llm | StrOutputParser()).invoke({"c": context, "q": query, "locs": active})
    final = (r_prompt | refiner_llm | StrOutputParser()).invoke({"c": context, "d": draft, "locs": active})
    
    return final, refined

# --- 6. UI & SIDEBAR ---
with st.sidebar:
    st.header("📊 Database Health")
    db_data = vector_db.get()
    if db_data['ids']:
        counts = {}
        for m in db_data['metadatas']:
            l = m.get('location', 'Unknown')
            counts[l] = counts.get(l, 0) + 1
        for loc, count in counts.items():
            st.write(f"📍 **{loc}**: {count} snippets")
    
    if st.button("🚨 Wipe & Re-index"):
        vector_db.delete_collection()
        st.cache_resource.clear()
        st.rerun()

st.title("⚖️ AI Regulations GPT")
if prompt := st.chat_input("Ask about regulations..."):
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant"):
        ans, docs = run_audit(prompt)
        st.markdown(ans)
        if docs:
            with st.expander("Sources"):
                for d in docs: st.write(f"📖 {d.metadata['source_file']} (P. {d.metadata.get('page', 0)+1})")