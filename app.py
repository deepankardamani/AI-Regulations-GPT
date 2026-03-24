# --- 1. ENVIRONMENT & DEPENDENCY CONFIGURATION ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from dotenv import load_dotenv

from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever

try:
    from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever
    from langchain_classic.storage import LocalFileStore, create_kv_docstore
    from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
except ImportError:
    from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
    from langchain_community.storage import LocalFileStore
    from langchain.storage import create_kv_docstore
    from langchain_community.document_compressors import FlashrankRerank

# --- 2. GLOBAL SYSTEM SETTINGS ---
load_dotenv()
st.set_page_config(page_title="AI Regulations GPT", page_icon="⚖️", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_location" not in st.session_state:
    st.session_state.last_location = []

# --- 3. ARCHITECTURAL COMPONENTS ---
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
def initialize_knowledge_base():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(collection_name="global_compliance_index", embedding_function=embeddings, persist_directory="./chroma_db")
    fs = LocalFileStore("./parent_store")
    docstore = create_kv_docstore(fs)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=400)
    child_splitter = SemanticSplitterAdapter(SemanticChunker(embeddings, breakpoint_threshold_type="standard_deviation"), max_chars=5000)

    h_retriever = ParentDocumentRetriever(vectorstore=vector_db, docstore=docstore, child_splitter=child_splitter, parent_splitter=parent_splitter, id_key="doc_id")
    reranker = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2")
    
    drafter = ChatGroq(model="qwen/qwen3-32b", temperature=0.1)
    refiner = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    
    return h_retriever, reranker, drafter, refiner, docstore, vector_db

retriever, reranker, drafter_llm, refiner_llm, docstore, vector_db = initialize_knowledge_base()

# --- 4. INGESTION PIPELINE ---
REGULATORY_MAP = {
    "EU_AI_ACT.pdf": "EU", "NIST_RAI.pdf": "USA", "SR11_7.pdf": "USA",
    "a-pro-innovation-approach-to-ai-regulation-UK.pdf": "UK",
    "OSFI E23.pdf": "Canada", "mgf-for-agentic-ai-sgp.pdf": "Singapore",
    "ISOIEC420012023.pdf": "Global"
}

if len(vector_db.get()['ids']) == 0:
    with st.status("Indexing Knowledge Base...", expanded=True) as status:
        data_path = "./data"
        if os.path.exists(data_path):
            for filename in os.listdir(data_path):
                if filename in REGULATORY_MAP:
                    loader = PyPDFLoader(os.path.join(data_path, filename))
                    pages = loader.load()
                    for page in pages:
                        page.metadata["location"] = REGULATORY_MAP[filename]
                        page.metadata["source_file"] = filename
                    retriever.add_documents(pages)
            status.update(label="System Ready", state="complete")

# --- 5. AUDIT ENGINE (With Parallel Disjoint Retrieval) ---
def run_compliance_audit(user_query):
    query_norm = user_query.lower()
    
    jurisdiction_db = {
        "uk": "UK", "united kingdom": "UK", "eu": "EU", "europe": "EU",
        "usa": "USA", "us": "USA", "canada": "Canada", "osfi": "Canada",
        "singapore": "Singapore", "sgp": "Singapore", "global": "Global"
    }
    
    CITATIONS = {
        "EU_AI_ACT.pdf": "Regulation (EU) 2024/1689 (EU AI Act)",
        "NIST_RAI.pdf": "NIST AI RMF 1.0",
        "SR11_7.pdf": "SR Letter 11-7 (US Fed/OCC)",
        "a-pro-innovation-approach-to-ai-regulation-UK.pdf": "UK AI Regulation Approach",
        "OSFI E23.pdf": "Guideline E-23 (Canada OSFI)",
        "mgf-for-agentic-ai-sgp.pdf": "Model AI Governance Framework (Singapore)",
        "ISOIEC420012023.pdf": "ISO/IEC 42001:2023"
    }

    # 1. Detect New Locations
    matches = set([v for k, v in jurisdiction_db.items() if k in query_norm])
    if matches:
        st.session_state.last_location = list(matches)
    
    active_locs = st.session_state.last_location
    if not active_locs:
        return "⚠️ Specify a jurisdiction (e.g., EU, USA, Singapore).", [], CITATIONS

    # 2. PARALLEL RETRIEVAL: Perform a dedicated search for EACH geography
    all_raw_results = []
    
    with st.spinner(f"Accessing separate vaults for: {', '.join(active_locs)}..."):
        for loc in active_locs:
            # Build a location-specific Lexical index
            parent_keys = list(docstore.yield_keys())
            loc_parents = [docstore.mget([k])[0] for k in parent_keys if docstore.mget([k])[0].metadata.get('location') == loc]
            
            if loc_parents:
                loc_lexical = BM25Retriever.from_documents(loc_parents)
                loc_lexical.k = 5
                
                # Create a temporary ensemble for ONLY this location
                loc_engine = EnsembleRetriever(retrievers=[retriever, loc_lexical], weights=[0.7, 0.3])
                
                # Dedicated Search for this location ONLY
                loc_results = loc_engine.invoke(
                    user_query, 
                    search_kwargs={"filter": {"location": loc}, "k": 8}
                )
                all_raw_results.extend(loc_results)

    if not all_raw_results:
        return "⚠️ No data found for the selected regions.", [], CITATIONS

    # 3. RERANK THE AGGREGATED SET
    refined_results = reranker.compress_documents(all_raw_results, user_query)

    # 4. DRAFTER-REFINER WITH SOURCE ATTRIBUTION
    context_payload = "\n\n".join([f"SOURCE [{d.metadata['location']}]: {d.page_content}" for d in refined_results])
    
    d_prompt = ChatPromptTemplate.from_template("""
    [ROLE: TECHNICAL SPECIALIST]
    Extract and group facts specifically by region. 
    Context: {context}
    Query: {question}
    Draft Response:""")
    
    r_prompt = ChatPromptTemplate.from_template("""
    [ROLE: SENIOR REGULATORY ADVISOR]
    STRICT ATTRIBUTION RULE:
    - Use data labeled [Singapore] ONLY for Singapore sections.
    - Use data labeled [USA] ONLY for USA sections.
    - If a document (like the UK) discusses another region (like Singapore), IGNORE it if the primary domestic document is available.
    - Focus strictly on: {active_locs}.

    Context: {context}
    Draft: {draft}
    Final Advisory Report:""")
    
    draft = (d_prompt | drafter_llm | StrOutputParser()).invoke({
        "context": context_payload, "question": user_query
    })
    final_output = (r_prompt | refiner_llm | StrOutputParser()).invoke({
        "context": context_payload, "draft": draft, "active_locs": active_locs
    })
    
    return final_output, refined_results, CITATIONS

# --- 6. UI LAYER ---
st.title("⚖️ AI Regulations GPT")
st.caption("Institutional-Grade Regulatory Intelligence | Parallel Retrieval Architecture")

col1, col2, col3 = st.columns(3)
with col1: st.info("🔍 **Hybrid Search:** Per-Region Quotas")
with col2: st.info("🌍 **Global Reach:** Isolated Jurisdictions")
with col3: st.info("🛡️ **Audit Grade:** Source-Specific Attribution")

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Verified Statutory Sources"):
                for src in message["sources"]: st.write(f"📖 {src}")

if prompt := st.chat_input("Ask about AI regulations..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        final_answer, sources, CITATIONS_DB = run_compliance_audit(prompt)
        st.markdown(final_answer)
        
        source_list = []
        if sources:
            with st.expander("Verified Statutory Sources"):
                for d in sources:
                    s_name = f"{CITATIONS_DB.get(d.metadata['source_file'], d.metadata['source_file'])} (Page {d.metadata.get('page', 0)+1})"
                    st.write(f"📖 {s_name}")
                    source_list.append(s_name)
        
        st.session_state.messages.append({"role": "assistant", "content": final_answer, "sources": source_list})