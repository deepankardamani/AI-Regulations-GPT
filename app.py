import streamlit as st
import os
from dotenv import load_dotenv

# Cloud-Ready RAG Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURATION & BRANDING ---
load_dotenv()
st.set_page_config(page_title="AI Regulations GPT", page_icon="⚖️", layout="wide")

st.title("⚖️ AI Regulations GPT")
st.markdown("### *Global Compliance Intelligence Navigator*")

# --- 2. THE ENGINE INITIALIZATION ---
@st.cache_resource
def init_rag_system():
    # 384-dimension embeddings for web-speed performance
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Vector Database & Parent Store
    # Note: On the web, these will be rebuilt from your /data folder
    vector_db = Chroma(
        collection_name="ai_reg_gpt_prod",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    store = LocalFileStore("./parent_store")
    docstore = create_kv_docstore(store)

    # Hierarchical Splitting Strategy
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="standard_deviation")

    # The Primary Retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vector_db,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    
    # 2026 Specialist-Auditor Models
    drafter_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.1)
    refiner_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    
    return retriever, drafter_llm, refiner_llm

retriever, drafter_llm, refiner_llm = init_rag_system()

# --- 3. AUDIT ADVISOR LOGIC ---
DOC_TITLES = {
    "EU_AI_ACT.pdf": "EU AI Act (Regulation (EU) 2024/1689)",
    "NIST_RAI.pdf": "NIST AI Risk Management Framework 1.0",
    "SR11_7.pdf": "US Fed/OCC SR 11-7: Model Risk Management",
    "a-pro-innovation-approach-to-ai-regulation-UK.pdf": "UK AI Regulation: A Pro-Innovation Approach",
    "mgf-for-agentic-ai-sgp.pdf": "Singapore Agentic AI Framework"
}

def get_regulatory_guidance(context, query):
    drafter_prompt = ChatPromptTemplate.from_template("[SPECIALIST] Extract mandates: {context}\nQuery: {question}")
    refiner_prompt = ChatPromptTemplate.from_template("[ADVISOR] Provide clear guidance. Context: {context}\nDraft: {draft}")
    
    draft = (drafter_prompt | drafter_llm | StrOutputParser()).invoke({"context": context, "question": query})
    return (refiner_prompt | refiner_llm | StrOutputParser()).invoke({"context": context, "draft": draft})

# --- 4. STATEFUL INTERFACE ---
if "last_location" not in st.session_state:
    st.session_state.last_location = None

query = st.text_input("Search regulations (e.g., 'What are the EU transparency rules?'):")

if query:
    with st.spinner("Analyzing Global Regulatory Store..."):
        # Jurisdiction Detection Logic (Simplified for Streamlit)
        # (This is where your run_global_audit logic lives)
        st.success("Analysis Complete")
        st.markdown("---")
        # Display results with official citations from DOC_TITLES