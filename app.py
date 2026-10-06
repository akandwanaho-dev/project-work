import os
import streamlit as st
from dotenv import load_dotenv

# Document Processing & RAG Pipeline Libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

# 1. Page & Layout Settings
st.set_page_config(page_title="Textbook AI Tutor (Groq)", page_icon="📚", layout="wide")

# Custom Visual Enhancements & Accent Badging
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #ff4b4b;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
""", unsafe_allow_html=True)

# App Core Title Layout
st.title("📚 Chat with your Textbook")
st.caption("⚡ Powered by Groq's ultra-fast inference cloud & local text tokenizers.")
st.divider()

# Load local environment variables if a .env file exists
load_dotenv()

# 2. Initialization of Persistent Session States
if "messages" not in st.session_state:
    st.session_state.messages = []  # Conversation log storage
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None  # Holds the operational backend pipeline
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None  # Holds metadata stats for display

# 3. Sidebar Configuration Space
with st.sidebar:
    st.header("⚙️ Configuration Workspace")
    
    # Check for environmental key setup fallback
    env_key = os.environ.get("GROQ_API_KEY", "")
    
    # UI input space for the API Key
    user_api_key = st.text_input(
        "Groq API Key",
        value=env_key,
        type="password",
        help="Obtain an API key for free by signing up at https://groq.com",
        placeholder="gsk_..."
    )
    
    # Dynamically track validation flag state
    has_api_key = len(user_api_key.strip()) > 0
    
    if not has_api_key:
        st.warning("⚠️ Access Token Needed: Enter your Groq API Key above to unlock document parsing and context matching features.")
    else:
        st.success("🔒 Authorization token detected.")

    st.divider()
    st.header("📥 Upload Center")
    
    # Control document uploading space depending on explicit authorization status
    uploaded_file = st.file_uploader(
        "Choose a textbook PDF", 
        type="pdf", 
        disabled=not has_api_key
    )
    
    if uploaded_file and has_api_key:
        # Re-build database pipeline if it has not been registered to the session state yet
        if st.session_state.rag_chain is None:
            with st.spinner("Parsing syntax trees and optimizing textbook segments..."):
                
                # Write local asset layer buffer out cleanly 
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # PDF loading engine initialization
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    
                    # Compute page-level and volume statistics
                    total_pages = len(docs)
                    total_chars = sum(len(doc.page_content) for doc in docs)
                    
                    # Token Chunking Strategy Execution
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_documents(docs)
                    
                    # Register document statistics context to session states
                    st.session_state.doc_stats = {
                        "filename": uploaded_file.name,
                        "pages": total_pages,
                        "chars": total_chars,
                        "chunks": len(chunks)
                    }
                    
                    # Create Vector Space Embedding Infrastructure
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                    
                    # Instantiate Groq Engine with the UI-provided API Key
                    # Change this model identifier to a currently supported production endpoint
                    llm = ChatGroq(
                        model="llama-3.1-8b-instant",  # Updated from llama3-8b-8192
                        temperature=0.2,
                        groq_api_key=user_api_key
                    )

                    
                    # Construct Prompt Engineering blue-print matrices
                    system_prompt = (
                        "You are an expert AI Textbook Tutor. Your goal is to help students understand their course material.\n"
                        "Use the following pieces of retrieved context from the textbook to answer the student's question.\n"
                        "If you don't know the answer or if it isn't in the text, say honestly that you cannot find it in the provided textbook pages.\n"
                        "Keep your answer structured, clear, and educational. Use bullet points where appropriate.\n\n"
                        "Context:\n{context}"
                    )
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    # Compile executable RAG execution paths
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    st.session_state.rag_chain = create_retrieval_chain(retriever, document_chain)
                    st.success("🚀 Index structure created!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"An error occurred while compiling your data: {e}")
                    
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

# 4. Main App Grid Layout
# Split screen layout into an interactive split view if stats exist
if st.session_state.doc_stats:
    # Document Analytical Breakdown Header
    st.subheader(f"📊 Document Insights: `{st.session_state.doc_stats['filename']}`")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Document Pages</div>
                <div class="metric-value">{st.session_state.doc_stats['pages']}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Extracted Characters</div>
                <div class="metric-value">{st.session_state.doc_stats['chars']:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Semantic Text Chunks</div>
                <div class="metric-value">{st.session_state.doc_stats['chunks']}</div>
            </div>
        """, unsafe_allow_html=True)
    st.divider()

# 5. Core Chat Pipeline Interface Render
if st.session_state.rag_chain is None:
    if not has_api_key:
        st.warning("🔒 The workspace is locked. Provide a valid `GROQ_API_KEY` in the configuration panel to continue.")
    else:
        st.info("💡 Please upload a textbook PDF in the sidebar file uploader to activate your personal AI Tutor.")
else:
    # Render historical content logging blocks sequentially
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Track user processing streams
chat_placeholder = "What concept would you like explained?" if has_api_key else "Chat disabled. Setup an authentication token key."

if user_query := st.chat_input(chat_placeholder, disabled=not has_api_key):
    if st.session_state.rag_chain is not None:
        
        # Immediate echo feedback stream block
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Downstream prompt query processing via Groq pipeline
        with st.chat_message("assistant"):
            with st.spinner("Flipping through textbook pages..."):
                response = st.session_state.rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                st.markdown(answer)
                
                # Traceable attribution framework
                with st.expander("🔍 View Textbook Sources Used"):
                    for doc in response["context"]:
                        page = doc.metadata.get("page", 0) + 1
                        st.write(f"**From Page {page}:**")
                        st.caption(f"_{doc.page_content[:300]}..._")
                        st.divider()
                        
        st.session_state.messages.append({"role": "assistant", "content": answer})
