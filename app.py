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
st.set_page_config(page_title="Textbook AI Tutor (Groq)", page_icon="📚", layout="centered")
st.title("📚 Chat with your Textbook")
st.write("Powered by Groq's ultra-fast inference engines.")

# Load local environment variables if a .env file exists
load_dotenv()

# 2. Check for Groq API Key
# It looks for GROQ_API_KEY in your system environment variables or a local .env file.
if "GROQ_API_KEY" not in os.environ:
    st.error("⚠️ `GROQ_API_KEY` not found! Please set it as an environment variable or create a `.env` file.")
    st.info("You can get a free API key by signing up at https://groq.com")
    st.stop()

# 3. Initialize Persistent Session States
if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores conversation logs
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None  # Holds the operational backend pipeline

# 4. Sidebar File Upload Section
with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Choose a textbook PDF", type="pdf")
    
    if uploaded_file:
        # Only build the database if it hasn't been built for this session yet
        if st.session_state.rag_chain is None:
            with st.spinner("Analyzing and parsing your textbook... Please wait."):
                
                # A. Write memory buffer out to a temporary local file so PyPDFLoader can access it
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # B. Load the PDF pages
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    
                    # C. Chunking: Split text into 1000-character segments with overlap 
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_documents(docs)
                    
                    # D. Generate Local Embeddings (Free, runs entirely on your CPU via HuggingFace)
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    
                    # E. Create and index Vector Database via FAISS
                    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # Fetch top 4 sources
                    
                    # F. Connect Groq (Using Llama 3 or Mixtral models for supreme quality)
                    llm = ChatGroq(
                        model="llama3-8b-8192", 
                        temperature=0.2
                    )
                    
                    # G. Design the Academic System Prompt Blueprint
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
                    
                    # H. Splice everything into an integrated executable RAG pipeline
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    st.session_state.rag_chain = create_retrieval_chain(retriever, document_chain)
                    st.success("✅ Textbook successfully indexed! Ask away below.")
                    
                except Exception as e:
                    st.error(f"An error occurred while compiling your data: {e}")
                    
                finally:
                    # I. Clean up the temporary file off your disk storage safely
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

# 5. Main Screen Layout & Active Chat Window
if st.session_state.rag_chain is None:
    st.info("💡 Please upload a textbook PDF in the sidebar file uploader to activate your personal AI Tutor.")
else:
    # Render historic conversation blocks
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Monitor user typing query field
    if user_query := st.chat_input("What concept would you like explained?"):
        
        # Display the prompt right away 
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Fire off query downstream into Groq RAG ecosystem
        with st.chat_message("assistant"):
            with st.spinner("Flipping through textbook pages..."):
                response = st.session_state.rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                st.markdown(answer)
                
                # Expose verifiable document text layers dynamically
                with st.expander("🔍 View Textbook Sources Used"):
                    for doc in response["context"]:
                        page = doc.metadata.get("page", 0) + 1
                        st.write(f"**From Page {page}:**")
                        st.caption(f"_{doc.page_content[:300]}..._")
                        st.divider()
                        
        st.session_state.messages.append({"role": "assistant", "content": answer})
