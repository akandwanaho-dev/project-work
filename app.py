import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==========================================================
# 1. SETUP FREE HUGGING FACE INFERENCE ENGINE
# ==========================================================
def load_huggingface_llm():
    """
    Initializes a free serverless LLM endpoint from Hugging Face Hub.
    """
    # Safely extract your key from Streamlit Cloud Secrets
    hf_token = st.secrets.get("HF_TOKEN")
    
    if not hf_token:
        st.error("🔑 **HF_TOKEN Missing:** Please add your Hugging Face token to your Streamlit App Secrets.")
        st.stop()

    try:
        # We use Llama-3.1-8B-Instruct because it offers excellent reasoning constraints for RAG pipelines
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            task="text-generation",
            max_new_tokens=512,
            temperature=0.6,
            huggingfacehub_api_token=hf_token,
            timeout=30
        )
        
        # Wrap it in ChatHuggingFace so it formats multi-turn chat dialogues correctly
        return ChatHuggingFace(llm=llm_endpoint)
        
    except Exception as e:
        st.error(f"Failed to connect to Hugging Face endpoint: {e}")
        st.stop()

# ==========================================================
# 2. YOUR VECTOR DATABASE RETRIEVER INTERFACE
# ==========================================================
def initialize_retriever():
    """
    Loads your document knowledge base retriever.
    NOTE: Replace the fallback class below with your actual FAISS/Chroma database tool!
    """
    # EXAMPLE SWAP:
    # db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # return db.as_retriever(search_kwargs={"k": 3})
    
    class LocalMockRetriever:
        def invoke(self, query):
            return "ISCC Uganda Grand Championship Finals and Innovation Bootcamp are scheduled for December."
            
    return LocalMockRetriever()

# ==========================================================
# 3. BUILD THE RAG DATA CHAIN
# ==========================================================
def build_rag_pipeline(llm, retriever):
    """
    Connects Context + Input via LangChain Expression Language (LCEL)
    """
    system_prompt = """You are a smart assistant for the ISCC Uganda Coding Competition. 
    Use the provided piece of retrieved context below to answer the user's question accurately. 
    If you do not know the answer based on the context, politely say that you do not know.

    Context:
    {context}
    """
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Executable analytical flow map
    chain = (
        {"context": retriever | RunnablePassthrough(), "input": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return chain

# ==========================================================
# 4. STREAMLIT APPLICATION SURFACE
# ==========================================================
st.set_page_config(page_title="ISCC AI Bot", page_icon="⚡", layout="centered")
st.title("⚡ Competition RAG Chatbot")
st.caption("Powered by Hugging Face Hub Serverless APIs & LangChain")

# Instantiate and cache components inside Streamlit's global session state
if "rag_chain" not in st.session_state:
    with st.spinner("Initializing Hugging Face model environment..."):
        chat_llm = load_huggingface_llm()
        data_retriever = initialize_retriever()
        st.session_state.rag_chain = build_rag_pipeline(chat_llm, data_retriever)

# Persistent Chat Log History Array
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation streams
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt field detection
if user_query := st.chat_input("Ask something about the competition files..."):
    
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # 2. Execute RAG pipeline safely
    with st.chat_message("assistant"):
        with st.spinner("Searching document layers..."):
            try:
                # Running the new safe hugging face engine loop
                response = st.session_state.rag_chain.invoke(user_query)
                st.markdown(response)
                
                # Append to history state
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Pipeline error running Hugging Face model: {e}")
                st.info("💡 Tip: Verify your HF_TOKEN permissions or network rate-limits on Hugging Face console.")
