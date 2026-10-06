import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ==========================================================
# 1. SETUP FREE HUGGING FACE INFERENCE ENGINE
# ==========================================================
def load_huggingface_llm(hf_token):
    """
    Initializes a free serverless LLM endpoint from Hugging Face Hub using the provided token.
    """
    try:
        # Using Llama-3.1-8B-Instruct for excellent RAG compliance
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            task="text-generation",
            max_new_tokens=512,
            temperature=0.6,
            huggingfacehub_api_token=hf_token,
            timeout=30
        )
        # Wrap it in ChatHuggingFace for correct multi-turn conversation formatting
        return ChatHuggingFace(llm=llm_endpoint)
    except Exception as e:
        st.error(f"⚠️ Error initializing Hugging Face model connection: {e}")
        return None

# ==========================================================
# 2. VECTOR DATABASE RETRIEVER INTERFACE
# ==========================================================
def initialize_retriever():
    """
    Loads your document knowledge base retriever.
    NOTE: Replace this with your actual FAISS/Chroma database tool!
    """
    def mock_retrieve(query: str):
        # Your actual vector DB will return a list of Documents; 
        # this mock returns a raw string for compatibility with your template.
        return "ISCC Uganda Grand Championship Finals and Innovation Bootcamp are scheduled for December."
            
    # Wrap it in a RunnableLambda so LangChain operators (|) work seamlessly
    return RunnableLambda(mock_retrieve)

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
    
    # FIX: Correct entry-point map configuration for LCEL
    chain = (
        {
            "context": retriever, 
            "input": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return chain

# ==========================================================
# 4. STREAMLIT APPLICATION SURFACE (UI Elements Always Render)
# ==========================================================
st.set_page_config(page_title="ISCC AI Bot", page_icon="⚡", layout="centered")
st.title("⚡ Competition RAG Chatbot")
st.caption("Powered by Hugging Face Hub Serverless APIs & LangChain")

# --- Security & Token Validation Gate ---
hf_token = st.secrets.get("HF_TOKEN")
api_is_valid = False

if not hf_token or hf_token.strip() == "":
    st.warning("⚠️ **System Configuration Alert:** The `HF_TOKEN` API key is missing. Please navigate to your App Settings -> Secrets panel on Streamlit Cloud to add it. Chat capabilities are currently locked.")
else:
    api_is_valid = True

# Instantiate and cache components inside Streamlit's global session state if key exists
if api_is_valid and "rag_chain" not in st.session_state:
    with st.spinner("Initializing Hugging Face model environment..."):
        chat_llm = load_huggingface_llm(hf_token)
        if chat_llm:
            data_retriever = initialize_retriever()
            st.session_state.rag_chain = build_rag_pipeline(chat_llm, data_retriever)

# Persistent Chat Log History Array
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation streams
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Conditional Input Lock ---
if api_is_valid:
    user_query = st.chat_input("Ask something about the competition files...")
else:
    user_query = st.chat_input("Chat disabled — missing API Configuration Token", disabled=True)

# 5. Process user prompt if available
if user_query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Execute RAG pipeline safely
    with st.chat_message("assistant"):
        with st.spinner("Searching document layers..."):
            try:
                if "rag_chain" in st.session_state:
                    response = st.session_state.rag_chain.invoke(user_query)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("RAG pipeline failed to initialize properly. Please check logs.")
            except Exception as e:
                st.error(f"Pipeline error running Hugging Face model: {e}")
