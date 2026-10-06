from pathlib import Path

app_code = r'''
import os
import io
import re
import json
import math
import time
import hashlib
from datetime import datetime
from collections import defaultdict

import streamlit as st
import pandas as pd
from pypdf import PdfReader
from docx import Document as DocxDocument
from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Textbook AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONSTANTS
# ============================================================
DEFAULT_LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ASR_MODEL = "openai/whisper-large-v3-turbo"

SUPPORTED_TEXTBOOK_TYPES = ["pdf", "docx", "txt", "md"]

STUDY_MODES = {
    "Normal Tutor": "Answer clearly and accurately. Explain enough for the student to understand.",
    "Explain Simply": "Explain in very simple language, use analogies, and avoid unnecessary jargon.",
    "Exam Mode": "Give an exam-ready answer with definition, key points, examples, and an exam tip.",
    "Summarize": "Give a concise but complete summary using headings and bullet points.",
    "Quiz Me": "Teach briefly, then end with 3 short questions for the student to answer.",
    "Flashcards": "Turn the answer into short question-and-answer flashcards.",
    "Revision Notes": "Produce organized revision notes with headings, definitions, key facts, and memory aids.",
    "Homework Helper": "Guide the student step by step. Show reasoning and method, not just the final result.",
    "Math Solver": "Show givens, formula, substitution, working, units, and final answer step by step.",
}

SUBJECT_PERSONAS = {
    "General": "You are an excellent multidisciplinary textbook tutor.",
    "Biology": "You are a biology tutor. Emphasize structure, function, process, importance, and examples.",
    "Chemistry": "You are a chemistry tutor. Show equations where useful and explain observations and chemical reasoning.",
    "Physics": "You are a physics tutor. Define quantities, show formulas, units, substitutions, and interpretations.",
    "Mathematics": "You are a mathematics tutor. Show complete working clearly and check the final answer.",
    "Computer Science": "You are a computer science tutor. Explain concepts clearly and use small code examples when useful.",
}

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --radius: 18px;
    }
    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,.18);
    }
    .hero {
        padding: 1.35rem 1.5rem;
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(92,94,255,.13), rgba(0,200,170,.08));
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
    }
    .hero p {
        margin: .35rem 0 0 0;
        opacity: .82;
    }
    .metric-card {
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 18px;
        padding: 1rem;
        min-height: 110px;
    }
    .source-card {
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 14px;
        padding: .75rem .9rem;
        margin: .35rem 0;
        background: rgba(128,128,128,.04);
    }
    .small-muted {
        opacity: .72;
        font-size: .88rem;
    }
    .book-chip {
        display:inline-block;
        padding:.28rem .58rem;
        border-radius:999px;
        border:1px solid rgba(128,128,128,.22);
        margin:.15rem .2rem .15rem 0;
        font-size:.82rem;
    }
    .stButton button {
        border-radius: 12px;
    }
    [data-testid="stChatMessage"] {
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.10);
        padding: .35rem .55rem;
        margin-bottom: .45rem;
    }
    div[data-testid="stExpander"] {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "books": {},                  # name -> metadata
        "documents": [],              # chunked LangChain docs
        "vectorstore": None,
        "messages": [],
        "chat_sessions": {"New Chat": []},
        "active_chat": "New Chat",
        "quiz": [],
        "quiz_submitted": False,
        "flashcards": [],
        "progress": {
            "questions_asked": 0,
            "quizzes_taken": 0,
            "quiz_correct": 0,
            "quiz_total": 0,
            "topics": defaultdict(int),
        },
        "last_sources": [],
        "last_search_results": [],
        "notice": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# ============================================================
# UTILITIES
# ============================================================
def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)

def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._ -]", "_", name).strip()

def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def detect_chapter(text: str, fallback="Unknown chapter"):
    candidates = [
        r"(?im)^\s*(chapter\s+\d+(?:\.\d+)?\s*[:\-–]?\s*[^\n]{0,90})",
        r"(?im)^\s*(unit\s+\d+(?:\.\d+)?\s*[:\-–]?\s*[^\n]{0,90})",
        r"(?im)^\s*(topic\s+\d+(?:\.\d+)?\s*[:\-–]?\s*[^\n]{0,90})",
    ]
    for pat in candidates:
        m = re.search(pat, text[:4000])
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    return fallback

def extract_pdf(data: bytes, source_name: str):
    docs = []
    reader = PdfReader(io.BytesIO(data))
    current_chapter = "Unknown chapter"
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue
        chapter_guess = detect_chapter(text, current_chapter)
        if chapter_guess != "Unknown chapter":
            current_chapter = chapter_guess
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "page": i,
                    "chapter": current_chapter,
                    "type": "pdf",
                },
            )
        )
    return docs

def extract_docx(data: bytes, source_name: str):
    docx = DocxDocument(io.BytesIO(data))
    docs = []
    page_like_parts = []
    current_chapter = "Unknown chapter"

    buffer = []
    section_idx = 1
    for p in docx.paragraphs:
        text = clean_text(p.text)
        if not text:
            continue
        if re.match(r"(?i)^(chapter|unit|topic)\s+\d+", text):
            if buffer:
                page_like_parts.append((section_idx, current_chapter, "\n".join(buffer)))
                section_idx += 1
                buffer = []
            current_chapter = text
        buffer.append(text)

    if buffer:
        page_like_parts.append((section_idx, current_chapter, "\n".join(buffer)))

    for section_no, chapter, text in page_like_parts:
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "page": section_no,
                    "chapter": chapter,
                    "type": "docx",
                },
            )
        )
    return docs

def extract_text_file(data: bytes, source_name: str, file_type: str):
    text = clean_text(data.decode("utf-8", errors="ignore"))
    if not text:
        return []

    # Split very long plain-text files into "sections" before chunking.
    raw_sections = re.split(r"(?im)(?=^\s*(?:chapter|unit|topic)\s+\d+)", text)
    docs = []
    current_chapter = "Unknown chapter"
    section_no = 1

    for section in raw_sections:
        section = clean_text(section)
        if not section:
            continue
        current_chapter = detect_chapter(section, current_chapter)
        docs.append(
            Document(
                page_content=section,
                metadata={
                    "source": source_name,
                    "page": section_no,
                    "chapter": current_chapter,
                    "type": file_type,
                },
            )
        )
        section_no += 1
    return docs

def load_uploaded_book(uploaded_file):
    data = uploaded_file.getvalue()
    name = safe_filename(uploaded_file.name)
    ext = name.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        raw_docs = extract_pdf(data, name)
    elif ext == "docx":
        raw_docs = extract_docx(data, name)
    elif ext in ("txt", "md"):
        raw_docs = extract_text_file(data, name, ext)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{file_hash(data)[:12]}-{idx}"
        chunk.metadata["file_hash"] = file_hash(data)

    chapters = sorted({d.metadata.get("chapter", "Unknown chapter") for d in chunks})
    pages = sorted({d.metadata.get("page") for d in raw_docs if d.metadata.get("page") is not None})

    meta = {
        "name": name,
        "hash": file_hash(data),
        "size_mb": round(len(data) / (1024 * 1024), 2),
        "chunks": len(chunks),
        "pages_or_sections": len(pages),
        "chapters": chapters,
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return chunks, meta

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def rebuild_vectorstore():
    if not st.session_state.documents:
        st.session_state.vectorstore = None
        return
    embeddings = load_embeddings()
    st.session_state.vectorstore = FAISS.from_documents(
        st.session_state.documents,
        embeddings,
    )

def add_uploaded_books(uploaded_files):
    if not uploaded_files:
        return

    existing_hashes = {meta["hash"] for meta in st.session_state.books.values()}
    added = 0
    errors = []

    for uploaded_file in uploaded_files:
        try:
            data = uploaded_file.getvalue()
            h = file_hash(data)
            if h in existing_hashes:
                continue
            chunks, meta = load_uploaded_book(uploaded_file)
            st.session_state.documents.extend(chunks)
            st.session_state.books[meta["name"]] = meta
            existing_hashes.add(h)
            added += 1
        except Exception as e:
            errors.append(f"{uploaded_file.name}: {e}")

    if added:
        with st.spinner("Building textbook search index..."):
            rebuild_vectorstore()
        st.session_state.notice = f"Indexed {added} textbook(s)."

    if errors:
        st.error("Some files could not be indexed:\n\n" + "\n".join(errors))

def delete_book(book_name):
    if book_name not in st.session_state.books:
        return
    st.session_state.documents = [
        d for d in st.session_state.documents
        if d.metadata.get("source") != book_name
    ]
    del st.session_state.books[book_name]
    rebuild_vectorstore()

def hf_client():
    token = get_secret("HF_TOKEN", "")
    if not token:
        return None
    return InferenceClient(token=token, timeout=90)

def call_llm(messages, model, temperature=0.4, max_tokens=900):
    client = hf_client()
    if client is None:
        raise RuntimeError(
            "HF_TOKEN is missing. Add it in Streamlit Secrets as HF_TOKEN='your_token'."
        )

    result = client.chat_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return result.choices[0].message.content.strip()

def retrieve_docs(query, selected_books, selected_chapters, k=7):
    if st.session_state.vectorstore is None:
        return []

    # Retrieve a wider candidate set first, then apply UI filters.
    candidates = st.session_state.vectorstore.similarity_search(query, k=max(24, k * 4))

    filtered = []
    seen = set()

    for d in candidates:
        src = d.metadata.get("source")
        chapter = d.metadata.get("chapter", "Unknown chapter")
        if selected_books and src not in selected_books:
            continue
        if selected_chapters and chapter not in selected_chapters:
            continue

        key = d.metadata.get("chunk_id") or (
            src,
            d.metadata.get("page"),
            d.page_content[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        filtered.append(d)

        if len(filtered) >= k:
            break

    return filtered

def format_context(docs):
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "Unknown source")
        page = d.metadata.get("page", "?")
        chapter = d.metadata.get("chapter", "Unknown chapter")
        blocks.append(
            f"[SOURCE {i} | {src} | page/section {page} | {chapter}]\n{d.page_content}"
        )
    return "\n\n".join(blocks)

def source_summary(docs):
    summaries = []
    seen = set()
    for d in docs:
        src = d.metadata.get("source", "Unknown source")
        page = d.metadata.get("page", "?")
        chapter = d.metadata.get("chapter", "Unknown chapter")
        key = (src, page, chapter)
        if key in seen:
            continue
        seen.add(key)
        summaries.append(
            {
                "source": src,
                "page": page,
                "chapter": chapter,
                "excerpt": d.page_content[:420].strip(),
            }
        )
    return summaries[:8]

def build_answer(
    query,
    docs,
    study_mode,
    subject,
    answer_length,
    source_mode,
    conversation,
    model,
    temperature,
):
    mode_instruction = STUDY_MODES[study_mode]
    persona = SUBJECT_PERSONAS[subject]

    length_instruction = {
        "Short": "Keep the answer concise.",
        "Medium": "Give a moderately detailed answer.",
        "Detailed": "Give a thorough, well-structured explanation.",
    }[answer_length]

    if docs:
        context = format_context(docs)
    else:
        context = "(No textbook passages were retrieved.)"

    textbook_only = source_mode == "Textbook only"

    if textbook_only:
        grounding_rule = """
Use the supplied textbook context as the only factual source.
If the context is insufficient, clearly say:
"I could not find enough information in the selected textbook material."
Do not invent page numbers or textbook claims.
"""
    else:
        grounding_rule = """
Use the supplied textbook context first.
You may use reliable general knowledge to explain missing background.
Clearly label information that is not directly supported by the supplied textbook context as "General knowledge".
Do not invent textbook citations.
"""

    history = []
    for m in conversation[-8:]:
        if m.get("role") in ("user", "assistant"):
            history.append({"role": m["role"], "content": m["content"]})

    system = f"""
{persona}

You are part of a textbook-learning RAG application.

Teaching mode: {study_mode}
Teaching instruction: {mode_instruction}
Answer length: {answer_length}
Length instruction: {length_instruction}

{grounding_rule}

Citation rules:
- When using textbook evidence, cite it inline using this exact style:
  [Source: filename, p. 12]
- For DOCX/TXT/MD files, "p." means section number.
- Only cite a page/section that is present in the provided context.
- Never fabricate a citation.
- Preserve important scientific and mathematical terminology.
- Use Markdown headings, bullets, equations, or tables when they improve clarity.

TEXTBOOK CONTEXT:
{context}
""".strip()

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    max_tokens = {"Short": 500, "Medium": 850, "Detailed": 1300}[answer_length]
    return call_llm(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

def create_quiz(topic, docs, count, difficulty, model):
    context = format_context(docs) if docs else "(No textbook context available.)"
    prompt = f"""
Create exactly {count} multiple-choice questions about: {topic}

Difficulty: {difficulty}

Use the textbook context whenever it contains relevant material.
Return ONLY valid JSON in this format:
[
  {{
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "answer_index": 0,
    "explanation": "...",
    "source": "filename, page/section X"
  }}
]

Rules:
- Each question must have exactly 4 options.
- answer_index must be 0, 1, 2, or 3.
- Do not wrap JSON in Markdown fences.
- If a source is not supported by context, use "General knowledge" instead of inventing a source.

CONTEXT:
{context}
"""
    text = call_llm(
        [
            {"role": "system", "content": "You generate reliable educational assessment content and valid JSON."},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.3,
        max_tokens=1800,
    )
    text = text.strip()
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError("The model did not return a valid quiz JSON array.")
    data = json.loads(match.group(0))

    valid = []
    for q in data:
        if (
            isinstance(q, dict)
            and isinstance(q.get("options"), list)
            and len(q["options"]) == 4
            and isinstance(q.get("answer_index"), int)
            and 0 <= q["answer_index"] <= 3
        ):
            valid.append(q)
    if not valid:
        raise ValueError("No valid quiz questions were returned.")
    return valid[:count]

def create_flashcards(topic, docs, count, model):
    context = format_context(docs) if docs else "(No textbook context available.)"
    prompt = f"""
Create exactly {count} study flashcards about: {topic}

Return ONLY valid JSON:
[
  {{
    "front": "question or term",
    "back": "clear answer",
    "source": "filename, page/section X"
  }}
]

Use textbook material when available.
Never invent textbook source details.
If unsupported by textbook context, set source to "General knowledge".

CONTEXT:
{context}
"""
    text = call_llm(
        [
            {"role": "system", "content": "You create accurate revision flashcards and valid JSON."},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.25,
        max_tokens=1500,
    )
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError("The model did not return valid flashcard JSON.")
    data = json.loads(match.group(0))
    valid = [x for x in data if isinstance(x, dict) and x.get("front") and x.get("back")]
    if not valid:
        raise ValueError("No valid flashcards were returned.")
    return valid[:count]

def transcribe_audio(audio_bytes):
    client = hf_client()
    if client is None:
        raise RuntimeError("HF_TOKEN is missing.")
    output = client.automatic_speech_recognition(
        audio_bytes,
        model=DEFAULT_ASR_MODEL,
    )
    return output.text.strip()

def web_search_tavily(query):
    # Optional real web research mode.
    # It activates only if TAVILY_API_KEY is configured.
    import requests

    key = get_secret("TAVILY_API_KEY", "")
    if not key:
        return []

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": key,
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
            "include_answer": False,
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("results", []):
        results.append(
            {
                "title": item.get("title", "Web result"),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
        )
    return results

def build_web_answer(query, docs, web_results, study_mode, subject, answer_length, model, temperature):
    textbook_context = format_context(docs) if docs else "(No textbook material retrieved.)"
    web_context = "\n\n".join(
        f"[WEB {i}] {r['title']}\nURL: {r['url']}\n{r['content']}"
        for i, r in enumerate(web_results, start=1)
    ) or "(No web results.)"

    system = f"""
{SUBJECT_PERSONAS[subject]}
You are a careful research tutor.

Teaching mode: {study_mode}
{STUDY_MODES[study_mode]}

Use both textbook and web material below.
Distinguish textbook evidence from web evidence.
Do not invent URLs, page numbers, or sources.
For textbook claims use [Source: filename, p. X].
For web claims use [Web: result number].

TEXTBOOK:
{textbook_context}

WEB:
{web_context}
"""
    max_tokens = {"Short": 500, "Medium": 850, "Detailed": 1300}[answer_length]
    return call_llm(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 📚 Textbook AI")
    st.caption("Your private study workspace")

    hf_token_present = bool(get_secret("HF_TOKEN", ""))
    if hf_token_present:
        st.success("AI connection ready", icon="✅")
    else:
        st.warning("Add `HF_TOKEN` in Streamlit Secrets.", icon="🔑")

    st.divider()

    st.markdown("### Library")
    uploads = st.file_uploader(
        "Upload textbooks",
        type=SUPPORTED_TEXTBOOK_TYPES,
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, TXT, MD. Scanned image-only PDFs may not extract text correctly.",
    )

    if st.button("📥 Index uploaded files", use_container_width=True, disabled=not uploads):
        add_uploaded_books(uploads)
        st.rerun()

    if st.session_state.notice:
        st.success(st.session_state.notice)
        st.session_state.notice = ""

    if st.session_state.books:
        for book_name, meta in list(st.session_state.books.items()):
            with st.expander(f"📘 {book_name}", expanded=False):
                st.caption(
                    f"{meta['size_mb']} MB • {meta['chunks']} chunks • "
                    f"{meta['pages_or_sections']} pages/sections"
                )
                st.caption(f"Uploaded: {meta['uploaded_at']}")
                if st.button("Remove book", key=f"remove_{meta['hash']}", use_container_width=True):
                    delete_book(book_name)
                    st.rerun()
    else:
        st.info("Upload one or more textbooks to enable RAG.")

    st.divider()
    st.markdown("### Chat sessions")

    new_chat_name = st.text_input("New chat name", placeholder="e.g. Cell Biology")
    if st.button("➕ Create chat", use_container_width=True):
        name = new_chat_name.strip() or f"Chat {len(st.session_state.chat_sessions)+1}"
        if name not in st.session_state.chat_sessions:
            st.session_state.chat_sessions[name] = []
        st.session_state.active_chat = name
        st.session_state.messages = st.session_state.chat_sessions[name]
        st.rerun()

    chats = list(st.session_state.chat_sessions.keys())
    active_index = chats.index(st.session_state.active_chat) if st.session_state.active_chat in chats else 0
    chosen_chat = st.selectbox("Open chat", chats, index=active_index)
    if chosen_chat != st.session_state.active_chat:
        st.session_state.active_chat = chosen_chat
        st.session_state.messages = st.session_state.chat_sessions[chosen_chat]
        st.rerun()

    if st.button("🗑️ Clear current chat", use_container_width=True):
        st.session_state.chat_sessions[st.session_state.active_chat] = []
        st.session_state.messages = st.session_state.chat_sessions[st.session_state.active_chat]
        st.rerun()

    st.divider()
    st.markdown("### Model settings")

    llm_model = st.text_input(
        "Hugging Face chat model",
        value=get_secret("HF_MODEL", DEFAULT_LLM_MODEL) or DEFAULT_LLM_MODEL,
    )
    temperature = st.slider("Creativity", 0.0, 1.0, 0.35, 0.05)
    answer_length = st.selectbox("Answer length", ["Short", "Medium", "Detailed"], index=1)
    teaching_level = st.selectbox(
        "Student level",
        ["Beginner", "Secondary School", "Advanced"],
        index=1,
    )

# ============================================================
# MAIN HEADER
# ============================================================
st.markdown(
    f"""
    <div class="hero">
        <h1>📚 Textbook AI</h1>
        <p>Upload textbooks, ask grounded questions, generate quizzes and flashcards, search your library, and track revision progress.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Selection controls
book_names = list(st.session_state.books.keys())
selected_books = st.multiselect(
    "Search in textbooks",
    options=book_names,
    default=book_names,
    placeholder="Choose one or more textbooks",
)

available_chapters = sorted({
    d.metadata.get("chapter", "Unknown chapter")
    for d in st.session_state.documents
    if not selected_books or d.metadata.get("source") in selected_books
})

with st.expander("🎛️ Study controls", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        subject = st.selectbox("Subject tutor", list(SUBJECT_PERSONAS.keys()))
    with c2:
        study_mode = st.selectbox("Study mode", list(STUDY_MODES.keys()))
    with c3:
        source_mode = st.selectbox(
            "Knowledge mode",
            ["Textbook only", "Textbook + general knowledge", "Textbook + web research"],
        )
    with c4:
        retrieval_k = st.slider("Retrieved passages", 3, 10, 6)

    selected_chapters = st.multiselect(
        "Limit to chapters/topics",
        available_chapters,
        default=[],
        placeholder="All chapters",
    )

    st.caption(f"Teaching level: **{teaching_level}**")

if source_mode == "Textbook + web research" and not get_secret("TAVILY_API_KEY", ""):
    st.info(
        "Web research mode needs a `TAVILY_API_KEY` in Streamlit Secrets. "
        "Without it, the app will fall back to textbook + general knowledge."
    )

tabs = st.tabs(["💬 Chat", "🔎 Search", "📝 Quiz", "🎴 Flashcards", "📊 Progress", "ℹ️ Help"])

# ============================================================
# CHAT TAB
# ============================================================
with tabs[0]:
    if not st.session_state.messages:
        st.markdown("### What would you like to study?")
        q1, q2, q3 = st.columns(3)
        q1.info("**Explain a topic**\n\n“Explain mitosis in simple terms.”")
        q2.info("**Prepare for exams**\n\n“Give me revision notes on electrolysis.”")
        q3.info("**Solve step-by-step**\n\n“Show how to solve this quadratic equation.”")

    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "📚"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📖 Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f"""
                            <div class="source-card">
                            <b>{src['source']}</b> — page/section {src['page']}<br>
                            <span class="small-muted">{src['chapter']}</span><br><br>
                            {src['excerpt']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            if msg.get("web_sources"):
                with st.expander("🌐 Web sources", expanded=False):
                    for i, src in enumerate(msg["web_sources"], start=1):
                        st.markdown(f"{i}. [{src['title']}]({src['url']})")

    voice_text = ""
    with st.expander("🎤 Ask by voice", expanded=False):
        audio = st.audio_input("Record a question")
        if audio is not None:
            if st.button("Transcribe recording"):
                try:
                    with st.spinner("Transcribing..."):
                        voice_text = transcribe_audio(audio.getvalue())
                    st.success(f"Transcript: {voice_text}")
                    st.session_state["voice_transcript"] = voice_text
                except Exception as e:
                    st.error(f"Voice transcription failed: {e}")

        if st.session_state.get("voice_transcript"):
            st.caption("Copy the transcript into the chat box below, or edit it first.")
            st.code(st.session_state["voice_transcript"])

    prompt = st.chat_input(
        "Ask about your textbooks...",
        accept_file=True,
        file_type=["png", "jpg", "jpeg"],
        disabled=not hf_token_present,
    )

    if prompt:
        # New Streamlit returns a ChatInputValue when attachments are enabled.
        if isinstance(prompt, str):
            user_query = prompt
            attachments = []
        else:
            user_query = prompt.text or ""
            attachments = list(prompt.files or [])

        if attachments:
            st.warning(
                "Image attachments are accepted in the interface, but this version's "
                "Llama text model cannot directly understand images. Type the question shown "
                "in the image, or switch the backend to a vision-language model."
            )

        if user_query.strip():
            user_query = user_query.strip()
            user_msg = {"role": "user", "content": user_query}
            st.session_state.messages.append(user_msg)
            st.session_state.chat_sessions[st.session_state.active_chat] = st.session_state.messages

            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(user_query)

            with st.chat_message("assistant", avatar="📚"):
                try:
                    with st.spinner("Searching your textbooks and preparing the answer..."):
                        docs = retrieve_docs(
                            user_query,
                            selected_books,
                            selected_chapters,
                            k=retrieval_k,
                        )
                        web_sources = []

                        if source_mode == "Textbook + web research" and get_secret("TAVILY_API_KEY", ""):
                            web_sources = web_search_tavily(user_query)
                            response = build_web_answer(
                                user_query,
                                docs,
                                web_sources,
                                study_mode,
                                subject,
                                answer_length,
                                llm_model,
                                temperature,
                            )
                        else:
                            effective_mode = source_mode
                            if source_mode == "Textbook + web research":
                                effective_mode = "Textbook + general knowledge"

                            response = build_answer(
                                user_query,
                                docs,
                                study_mode,
                                subject,
                                answer_length,
                                effective_mode,
                                st.session_state.messages[:-1],
                                llm_model,
                                temperature,
                            )

                    st.markdown(response)
                    sources = source_summary(docs)
                    st.session_state.last_sources = sources

                    if sources:
                        with st.expander("📖 Sources used", expanded=False):
                            for src in sources:
                                st.markdown(
                                    f"**{src['source']}** — page/section {src['page']}  \n"
                                    f"*{src['chapter']}*"
                                )

                    if web_sources:
                        with st.expander("🌐 Web sources used", expanded=False):
                            for i, src in enumerate(web_sources, start=1):
                                st.markdown(f"{i}. [{src['title']}]({src['url']})")

                    assistant_msg = {
                        "role": "assistant",
                        "content": response,
                        "sources": sources,
                        "web_sources": web_sources,
                    }
                    st.session_state.messages.append(assistant_msg)
                    st.session_state.chat_sessions[st.session_state.active_chat] = st.session_state.messages

                    st.session_state.progress["questions_asked"] += 1
                    topic_key = user_query[:50]
                    st.session_state.progress["topics"][topic_key] += 1

                except Exception as e:
                    st.error(f"AI error: {e}")

# ============================================================
# SEARCH TAB
# ============================================================
with tabs[1]:
    st.markdown("### 🔎 Search your textbook library")
    search_query = st.text_input("Search phrase or concept", placeholder="e.g. osmosis, Newton's second law")
    search_count = st.slider("Number of results", 3, 15, 8, key="search_count")

    if st.button("Search textbooks", disabled=not search_query.strip()):
        if st.session_state.vectorstore is None:
            st.warning("Upload and index at least one textbook first.")
        else:
            st.session_state.last_search_results = retrieve_docs(
                search_query,
                selected_books,
                selected_chapters,
                k=search_count,
            )

    if st.session_state.last_search_results:
        for i, d in enumerate(st.session_state.last_search_results, start=1):
            with st.expander(
                f"{i}. {d.metadata.get('source')} — page/section {d.metadata.get('page')}",
                expanded=i <= 3,
            ):
                st.caption(d.metadata.get("chapter", "Unknown chapter"))
                st.write(d.page_content)

# ============================================================
# QUIZ TAB
# ============================================================
with tabs[2]:
    st.markdown("### 📝 Generate a textbook quiz")
    qc1, qc2, qc3 = st.columns([2, 1, 1])
    with qc1:
        quiz_topic = st.text_input("Quiz topic", placeholder="e.g. Cell division")
    with qc2:
        quiz_count = st.selectbox("Questions", [3, 5, 8, 10], index=1)
    with qc3:
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)

    if st.button("✨ Generate quiz", disabled=not quiz_topic.strip()):
        try:
            docs = retrieve_docs(
                quiz_topic,
                selected_books,
                selected_chapters,
                k=min(10, retrieval_k + 2),
            )
            with st.spinner("Generating quiz..."):
                st.session_state.quiz = create_quiz(
                    quiz_topic,
                    docs,
                    quiz_count,
                    difficulty,
                    llm_model,
                )
            st.session_state.quiz_submitted = False
            st.rerun()
        except Exception as e:
            st.error(f"Could not generate quiz: {e}")

    if st.session_state.quiz:
        with st.form("quiz_form"):
            answers = []
            for i, q in enumerate(st.session_state.quiz):
                st.markdown(f"**{i+1}. {q['question']}**")
                choice = st.radio(
                    "Choose an answer",
                    q["options"],
                    key=f"quiz_answer_{i}",
                    index=None,
                    label_visibility="collapsed",
                )
                answers.append(choice)
                st.divider()

            submitted = st.form_submit_button("Submit answers", use_container_width=True)

        if submitted:
            correct = 0
            for i, q in enumerate(st.session_state.quiz):
                selected = answers[i]
                correct_option = q["options"][q["answer_index"]]
                if selected == correct_option:
                    correct += 1

            st.session_state.progress["quizzes_taken"] += 1
            st.session_state.progress["quiz_correct"] += correct
            st.session_state.progress["quiz_total"] += len(st.session_state.quiz)
            st.session_state.quiz_submitted = True
            st.session_state["last_quiz_answers"] = answers

        if st.session_state.quiz_submitted:
            answers = st.session_state.get("last_quiz_answers", [])
            correct = 0
            for i, q in enumerate(st.session_state.quiz):
                selected = answers[i] if i < len(answers) else None
                correct_option = q["options"][q["answer_index"]]
                is_correct = selected == correct_option
                correct += int(is_correct)

                if is_correct:
                    st.success(f"Question {i+1}: Correct")
                else:
                    st.error(
                        f"Question {i+1}: Correct answer — {correct_option}"
                    )
                st.write(q.get("explanation", ""))
                if q.get("source"):
                    st.caption(f"Source: {q['source']}")

            score = round(100 * correct / len(st.session_state.quiz))
            st.metric("Quiz score", f"{score}%")

# ============================================================
# FLASHCARDS TAB
# ============================================================
with tabs[3]:
    st.markdown("### 🎴 Generate revision flashcards")
    fc1, fc2 = st.columns([3, 1])
    with fc1:
        flash_topic = st.text_input("Flashcard topic", placeholder="e.g. Organic chemistry reactions")
    with fc2:
        flash_count = st.selectbox("Cards", [5, 8, 10, 15], index=1)

    if st.button("✨ Generate flashcards", disabled=not flash_topic.strip()):
        try:
            docs = retrieve_docs(
                flash_topic,
                selected_books,
                selected_chapters,
                k=min(10, retrieval_k + 2),
            )
            with st.spinner("Creating flashcards..."):
                st.session_state.flashcards = create_flashcards(
                    flash_topic,
                    docs,
                    flash_count,
                    llm_model,
                )
            st.rerun()
        except Exception as e:
            st.error(f"Could not create flashcards: {e}")

    if st.session_state.flashcards:
        for i, card in enumerate(st.session_state.flashcards, start=1):
            with st.expander(f"Card {i}: {card['front']}", expanded=False):
                st.markdown(card["back"])
                st.caption(f"Source: {card.get('source', 'Unknown')}")

# ============================================================
# PROGRESS TAB
# ============================================================
with tabs[4]:
    st.markdown("### 📊 Study progress")
    progress = st.session_state.progress

    accuracy = (
        100 * progress["quiz_correct"] / progress["quiz_total"]
        if progress["quiz_total"] else 0
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Questions asked", progress["questions_asked"])
    m2.metric("Quizzes taken", progress["quizzes_taken"])
    m3.metric("Quiz accuracy", f"{accuracy:.0f}%")
    m4.metric("Textbooks", len(st.session_state.books))

    st.markdown("#### Library overview")
    if st.session_state.books:
        rows = []
        for meta in st.session_state.books.values():
            rows.append(
                {
                    "Textbook": meta["name"],
                    "Pages/sections": meta["pages_or_sections"],
                    "Chunks": meta["chunks"],
                    "Size (MB)": meta["size_mb"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No books indexed yet.")

    st.markdown("#### Recently studied questions")
    topics = dict(progress["topics"])
    if topics:
        topic_rows = [
            {"Question/topic": k, "Times studied": v}
            for k, v in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        st.dataframe(pd.DataFrame(topic_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Ask some questions to build your progress history.")

# ============================================================
# HELP TAB
# ============================================================
with tabs[5]:
    st.markdown("### ℹ️ Setup and features")
    st.markdown(
        """
        **Core features included**
        - Multiple PDF, DOCX, TXT, and Markdown textbook uploads
        - Real FAISS vector search with local sentence-transformer embeddings
        - Page/section metadata and textbook source citations
        - Textbook-only and textbook + general-knowledge modes
        - Optional live web research with Tavily
        - Subject-specific tutor personalities
        - Explain Simply, Exam, Summary, Quiz, Flashcard, Revision, Homework, and Math modes
        - Multi-chat session history during the current Streamlit session
        - Search across uploaded textbooks
        - Quiz generation, scoring, and progress tracking
        - Flashcard generation
        - Voice-question transcription through Hugging Face ASR
        - Image attachment UI for future vision-model support
        """
    )

    st.markdown("#### Streamlit Secrets")
    st.code(
       HF_TOKEN = "hf_your_actual_token"

        # Optional
        HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
        
        # Optional — only needed for web research
        TAVILY_API_KEY = "tvly_your_key"
    st.warning(
        "Do not place API keys directly inside `app.py` or commit them to GitHub."
    )

    st.markdown("#### Important limitations")
    st.markdown(
        """
        - **Scanned PDFs:** `pypdf` extracts embedded text, not OCR. Image-only/scanned textbooks need an OCR pipeline.
        - **Image questions:** the current Llama text model cannot inspect uploaded photos. The UI accepts them, but true image understanding requires a vision-language model.
        - **Persistence:** chats, books, quiz scores, and the FAISS index are stored in Streamlit session memory. A production app should save them to a database/object store.
        - **Hugging Face availability:** serverless model/provider availability can change. If a model is unavailable, set `HF_MODEL` to another chat-capable model available to your account.
        """
    )
'''

requirements = '''streamlit>=1.63,<2
huggingface-hub>=0.35
langchain-core>=0.3,<2
langchain-community>=0.3,<2
langchain-text-splitters>=0.3,<2
langchain-huggingface>=0.3,<2
sentence-transformers>=3.0
faiss-cpu>=1.8
pypdf>=5.0
python-docx>=1.1
pandas>=2.2
requests>=2.32
'''

Path("/mnt/data/app.py").write_text(app_code, encoding="utf-8")
Path("/mnt/data/requirements.txt").write_text(requirements, encoding="utf-8")

# basic syntax check
compile(app_code, "/mnt/data/app.py", "exec")

print("Created app.py and requirements.txt successfully.")
