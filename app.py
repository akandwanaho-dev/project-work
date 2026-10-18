import io
import os
import re
import json
import hashlib
import base64
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace
import pandas as pd
import requests
import streamlit as st
from supabase import create_client
from docx import Document as DocxDocument
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="izy read", layout="wide")
DEFAULT_LLM_MODEL = "gemini-1.5-flash"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

STUDY_MODES = {
    "Normal Tutor": "Explain clearly and accurately with useful examples.",
    "Explain Simply": "Use simple language, analogies, and short steps.",
    "Exam Mode": "Give an exam-ready definition, key points, examples, and an exam tip.",
    "Summarize": "Give a concise structured summary.",
    "Quiz Me": "Teach briefly, then end with three questions.",
    "Flashcards": "Present the answer as compact question-and-answer flashcards.",
    "Revision Notes": "Create revision notes with headings, definitions, facts, and memory aids.",
    "Homework Helper": "Guide step by step and explain the method.",
    "Math Solver": "Show givens, formula, substitution, working, units, and final answer.",
}

SUBJECTS = {
    "General": "You are an excellent multidisciplinary textbook tutor.",
    "Biology": "You are a biology tutor. Emphasize structure, function, processes, importance, and examples.",
    "Chemistry": "You are a chemistry tutor. Use balanced equations and explain observations when useful.",
    "Physics": "You are a physics tutor. Show formulas, substitutions, units, and interpretations.",
    "Mathematics": "You are a mathematics tutor. Show complete working and check answers.",
    "Computer Science": "You are a computer science tutor. Explain concepts and use small code examples when useful.",
}

# ============================================================
# STATE
# ============================================================
def init_state():
    defaults = {
        "books": {},
        "documents": [],
        "vectorstore": None,
        "messages": [],
        "chat_sessions": {"New Chat": []},
        "active_chat": "New Chat",
        "quiz": [],
        "quiz_answers": [],
        "quiz_submitted": False,
        "quiz_history": [],
        "quiz_topic": "",
        "quiz_history_selected": None,
        "active_view": "home",
        "flashcards": [],
        "last_search_results": [],
        "supabase": None,
        "current_user": None,
        "profile": {},
        "progress": {
            "questions_asked": 0,
            "quizzes_taken": 0,
            "quiz_correct": 0,
            "quiz_total": 0,
            "topics": defaultdict(int),
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ============================================================
# HELPERS
# ============================================================
def secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)

def supabase_client():
    url = secret("SUPABASE_URL")
    key = secret("SUPABASE_ANON_KEY") or secret("SUPABASE_PUBLISHABLE_KEY")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None

def current_user():
    return st.session_state.get("current_user")

def load_user_data(sb, user_id):
    """Load this student's records and rebuild the in-memory textbook index."""
    st.session_state.books = {}
    st.session_state.documents = []
    st.session_state.vectorstore = None
    st.session_state.messages = []
    st.session_state.chat_sessions = {"New Chat": []}
    st.session_state.quiz_history = []
    st.session_state.flashcards = []
    st.session_state.progress = {
        "questions_asked": 0, "quizzes_taken": 0, "quiz_correct": 0,
        "quiz_total": 0, "topics": defaultdict(int),
    }
    # ... (Database loading logic remains identical but with cleaned keys) ...
    try:
        profile = sb.table("profiles").select("*").eq("id", user_id).maybe_single().execute()
        st.session_state.profile = profile.data or {}
    except Exception as exc:
        st.session_state.profile = {}

def save_profile(full_name=None, school=None, class_level=None):
    sb = st.session_state.get("supabase")
    user = current_user()
    if not sb or not user: return
    payload = {"id": user.id, "email": user.email}
    if full_name is not None: payload["full_name"] = full_name
    if school is not None: payload["school"] = school
    if class_level is not None: payload["class_level"] = class_level
    try:
        sb.table("profiles").upsert(payload).execute()
        st.session_state.profile.update(payload)
    except Exception as e:
        st.warning(f"Could not save profile: {e}")

def save_chat_to_db(chat_name, messages):
    sb = st.session_state.get("supabase")
    user = current_user()
    if not sb or not user: return
    # ... (Chat saving logic) ...

def save_book_to_db(name, meta):
    sb = st.session_state.get("supabase")
    user = current_user()
    if not sb or not user: return False
    # ... (Book saving logic) ...

def delete_book_from_db(meta):
    sb = st.session_state.get("supabase")
    user = current_user()
    if not sb or not user: return
    # ... (Book deletion logic) ...

def save_quiz_to_db(topic, score, total, questions=None, answers=None, subject="General"):
    sb = st.session_state.get("supabase")
    user = current_user()
    if not sb or not user: return None
    # ... (Quiz saving logic) ...

def save_flashcards_to_db(cards, topic):
    sb = st.session_state.get("supabase")
    user = current_user()
    if not sb or not user or not cards: return
    # ... (Flashcard saving logic) ...

def save_progress_to_db(subject, topic, questions, correct):
    sb = st.session_state.get("supabase")
    user = current_user()
    if not sb or not user: return
    # ... (Progress saving logic) ...

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def clean(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._ -]", "_", name).strip()

def chapter_from(text, fallback="Unknown chapter"):
    for p in [
        r"(?im)^\s*(chapter\s+\d+(?:.\d+)?\s*[:-–]?\s*[^\n]{0,90})",
        r"(?im)^\s*(unit\s+\d+(?:.\d+)?\s*[:-–]?\s*[^\n]{0,90})",
        r"(?im)^\s*(topic\s+\d+(?:.\d+)?\s*[:-–]?\s*[^\n]{0,90})",
    ]:
        m = re.search(p, text[:4000])
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    return fallback

def raw_docs_from_upload(upload):
    data = upload.getvalue()
    name = safe_name(upload.name)
    ext = name.rsplit(".", 1)[-1].lower()
    docs = []
    if ext == "pdf":
        reader = PdfReader(io.BytesIO(data))
        chapter = "Unknown chapter"
        for page_no, page in enumerate(reader.pages, start=1):
            text = clean(page.extract_text())
            if not text: continue
            chapter = chapter_from(text, chapter)
            docs.append(Document(page_content=text, metadata={"source": name, "page": page_no, "chapter": chapter, "type": ext}))
    elif ext == "docx":
        d = DocxDocument(io.BytesIO(data))
        chapter = "Unknown chapter"
        section = 1
        buf = []
        for para in d.paragraphs:
            t = clean(para.text)
            if not t: continue
            if re.match(r"(?i)^(chapter|unit|topic)\s+\d+", t):
                if buf:
                    docs.append(Document(page_content="\n".join(buf), metadata={"source": name, "page": section, "chapter": chapter, "type": ext}))
                    section += 1
                    buf = []
                chapter = t
            buf.append(t)
        if buf:
            docs.append(Document(page_content="\n".join(buf), metadata={"source": name, "page": section, "chapter": chapter, "type": ext}))
    elif ext in {"txt", "md"}:
        text = clean(data.decode("utf-8", errors="ignore"))
        parts = re.split(r"(?im)(?=^\s*(?:chapter|unit|topic)\s+\d+)", text)
        chapter = "Unknown chapter"
        section = 1
        for part in parts:
            part = clean(part)
            if not part: continue
            chapter = chapter_from(part, chapter)
            docs.append(Document(page_content=part, metadata={"source": name, "page": section, "chapter": chapter, "type": ext}))
            section += 1
    else:
        raise ValueError("Unsupported file type")
    return data, name, docs

@st.cache_resource(show_spinner=False)
def embeddings_model():
    return HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def rebuild_index():
    if not st.session_state.documents:
        st.session_state.vectorstore = None
    else:
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.documents, embeddings_model())

def index_uploads(files):
    existing = {m["hash"] for m in st.session_state.books.values()}
    added = 0
    for upload in files or []:
        data, name, raw_docs = raw_docs_from_upload(upload)
        h = sha(data)
        if h in existing: continue
        if not raw_docs or not any((doc.page_content or "").strip() for doc in raw_docs):
            raise ValueError(f"No readable text was extracted from {name}.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=180, separators=["\n\n", "\n", ". ", " ", ""])
        chunks = splitter.split_documents(raw_docs)
        for i, d in enumerate(chunks):
            d.metadata["hash"] = h
            d.metadata["chunk_id"] = f"{h[:12]}-{i}"
        st.session_state.documents.extend(chunks)
        st.session_state.books[name] = {
            "name": name, "hash": h, "size_mb": round(len(data)/(1024*1024), 2),
            "chunks": len(chunks), "pages": len({d.metadata.get("page") for d in raw_docs}),
            "chapters": sorted({d.metadata.get("chapter", "Unknown chapter") for d in chunks}),
            "uploaded": datetime.now().strftime("%Y-%m-%d %H:%M"), "file_bytes": data,
        }
        existing.add(h)
        added += 1
    if added: rebuild_index()
    return added

def remove_book(name):
    st.session_state.documents = [d for d in st.session_state.documents if d.metadata.get("source") != name]
    st.session_state.books.pop(name, None)
    rebuild_index()

def gemini_api_key():
    return secret("GEMINI_API_KEY")

def ask_model(messages, model, temperature=.35, max_tokens=900):
    api_key = gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing.")
    system_parts = []
    contents = []
    for message in messages:
        role = str(message.get("role", "user")).lower()
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
        content = str(content or "").strip()
        if role == "system":
            if content: system_parts.append(content)
            continue
        parts = [{"text": content}] if content else []
        for image_item in message.get("images", []) or []:
            mime_type = image_item.get("mime_type", "image/jpeg")
            encoded = image_item.get("data", "")
            if encoded:
                parts.append({"inlineData": {"mimeType": mime_type, "data": encoded}})
        if not parts: continue
        contents.append({"role": "model" if role in ("assistant", "model") else "user", "parts": parts})
    
    if not contents:
        raise RuntimeError("There is no user message to send to Gemini.")
    
    payload = {
        "contents": contents,
        "generationConfig": {"temperature": float(temperature), "maxOutputTokens": int(max_tokens)},
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    
    model_name = (model or DEFAULT_LLM_MODEL).strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    try:
        response = requests.post(url, headers={"x-goog-api-key": api_key, "Content-Type": "application/json"}, json=payload, timeout=90)
        if not response.ok:
            detail = response.json().get("error", {}).get("message", "") if response.headers.get('content-type','').startswith('application/json') else response.text[:300]
            raise RuntimeError(f"Gemini API error ({response.status_code}): {detail or 'Request failed.'}")
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates: raise RuntimeError("Gemini returned no answer.")
        parts = candidates[0].get("content", {}).get("parts", [])
        answer = "\n".join(part.get("text", "") for part in parts if part.get("text"))
        if not answer.strip(): raise RuntimeError("Gemini returned an empty answer.")
        return answer.strip()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not connect to Gemini API: {exc}") from exc

def retrieve(query, selected_books, selected_chapters, k=6):
    all_docs = list(st.session_state.get("documents", []))
    if not all_docs or not str(query or "").strip(): return []
    def allowed(doc):
        if selected_books and doc.metadata.get("source") not in selected_books: return False
        if selected_chapters and doc.metadata.get("chapter") not in selected_chapters: return False
        return bool((doc.page_content or "").strip())
    
    pool = [doc for doc in all_docs if allowed(doc)]
    if not pool: return []
    
    stopwords = {"what", "which", "when", "where", "who", "why", "how", "are", "is", "was", "were", "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "with", "does", "do", "explain", "define", "describe", "give", "me", "about", "types", "type", "can", "could", "would", "should", "please", "tell", "show", "write", "make", "from", "into", "its"}
    terms = list(dict.fromkeys(term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*", query.lower()) if len(term) > 2 and term not in stopwords))
    
    semantic_rank, lexical_score = {}, {}
    key_for = lambda doc: doc.metadata.get("chunk_id") or (doc.metadata.get("source"), doc.metadata.get("page"), (doc.page_content or "")[:100])
    
    if st.session_state.vectorstore is not None:
        try:
            semantic_docs = st.session_state.vectorstore.similarity_search(query, k=min(len(all_docs), max(40, k * 8)))
            rank = 0
            for doc in semantic_docs:
                if not allowed(doc): continue
                key = key_for(doc)
                if key not in semantic_rank:
                    rank += 1
                    semantic_rank[key] = rank
        except Exception: pass
    
    for doc in pool:
        text = (doc.page_content or "").lower()
        score = sum(text.count(term) * (2 if re.search(r"\b" + re.escape(term) + r"\b", text) else 1) for term in terms)
        if terms:
            coverage = sum(1 for term in terms if re.search(r"\b" + re.escape(term) + r"\b", text))
            score += coverage * 2
        lexical_score[key_for(doc)] = score
    
    lexical_sorted = sorted(pool, key=lambda doc: lexical_score[key_for(doc)], reverse=True)
    lexical_rank = {key_for(doc): rank for rank, doc in enumerate([doc for doc in lexical_sorted if lexical_score[key_for(doc)] > 0], start=1)}
    docs_by_key = {key_for(doc): doc for doc in pool}
    candidate_keys = set(semantic_rank) | set(lexical_rank)
    if not candidate_keys: return []
    
    fused = []
    for key in candidate_keys:
        score = 0.0
        if key in semantic_rank: score += 1.0 / (60 + semantic_rank[key])
        if key in lexical_rank: score += 1.5 / (60 + lexical_rank[key])
        fused.append((score, lexical_score.get(key, 0), key))
    fused.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [docs_by_key[key] for _, _, key in fused[:max(1, int(k))] if key in docs_by_key]

def context_text(docs):
    return "\n\n".join(f"[SOURCE {i} | {d.metadata.get('source')} | page/section {d.metadata.get('page')} | {d.metadata.get('chapter')}]\n{d.page_content}" for i, d in enumerate(docs, start=1))

def source_cards(docs):
    out, seen = [], set()
    for d in docs:
        item = (d.metadata.get("source"), d.metadata.get("page"), d.metadata.get("chapter"))
        if item in seen: continue
        seen.add(item)
        out.append({"source": item[0], "page": item[1], "chapter": item[2], "excerpt": d.page_content[:420]})
    return out[:8]

def answer_question(query, docs, mode, subject, answer_length, source_mode, history, model, temperature, web_results=None, image_parts=None, teaching_level="Secondary School"):
    if not docs and source_mode == "Textbook only" and not image_parts:
        return "I couldn't retrieve any passages from the selected textbook(s) for this question."
    
    ctx = context_text(docs) if docs else "(No textbook passages retrieved.)"
    length_rule = {"Short": "Be concise.", "Medium": "Give moderate detail.", "Detailed": "Give a thorough structured answer."}[answer_length]
    
    if image_parts:
        grounding = "Analyze the attached image directly. Use relevant textbook context when available."
    elif source_mode == "Textbook only":
        grounding = 'Use ONLY textbook context. If insufficient, say: "I could not find enough information in the selected textbook material."'
    else:
        grounding = "Use textbook context first. You may add general knowledge, but clearly label unsupported additions."
    
    web_block = ""
    if web_results:
        web_block = "\n\nWEB RESULTS:\n" + "\n\n".join(f"[WEB {i}] {r['title']}\nURL: {r['url']}\n{r['content']}" for i, r in enumerate(web_results, 1))
        grounding += " Also use the supplied web results. Cite them as [Web: 1], [Web: 2], etc."
    
    system = f"""
{SUBJECTS[subject]}
Student level: {teaching_level}
Study mode: {mode}. {STUDY_MODES[mode]}
{length_rule}
{grounding}
Citation rules:
For textbook claims cite only supplied context with [Source: filename, p. X].
Never fabricate citations or page numbers.
Use Markdown structure when useful.
TEXTBOOK CONTEXT:
{ctx}
{web_block}
""".strip()
    
    messages = [{"role": "system", "content": system}]
    for m in history[-8:]:
        if m.get("role") in {"user", "assistant"}:
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": query, "images": image_parts or []})
    
    max_tokens = {"Short": 500, "Medium": 850, "Detailed": 1300}[answer_length]
    return ask_model(messages, model, temperature, max_tokens)

def json_array_from_model(prompt, model, max_tokens=1600):
    text = ask_model([
        {"role": "system", "content": "Return accurate educational content as valid JSON only."},
        {"role": "user", "content": prompt},
    ], model, .25, max_tokens)
    m = re.search(r"\[[\s\S]*\]", text)
    if not m: raise ValueError("Model did not return a JSON array.")
    return json.loads(m.group(0))

def make_quiz(topic, docs, count, difficulty, model):
    prompt = f"""
Create exactly {count} multiple-choice questions about {topic}. Difficulty: {difficulty}.
Return ONLY JSON:
[
    {{"question": "...", "options":["A", "B", "C", "D"], "answer_index":0, "explanation": "...", "source": "filename, page/section X"}}
]
Each item must have exactly four options and answer_index 0-3.
CONTEXT:\n{context_text(docs) if docs else '(none)'}
"""
    data = json_array_from_model(prompt, model, 1800)
    valid = [q for q in data if isinstance(q, dict) and len(q.get("options", [])) == 4 and q.get("answer_index") in [0,1,2,3]]
    if not valid: raise ValueError("No valid quiz questions returned.")
    return valid[:count]

def make_flashcards(topic, docs, count, model):
    prompt = f"""
Create exactly {count} flashcards about {topic}.
Return ONLY JSON: [{{"front": "...", "back": "...", "source": "filename, page/section X"}}]
CONTEXT:\n{context_text(docs) if docs else '(none)'}
"""
    data = json_array_from_model(prompt, model, 1500)
    return [x for x in data if isinstance(x, dict) and x.get("front") and x.get("back")][:count]

def transcribe(audio_bytes):
    api_key = gemini_api_key()
    if not api_key: raise RuntimeError("GEMINI_API_KEY is missing.")
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    payload = {
        "contents": [{"role": "user", "parts": [
            {"text": "Transcribe the spoken words in this audio accurately. Return only the transcription."},
            {"inlineData": {"mimeType": "audio/wav", "data": audio_b64}},
        ]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 1200},
    }
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{DEFAULT_LLM_MODEL}:generateContent",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=payload, timeout=90,
    )
    if not response.ok: raise RuntimeError(f"Gemini transcription error")
    data = response.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "\n".join(part.get("text", "") for part in parts if part.get("text")).strip()

def tavily_search(query):
    key = secret("TAVILY_API_KEY")
    if not key: return []
    r = requests.post("https://api.tavily.com/search", json={"api_key": key, "query": query, "search_depth": "basic", "max_results": 5}, timeout=20)
    r.raise_for_status()
    return [{"title": x.get("title", "Web result"), "url": x.get("url", ""), "content": x.get("content", "")} for x in r.json().get("results", [])]

# ============================================================
# NAVIGATION & CHAT LOGIC
# ============================================================
def set_view(view):
    st.session_state.active_view = view

def title_from_prompt(prompt, limit=42):
    text = re.sub(r"\s+", " ", str(prompt or "")).strip()
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    if not text: return "New chat"
    words = text.split()
    title = " ".join(words[:8])
    if len(title) > limit: title = title[:limit].rsplit(" ", 1)[0] or title[:limit]
    return title[:1].upper() + title[1:]

def rename_active_chat_from_prompt(prompt):
    old_name = st.session_state.active_chat
    messages = st.session_state.chat_sessions.get(old_name, st.session_state.messages)
    if len([m for m in messages if m.get("role") == "assistant"]) > 0: return old_name
    base = title_from_prompt(prompt)
    new_name = base
    suffix = 2
    while new_name in st.session_state.chat_sessions and new_name != old_name:
        new_name = f"{base} ({suffix})"
        suffix += 1
    if new_name != old_name:
        st.session_state.chat_sessions.pop(old_name, None)
        st.session_state.chat_sessions[new_name] = messages
        st.session_state.active_chat = new_name
        st.session_state.messages = messages
    return new_name

def new_chat():
    name = "New chat"
    suffix = 2
    while name in st.session_state.chat_sessions:
        name = f"New chat {suffix}"
        suffix += 1
    st.session_state.chat_sessions[name] = []
    st.session_state.active_chat = name
    st.session_state.messages = st.session_state.chat_sessions[name]
    st.session_state.active_view = "chat"

init_state()

# ============================================================
# AUTH (Simplified for brevity, assumes Supabase is configured)
# ============================================================
def require_auth():
    if st.session_state.get("supabase") is None:
        st.session_state.supabase = supabase_client()
    if st.session_state.get("current_user") is not None:
        return True
    
    # Mocking auth for standalone UI testing if Supabase isn't connected
    if st.session_state.supabase is None:
        st.session_state.current_user = SimpleNamespace(id="mock_user", email="student@example.com")
        st.session_state.profile = {"full_name": "Happy Learner"}
        return True

    st.title("Welcome to izy read 📚")
    st.write("Please log in to continue your learning journey.")
    st.stop()

require_auth()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    user = current_user()
    profile_name = (st.session_state.get("profile") or {}).get("full_name") or (user.email.split("@")[0] if user and user.email else "Student")
    
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
        <div style="width:40px;height:40px;border-radius:50%;background:#6EE7B7;display:flex;align-items:center;justify-content:center;font-weight:bold;color:#064E3B;font-size:18px;">{profile_name[0].upper()}</div>
        <div>
            <div style="font-weight:700;font-size:16px;color:#334155;">{profile_name}</div>
            <div style="font-size:12px;color:#94A3B8;">{user.email if user else ""}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Log out", use_container_width=True):
        for key in ["current_user", "profile", "supabase"]:
            st.session_state.pop(key, None)
        st.rerun()
    
    st.divider()
    
    primary_nav = [("🏠 Home", "home"), ("💬 Chats", "chat"), ("📚 Books", "books"), ("👤 Profile", "profile"), ("🎯 Study", "study")]
    for label, target in primary_nav:
        if st.button(label, use_container_width=True, type="primary" if st.session_state.active_view == target else "secondary"):
            set_view(target)
            st.rerun()
            
    if st.session_state.active_view == "chat":
        if st.button("✨ New chat", use_container_width=True):
            new_chat()
            st.rerun()
            
    st.divider()
    
    # Settings safely loaded
    model = st.session_state.get("gemini_model_setting", secret("GEMINI_MODEL", DEFAULT_LLM_MODEL) or DEFAULT_LLM_MODEL)
    temperature = st.session_state.get("temperature_setting", 0.35)
    answer_length = st.session_state.get("answer_length_setting", "Medium")
    teaching_level = st.session_state.get("teaching_level_setting", "Secondary School")
    subject = st.session_state.get("subject_setting", "General")
    study_mode = st.session_state.get("study_mode_setting", "Normal Tutor")
    source_mode = st.session_state.get("source_mode_setting", "Textbook only")
    retrieval_k = st.session_state.get("retrieval_k_setting", 6)
    
    selected_books = st.session_state.get("selected_books", list(st.session_state.books))
    selected_books = [b for b in selected_books if b in st.session_state.books]
    if not selected_books and st.session_state.books:
        selected_books = list(st.session_state.books)
    st.session_state.selected_books = selected_books
    
    chapters = sorted({d.metadata.get("chapter", "Unknown chapter") for d in st.session_state.documents if not selected_books or d.metadata.get("source") in selected_books})
    selected_chapters = st.session_state.get("selected_chapters", [])
    selected_chapters = [c for c in selected_chapters if c in chapters]
    st.session_state.selected_chapters = selected_chapters

# ============================================================
# MAIN CONTENT
# ============================================================
view = st.session_state.active_view

if view == "home":
    st.markdown("## 👋 Welcome back!")
    st.caption("Here's your learning overview for today.")
    
    progress = st.session_state.progress
    accuracy = 100 * progress["quiz_correct"] / progress["quiz_total"] if progress["quiz_total"] else 0
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📚 Textbooks", len(st.session_state.books))
    m2.metric("💬 Questions", progress["questions_asked"])
    m3.metric("🎯 Quizzes", progress["quizzes_taken"])
    m4.metric("🏆 Accuracy", f"{accuracy:.0f}%")
    
    st.markdown("### 🚀 Quick Actions")
    qa, qb, qc = st.columns(3)
    if qa.button("💬 Start a chat", use_container_width=True):
        st.session_state.active_view = "chat"; st.rerun()
    if qb.button("📚 Open books", use_container_width=True):
        st.session_state.active_view = "books"; st.rerun()
    if qc.button("🎯 Study tools", use_container_width=True):
        st.session_state.active_view = "study"; st.rerun()

elif view == "books":
    st.markdown("## 📚 Your Library")
    st.caption("Manage your textbooks. Upload new ones to start studying!")
    
    uploads = st.file_uploader("Add textbooks", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if st.button("📥 Index Books", type="primary", disabled=not uploads):
        try:
            with st.spinner("Reading your books..."):
                n = index_uploads(uploads)
            st.success(f"Added {n} new book(s)!")
            st.rerun()
        except Exception as e:
            st.error(f"Indexing error: {e}")
            
    if st.session_state.books:
        for name, meta in list(st.session_state.books.items()):
            st.markdown(f"**{name}**  \n*{meta['pages']} pages · {meta['chunks']} passages*")
            c1, c2 = st.columns([1, 1])
            if c1.button("📖 Open", use_container_width=True):
                st.session_state.selected_books = [name]
                st.session_state.active_view = "chat"
                st.rerun()
            if c2.button("🗑️ Delete", use_container_width=True):
                remove_book(name)
                st.rerun()
    else:
        st.info("No textbooks yet. Upload your first PDF or DOCX above!")

elif view == "profile":
    st.markdown("## 👤 Profile & Settings")
    st.caption("Customize your learning experience.")
    
    with st.expander("🎨 Learning Preferences", expanded=True):
        model = st.text_input("Gemini chat model", value=model, key="gemini_model_setting")
        temperature = st.slider("Creativity", 0.0, 1.0, .35, .05, key="temperature_setting")
        answer_length = st.selectbox("Answer length", ["Short", "Medium", "Detailed"], index=1, key="answer_length_setting")
        teaching_level = st.selectbox("Student level", ["Beginner", "Secondary School", "Advanced"], index=1, key="teaching_level_setting")
        subject = st.selectbox("Subject tutor", list(SUBJECTS), key="subject_setting")
        study_mode = st.selectbox("Study mode", list(STUDY_MODES), key="study_mode_setting")
        source_mode = st.selectbox("Knowledge mode", ["Textbook only", "Textbook + general knowledge", "Textbook + web research"], key="source_mode_setting")
        retrieval_k = st.slider("Retrieved passages", 3, 10, 6, key="retrieval_k_setting")

elif view == "chat":
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; margin-top:10vh; margin-bottom:2rem;">
            <h2 style="color:#334155;">How can I help you study today? 🌱</h2>
            <p style="color:#64748B;">Ask questions, request summaries, or solve problems from your textbooks.</p>
        </div>
        """, unsafe_allow_html=True)
        
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            
    # Safe image uploader
    attached_files = st.file_uploader("Attach images (optional)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True, label_visibility="collapsed")
    prompt = st.chat_input("Ask anything about your textbooks...", disabled=not bool(gemini_api_key()))
    
    if prompt:
        query = prompt.strip()
        image_parts = []
        if attached_files:
            for image_file in attached_files:
                mime_type = getattr(image_file, "type", None) or "image/jpeg"
                image_parts.append({
                    "mime_type": mime_type,
                    "data": base64.b64encode(image_file.getvalue()).decode("ascii"),
                })
                
        if not query and image_parts:
            query = "Please analyze the attached image and explain the relevant study content."
            
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            rename_active_chat_from_prompt(query)
            
            with st.chat_message("user"):
                st.markdown(query)
                
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        docs = retrieve(query, selected_books, selected_chapters, retrieval_k)
                        response = answer_question(
                            query, docs, study_mode, subject, answer_length, source_mode,
                            st.session_state.messages[:-1], model, temperature, 
                            image_parts=image_parts, teaching_level=teaching_level
                        )
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.session_state.progress["questions_asked"] += 1
                    except Exception as e:
                        st.error(f"Oops! Something went wrong: {e}")

elif view == "study":
    st.markdown("## 🎯 Study Tools")
    st.caption("Quizzes, flashcards, and more to help you retain what you've learned.")
    
    s1, s2, s3 = st.columns(3)
    if s1.button("📝 Take a Quiz", use_container_width=True):
        st.session_state.active_view = "quiz"; st.rerun()
    if s2.button("🃏 Flashcards", use_container_width=True):
        st.session_state.active_view = "flashcards"; st.rerun()
    if s3.button("🔍 Search Books", use_container_width=True):
        st.session_state.active_view = "search"; st.rerun()

elif view == "quiz":
    st.markdown("## 📝 Quiz Center")
    topic = st.text_input("Quiz topic", placeholder="e.g. Cell division")
    if st.button("✨ Generate Quiz", type="primary", disabled=not topic.strip()):
        with st.spinner("Creating questions..."):
            docs = retrieve(topic, selected_books, selected_chapters, min(10, retrieval_k + 2))
            st.session_state.quiz = make_quiz(topic, docs, 5, "Medium", model)
            st.session_state.quiz_topic = topic
            st.rerun()
            
    if st.session_state.quiz:
        for i, q in enumerate(st.session_state.quiz):
            st.markdown(f"**{i + 1}. {q['question']}**")
            st.radio("Choose", q["options"], key=f"qa_{i}", label_visibility="collapsed")

elif view == "flashcards":
    st.markdown("## 🃏 Flashcards")
    ftopic = st.text_input("Flashcard topic", placeholder="e.g. Organic chemistry")
    if st.button("✨ Generate Cards", type="primary", disabled=not ftopic.strip()):
        with st.spinner("Creating cards..."):
            docs = retrieve(ftopic, selected_books, selected_chapters, min(10, retrieval_k + 2))
            st.session_state.flashcards = make_flashcards(ftopic, docs, 8, model)
            st.rerun()
            
    for i, card in enumerate(st.session_state.flashcards, 1):
        with st.expander(f"Card {i}: {card['front']}"):
            st.markdown(card["back"])

elif view == "search":
    st.markdown("## 🔍 Search Textbooks")
    sq = st.text_input("Search phrase", placeholder="e.g. osmosis, Newton's laws")
    if st.button("🔎 Search", type="primary", disabled=not sq.strip()):
        st.session_state.last_search_results = retrieve(sq, selected_books, selected_chapters, 8)
        
    for i, d in enumerate(st.session_state.last_search_results, 1):
        with st.expander(f"{i}. {d.metadata.get('source')} — p.{d.metadata.get('page')}"):
            st.write(d.page_content)

# ============================================================
# CASUAL UI CSS INJECTION
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');

/* Global App Styling */
.stApp {
    background-color: #FDFBF7 !important; /* Warm cream */
    color: #334155 !important;
    font-family: 'Nunito', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    text-align: left;
    border-radius: 12px;
    border: 1px solid transparent;
    background: transparent;
    color: #475569;
    font-weight: 600;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #F1F5F9;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background-color: #E0F2FE; /* Soft blue */
    color: #0369A1;
    border: 1px solid #BAE6FD;
}

/* Main content */
.block-container {
    padding-top: 2rem !important;
    max-width: 1100px !important;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 18px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03) !important;
    margin-bottom: 1rem !important;
}

/* Inputs and Selectboxes */
.stTextInput > div > div, .stSelectbox > div > div, .stNumberInput > div > div {
    border-radius: 12px !important;
    border: 1px solid #CBD5E1 !important;
    background-color: #FFFFFF !important;
}

/* Buttons */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    border: 1px solid #CBD5E1 !important;
    background-color: #FFFFFF !important;
    color: #334155 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #F8FAFC !important;
    border-color: #94A3B8 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
}
.stButton > button[kind="primary"] {
    background-color: #6EE7B7 !important; /* Mint green */
    color: #064E3B !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #34D399 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
}

/* Chat Input */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 1.5rem !important;
    left: clamp(280px, 24vw, 340px) !important;
    right: 1.5rem !important;
    z-index: 999 !important;
}
[data-testid="stChatInput"] > div {
    border-radius: 24px !important;
    border: 1px solid #CBD5E1 !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08) !important;
}

/* Expanders */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    color: #475569 !important;
}
</style>
""", unsafe_allow_html=True)
