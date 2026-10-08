import io
import os
import re
import json
import hashlib
from collections import defaultdict
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from supabase import create_client
from docx import Document as DocxDocument
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Textbook AI", page_icon="📚", layout="wide")

DEFAULT_LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ASR_MODEL = "openai/whisper-large-v3-turbo"

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

st.markdown("""
<style>
/* ---------- Global ---------- */
.block-container{max-width:1450px;padding-top:1.25rem;padding-bottom:2.5rem}
[data-testid="stSidebar"]{border-right:1px solid rgba(128,128,128,.16)}
[data-testid="stSidebar"] > div:first-child{padding:1rem .8rem 1.2rem}
[data-testid="stSidebar"] .block-container{padding:0}
[data-testid="stSidebar"] hr{margin:.8rem 0;border-color:rgba(128,128,128,.14)}

/* ---------- Sidebar ---------- */
.sidebar-brand{display:flex;align-items:center;gap:.7rem;padding:.45rem .35rem .2rem}
.sidebar-logo{width:42px;height:42px;border-radius:13px;display:flex;align-items:center;justify-content:center;font-size:1.45rem;background:linear-gradient(135deg,rgba(255,255,255,.12),rgba(120,100,255,.22));border:1px solid rgba(255,255,255,.10);box-shadow:0 8px 22px rgba(0,0,0,.10)}
.sidebar-title{font-size:1.12rem;font-weight:750;line-height:1.1}
.sidebar-subtitle{font-size:.74rem;opacity:.58;margin-top:.16rem}
.status-pill{display:flex;align-items:center;gap:.45rem;margin:.8rem .2rem .45rem;padding:.48rem .65rem;border-radius:10px;font-size:.78rem;border:1px solid rgba(128,128,128,.16);background:rgba(128,128,128,.055)}
.status-dot{width:8px;height:8px;border-radius:50%;background:#2ecc71;box-shadow:0 0 0 3px rgba(46,204,113,.12)}
.status-dot.off{background:#f39c12;box-shadow:0 0 0 3px rgba(243,156,18,.12)}
.status-text{font-weight:650}
.sidebar-section-label{text-transform:uppercase;letter-spacing:.08em;font-size:.67rem;font-weight:750;opacity:.52;margin:.95rem .35rem .4rem}
.sidebar-help{font-size:.73rem;line-height:1.4;opacity:.58;margin:.45rem .25rem}
.sidebar-count{float:right;opacity:.55;font-weight:600}

/* Make sidebar buttons compact and consistent */
[data-testid="stSidebar"] .stButton>button{width:100%;border-radius:10px;min-height:2.25rem;padding:.35rem .65rem;font-weight:600;border:1px solid rgba(128,128,128,.16);background:rgba(128,128,128,.045)}
[data-testid="stSidebar"] .stButton>button:hover{border-color:rgba(120,100,255,.45);background:rgba(120,100,255,.08)}
[data-testid="stSidebar"] .stTextInput input,[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"]>div{border-radius:10px}
[data-testid="stSidebar"] [data-testid="stFileUploader"]{border:1px dashed rgba(128,128,128,.25);border-radius:12px;padding:.15rem}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{background:transparent}
[data-testid="stSidebar"] [data-testid="stExpander"]{border:1px solid rgba(128,128,128,.14);border-radius:11px;background:rgba(128,128,128,.025)}

/* ChatGPT-like recent chat buttons */
.chat-row button{font-size:.82rem;text-align:left}

/* ---------- Main ---------- */
.hero{padding:1.5rem 1.65rem;border:1px solid rgba(128,128,128,.18);border-radius:24px;background:linear-gradient(135deg,rgba(100,90,255,.14),rgba(0,190,160,.07));margin-bottom:1rem;box-shadow:0 10px 35px rgba(0,0,0,.05)}
.hero h1{margin:0;font-size:2rem;letter-spacing:-.03em}.hero p{margin:.4rem 0 0;opacity:.72;max-width:850px;line-height:1.5}
.source-card{border:1px solid rgba(128,128,128,.18);border-radius:14px;padding:.8rem;margin:.35rem 0;background:rgba(128,128,128,.04)}
.small-muted{opacity:.7;font-size:.88rem}
[data-testid="stChatMessage"]{border:1px solid rgba(128,128,128,.10);border-radius:18px;padding:.45rem .65rem;margin-bottom:.55rem;background:rgba(128,128,128,.018)}

/* ---------- ChatGPT-style fixed composer ---------- */
/* Streamlit already treats st.chat_input as a bottom composer. These rules
   make the behavior visually consistent and keep it above the page content. */
[data-testid="stChatInput"]{
    position:fixed !important;
    left:clamp(18rem, 22vw, 24rem) !important;
    right:1.5rem !important;
    bottom:1rem !important;
    z-index:999 !important;
    border-radius:18px !important;
    padding:0 !important;
    background:transparent !important;
}
[data-testid="stChatInput"] > div{
    border-radius:20px !important;
    border:1px solid rgba(128,128,128,.24) !important;
    background:var(--background-color, #17181c) !important;
    box-shadow:0 8px 30px rgba(0,0,0,.24), 0 1px 3px rgba(0,0,0,.12) !important;
}
[data-testid="stChatInput"] textarea{
    min-height:48px !important;
    max-height:180px !important;
    padding:13px 52px 13px 16px !important;
    font-size:.98rem !important;
}
[data-testid="stChatInput"] button{
    border-radius:12px !important;
}
/* Keep the conversation from being hidden behind the fixed composer. */
[data-testid="stAppViewContainer"] .main .block-container{
    padding-bottom:7rem !important;
}
.stButton button{border-radius:11px}
[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.12);border-radius:14px;padding:.7rem;background:rgba(128,128,128,.025)}
</style>
""", unsafe_allow_html=True)

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
        "active_view": "chat",
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

init_state()
require_auth()

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
    try:
        profile = sb.table("profiles").select("*").eq("id", user_id).maybe_single().execute()
        st.session_state.profile = profile.data or {}
    except Exception:
        st.session_state.profile = {}
    try:
        chats = sb.table("chats").select("*").eq("user_id", user_id).order("updated_at", desc=True).execute().data or []
        sessions = {}
        for chat in chats:
            rows = sb.table("messages").select("*").eq("chat_id", chat["id"]).order("created_at").execute().data or []
            sessions[chat["title"]] = [{"role": r["role"], "content": r["content"]} for r in rows if r["role"] in ("user", "assistant")]
        if sessions:
            st.session_state.chat_sessions = sessions
            first = next(iter(sessions))
            st.session_state.active_chat = first
            st.session_state.messages = sessions[first]
    except Exception:
        pass
    try:
        books = sb.table("books").select("*").eq("user_id", user_id).order("created_at", desc=True).execute().data or []
        for b in books:
            name = b.get("file_name") or b.get("title") or "Untitled book"
            st.session_state.books[name] = {"name": name, "hash": b.get("id", sha(name.encode())), "pages": b.get("page_count") or 0, "chunks": 0, "size_mb": 0, "subject": b.get("subject") or "", "db_id": b.get("id"), "persisted_only": True}
    except Exception:
        pass
    try:
        quizzes = sb.table("quizzes").select("*").eq("user_id", user_id).order("created_at", desc=True).execute().data or []
        st.session_state.quiz_history = [{"id": q["id"], "topic": q.get("topic") or q.get("title") or "Untitled quiz", "score": q.get("score",0), "questions": [], "answers": [], "submitted": q.get("completed",False), "date": (q.get("created_at") or "")[:16].replace("T"," ")} for q in quizzes]
    except Exception:
        pass
    try:
        cards = sb.table("flashcards").select("*").eq("user_id", user_id).order("created_at", desc=True).execute().data or []
        st.session_state.flashcards = [{"front": c["question"], "back": c["answer"], "source": c.get("subject") or "Textbook"} for c in cards]
    except Exception:
        pass
    try:
        rows = sb.table("progress").select("*").eq("user_id", user_id).execute().data or []
        st.session_state.progress["topics"] = defaultdict(int)
        for row in rows:
            st.session_state.progress["topics"][f'{row.get("subject","General")}: {row.get("topic","Unknown")}'] = row.get("questions_answered",0)
    except Exception:
        pass


def save_profile(full_name=None, school=None, class_level=None):
    sb=st.session_state.get("supabase"); user=current_user()
    if not sb or not user: return
    payload={"id":user.id,"email":user.email}
    if full_name is not None: payload["full_name"]=full_name
    if school is not None: payload["school"]=school
    if class_level is not None: payload["class_level"]=class_level
    try:
        sb.table("profiles").upsert(payload).execute(); st.session_state.profile.update(payload)
    except Exception as e: st.warning(f"Could not save profile: {e}")


def save_chat_to_db(chat_name, messages):
    sb=st.session_state.get("supabase"); user=current_user()
    if not sb or not user: return
    try:
        existing=sb.table("chats").select("id").eq("user_id",user.id).eq("title",chat_name).limit(1).execute().data or []
        if existing:
            chat_id=existing[0]["id"]; sb.table("chats").update({"updated_at":datetime.now().isoformat()}).eq("id",chat_id).execute(); sb.table("messages").delete().eq("chat_id",chat_id).execute()
        else:
            chat_id=sb.table("chats").insert({"user_id":user.id,"title":chat_name}).execute().data[0]["id"]
        payload=[{"chat_id":chat_id,"user_id":user.id,"role":m["role"],"content":m["content"]} for m in messages if m.get("role") in ("user","assistant") and m.get("content")]
        if payload: sb.table("messages").insert(payload).execute()
    except Exception: pass


def save_book_to_db(name, meta):
    sb=st.session_state.get("supabase"); user=current_user()
    if not sb or not user: return
    try:
        result=sb.table("books").insert({"user_id":user.id,"title":name,"file_name":name,"file_path":"","subject":meta.get("subject",""),"page_count":int(meta.get("pages",0) or 0),"indexed":True}).execute()
        if result.data: meta["db_id"]=result.data[0]["id"]
    except Exception: pass


def delete_book_from_db(meta):
    sb=st.session_state.get("supabase"); user=current_user()
    if not sb or not user: return
    try:
        if meta.get("db_id"): sb.table("books").delete().eq("id",meta["db_id"]).eq("user_id",user.id).execute()
    except Exception: pass


def save_quiz_to_db(topic, score, total):
    sb=st.session_state.get("supabase"); user=current_user()
    if not sb or not user: return None
    try:
        row=sb.table("quizzes").insert({"user_id":user.id,"title":topic or "Untitled quiz","topic":topic or "General","score":int(score),"total_questions":int(total),"completed":True}).execute().data
        return row[0]["id"] if row else None
    except Exception: return None


def save_flashcards_to_db(cards, topic):
    sb=st.session_state.get("supabase"); user=current_user()
    if not sb or not user or not cards: return
    try:
        rows=[{"user_id":user.id,"question":c.get("front",""),"answer":c.get("back",""),"subject":st.session_state.get("subject_setting","General"),"topic":topic} for c in cards if c.get("front") and c.get("back")]
        if rows: sb.table("flashcards").insert(rows).execute()
    except Exception: pass


def save_progress_to_db(subject, topic, questions, correct):
    sb=st.session_state.get("supabase"); user=current_user()
    if not sb or not user: return
    try:
        existing=sb.table("progress").select("*").eq("user_id",user.id).eq("subject",subject).eq("topic",topic).limit(1).execute().data or []
        if existing:
            row=existing[0]; qa=int(row.get("questions_answered",0))+int(questions); ca=int(row.get("correct_answers",0))+int(correct)
            sb.table("progress").update({"questions_answered":qa,"correct_answers":ca,"mastery":round(100*ca/max(1,qa),2),"last_studied":datetime.now().isoformat()}).eq("id",row["id"]).execute()
        else:
            sb.table("progress").insert({"user_id":user.id,"subject":subject,"topic":topic or "General","questions_answered":int(questions),"correct_answers":int(correct),"mastery":round(100*int(correct)/max(1,int(questions)),2)}).execute()
    except Exception: pass


def auth_screen():
    sb=st.session_state.get("supabase")
    st.markdown('<div style="max-width:520px;margin:8vh auto 0;text-align:center"><div style="font-size:3rem">📚</div><h1 style="margin:.25rem 0">Textbook AI</h1><p style="opacity:.65">Your personal textbook study workspace</p></div>',unsafe_allow_html=True)
    if sb is None:
        st.error("Supabase is not connected. Add SUPABASE_URL and SUPABASE_ANON_KEY to Streamlit Secrets.")
        st.code('SUPABASE_URL = "https://your-project.supabase.co"\nSUPABASE_ANON_KEY = "your-client-safe-key"',language="toml"); st.stop()
    login_tab,signup_tab=st.tabs(["Log in","Create account"])
    with login_tab:
        with st.form("login_form"):
            email=st.text_input("Email",placeholder="student@example.com"); password=st.text_input("Password",type="password")
            submitted=st.form_submit_button("Log in",type="primary",use_container_width=True)
        if submitted:
            if not email or not password: st.error("Enter your email and password.")
            else:
                try:
                    result=sb.auth.sign_in_with_password({"email":email.strip(),"password":password})
                    if result.user:
                        st.session_state.current_user=result.user; st.session_state.supabase=sb; load_user_data(sb,result.user.id); st.rerun()
                except Exception as e: st.error(f"Login failed: {e}")
    with signup_tab:
        with st.form("signup_form"):
            full_name=st.text_input("Full name",placeholder="Your name"); email=st.text_input("Email",placeholder="student@example.com",key="signup_email"); password=st.text_input("Password",type="password",key="signup_password"); confirm=st.text_input("Confirm password",type="password")
            submitted=st.form_submit_button("Create account",type="primary",use_container_width=True)
        if submitted:
            if not full_name.strip() or not email.strip() or not password: st.error("Complete all fields.")
            elif password!=confirm: st.error("Passwords do not match.")
            elif len(password)<6: st.error("Use a password with at least 6 characters.")
            else:
                try:
                    result=sb.auth.sign_up({"email":email.strip(),"password":password,"options":{"data":{"full_name":full_name.strip()}}})
                    if result.user and result.session:
                        st.session_state.current_user=result.user; st.session_state.supabase=sb; load_user_data(sb,result.user.id); st.success("Account created."); st.rerun()
                    else: st.success("Account created. Check your email to confirm your account, then log in.")
                except Exception as e: st.error(f"Sign-up failed: {e}")


def require_auth():
    if st.session_state.get("supabase") is None: st.session_state.supabase=supabase_client()
    if st.session_state.get("current_user") is not None: return True
    auth_screen(); st.stop()


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
        r"(?im)^\s*(chapter\s+\d+(?:\.\d+)?\s*[:\-–]?\s*[^\n]{0,90})",
        r"(?im)^\s*(unit\s+\d+(?:\.\d+)?\s*[:\-–]?\s*[^\n]{0,90})",
        r"(?im)^\s*(topic\s+\d+(?:\.\d+)?\s*[:\-–]?\s*[^\n]{0,90})",
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
            if not text:
                continue
            chapter = chapter_from(text, chapter)
            docs.append(Document(page_content=text, metadata={"source": name, "page": page_no, "chapter": chapter, "type": ext}))

    elif ext == "docx":
        d = DocxDocument(io.BytesIO(data))
        chapter = "Unknown chapter"
        section = 1
        buf = []
        for para in d.paragraphs:
            t = clean(para.text)
            if not t:
                continue
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
            if not part:
                continue
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
        if h in existing:
            continue
        splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=180, separators=["\n\n", "\n", ". ", " ", ""])
        chunks = splitter.split_documents(raw_docs)
        for i, d in enumerate(chunks):
            d.metadata["hash"] = h
            d.metadata["chunk_id"] = f"{h[:12]}-{i}"
        st.session_state.documents.extend(chunks)
        st.session_state.books[name] = {
            "name": name,
            "hash": h,
            "size_mb": round(len(data)/(1024*1024), 2),
            "chunks": len(chunks),
            "pages": len({d.metadata.get("page") for d in raw_docs}),
            "chapters": sorted({d.metadata.get("chapter", "Unknown chapter") for d in chunks}),
            "uploaded": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        existing.add(h)
        added += 1
    if added:
        rebuild_index()
    return added


def remove_book(name):
    st.session_state.documents = [d for d in st.session_state.documents if d.metadata.get("source") != name]
    st.session_state.books.pop(name, None)
    rebuild_index()


def hf_client():
    token = secret("HF_TOKEN")
    if not token:
        return None
    return InferenceClient(token=token, timeout=90)


def ask_model(messages, model, temperature=.35, max_tokens=900):
    c = hf_client()
    if c is None:
        raise RuntimeError("HF_TOKEN is missing. Add it to Streamlit Secrets.")
    out = c.chat_completion(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    return out.choices[0].message.content.strip()


def retrieve(query, selected_books, selected_chapters, k=6):
    if st.session_state.vectorstore is None:
        return []
    candidates = st.session_state.vectorstore.similarity_search(query, k=max(24, k*4))
    out, seen = [], set()
    for d in candidates:
        if selected_books and d.metadata.get("source") not in selected_books:
            continue
        if selected_chapters and d.metadata.get("chapter") not in selected_chapters:
            continue
        key = d.metadata.get("chunk_id") or (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:100])
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
        if len(out) >= k:
            break
    return out


def context_text(docs):
    return "\n\n".join(
        f"[SOURCE {i} | {d.metadata.get('source')} | page/section {d.metadata.get('page')} | {d.metadata.get('chapter')}]\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def source_cards(docs):
    out, seen = [], set()
    for d in docs:
        item = (d.metadata.get("source"), d.metadata.get("page"), d.metadata.get("chapter"))
        if item in seen:
            continue
        seen.add(item)
        out.append({"source": item[0], "page": item[1], "chapter": item[2], "excerpt": d.page_content[:420]})
    return out[:8]


def answer_question(query, docs, mode, subject, answer_length, source_mode, history, model, temperature, web_results=None):
    ctx = context_text(docs) if docs else "(No textbook passages retrieved.)"
    length_rule = {"Short": "Be concise.", "Medium": "Give moderate detail.", "Detailed": "Give a thorough structured answer."}[answer_length]

    if source_mode == "Textbook only":
        grounding = 'Use ONLY textbook context. If insufficient, say: "I could not find enough information in the selected textbook material."'
    else:
        grounding = "Use textbook context first. You may add general knowledge, but clearly label unsupported additions as General knowledge."

    web_block = ""
    if web_results:
        web_block = "\n\nWEB RESULTS:\n" + "\n\n".join(
            f"[WEB {i}] {r['title']}\nURL: {r['url']}\n{r['content']}" for i, r in enumerate(web_results, 1)
        )
        grounding += " Also use the supplied web results. Cite them as [Web: 1], [Web: 2], etc. Never invent URLs."

    system = f"""
{SUBJECTS[subject]}
Student level: {teaching_level}
Study mode: {mode}. {STUDY_MODES[mode]}
{length_rule}
{grounding}

Citation rules:
- For textbook claims cite only supplied context with [Source: filename, p. X].
- For DOCX/TXT/MD, p. means section number.
- Never fabricate citations or page numbers.
- Use Markdown structure when useful.

TEXTBOOK CONTEXT:
{ctx}
{web_block}
""".strip()

    messages = [{"role": "system", "content": system}]
    for m in history[-8:]:
        if m.get("role") in {"user", "assistant"}:
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": query})
    max_tokens = {"Short": 500, "Medium": 850, "Detailed": 1300}[answer_length]
    return ask_model(messages, model, temperature, max_tokens)


def json_array_from_model(prompt, model, max_tokens=1600):
    text = ask_model([
        {"role": "system", "content": "Return accurate educational content as valid JSON only."},
        {"role": "user", "content": prompt},
    ], model, .25, max_tokens)
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        raise ValueError("Model did not return a JSON array.")
    return json.loads(m.group(0))


def make_quiz(topic, docs, count, difficulty, model):
    prompt = f"""
Create exactly {count} multiple-choice questions about {topic}. Difficulty: {difficulty}.
Return ONLY JSON:
[
 {{"question":"...","options":["A","B","C","D"],"answer_index":0,"explanation":"...","source":"filename, page/section X"}}
]
Each item must have exactly four options and answer_index 0-3. Never invent textbook source details; use General knowledge if needed.
CONTEXT:\n{context_text(docs) if docs else '(none)'}
"""
    data = json_array_from_model(prompt, model, 1800)
    valid = [q for q in data if isinstance(q, dict) and len(q.get("options", [])) == 4 and q.get("answer_index") in [0,1,2,3]]
    if not valid:
        raise ValueError("No valid quiz questions returned.")
    return valid[:count]


def make_flashcards(topic, docs, count, model):
    prompt = f"""
Create exactly {count} flashcards about {topic}.
Return ONLY JSON: [{{"front":"...","back":"...","source":"filename, page/section X"}}]
Use textbook context when possible. Never invent sources; use General knowledge when needed.
CONTEXT:\n{context_text(docs) if docs else '(none)'}
"""
    data = json_array_from_model(prompt, model, 1500)
    return [x for x in data if isinstance(x, dict) and x.get("front") and x.get("back")][:count]


def transcribe(audio_bytes):
    c = hf_client()
    if c is None:
        raise RuntimeError("HF_TOKEN is missing.")
    return c.automatic_speech_recognition(audio_bytes, model=DEFAULT_ASR_MODEL).text.strip()


def tavily_search(query):
    key = secret("TAVILY_API_KEY")
    if not key:
        return []
    r = requests.post("https://api.tavily.com/search", json={"api_key": key, "query": query, "search_depth": "basic", "max_results": 5}, timeout=20)
    r.raise_for_status()
    return [{"title": x.get("title", "Web result"), "url": x.get("url", ""), "content": x.get("content", "")} for x in r.json().get("results", [])]

# ============================================================
# CHATGPT-STYLE WORKSPACE
# ============================================================
# Navigation state is kept in session so the main area is focused on the
# selected workspace while the sidebar contains the controls/features.

def set_view(view):
    st.session_state.active_view = view


def new_chat():
    name = f"New chat {len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[name] = []
    st.session_state.active_chat = name
    st.session_state.messages = st.session_state.chat_sessions[name]
    st.session_state.active_view = "chat"


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    user=current_user()
    profile_name=(st.session_state.get("profile") or {}).get("full_name") or (user.email.split("@")[0] if user and user.email else "Student")
    st.markdown("""
    <div class="chatgpt-brand">
        <div class="brand-mark">📚</div>
        <div class="brand-name">Textbook AI</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="student-account"><div class="student-avatar">{profile_name[:1].upper()}</div><div><div class="student-name">{profile_name}</div><div class="student-email">{user.email if user else ""}</div></div></div>', unsafe_allow_html=True)
    if st.button("🚪  Log out", key="logout_button", use_container_width=True):
        try: st.session_state.supabase.auth.sign_out()
        except Exception: pass
        for key in ["current_user","profile","supabase"]: st.session_state.pop(key,None)
        st.session_state.books={}; st.session_state.documents=[]; st.session_state.vectorstore=None; st.session_state.messages=[]; st.session_state.chat_sessions={"New Chat":[]}; st.session_state.active_chat="New Chat"
        st.rerun()

    token_ready = bool(secret("HF_TOKEN"))

    if st.button("✎  New chat", key="sidebar_new_chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.markdown('<div class="nav-label">Study</div>', unsafe_allow_html=True)
    nav_items = [
        ("💬", "Chat", "chat"),
        ("🔎", "Search textbooks", "search"),
        ("🎴", "Flashcards", "flashcards"),
        ("📊", "Progress", "progress"),
        ("ℹ️", "Help & setup", "help"),
    ]
    for icon, label, view in nav_items:
        active = st.session_state.active_view == view
        if st.button(f"{icon}  {label}", key=f"nav_{view}", use_container_width=True, type="primary" if active else "secondary"):
            set_view(view)
            st.rerun()

    # --------------------------------------------------------
    # BOOKS — all textbook management lives here
    # --------------------------------------------------------
    st.markdown('<div class="nav-section-title">📚 Books</div>', unsafe_allow_html=True)
    with st.expander(f"Your books  ·  {len(st.session_state.books)}", expanded=True):
        uploads = st.file_uploader(
            "Add textbooks",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key="textbook_uploader",
            label_visibility="visible",
        )
        if uploads:
            st.caption(f"{len(uploads)} file(s) ready to add")
        if st.button("＋ Add & index books", key="index_textbooks", use_container_width=True, disabled=not uploads):
            try:
                with st.spinner("Indexing textbooks..."):
                    n = index_uploads(uploads)
                    for uploaded_name, meta in list(st.session_state.books.items()):
                        if not meta.get("db_id"): save_book_to_db(uploaded_name, meta)
                st.success(f"Added {n} new book(s)." if n else "No new books added.")
                st.rerun()
            except Exception as e:
                st.error(f"Indexing error: {e}")

        if st.session_state.books:
            st.markdown('<div class="book-list">', unsafe_allow_html=True)
            for name, meta in list(st.session_state.books.items()):
                st.markdown(f'<div class="book-item"><div class="book-icon">📘</div><div class="book-info"><div class="book-name">{name}</div><div class="book-meta">{meta["pages"]} pages · {meta["chunks"]} passages</div></div></div>', unsafe_allow_html=True)
                bc1, bc2 = st.columns(2)
                if bc1.button("Open", key=f"open_book_{meta['hash']}", use_container_width=True):
                    st.session_state.active_view = "chat"
                    st.session_state.selected_books = [name]
                    st.rerun()
                if bc2.button("Delete", key=f"rm_{meta['hash']}", use_container_width=True):
                    delete_book_from_db(meta)
                    remove_book(name)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.caption("No books yet. Upload your school textbooks to build your private study library.")

    # --------------------------------------------------------
    # QUIZZES — history + new quiz controls live here
    # --------------------------------------------------------
    st.markdown('<div class="nav-section-title">📝 Quizzes</div>', unsafe_allow_html=True)
    with st.expander(f"Quiz center  ·  {len(st.session_state.quiz_history)} saved", expanded=True):
        if st.button("＋ New quiz", key="sidebar_new_quiz", use_container_width=True):
            st.session_state.active_view = "quiz"
            st.session_state.quiz = []
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = []
            st.rerun()

        if st.session_state.quiz_history:
            st.caption("Previous quizzes")
            for i, item in enumerate(st.session_state.quiz_history[-6:][::-1]):
                title = item.get("topic", "Untitled quiz")
                score = item.get("score")
                label = f"{title[:27]}" + (f"  ·  {score}%" if score is not None else "")
                if st.button(f"📋  {label}", key=f"quiz_history_{i}_{item.get('id', i)}", use_container_width=True):
                    st.session_state.quiz = item.get("questions", [])
                    st.session_state.quiz_answers = item.get("answers", [])
                    st.session_state.quiz_submitted = item.get("submitted", False)
                    st.session_state.quiz_topic = title
                    st.session_state.quiz_history_selected = item.get("id")
                    st.session_state.active_view = "quiz"
                    st.rerun()
        else:
            st.caption("Your completed quizzes will appear here.")

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------
    st.markdown('<div class="nav-section-title">💬 Chats</div>', unsafe_allow_html=True)
    chat_names = [n for n in st.session_state.chat_sessions.keys() if st.session_state.chat_sessions[n] or n == st.session_state.active_chat]
    if not chat_names:
        st.caption("No conversations yet.")
    else:
        for chat_name in chat_names[-8:][::-1]:
            active = chat_name == st.session_state.active_chat
            prefix = "●  " if active else "   "
            if st.button(prefix + chat_name, key=f"chat_open_{hashlib.md5(chat_name.encode()).hexdigest()}", use_container_width=True, type="primary" if active else "secondary"):
                st.session_state.active_chat = chat_name
                st.session_state.messages = st.session_state.chat_sessions[chat_name]
                st.session_state.active_view = "chat"
                st.rerun()

    if st.button("🗑  Clear current chat", key="clear_chat", use_container_width=True):
        st.session_state.chat_sessions[st.session_state.active_chat] = []
        st.session_state.messages = st.session_state.chat_sessions[st.session_state.active_chat]
        st.rerun()

    # --------------------------------------------------------
    # SETTINGS — compact, at the bottom of the sidebar
    # --------------------------------------------------------
    with st.expander("👤  Student profile", expanded=False):
        new_name=st.text_input("Full name",value=profile_name,key="profile_name_setting")
        new_school=st.text_input("School",value=(st.session_state.get("profile") or {}).get("school", ""),key="profile_school_setting")
        new_class=st.text_input("Class / level",value=(st.session_state.get("profile") or {}).get("class_level", ""),key="profile_class_setting")
        if st.button("Save profile",key="save_profile_button",use_container_width=True):
            save_profile(new_name.strip(),new_school.strip(),new_class.strip()); st.success("Profile saved.")

    with st.expander("⚙️  Settings", expanded=False):
        model = st.text_input(
            "Hugging Face chat model",
            value=secret("HF_MODEL", DEFAULT_LLM_MODEL) or DEFAULT_LLM_MODEL,
            key="hf_model_setting",
        )
        temperature = st.slider("Creativity", 0.0, 1.0, .35, .05, key="temperature_setting")
        answer_length = st.selectbox("Answer length", ["Short", "Medium", "Detailed"], index=1, key="answer_length_setting")
        teaching_level = st.selectbox("Student level", ["Beginner", "Secondary School", "Advanced"], index=1, key="teaching_level_setting")
        subject = st.selectbox("Subject tutor", list(SUBJECTS), key="subject_setting")
        study_mode = st.selectbox("Study mode", list(STUDY_MODES), key="study_mode_setting")
        source_mode = st.selectbox("Knowledge mode", ["Textbook only", "Textbook + general knowledge", "Textbook + web research"], key="source_mode_setting")
        retrieval_k = st.slider("Retrieved passages", 3, 10, 6, key="retrieval_k_setting")

    if token_ready:
        st.markdown('<div class="connection"><span class="connection-dot"></span> AI connection ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="connection connection-off"><span class="connection-dot"></span> Add HF_TOKEN in Secrets</div>', unsafe_allow_html=True)

# Keep settings accessible to retrieval/chat logic.
selected_books = st.session_state.get("selected_books", list(st.session_state.books))
selected_books = [b for b in selected_books if b in st.session_state.books]
if not selected_books and st.session_state.books:
    selected_books = list(st.session_state.books)
st.session_state.selected_books = selected_books

chapters = sorted({d.metadata.get("chapter", "Unknown chapter") for d in st.session_state.documents if not selected_books or d.metadata.get("source") in selected_books})
selected_chapters = st.session_state.get("selected_chapters", [])
selected_chapters = [c for c in selected_chapters if c in chapters]
st.session_state.selected_chapters = selected_chapters

if source_mode == "Textbook + web research" and not secret("TAVILY_API_KEY"):
    # Don't interrupt the ChatGPT-like main chat. The settings panel is enough.
    pass

# ============================================================
# MAIN CONTENT
# ============================================================
view = st.session_state.active_view

if view == "chat":
    # Clean ChatGPT-like conversation surface: no tabs, no study controls.
    if not st.session_state.messages:
        st.markdown('<div class="welcome"><h1>📚</h1><h2>How can I help you study?</h2><p>Ask questions about your textbooks, request explanations, solve problems, or prepare for exams.</p></div>', unsafe_allow_html=True)
        if not st.session_state.books:
            st.info("Start by adding a textbook from **📚 Books** in the sidebar.")
        else:
            s1, s2, s3 = st.columns(3)
            s1.markdown("**Explain a topic**\n\nTry: *Explain mitosis simply.*")
            s2.markdown("**Prepare for an exam**\n\nTry: *Make revision notes on electrolysis.*")
            s3.markdown("**Solve a problem**\n\nTry: *Show the steps for this quadratic.*")

    for m in st.session_state.messages:
        avatar = "🧑‍🎓" if m["role"] == "user" else "📚"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("📖 Sources"):
                    for s in m["sources"]:
                        st.markdown(f"<div class='source-card'><b>{s['source']}</b> — page/section {s['page']}<br><span class='small-muted'>{s['chapter']}</span><br><br>{s['excerpt']}</div>", unsafe_allow_html=True)
            if m.get("web_sources"):
                with st.expander("🌐 Web sources"):
                    for i, s in enumerate(m["web_sources"], 1):
                        st.markdown(f"{i}. [{s['title']}]({s['url']})")

    with st.expander("🎤 Voice question", expanded=False):
        audio = st.audio_input("Record a question")
        if audio and st.button("Transcribe recording", key="transcribe_voice"):
            try:
                with st.spinner("Transcribing..."):
                    st.session_state.voice_transcript = transcribe(audio.getvalue())
                st.success(st.session_state.voice_transcript)
            except Exception as e:
                st.error(f"Voice transcription failed: {e}")

    prompt = st.chat_input("Ask anything about your textbooks...", accept_file=True, file_type=["png", "jpg", "jpeg"], disabled=not token_ready)
    if prompt:
        if isinstance(prompt, str):
            query, attached = prompt, []
        else:
            query, attached = prompt.text or "", list(prompt.files or [])
        if attached:
            st.warning("Image attachment received, but this text-only Llama backend cannot inspect images yet.")
        if query.strip():
            query = query.strip()
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.chat_sessions[st.session_state.active_chat] = st.session_state.messages
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(query)
            with st.chat_message("assistant", avatar="📚"):
                try:
                    with st.spinner("Searching your textbooks..."):
                        docs = retrieve(query, selected_books, selected_chapters, retrieval_k)
                        web_results = []
                        effective = source_mode
                        if source_mode == "Textbook + web research" and secret("TAVILY_API_KEY"):
                            web_results = tavily_search(query)
                        elif source_mode == "Textbook + web research":
                            effective = "Textbook + general knowledge"
                        response = answer_question(query, docs, study_mode, subject, answer_length, effective, st.session_state.messages[:-1], model, temperature, web_results)
                    st.markdown(response)
                    sources = source_cards(docs)
                    if sources:
                        with st.expander("📖 Sources used"):
                            for src in sources:
                                st.markdown(f"**{src['source']}** — page/section {src['page']}  \n*{src['chapter']}*")
                    if web_results:
                        with st.expander("🌐 Web sources used"):
                            for i, src in enumerate(web_results, 1):
                                st.markdown(f"{i}. [{src['title']}]({src['url']})")
                    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources, "web_sources": web_results})
                    st.session_state.chat_sessions[st.session_state.active_chat] = st.session_state.messages
                    save_chat_to_db(st.session_state.active_chat, st.session_state.messages)
                    st.session_state.progress["questions_asked"] += 1
                    st.session_state.progress["topics"][query[:60]] += 1
                except Exception as e:
                    st.error(f"AI error: {e}")

elif view == "search":
    st.title("🔎 Search textbooks")
    st.caption("Search across the books in your library. Book selection is managed from the sidebar.")
    sq = st.text_input("Search phrase or concept", placeholder="e.g. osmosis, Newton's laws, quadratic equations")
    nres = st.slider("Number of results", 3, 15, 8)
    if st.button("Search", type="primary", disabled=not sq.strip()):
        if st.session_state.vectorstore is None:
            st.warning("Add and index a textbook from the sidebar first.")
        else:
            st.session_state.last_search_results = retrieve(sq, selected_books, selected_chapters, nres)
    for i, d in enumerate(st.session_state.last_search_results, 1):
        with st.expander(f"{i}. {d.metadata.get('source')} — page/section {d.metadata.get('page')}", expanded=i <= 3):
            st.caption(d.metadata.get("chapter"))
            st.write(d.page_content)

elif view == "quiz":
    st.title("📝 Quiz center")
    st.caption("Generate quizzes from your textbook library or reopen one of your previous quizzes from the sidebar.")
    q1, q2, q3 = st.columns([2, 1, 1])
    topic = q1.text_input("Quiz topic", value=st.session_state.get("quiz_topic", ""), placeholder="e.g. Cell division")
    qcount = q2.selectbox("Questions", [3, 5, 8, 10], index=1)
    difficulty = q3.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    if st.button("✨ Generate new quiz", type="primary", disabled=not topic.strip()):
        try:
            docs = retrieve(topic, selected_books, selected_chapters, min(10, retrieval_k + 2))
            with st.spinner("Generating quiz..."):
                st.session_state.quiz = make_quiz(topic, docs, qcount, difficulty, model)
            st.session_state.quiz_topic = topic
            st.session_state.quiz_answers = []
            st.session_state.quiz_submitted = False
            st.session_state.quiz_history_selected = None
            st.rerun()
        except Exception as e:
            st.error(f"Could not generate quiz: {e}")

    if st.session_state.quiz:
        with st.form("quiz_form"):
            answers = []
            for i, q in enumerate(st.session_state.quiz):
                st.markdown(f"**{i + 1}. {q['question']}**")
                answers.append(st.radio("Choose", q["options"], key=f"qa_{i}", index=None, label_visibility="collapsed"))
                st.divider()
            submitted = st.form_submit_button("Submit answers", use_container_width=True)
        if submitted:
            st.session_state.quiz_answers = answers
            st.session_state.quiz_submitted = True
            correct = sum(a == q["options"][q["answer_index"]] for a, q in zip(answers, st.session_state.quiz))
            total = len(st.session_state.quiz)
            score = round(100 * correct / total) if total else 0
            st.session_state.progress["quizzes_taken"] += 1
            st.session_state.progress["quiz_correct"] += correct
            st.session_state.progress["quiz_total"] += total
            save_progress_to_db(subject, st.session_state.quiz_topic or "Quiz", total, correct)
            db_quiz_id = save_quiz_to_db(st.session_state.quiz_topic or "Untitled quiz", score, total)
            quiz_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
            st.session_state.quiz_history.append({
                "id": db_quiz_id or quiz_id,
                "topic": st.session_state.quiz_topic or "Untitled quiz",
                "score": score,
                "questions": st.session_state.quiz,
                "answers": answers,
                "submitted": True,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            st.rerun()

        if st.session_state.quiz_submitted:
            correct = 0
            for i, q in enumerate(st.session_state.quiz):
                selected = st.session_state.quiz_answers[i] if i < len(st.session_state.quiz_answers) else None
                right = q["options"][q["answer_index"]]
                if selected == right:
                    correct += 1
                    st.success(f"Question {i + 1}: Correct")
                else:
                    st.error(f"Question {i + 1}: Correct answer — {right}")
                st.write(q.get("explanation", ""))
                st.caption(f"Source: {q.get('source', 'Unknown')}")
            st.metric("Quiz score", f"{round(100 * correct / len(st.session_state.quiz))}%")

elif view == "flashcards":
    st.title("🎴 Flashcards")
    st.caption("Create compact revision cards from the textbooks in your library.")
    f1, f2 = st.columns([3, 1])
    ftopic = f1.text_input("Flashcard topic", placeholder="e.g. Organic chemistry reactions")
    fcount = f2.selectbox("Cards", [5, 8, 10, 15], index=1)
    if st.button("✨ Generate flashcards", type="primary", disabled=not ftopic.strip()):
        try:
            docs = retrieve(ftopic, selected_books, selected_chapters, min(10, retrieval_k + 2))
            with st.spinner("Creating flashcards..."):
                st.session_state.flashcards = make_flashcards(ftopic, docs, fcount, model)
                save_flashcards_to_db(st.session_state.flashcards, ftopic)
            st.rerun()
        except Exception as e:
            st.error(f"Could not create flashcards: {e}")
    for i, card in enumerate(st.session_state.flashcards, 1):
        with st.expander(f"Card {i}: {card['front']}"):
            st.markdown(card["back"])
            st.caption(f"Source: {card.get('source', 'Unknown')}")

elif view == "progress":
    st.title("📊 Study progress")
    p = st.session_state.progress
    acc = 100 * p["quiz_correct"] / p["quiz_total"] if p["quiz_total"] else 0
    a, b, c, d = st.columns(4)
    a.metric("Questions asked", p["questions_asked"])
    b.metric("Quizzes taken", p["quizzes_taken"])
    c.metric("Quiz accuracy", f"{acc:.0f}%")
    d.metric("Textbooks", len(st.session_state.books))
    if st.session_state.books:
        rows = [{"Textbook": m["name"], "Pages/sections": m["pages"], "Chunks": m["chunks"], "Size (MB)": m["size_mb"]} for m in st.session_state.books.values()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if st.session_state.quiz_history:
        st.subheader("Quiz history")
        rows = [{"Topic": x["topic"], "Score": f"{x['score']}%", "Date": x["date"]} for x in st.session_state.quiz_history[::-1]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    topics = dict(p["topics"])
    if topics:
        st.subheader("Most studied topics")
        rows = [{"Question/topic": k, "Times studied": v} for k, v in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

elif view == "help":
    st.title("ℹ️ Help & setup")
    st.markdown("""
### Getting started
1. Open **📚 Books** in the sidebar and add your PDF, DOCX, TXT or Markdown textbooks.
2. Ask questions from the main chat. Answers are grounded in the indexed textbook passages.
3. Open **📝 Quizzes** to create a new quiz or reopen previous quizzes.
4. Use **🔎 Search textbooks**, **🎴 Flashcards**, and **📊 Progress** from the sidebar.
5. Use **⚙️ Settings** for subject, study mode, answer length and model options.

### Streamlit Secrets
```toml
HF_TOKEN = "hf_your_token_here"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TAVILY_API_KEY = "tvly_your_key_here"
```

**Important:** never hard-code API keys in `app.py` or commit them to GitHub.

**Limitations:** scanned/image-only PDFs need OCR, the current Llama backend is text-only for images, student accounts and core metadata are persisted in Supabase; textbook file contents/indexes still need Supabase Storage for full cross-device RAG persistence.
""")

# ============================================================
# CHATGPT-STYLE CSS
# ============================================================
# CSS is injected late as well so it reliably wins over Streamlit defaults.
st.markdown("""
<style>
/* Overall ChatGPT-like dark workspace */
.block-container{max-width:1050px;padding-top:1.6rem;padding-bottom:7.5rem}
[data-testid="stSidebar"]{border-right:1px solid rgba(255,255,255,.08);background:#171717}
[data-testid="stSidebar"]>div:first-child{padding:.7rem .55rem 1rem}
[data-testid="stSidebar"] .block-container{padding:0}
[data-testid="stSidebar"] hr{margin:.65rem .35rem;border-color:rgba(255,255,255,.08)}
[data-testid="stSidebar"] .stButton>button{width:100%;border:0;border-radius:9px;background:transparent;text-align:left;min-height:2.35rem;padding:.48rem .65rem;font-weight:450;color:inherit;box-shadow:none}
[data-testid="stSidebar"] .stButton>button:hover{background:rgba(255,255,255,.08)}
[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:rgba(255,255,255,.10)}
[data-testid="stSidebar"] .stButton>button[kind="primary"]:hover{background:rgba(255,255,255,.13)}
[data-testid="stSidebar"] .stExpander{border:0;background:transparent}
[data-testid="stSidebar"] [data-testid="stExpanderDetails"]{padding:.35rem .25rem .55rem}
[data-testid="stSidebar"] [data-testid="stExpanderToggleIcon"]{opacity:.7}
[data-testid="stSidebar"] .stFileUploader{border-radius:9px}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{padding:.55rem;background:rgba(255,255,255,.035);border:1px dashed rgba(255,255,255,.15);border-radius:9px}
[data-testid="stSidebar"] .stCaption{font-size:.72rem}
.chatgpt-brand{display:flex;align-items:center;gap:.55rem;padding:.45rem .55rem .85rem}
.brand-mark{width:30px;height:30px;display:flex;align-items:center;justify-content:center;font-size:1.2rem}
.brand-name{font-size:1.05rem;font-weight:700;letter-spacing:-.02em}
.student-account{display:flex;align-items:center;gap:.55rem;margin:.15rem .35rem .7rem;padding:.55rem .55rem;border-radius:10px;background:rgba(255,255,255,.045)}
.student-avatar{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#2f6feb;color:white;font-weight:700;font-size:.78rem}.student-name{font-size:.78rem;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:155px}.student-email{font-size:.62rem;opacity:.45;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:155px}
.nav-label{font-size:.7rem;font-weight:650;opacity:.45;text-transform:uppercase;letter-spacing:.08em;padding:.35rem .65rem .25rem}
.nav-section-title{font-size:.86rem;font-weight:650;padding:.85rem .65rem .35rem;color:rgba(255,255,255,.82)}
.book-item{display:flex;align-items:center;gap:.55rem;padding:.45rem .25rem}
.book-icon{width:29px;height:29px;border-radius:7px;background:rgba(255,255,255,.08);display:flex;align-items:center;justify-content:center}
.book-info{min-width:0}.book-name{font-size:.77rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.book-meta{font-size:.64rem;opacity:.45;margin-top:.08rem}
.connection{margin:.75rem .35rem .2rem;padding:.45rem .6rem;border-radius:8px;background:rgba(46,204,113,.10);color:#7ee2a8;font-size:.7rem;display:flex;align-items:center;gap:.45rem}
.connection-off{background:rgba(243,156,18,.10);color:#f3c36b}
.connection-dot{width:7px;height:7px;border-radius:50%;background:#36d77f;display:inline-block;box-shadow:0 0 0 3px rgba(54,215,127,.10)}
.connection-off .connection-dot{background:#f0a52c}
/* Main welcome */
.welcome{text-align:center;margin:18vh auto 2rem;max-width:650px}.welcome h1{font-size:2.2rem;margin:0}.welcome h2{font-size:1.8rem;margin:.35rem 0}.welcome p{opacity:.6;font-size:1rem}
.source-card{border:1px solid rgba(128,128,128,.18);border-radius:12px;padding:.75rem;margin:.35rem 0;background:rgba(128,128,128,.04)}
.small-muted{opacity:.7;font-size:.88rem}
[data-testid="stChatMessage"]{border:0;background:transparent;padding:.2rem 0;margin-bottom:.75rem}
/* Fixed bottom composer */
[data-testid="stChatInput"]{position:fixed!important;left:calc(50% - min(525px, 45vw))!important;right:auto!important;width:min(1050px, 90vw)!important;bottom:1rem!important;z-index:999!important;padding:0!important;background:transparent!important}
[data-testid="stChatInput"]>div{border-radius:20px!important;border:1px solid rgba(255,255,255,.12)!important;background:#212121!important;box-shadow:0 8px 30px rgba(0,0,0,.35)!important}
[data-testid="stChatInput"] textarea{min-height:50px!important;max-height:180px!important;padding:14px 58px 14px 16px!important;font-size:.96rem!important}
[data-testid="stChatInput"] button{border-radius:12px!important}
</style>
""", unsafe_allow_html=True)
