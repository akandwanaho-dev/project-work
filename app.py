"""Izy Read — clean, responsive study workspace. Run: streamlit run app.py"""
import io, os, re, json, hashlib
from collections import defaultdict
from datetime import datetime
import pandas as pd
import requests
import streamlit as st
from supabase import create_client
from docx import Document as DocxDocument
from groq import Groq
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ASR_MODEL = "whisper-large-v3-turbo"

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
    sb = st.session_state.get("supabase")
    slides = [
        ("learnwithai.jpg", "Learn with AI", "Get clear explanations, whenever you need them."),
        ("students.jpg", "Make progress with confidence", "Build stronger study habits, one topic at a time."),
        ("study.jpg", "Make every session count", "Turn your textbooks into a personal learning plan."),
        ("determined.jpg", "Stay curious. Stay determined.", "Small steps today lead to big results."),
    ]
    # Auto-advance every 6 seconds; CSS animation softens each image transition.
    if "auth_slide" not in st.session_state:
        st.session_state.auth_slide = 0
    try:
        from streamlit_autorefresh import st_autorefresh
        tick = st_autorefresh(interval=6000, limit=None, key="auth_slide_timer")
        if tick:
            st.session_state.auth_slide = tick % len(slides)
    except Exception:
        pass
    slide = slides[st.session_state.auth_slide % len(slides)]
    st.markdown('<div class="eyebrow">WELCOME TO IZY READ</div><h1 class="auth-heading">Your next big idea starts here.</h1><p class="auth-subtitle">Read smarter, practise with purpose, and make progress every day.</p>', unsafe_allow_html=True)
    image_col, form_col = st.columns([1.05, 0.95], gap="large")
    with image_col:
        with st.container(border=True):
            if Path(slide[0]).exists():
                st.image(slide[0], use_container_width=True)
            elif Path("logo.jpg").exists():
                st.image("logo.jpg", use_container_width=True)
            else:
                st.markdown('<div class="empty-card">📚</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="auth-slide-caption"><div class="eyebrow">YOUR STUDY COMPANION</div><h2>{slide[1]}</h2><p>{slide[2]}</p></div>', unsafe_allow_html=True)
            dots = st.columns([1, 1, 1, 1, 4])
            for idx in range(4):
                if dots[idx].button("●" if idx == st.session_state.auth_slide else "○", key=f"auth_dot_{idx}"):
                    st.session_state.auth_slide = idx
                    st.rerun()
    with form_col:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">YOUR LEARNING SPACE</div><h2>Welcome back</h2><p class="muted">Sign in or create an account to continue.</p>', unsafe_allow_html=True)
            if sb is None:
                st.error("Supabase is not connected. Add SUPABASE_URL and SUPABASE_ANON_KEY to Streamlit Secrets.")
                st.code('SUPABASE_URL = "https://your-project.supabase.co"\\nSUPABASE_ANON_KEY = "your-client-safe-key"', language="toml")
                st.stop()
            login_tab, signup_tab = st.tabs(["Log in", "Create account"])
            with login_tab:
                with st.form("login_form"):
                    email = st.text_input("Email", placeholder="student@example.com")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")
                    submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
                if submitted:
                    if not email or not password:
                        st.error("Enter your email and password.")
                    else:
                        try:
                            result = sb.auth.sign_in_with_password({"email": email.strip(), "password": password})
                            if result.user:
                                st.session_state.current_user = result.user
                                st.session_state.supabase = sb
                                load_user_data(sb, result.user.id)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Login failed: {e}")
            with signup_tab:
                with st.form("signup_form"):
                    full_name = st.text_input("Full name", placeholder="Your name")
                    email = st.text_input("Email", placeholder="student@example.com", key="signup_email")
                    password = st.text_input("Password", type="password", placeholder="At least 6 characters", key="signup_password")
                    confirm = st.text_input("Confirm password", type="password", placeholder="Repeat your password")
                    submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
                if submitted:
                    if not full_name.strip() or not email.strip() or not password:
                        st.error("Complete all fields.")
                    elif password != confirm:
                        st.error("Passwords do not match.")
                    elif len(password) < 6:
                        st.error("Use a password with at least 6 characters.")
                    else:
                        try:
                            result = sb.auth.sign_up({"email": email.strip(), "password": password, "options": {"data": {"full_name": full_name.strip()}}})
                            if result.user and result.session:
                                st.session_state.current_user = result.user
                                st.session_state.supabase = sb
                                load_user_data(sb, result.user.id)
                                st.success("Account created.")
                                st.rerun()
                            else:
                                st.success("Account created. Check your email to confirm your account, then log in.")
                        except Exception as e:
                            st.error(f"Sign-up failed: {e}")


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

def groq_client():
    api_key = secret("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key, timeout=90.0, max_retries=2)


def ask_model(messages, model, temperature=.35, max_tokens=900):
    c = groq_client()
    if c is None:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to Streamlit Secrets.")
    out = c.chat.completions.create(
        model=model or DEFAULT_LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (out.choices[0].message.content or "").strip()


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


def answer_question(query, docs, mode, subject, answer_length, source_mode, history, model, temperature, web_results=None, teaching_level="Secondary School"):
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
    c = groq_client()
    if c is None:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to Streamlit Secrets.")
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "recording.wav"
    result = c.audio.transcriptions.create(
        file=(audio_file.name, audio_file.getvalue()),
        model=DEFAULT_ASR_MODEL,
        response_format="text",
    )
    return str(result).strip()


def tavily_search(query):
    key = secret("TAVILY_API_KEY")
    if not key:
        return []
    r = requests.post("https://api.tavily.com/search", json={"api_key": key, "query": query, "search_depth": "basic", "max_results": 5}, timeout=20)
    r.raise_for_status()
    return [{"title": x.get("title", "Web result"), "url": x.get("url", ""), "content": x.get("content", "")} for x in r.json().get("results", [])]

def set_view(view):
    st.session_state.active_view = view


def new_chat():
    name = f"New chat {len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[name] = []
    st.session_state.active_chat = name
    st.session_state.messages = st.session_state.chat_sessions[name]
    st.session_state.active_view = "chat"

st.set_page_config(page_title="Izy Read — AI Study Assistant", page_icon="📚", layout="wide")
st.markdown(r"""
<style>
:root{--ink:#202938;--muted:#64748b;--purple:#4f46e5;--purple2:#6366f1;--line:#e2e8f0;--canvas:#f8fafc;--surface:#ffffff;--lav:#eef2ff;transition:background-color .35s ease,color .35s ease}
html,body,[data-testid="stAppViewContainer"]{background:var(--canvas);color:var(--ink)}
.block-container{max-width:1450px;padding-top:1.4rem;padding-bottom:3rem}
h1,h2,h3{color:var(--ink);letter-spacing:-.035em}
p,.muted,.stCaption{color:var(--muted)}
[data-testid="stSidebar"]{background:#182235;border-right:1px solid rgba(148,163,184,.18)}
[data-testid="stSidebar"] *{color:#f6f5ff}
[data-testid="stSidebar"]>div:first-child{padding:1.25rem .95rem}
[data-testid="stSidebar"] .stButton>button{border:0;border-radius:12px;background:transparent;color:#eeedff;text-align:left;min-height:2.8rem;padding:.65rem .85rem;font-weight:550;box-shadow:none}
[data-testid="stSidebar"] .stButton>button:hover{background:rgba(255,255,255,.10)}
[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:linear-gradient(110deg,#4f46e5,#4338ca);color:white}
[data-testid="stSidebar"] [data-testid="stImage"]{margin-bottom:.3rem}
.brand-title{font-size:1.45rem;font-weight:800;color:#fff;letter-spacing:-.04em}.brand-subtitle{font-size:.83rem;color:#bdbce5;margin-top:.1rem}.side-rule{height:1px;background:rgba(255,255,255,.13);margin:1rem 0}
.sidebar-bottom{height:35vh;min-height:60px}
[data-testid="stMetric"],div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--surface);border:1px solid var(--line);border-radius:18px;box-shadow:0 5px 18px rgba(36,34,88,.035)}
[data-testid="stMetric"]{padding:1rem 1.15rem}
[data-testid="stMetricLabel"]{color:var(--muted)}[data-testid="stMetricValue"]{color:var(--ink);font-weight:750}
.stButton>button,[data-testid="stFormSubmitButton"]>button{border-radius:11px;min-height:2.6rem;font-weight:600;border-color:#e5e3f1}
.stButton>button[kind="primary"],[data-testid="stFormSubmitButton"]>button[kind="primary"]{background:linear-gradient(110deg,#4f46e5,#4338ca);border:0;color:#fff}
.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"]>div,.stNumberInput input{border-radius:10px!important;border-color:var(--line)!important;background:var(--surface)!important}
[data-testid="stChatMessage"]{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:.5rem .8rem}
[data-testid="stChatInput"]{background:var(--surface);border-radius:14px}
.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:.72rem;font-weight:800;color:#7758df;margin-bottom:.35rem}
.hero{padding:1.8rem 2rem;border-radius:24px;background:linear-gradient(115deg,#e7e3ff,#f3edff 55%,#e7f5ff);border:1px solid #e8e1ff;margin-bottom:1.2rem}
.hero h1{margin:0;color:#191b43;font-size:2.1rem}.hero p{font-size:1rem;margin:.45rem 0 0;color:#565d80}
.empty-card{padding:2.5rem 1rem;text-align:center;background:white;border:1px dashed #d9d6ed;border-radius:18px;color:var(--muted)}
[data-testid="stFileUploaderDropzone"]{border:1px dashed #c9c3e9!important;border-radius:14px;background:var(--surface)}
.auth-heading{font-size:clamp(2rem,3vw,2.8rem);letter-spacing:-.04em;margin-bottom:.25rem}
.auth-subtitle{color:var(--muted);margin-bottom:1.2rem}
.auth-slide-caption{padding:.5rem .35rem .2rem;animation:fadeSlide .7s ease both}
[data-testid="stImage"] img{animation:fadeSlide .7s ease both;border-radius:14px}
@keyframes fadeSlide{from{opacity:.25;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
.stTextInput input,.stTextArea textarea{min-height:2.8rem!important;padding:.65rem .85rem!important;transition:border-color .2s,box-shadow .2s!important}
.stTextInput input:focus,.stTextArea textarea:focus{box-shadow:0 0 0 3px rgba(79,70,229,.14)!important;border-color:#6366f1!important}
[data-testid="stChatInput"]{border:1px solid var(--line);box-shadow:0 8px 28px rgba(15,23,42,.07);transition:box-shadow .2s,border-color .2s}
[data-testid="stChatInput"]:focus-within{border-color:#6366f1;box-shadow:0 0 0 3px rgba(79,70,229,.12)}
@media(max-width:800px){.block-container{padding-left:1rem;padding-right:1rem}.hero{padding:1.2rem}.hero h1{font-size:1.6rem}}
body:has(button[kind="secondary"]){ }

</style>
""", unsafe_allow_html=True)

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
        "active_view": "study",
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

with st.sidebar:
    user = current_user()
    profile = st.session_state.get("profile") or {}
    profile_name = profile.get("full_name") or (user.email.split("@")[0] if user and user.email else "Student")
    logo_path = Path("logo.jpg")
    if logo_path.exists():
        st.image(str(logo_path), width=62)
    st.markdown('<div class="brand-title">Izy Read</div><div class="brand-subtitle">AI Study Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-rule"></div>', unsafe_allow_html=True)
    nav = [("💬", "Chats", "chat"), ("📚", "Books", "books"), ("👤", "Profile", "profile"), ("✦", "Study", "study")]
    for icon, label, target in nav:
        active = (st.session_state.active_view == target) or (target == "study" and st.session_state.active_view in {"search", "quiz", "flashcards", "progress", "help"})
        if st.button(f"{icon}   {label}", key=f"mainnav_{target}", use_container_width=True, type="primary" if active else "secondary"):
            st.session_state.active_view = target
            st.rerun()
    st.markdown('<div class="sidebar-bottom"></div>', unsafe_allow_html=True)
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "light"
    if st.button("☾  " + ("Switch to light mode" if st.session_state.ui_theme == "dark" else "Switch to dark mode"), key="theme_toggle", use_container_width=True):
        st.session_state.ui_theme = "light" if st.session_state.ui_theme == "dark" else "dark"
        st.rerun()
    if st.session_state.ui_theme == "dark":
        st.markdown("""<style>:root{--ink:#e5eaf3;--muted:#a5b1c2;--line:#303b4d;--canvas:#0b1220;--surface:#121c2d;--lav:#1c2540} html,body,[data-testid="stAppViewContainer"]{background:var(--canvas)!important;color:var(--ink)!important} h1,h2,h3,p,.stMarkdown,.stCaption{color:var(--ink)} [data-testid="stMetric"],div[data-testid="stVerticalBlockBorderWrapper"],[data-testid="stChatMessage"],[data-testid="stChatInput"]{background:var(--surface)!important;color:var(--ink)!important;border-color:var(--line)!important} .stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"]>div,.stNumberInput input{background:var(--surface)!important;color:var(--ink)!important;border-color:var(--line)!important} .hero{background:linear-gradient(115deg,#202044,#17283c)!important;border-color:#303b4d!important}.hero h1,.hero p{color:#e5eaf3!important} .empty-card{background:var(--surface)!important;border-color:var(--line)!important} [data-testid="stFileUploaderDropzone"]{background:var(--surface)!important;border-color:var(--line)!important}</style>""", unsafe_allow_html=True)
    st.caption(f"Signed in as **{profile_name}**")
    if st.button("Log out", key="logout_button", use_container_width=True):
        try:
            st.session_state.supabase.auth.sign_out()
        except Exception:
            pass
        for key in ["current_user", "profile", "supabase"]:
            st.session_state.pop(key, None)
        st.session_state.books = {}; st.session_state.documents = []; st.session_state.vectorstore = None
        st.session_state.messages = []; st.session_state.chat_sessions = {"New Chat": []}; st.session_state.active_chat = "New Chat"
        st.rerun()

# Persisted study preferences and textbook filters used by every study tool.
model = st.session_state.get("hf_model_setting", secret("GROQ_MODEL", DEFAULT_LLM_MODEL) or DEFAULT_LLM_MODEL)
temperature = st.session_state.get("temperature_setting", .35)
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

view = st.session_state.active_view

if view == "books":
    st.markdown('<div class="eyebrow">YOUR LIBRARY</div>', unsafe_allow_html=True)
    st.title("My Books")
    st.write("Upload your textbooks and build a searchable study library.")
    uploads = st.file_uploader("Upload textbooks", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True, key="library_upload")
    if st.button("Add and index books", type="primary", disabled=not uploads, use_container_width=False):
        try:
            with st.spinner("Reading and indexing your books…"):
                n = index_uploads(uploads)
                for book_name, meta in list(st.session_state.books.items()):
                    if not meta.get("db_id"):
                        save_book_to_db(book_name, meta)
            st.success(f"Added {n} new book(s).")
            st.rerun()
        except Exception as e:
            st.error(f"Could not index books: {e}")
    if st.session_state.books:
        cols = st.columns(2)
        for i, (name, meta) in enumerate(st.session_state.books.items()):
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown("📘 **" + name + "**")
                    st.caption(f"{meta.get('pages', 0)} pages · {meta.get('chunks', 0)} passages")
                    a, b = st.columns(2)
                    if a.button("Ask about it", key=f"book_open_{meta['hash']}", use_container_width=True):
                        st.session_state.selected_books = [name]
                        st.session_state.active_view = "chat"
                        st.rerun()
                    if b.button("Remove", key=f"book_delete_{meta['hash']}", use_container_width=True):
                        delete_book_from_db(meta); remove_book(name); st.rerun()
    else:
        st.markdown('<div class="empty-card"><div style="font-size:2rem">📚</div><h3>Your library starts here</h3><p>Add a PDF, Word document, text file or Markdown textbook to study with Izy Read.</p></div>', unsafe_allow_html=True)

elif view == "profile":
    st.markdown('<div class="eyebrow">YOUR ACCOUNT</div>', unsafe_allow_html=True)
    st.title("Student profile")
    profile = st.session_state.get("profile") or {}
    user = current_user()
    with st.container(border=True):
        full_name = st.text_input("Full name", value=profile.get("full_name") or (user.email.split("@")[0] if user and user.email else ""), key="profile_full_name")
        school = st.text_input("School", value=profile.get("school", ""), key="profile_school")
        class_level = st.text_input("Class / level", value=profile.get("class_level", ""), key="profile_level")
        st.text_input("Email address", value=user.email if user else "", disabled=True)
        if st.button("Save profile", type="primary"):
            save_profile(full_name.strip(), school.strip(), class_level.strip())
            st.success("Your profile has been saved.")
    st.subheader("Preferences")
    st.selectbox("Subject", list(SUBJECTS), key="subject_setting")
    st.selectbox("Teaching style", list(STUDY_MODES), key="study_mode_setting")
    st.selectbox("Answer length", ["Short", "Medium", "Detailed"], index=1, key="answer_length_setting")
    st.slider("Creativity", 0.0, 1.0, .35, .05, key="temperature_setting")

elif view == "study":
    st.markdown('<div class="eyebrow">YOUR STUDY SPACE</div>', unsafe_allow_html=True)
    st.title("Study tools")
    st.write("Everything you need to learn, practise and track your progress — in one place.")
    tools = [
        ("🔎", "Textbook search", "Find explanations inside your uploaded books.", "search"),
        ("🎴", "Flashcards", "Turn topics into quick revision cards.", "flashcards"),
        ("✍️", "Practice quizzes", "Test your understanding with generated questions.", "quiz"),
        ("📈", "Study progress", "Review quiz results and your study activity.", "progress"),
        ("💡", "Help & setup", "Learn how to get the best from Izy Read.", "help"),
    ]
    c1, c2 = st.columns(2)
    for i, (icon, title, desc, target) in enumerate(tools):
        with (c1 if i % 2 == 0 else c2):
            with st.container(border=True):
                st.markdown(f"<div style='font-size:1.8rem'>{icon}</div><h3 style='margin:.35rem 0'>{title}</h3><p class='muted'>{desc}</p>", unsafe_allow_html=True)
                if st.button("Open tool →", key=f"study_tool_{target}", use_container_width=True):
                    st.session_state.active_view = target
                    st.rerun()

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

    token_ready = bool(secret("GROQ_API_KEY"))
    prompt = st.chat_input("Ask a question about your textbooks…", accept_file=True, file_type=["png", "jpg", "jpeg"], disabled=not token_ready)
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
                        response = answer_question(query, docs, study_mode, subject, answer_length, effective, st.session_state.messages[:-1], model, temperature, web_results, teaching_level=teaching_level)
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
                    st.error(f"Response error: {e}")

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
    if st.button("Generate new quiz", type="primary", disabled=not topic.strip()):
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
    if st.button("Generate flashcards", type="primary", disabled=not ftopic.strip()):
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
GROQ_API_KEY = "gsk_your_groq_api_key_here"
GROQ_MODEL = "openai/gpt-oss-120b"
TAVILY_API_KEY = "tvly_your_key_here"
```

**Important:** never hard-code API keys in `app.py` or commit them to GitHub.

**Limitations:** scanned/image-only PDFs need OCR, the current Groq chat backend is text-only for images, student accounts and core metadata are persisted in Supabase; textbook file contents/indexes still need Supabase Storage for full cross-device RAG persistence.
""")

