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
.block-container{max-width:1500px;padding-top:1.2rem;padding-bottom:2rem}
[data-testid="stSidebar"]{border-right:1px solid rgba(128,128,128,.2)}
.hero{padding:1.35rem 1.5rem;border:1px solid rgba(128,128,128,.18);border-radius:24px;background:linear-gradient(135deg,rgba(100,90,255,.14),rgba(0,190,160,.08));margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem}.hero p{margin:.35rem 0 0;opacity:.8}
.source-card{border:1px solid rgba(128,128,128,.18);border-radius:14px;padding:.8rem;margin:.35rem 0;background:rgba(128,128,128,.04)}
.small-muted{opacity:.7;font-size:.88rem}
[data-testid="stChatMessage"]{border:1px solid rgba(128,128,128,.10);border-radius:18px;padding:.35rem .55rem;margin-bottom:.45rem}
.stButton button{border-radius:12px}
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
        "flashcards": [],
        "last_search_results": [],
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

# ============================================================
# HELPERS
# ============================================================
def secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)


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
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 📚 Textbook AI")
    st.caption("Your textbook RAG study workspace")

    token_ready = bool(secret("HF_TOKEN"))
    st.success("AI connection ready", icon="✅") if token_ready else st.warning("Add HF_TOKEN in Secrets", icon="🔑")

    st.divider()
    st.markdown("### Library")
    uploads = st.file_uploader("Upload textbooks", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if st.button("📥 Index uploaded files", use_container_width=True, disabled=not uploads):
        try:
            n = index_uploads(uploads)
            st.success(f"Indexed {n} new textbook(s)." if n else "No new files to index.")
        except Exception as e:
            st.error(f"Indexing error: {e}")

    for name, meta in list(st.session_state.books.items()):
        with st.expander(f"📘 {name}"):
            st.caption(f"{meta['size_mb']} MB • {meta['pages']} pages/sections • {meta['chunks']} chunks")
            st.caption(f"Uploaded {meta['uploaded']}")
            if st.button("Remove", key=f"rm_{meta['hash']}", use_container_width=True):
                remove_book(name)
                st.rerun()

    st.divider()
    st.markdown("### Chats")
    new_name = st.text_input("New chat name", placeholder="e.g. Organic Chemistry")
    if st.button("➕ Create chat", use_container_width=True):
        name = new_name.strip() or f"Chat {len(st.session_state.chat_sessions)+1}"
        st.session_state.chat_sessions.setdefault(name, [])
        st.session_state.active_chat = name
        st.session_state.messages = st.session_state.chat_sessions[name]
        st.rerun()

    chat_names = list(st.session_state.chat_sessions)
    idx = chat_names.index(st.session_state.active_chat) if st.session_state.active_chat in chat_names else 0
    chosen = st.selectbox("Open chat", chat_names, index=idx)
    if chosen != st.session_state.active_chat:
        st.session_state.active_chat = chosen
        st.session_state.messages = st.session_state.chat_sessions[chosen]
        st.rerun()

    if st.button("🗑️ Clear current chat", use_container_width=True):
        st.session_state.chat_sessions[st.session_state.active_chat] = []
        st.session_state.messages = st.session_state.chat_sessions[st.session_state.active_chat]
        st.rerun()

    st.divider()
    st.markdown("### Model settings")
    model = st.text_input("Hugging Face chat model", value=secret("HF_MODEL", DEFAULT_LLM_MODEL) or DEFAULT_LLM_MODEL)
    temperature = st.slider("Creativity", 0.0, 1.0, .35, .05)
    answer_length = st.selectbox("Answer length", ["Short", "Medium", "Detailed"], index=1)
    teaching_level = st.selectbox("Student level", ["Beginner", "Secondary School", "Advanced"], index=1)

# ============================================================
# MAIN
# ============================================================
st.markdown("""
<div class="hero"><h1>📚 Textbook AI</h1><p>Upload textbooks, ask grounded questions, generate quizzes and flashcards, search your library, and track revision progress.</p></div>
""", unsafe_allow_html=True)

book_names = list(st.session_state.books)
selected_books = st.multiselect("Search in textbooks", book_names, default=book_names)
chapters = sorted({d.metadata.get("chapter", "Unknown chapter") for d in st.session_state.documents if not selected_books or d.metadata.get("source") in selected_books})

with st.expander("🎛️ Study controls"):
    c1, c2, c3, c4 = st.columns(4)
    subject = c1.selectbox("Subject tutor", list(SUBJECTS))
    study_mode = c2.selectbox("Study mode", list(STUDY_MODES))
    source_mode = c3.selectbox("Knowledge mode", ["Textbook only", "Textbook + general knowledge", "Textbook + web research"])
    retrieval_k = c4.slider("Retrieved passages", 3, 10, 6)
    selected_chapters = st.multiselect("Limit to chapters/topics", chapters, default=[])

if source_mode == "Textbook + web research" and not secret("TAVILY_API_KEY"):
    st.info("Web research requires TAVILY_API_KEY in Streamlit Secrets. Without it, the app falls back to textbook + general knowledge.")

tab_chat, tab_search, tab_quiz, tab_cards, tab_progress, tab_help = st.tabs(["💬 Chat", "🔎 Search", "📝 Quiz", "🎴 Flashcards", "📊 Progress", "ℹ️ Help"])

# CHAT
with tab_chat:
    if not st.session_state.messages:
        a,b,c = st.columns(3)
        a.info("**Explain a topic**\n\nExplain mitosis in simple terms.")
        b.info("**Exam preparation**\n\nGive me revision notes on electrolysis.")
        c.info("**Step-by-step**\n\nShow how to solve this quadratic equation.")

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
                    for i,s in enumerate(m["web_sources"],1):
                        st.markdown(f"{i}. [{s['title']}]({s['url']})")

    with st.expander("🎤 Ask by voice"):
        audio = st.audio_input("Record a question")
        if audio and st.button("Transcribe recording"):
            try:
                with st.spinner("Transcribing..."):
                    st.session_state.voice_transcript = transcribe(audio.getvalue())
                st.success(st.session_state.voice_transcript)
            except Exception as e:
                st.error(f"Voice transcription failed: {e}")
        if st.session_state.get("voice_transcript"):
            st.code(st.session_state.voice_transcript)

    prompt = st.chat_input("Ask about your textbooks...", accept_file=True, file_type=["png","jpg","jpeg"], disabled=not token_ready)
    if prompt:
        if isinstance(prompt, str):
            query, attached = prompt, []
        else:
            query, attached = prompt.text or "", list(prompt.files or [])
        if attached:
            st.warning("Image attachment received, but this text-only Llama backend cannot inspect images yet. Type the question shown in the image or switch to a vision-language model.")
        if query.strip():
            query = query.strip()
            st.session_state.messages.append({"role":"user","content":query})
            st.session_state.chat_sessions[st.session_state.active_chat] = st.session_state.messages
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(query)
            with st.chat_message("assistant", avatar="📚"):
                try:
                    with st.spinner("Searching your textbooks..."):
                        docs = retrieve(query, selected_books, selected_chapters, retrieval_k)
                        web_results = []
                        effective = source_mode
                        if source_mode == "Textbook + web research":
                            if secret("TAVILY_API_KEY"):
                                web_results = tavily_search(query)
                            else:
                                effective = "Textbook + general knowledge"
                        response = answer_question(query, docs, study_mode, subject, answer_length, effective, st.session_state.messages[:-1], model, temperature, web_results)
                    st.markdown(response)
                    sources = source_cards(docs)
                    if sources:
                        with st.expander("📖 Sources used"):
                            for s in sources:
                                st.markdown(f"**{s['source']}** — page/section {s['page']}  \n*{s['chapter']}*")
                    if web_results:
                        with st.expander("🌐 Web sources used"):
                            for i,s in enumerate(web_results,1):
                                st.markdown(f"{i}. [{s['title']}]({s['url']})")
                    st.session_state.messages.append({"role":"assistant","content":response,"sources":sources,"web_sources":web_results})
                    st.session_state.chat_sessions[st.session_state.active_chat] = st.session_state.messages
                    st.session_state.progress["questions_asked"] += 1
                    st.session_state.progress["topics"][query[:60]] += 1
                except Exception as e:
                    st.error(f"AI error: {e}")

# SEARCH
with tab_search:
    st.markdown("### 🔎 Search your textbook library")
    sq = st.text_input("Search phrase or concept", placeholder="e.g. osmosis")
    nres = st.slider("Number of results", 3, 15, 8, key="nres")
    if st.button("Search textbooks", disabled=not sq.strip()):
        if st.session_state.vectorstore is None:
            st.warning("Upload and index a textbook first.")
        else:
            st.session_state.last_search_results = retrieve(sq, selected_books, selected_chapters, nres)
    for i,d in enumerate(st.session_state.last_search_results,1):
        with st.expander(f"{i}. {d.metadata.get('source')} — page/section {d.metadata.get('page')}", expanded=i<=3):
            st.caption(d.metadata.get("chapter"))
            st.write(d.page_content)

# QUIZ
with tab_quiz:
    st.markdown("### 📝 Generate a textbook quiz")
    q1,q2,q3 = st.columns([2,1,1])
    topic = q1.text_input("Quiz topic", placeholder="e.g. Cell division")
    qcount = q2.selectbox("Questions", [3,5,8,10], index=1)
    difficulty = q3.selectbox("Difficulty", ["Easy","Medium","Hard"], index=1)
    if st.button("✨ Generate quiz", disabled=not topic.strip()):
        try:
            docs = retrieve(topic, selected_books, selected_chapters, min(10,retrieval_k+2))
            with st.spinner("Generating quiz..."):
                st.session_state.quiz = make_quiz(topic, docs, qcount, difficulty, model)
            st.session_state.quiz_submitted = False
            st.rerun()
        except Exception as e:
            st.error(f"Could not generate quiz: {e}")

    if st.session_state.quiz:
        with st.form("quiz_form"):
            answers = []
            for i,q in enumerate(st.session_state.quiz):
                st.markdown(f"**{i+1}. {q['question']}**")
                answers.append(st.radio("Choose", q["options"], key=f"qa_{i}", index=None, label_visibility="collapsed"))
                st.divider()
            submitted = st.form_submit_button("Submit answers", use_container_width=True)
        if submitted:
            st.session_state.quiz_answers = answers
            st.session_state.quiz_submitted = True
            correct = sum(a == q["options"][q["answer_index"]] for a,q in zip(answers, st.session_state.quiz))
            st.session_state.progress["quizzes_taken"] += 1
            st.session_state.progress["quiz_correct"] += correct
            st.session_state.progress["quiz_total"] += len(st.session_state.quiz)
        if st.session_state.quiz_submitted:
            correct = 0
            for i,q in enumerate(st.session_state.quiz):
                selected = st.session_state.quiz_answers[i] if i < len(st.session_state.quiz_answers) else None
                right = q["options"][q["answer_index"]]
                if selected == right:
                    correct += 1
                    st.success(f"Question {i+1}: Correct")
                else:
                    st.error(f"Question {i+1}: Correct answer — {right}")
                st.write(q.get("explanation", ""))
                st.caption(f"Source: {q.get('source','Unknown')}")
            st.metric("Quiz score", f"{round(100*correct/len(st.session_state.quiz))}%")

# FLASHCARDS
with tab_cards:
    st.markdown("### 🎴 Generate revision flashcards")
    f1,f2 = st.columns([3,1])
    ftopic = f1.text_input("Flashcard topic", placeholder="e.g. Organic chemistry reactions")
    fcount = f2.selectbox("Cards", [5,8,10,15], index=1)
    if st.button("✨ Generate flashcards", disabled=not ftopic.strip()):
        try:
            docs = retrieve(ftopic, selected_books, selected_chapters, min(10,retrieval_k+2))
            with st.spinner("Creating flashcards..."):
                st.session_state.flashcards = make_flashcards(ftopic, docs, fcount, model)
            st.rerun()
        except Exception as e:
            st.error(f"Could not create flashcards: {e}")
    for i,c in enumerate(st.session_state.flashcards,1):
        with st.expander(f"Card {i}: {c['front']}"):
            st.markdown(c["back"])
            st.caption(f"Source: {c.get('source','Unknown')}")

# PROGRESS
with tab_progress:
    st.markdown("### 📊 Study progress")
    p = st.session_state.progress
    acc = 100*p["quiz_correct"]/p["quiz_total"] if p["quiz_total"] else 0
    a,b,c,d = st.columns(4)
    a.metric("Questions asked", p["questions_asked"])
    b.metric("Quizzes taken", p["quizzes_taken"])
    c.metric("Quiz accuracy", f"{acc:.0f}%")
    d.metric("Textbooks", len(st.session_state.books))
    if st.session_state.books:
        rows = [{"Textbook":m["name"],"Pages/sections":m["pages"],"Chunks":m["chunks"],"Size (MB)":m["size_mb"]} for m in st.session_state.books.values()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    topics = dict(p["topics"])
    if topics:
        rows = [{"Question/topic":k,"Times studied":v} for k,v in sorted(topics.items(), key=lambda x:x[1], reverse=True)[:10]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# HELP
with tab_help:
    st.markdown("### ℹ️ Setup")
    st.markdown("""
**Included:** multiple textbook uploads, FAISS RAG, local embeddings, page/section metadata, citations, subject tutors, study modes, quiz generation, flashcards, textbook search, session chat history, progress tracking, voice transcription, optional live web research, and image-attachment UI.

**Streamlit Secrets:**
```toml
HF_TOKEN = "hf_your_token_here"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # optional
TAVILY_API_KEY = "tvly_your_key_here"         # optional, for live web search
```

**Important limitations:**
- Scanned/image-only PDFs need OCR; `pypdf` only extracts embedded text.
- The current Llama backend is text-only, so photo understanding needs a vision-language model.
- Library data, chats, FAISS index, and progress are stored in Streamlit session memory. Add a database/object store for true persistent accounts.
- Hugging Face serverless model availability can change; set `HF_MODEL` to another chat-capable model if needed.
""")
    st.warning("Never hard-code API keys in app.py or commit them to GitHub.")
