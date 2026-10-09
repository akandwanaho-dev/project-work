"""Main entry point for izy read. Run with: streamlit run app.py"""
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
from groq import Groq
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DEFAULT_LLM_MODEL, DEFAULT_EMBED_MODEL, DEFAULT_ASR_MODEL, STUDY_MODES, SUBJECTS
from auth import secret, supabase_client, current_user, load_user_data, save_profile, save_chat_to_db, save_book_to_db, delete_book_from_db, save_quiz_to_db, save_flashcards_to_db, save_progress_to_db, auth_screen, require_auth
from textbooks import sha, clean, safe_name, chapter_from, raw_docs_from_upload, embeddings_model, rebuild_index, index_uploads, remove_book
from tutor import groq_client, ask_model, retrieve, context_text, source_cards, answer_question, json_array_from_model, make_quiz, make_flashcards, transcribe, tavily_search
from navigation import set_view, new_chat

st.set_page_config(page_title="izy read", page_icon="📚", layout="wide")

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

with st.sidebar:
    user=current_user()
    profile_name=(st.session_state.get("profile") or {}).get("full_name") or (user.email.split("@")[0] if user and user.email else "Student")
    st.markdown("""
    <div class="chatgpt-brand">
        <div class="brand-mark">📚</div>
        <div class="brand-name">izy read</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="student-account"><div class="student-avatar">{profile_name[:1].upper()}</div><div><div class="student-name">{profile_name}</div><div class="student-email">{user.email if user else ""}</div></div></div>', unsafe_allow_html=True)
    if st.button("🚪  Log out", key="logout_button", use_container_width=True):
        try: st.session_state.supabase.auth.sign_out()
        except Exception: pass
        for key in ["current_user","profile","supabase"]: st.session_state.pop(key,None)
        st.session_state.books={}; st.session_state.documents=[]; st.session_state.vectorstore=None; st.session_state.messages=[]; st.session_state.chat_sessions={"New Chat":[]}; st.session_state.active_chat="New Chat"
        st.rerun()

    token_ready = bool(secret("GROQ_API_KEY"))

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
            "Groq chat model",
            value=secret("GROQ_MODEL", DEFAULT_LLM_MODEL) or DEFAULT_LLM_MODEL,
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
        st.markdown('<div class="connection"><span class="connection-dot"></span> Study services ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="connection connection-off"><span class="connection-dot"></span> Add GROQ_API_KEY in Secrets</div>', unsafe_allow_html=True)

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
