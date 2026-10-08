"""Supabase authentication and user-data persistence."""
import os
import streamlit as st
from supabase import create_client
from collections import defaultdict
from datetime import datetime

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
    st.markdown('<div style="max-width:520px;margin:8vh auto 0;text-align:center"><div style="font-size:3rem">📚</div><h1 style="margin:.25rem 0">izy read</h1><p style="opacity:.65">Your personal textbook study workspace</p></div>',unsafe_allow_html=True)
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

