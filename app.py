elif view == "progress":
    st.title(" Study progress")
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
    st.title("ℹ Help & setup")
    st.markdown("""
### Getting started
1. Open ** Books** in the sidebar and add your PDF, DOCX, TXT or Markdown textbooks.
2. Ask questions from the main chat. Answers are grounded in the indexed textbook passages.
3. Open ** Quizzes** to create a new quiz or reopen previous quizzes.
4. Use ** Search textbooks**, ** Flashcards**, and ** Progress** from the sidebar.
5. Use ** Settings** for subject, study mode, answer length and model options.

### Streamlit Secrets
```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
GEMINI_MODEL = "gemini-2.5-flash"
TAVILY_API_KEY = "tvly_your_key_here"
```

**Important:** never hard-code API keys in `app.py` or commit them to GitHub.

**Limitations:** scanned/image-only PDFs need OCR, Gemini is used for chat and voice transcription, student accounts and core metadata are persisted in Supabase; textbook file contents/indexes still need Supabase Storage for full cross-device RAG persistence.
""")

# ============================================================
# CHATGPT-STYLE CSS
# ============================================================
# CSS is injected late as well so it reliably wins over Streamlit defaults.
st.markdown("""
<style>
/* Overall ChatGPT-like dark workspace with subtle study-themed backgrounds */
.stApp{
  background-color:#101114;
  background-image:
    linear-gradient(rgba(15,16,20,.91),rgba(15,16,20,.94)),
    url("https://images.unsplash.com/photo-1455390582262-044c596d4f4e?auto=format&fit=crop&w=2200&q=85");
  background-size:cover;
  background-position:center;
  background-attachment:fixed;
}
.block-container{max-width:1050px;padding-top:1.6rem;padding-bottom:7.5rem}
[data-testid="stSidebar"]{
  border-right:1px solid rgba(255,255,255,.10);
  background-image:linear-gradient(rgba(20,21,27,.94),rgba(20,21,27,.97)),url("https://images.unsplash.com/photo-1507842217343-583bb7270b66?auto=format&fit=crop&w=900&q=80");
  background-size:cover;background-position:center;
}
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
.welcome{text-align:center;margin:18vh auto 2rem;max-width:650px;padding:2.1rem 1.5rem;border:1px solid rgba(255,255,255,.10);border-radius:26px;background:linear-gradient(135deg,rgba(17,20,28,.86),rgba(26,29,39,.76)),url("https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=1400&q=80");background-size:cover;background-position:center;box-shadow:0 18px 55px rgba(0,0,0,.18)}.welcome h1{font-size:2.2rem;margin:0}.welcome h2{font-size:1.8rem;margin:.35rem 0}.welcome p{opacity:.72;font-size:1rem}
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
