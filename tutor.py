"""Groq-powered tutoring, quiz generation, and voice transcription."""
import io
import json
import re
import requests
import streamlit as st
from groq import Groq
from auth import secret
from config import DEFAULT_LLM_MODEL, DEFAULT_ASR_MODEL, SUBJECTS, STUDY_MODES

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

