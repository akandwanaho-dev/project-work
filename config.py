"""Application configuration for izy read."""

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
