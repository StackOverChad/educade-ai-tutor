# grade_detect.py
import os
import textstat
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def heuristic_grade_from_text(text: str):
    """
    Use Flesch–Kincaid approximate mapping:
    - Score < 1.5 -> nursery/LKG
    - 1.5–2.5 -> UKG/Grade1
    - 2.5–3.5 -> Grade2
    - 3.5–4.5 -> Grade3
    - >4.5 -> Grade4+
    """
    try:
        score = textstat.flesch_kincaid_grade(text)
    except Exception:
        return None
    if score < 1.5:
        return 0
    if score < 2.5:
        return 1
    if score < 3.5:
        return 2
    if score < 4.5:
        return 3
    return 4

def llm_detect_grade(question: str, context: str = "") -> int:
    """
    Use a short LLM prompt to decide grade (0..4). Returns int or None.
    """
    if client is None:
        return None
    prompt = f"""
You're a helpful assistant. Classify the appropriate school level for the following child's question and short context.
Return only a single integer between 0 and 4, where:
0 = nursery / pre-reader,
1 = LKG (pre-school),
2 = UKG / Grade 1,
3 = Grade 2–3,
4 = Grade 4

Question:
{question}

Context (short):
{context}

Return just the digit (0,1,2,3 or 4).
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user", "content": prompt}],
            max_tokens=4,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        digit = int(text[0]) if text and text[0].isdigit() else None
        if digit is not None and 0 <= digit <= 4:
            return digit
    except Exception:
        return None
    return None

def detect_grade(question: str, retrieved_text: str = "") -> int:
    # Try heuristic on question + snippet
    combined = (question + "\n\n" + (retrieved_text or "")).strip()
    if combined:
        h = heuristic_grade_from_text(combined)
        if h is not None:
            return h
    # fallback to LLM if available
    g = llm_detect_grade(question, retrieved_text)
    if g is not None:
        return g
    # default mid primary
    return 2
