# simplify.py
import os
import textstat
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def needs_simplify(text: str, target_grade: int) -> bool:
    try:
        fk = textstat.flesch_kincaid_grade(text)
    except Exception:
        return True
    # if fk grade is more than 1 above target, simplify
    return fk > (target_grade + 1.0)

def simplify_with_llm(text: str, target_grade: int) -> str:
    if client is None:
        return text
    prompt = f"""
You are a very friendly primary school teacher who explains things simply.
Rewrite the following answer so a child at Grade {target_grade} can read and understand it.
- Use short sentences (<= 10 words per sentence).
- Use simple words.
- Give one short example or tiny activity.
- Keep it cheerful and encouraging.

Answer:
{text}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user", "content": prompt}],
        max_tokens=400,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def adjust_for_grade(answer: str, target_grade: int) -> str:
    if not needs_simplify(answer, target_grade):
        return answer
    simplified = simplify_with_llm(answer, target_grade)
    return simplified
