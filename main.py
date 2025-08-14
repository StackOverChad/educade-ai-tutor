# main.py
import os
import shutil
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
# Assuming answer_query is now in rag.py or you adapt it
from rag import get_answer # Changed from answer_query
from tts import text_to_speech

app = FastAPI(title="Kids AI Helper")

# serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

BOOKS_DIR = Path("books")
BOOKS_DIR.mkdir(exist_ok=True)
AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

@app.post("/upload-book")
async def upload_book(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")
    out_path = BOOKS_DIR / file.filename
    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"message": f"Uploaded {file.filename}"}

@app.post("/ask")
async def ask_ai(question: str = Form(...), grade: int = Form(None), lang: str = Form("en")):
    """
    question: text question from child
    grade: optional integer 0..4; if omitted, system will auto-detect
    lang: 'en' or 'hi' etc (for TTS and response language)
    """
    # This is a placeholder for the full chat history that rag.get_answer expects
    # In a real FastAPI chat app, you'd manage conversation state.
    messages = [
        {"role": "user", "content": question}
    ]
    
    # You will need to determine subject and grade based on the request
    # For now, let's assume a default or detected grade/subject
    # The new get_answer function requires grade and subject
    # This part might need more logic depending on your FastAPI app's flow
    detected_grade_str = f"Grade{grade}" if grade is not None else "Grade1" # Example
    detected_subject = "English" # Example, you might need to detect this too

    res = get_answer(messages, grade=detected_grade_str, subject=detected_subject, lang=lang)
    
    audio_path = text_to_speech(res["answer"], lang=lang)
    
    return {
        "answer": res["answer"],
        "grade": grade, # Return the grade used
        "sources": res["sources"],
        "audio_file": audio_path
    }

@app.get("/audio/{fname}")
def get_audio(fname: str):
    p = AUDIO_DIR / fname
    if p.exists():
        return FileResponse(str(p), media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="Audio not found")