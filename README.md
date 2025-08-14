Kids AI Helper - local dev setup

1. Create & activate venv:
   python3 -m venv venv
   source venv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt
   pip install pymupdf

3. Run Qdrant locally:
   docker run -p 6333:6333 qdrant/qdrant

4. Place PDFs in books/ directory.

5. Ingest books:
   python ingest.py

6. Set OPENAI_API_KEY environment variable (needed for grade detection & simplification):
   export OPENAI_API_KEY="sk-..."

7. Run app:
   uvicorn main:app --reload --port 8000

8. Open http://localhost:8000/static/index.html
