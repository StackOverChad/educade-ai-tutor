import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
import uuid

# --- BARE-BONES INGESTION SCRIPT ---
print("--- Starting Bare-Bones Ingestion ---")
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "educade_final_db_v2" # A BRAND NEW, CLEAN NAME

if not QDRANT_URL or not QDRANT_API_KEY:
    print("🛑 FATAL ERROR: Credentials not found in .env file.")
    exit()

print(f"✅ Connecting to Qdrant Cloud. Targeting NEW collection: '{COLLECTION_NAME}'")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

print(f"Recreating collection '{COLLECTION_NAME}' to ensure it is clean...")
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    # By not specifying other configs, we let Qdrant use its most stable defaults.
)
print("✅ Collection created successfully.")

base_path = "books"
pdf_files = []
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".pdf"):
            full_path = os.path.join(root, file)
            try:
                parts = full_path.split(os.sep)
                pdf_files.append({"path": full_path, "grade": parts[-3], "subject": parts[-2], "filename": file})
            except IndexError: continue

if not pdf_files:
    print("🛑 FATAL ERROR: No PDFs found in 'books/GRADE/SUBJECT/' structure.")
    exit()

total_points_uploaded = 0
for pdf_info in pdf_files:
    print(f"📄 Processing: {pdf_info['path']}")
    loader = PyPDFLoader(pdf_info["path"])
    documents = loader.load()
    docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
    
    points_batch = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2").embed_query(doc.page_content),
            payload={ "text": doc.page_content, "source": pdf_info["filename"], "grade": pdf_info["grade"], "subject": pdf_info["subject"] }
        ) for doc in docs
    ]
    if points_batch:
        client.upsert(collection_name=COLLECTION_NAME, points=points_batch, wait=True)
        total_points_uploaded += len(points_batch)
        print(f"  - ✅ Uploaded {len(points_batch)} data points.")

print(f"\n✅✅✅ INGESTION COMPLETE! Total points in DB: {total_points_uploaded} ✅✅✅")