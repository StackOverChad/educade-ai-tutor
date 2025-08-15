import os
import time
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models

# --- BULLETPROOF INGESTION SCRIPT ---
print("--- Starting Bulletproof Data Ingestion ---")
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "educade_production_db" # A new, final, clean name

if not QDRANT_URL or not QDRANT_API_KEY:
    print("🛑 FATAL ERROR: Credentials not found in .env file.")
    exit()

print(f"✅ Connecting to Qdrant Cloud. Targeting NEW collection: '{COLLECTION_NAME}'")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# --- 1. Recreate Collection ---
print(f"Recreating collection '{COLLECTION_NAME}' to ensure it is clean...")
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)
print("✅ Collection created successfully.")

# --- 2. Find PDF Files ---
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
print(f"✅ Found {len(pdf_files)} PDF files.")

# --- 3. Process and Upload in Small Batches ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
total_points_to_upload = 0
all_points = []

for pdf_info in pdf_files:
    print(f"📄 Processing: {pdf_info['path']}")
    loader = PyPDFLoader(pdf_info["path"])
    documents = loader.load()
    docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
    
    for doc in docs:
        all_points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings.embed_query(doc.page_content),
            payload={ "text": doc.page_content, "source": pdf_info["filename"], "grade": pdf_info["grade"], "subject": pdf_info["subject"] }
        ))
total_points_to_upload = len(all_points)
print(f"\nTotal data points to upload: {total_points_to_upload}")

BATCH_SIZE = 32 # Use a small batch size to be gentle on the server
for i in range(0, len(all_points), BATCH_SIZE):
    batch = all_points[i:i + BATCH_SIZE]
    print(f"  - ⬆️ Uploading batch {i//BATCH_SIZE + 1} ({len(batch)} points)...")
    client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
    time.sleep(1) # Give the server a 1-second break between batches

print("\n✅ All batches uploaded.")

# --- 4. Final Verification ---
print("\nVerifying data integrity on the server...")
wait_time = 10
print(f"Waiting {wait_time} seconds for the server to finish indexing...")
time.sleep(wait_time)

try:
    info = client.get_collection(collection_name=COLLECTION_NAME)
    points = info.points_count
    indexed = info.indexed_vectors_count

    print("\n--- FINAL DATABASE STATUS ---")
    print(f"   Points Count: {points}")
    print(f"   Indexed Vectors: {indexed}")
    print("---------------------------")

    if points == total_points_to_upload and indexed == total_points_to_upload:
        print("\n✅✅✅ INGESTION SUCCESSFUL AND VERIFIED! ✅✅✅")
    else:
        print("\n❌❌❌ VERIFICATION FAILED! ❌❌❌")
        print("   The number of points/indexed vectors on the server does not match the number of points uploaded.")
        print("   This indicates a server-side indexing issue with Qdrant Cloud. Please try again or contact their support.")

except Exception as e:
    print(f"\n❌ FAILED during final verification. Error: {e}")