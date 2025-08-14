import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
import uuid
import time

print("--- Starting Final Ingestion Script ---")

# --- 1. Load Credentials ---
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    print("🛑 FATAL ERROR: Credentials not found in .env file.")
    exit()
print("✅ Credentials loaded.")

# --- 2. Initialize Clients ---
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# --- Use a BRAND NEW collection name to ensure no conflicts ---
COLLECTION_NAME = "educade_data_v1" 
print(f"✅ Clients initialized. Targeting NEW collection: '{COLLECTION_NAME}'")

# --- 3. Recreate Collection ---
# We will explicitly delete the old and create a new one to guarantee a clean slate.
try:
    print(f"Deleting collection '{COLLECTION_NAME}' if it exists...")
    client.delete_collection(collection_name=COLLECTION_NAME)
    time.sleep(1) # Give the server a moment
    print("Old collection deleted (or did not exist).")
except Exception:
    print("Could not delete old collection (it likely did not exist).")

print(f"Creating new, clean collection: '{COLLECTION_NAME}'...")
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)
print("✅ Collection created successfully.")

# --- 4. Find All PDF Files Recursively ---
base_path = "books"
pdf_files = []
print(f"\n🔎 Searching for all PDF files inside '{base_path}'...")
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".pdf"):
            full_path = os.path.join(root, file)
            try:
                parts = full_path.split(os.sep)
                subject = parts[-2]
                grade = parts[-3]
                pdf_files.append({"path": full_path, "grade": grade, "subject": subject, "filename": file})
            except IndexError:
                print(f"  - ⚠️ Skipping file with unexpected path: {full_path}")

if not pdf_files:
    print("\n🛑 FATAL ERROR: Found 0 PDF files to process in 'books/GRADE/SUBJECT/' structure.")
    exit()
print(f"✅ Found {len(pdf_files)} PDF files.")

# --- 5. Process and Upload Data ---
total_points_uploaded = 0
for pdf_info in pdf_files:
    file_path = pdf_info["path"]
    print(f"\n📄 Processing: {file_path}")
    loader = PyPDFLoader(file_path)
    try:
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        points_batch = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings.embed_query(doc.page_content),
                payload={
                    "text": doc.page_content, "source": pdf_info["filename"],
                    "grade": pdf_info["grade"], "subject": pdf_info["subject"],
                }
            ) for doc in docs
        ]

        if points_batch:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_batch,
                wait=True
            )
            total_points_uploaded += len(points_batch)
            print(f"  - ✅ Uploaded {len(points_batch)} data points.")

    except Exception as e:
        print(f"  - ❌ ERROR processing or uploading file. Skipping. Error: {e}")

print("\n----------------------------------------------------")
print("✅✅✅ INGESTION COMPLETE! ✅✅✅")
print(f"   Total data points uploaded to Qdrant Cloud: {total_points_uploaded}")
print("----------------------------------------------------")