import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import time

# --- PINEECONE FINAL INGESTION SCRIPT (v4) ---
print("--- Starting Pinecone Data Ingestion ---")
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "educade-prod-db"

if not PINECONE_API_KEY:
    print("🛑 FATAL ERROR: Pinecone API Key not found in .env file.")
    exit()

print(f"✅ Credentials loaded. Targeting Pinecone index: '{INDEX_NAME}'")
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 1. Check if index exists, if not, create it ---
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{INDEX_NAME}' not found. Creating it now...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("✅ Index created. Waiting for it to initialize...")
    time.sleep(60)
else:
    print(f"✅ Index '{INDEX_NAME}' already exists. Clearing it out...")
    index = pc.Index(INDEX_NAME)
    index.delete(delete_all=True) # Clear the index to ensure a fresh start
    print("✅ Index cleared.")

index = pc.Index(INDEX_NAME)

# --- 2. Find and Process PDF Files ---
base_path = "books"
all_vectors = []
print(f"\n🔎 Searching for all PDF files...")
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".pdf"):
            full_path = os.path.join(root, file)
            print(f"📄 Processing: {full_path}")
            try:
                parts = full_path.split(os.sep)
                grade, subject, filename = parts[-3], parts[-2], file
                
                loader = PyPDFLoader(full_path)
                documents = loader.load()
                docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
                
                for i, doc in enumerate(docs):
                    vector = embeddings.embed_query(doc.page_content)
                    metadata = { "text": doc.page_content, "source": filename, "grade": grade, "subject": subject }
                    all_vectors.append({"id": f"{filename}-{i}", "values": vector, "metadata": metadata})
            except Exception as e:
                print(f"  - ❌ ERROR processing file {filename}. Skipping. Error: {e}")

if not all_vectors:
    print("\n🛑 FATAL ERROR: No data points were created from your PDFs.")
    exit()

print(f"\n✅ Total data points to upload: {len(all_vectors)}")

# --- 3. Upload to Pinecone in Batches ---
BATCH_SIZE = 100
for i in range(0, len(all_vectors), BATCH_SIZE):
    batch = all_vectors[i:i + BATCH_SIZE]
    print(f"  - ⬆️ Uploading batch {i//BATCH_SIZE + 1} ({len(batch)} points)...")
    index.upsert(vectors=batch)

print("\n✅ All batches uploaded.")

# --- 4. Final Verification ---
time.sleep(10) # Give the index a moment to update
index_stats = index.describe_index_stats()
total_vectors = index_stats.get('total_vector_count', 0)

print("\n--- FINAL DATABASE STATUS ---")
print(f"   Total Vectors in Index: {total_vectors}")
print("---------------------------")

if total_vectors == len(all_vectors):
    print("\n✅✅✅ INGESTION SUCCESSFUL AND VERIFIED! ✅✅✅")
else:
    print("\n❌❌❌ VERIFICATION FAILED! ❌❌❌")