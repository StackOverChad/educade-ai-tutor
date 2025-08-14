import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader  # updated import per warning
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "kids_ai"

# Initialize Qdrant client with optional timeout (adjust as needed)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

# Set embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create collection if not exists
existing_collections = [col.name for col in client.get_collections().collections]
if COLLECTION_NAME not in existing_collections:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        shard_number=1,
        replication_factor=1,
        write_consistency_factor=1,
        on_disk_payload=False,
    )

base_path = "books"
grades = ["Grade1"]
subjects = ["English"]

BATCH_SIZE = 50  # batch size for upsert

for grade in grades:
    for subject in subjects:
        folder_path = os.path.join(base_path, grade, subject)
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                print(f"📄 Loading {os.path.join(folder_path, filename)}")
                loader = PyPDFLoader(os.path.join(folder_path, filename))
                documents = loader.load()

                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = text_splitter.split_documents(documents)

                points = []
                for i, doc in enumerate(docs):
                    vector = embeddings.embed_query(doc.page_content)
                    point_id = i
                    payload = {
                        "text": doc.page_content,
                        "source": filename,
                        "grade": grade,
                        "subject": subject,
                    }
                    points.append(PointStruct(id=point_id, vector=vector, payload=payload))

                # Upload in batches to avoid timeout
                for i in range(0, len(points), BATCH_SIZE):
                    batch = points[i:i + BATCH_SIZE]
                    print(f"⬆️ Uploading batch {i // BATCH_SIZE + 1} with {len(batch)} chunks...")
                    client.upsert(collection_name=COLLECTION_NAME, points=batch)

print("✅ Ingestion complete!")
