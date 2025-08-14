import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

print("--- Qdrant Cloud Verification Script ---")
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    print("🛑 ERROR: Credentials not found in .env file.")
else:
    print("✅ Credentials loaded successfully.")
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print("✅ Connection to Qdrant Cloud successful.")
        
        response = client.get_collections()
        collections = [col.name for col in response.collections]
        
        print("\n--- DATABASE STATE ---")
        if collections:
            print(f"🎉 Found the following collections: {collections}")
            if "kids_ai" in collections:
                info = client.get_collection(collection_name="kids_ai")
                print(f"   - Collection 'kids_ai' has {info.vectors_count or 0} data points.")
            else:
                print("   - CRITICAL: The required collection 'kids_ai' was NOT found.")
        else:
            print("   - CRITICAL: No collections were found in your database. It is empty.")
        print("----------------------")

    except Exception as e:
        print(f"\n❌ FAILED to connect or get collections. Error: {e}")