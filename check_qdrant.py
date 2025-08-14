import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load the credentials from your local .env file
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Check if the credentials were loaded
if not QDRANT_URL or not QDRANT_API_KEY:
    print("🛑 ERROR: QDRANT_URL or QDRANT_API_KEY not found in your .env file.")
    print("Please make sure your .env file contains the correct cloud credentials.")
else:
    print("✅ Credentials loaded from .env file.")
    print(f"   Connecting to Qdrant Cloud at: {QDRANT_URL[:30]}...") # Print partial URL for security

try:
    # Connect to the Qdrant Cloud database
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    collection_name = "kids_ai"
    
    print(f"\nChecking for collection '{collection_name}'...")
    
    # Get information about the collection
    collection_info = client.get_collection(collection_name=collection_name)
    
    # Get the number of vectors (the data points)
    vector_count = collection_info.vectors_count
    
    print("\n🎉 SUCCESS! Connection to Qdrant Cloud is working.")
    print("===================================================")
    print(f"Collection Name: {collection_name}")
    print(f"Number of Data Points (Vectors): {vector_count}")
    print("===================================================")
    
    if vector_count > 0:
        print("\n✅ Your data has been successfully uploaded to the cloud!")
    else:
        print("\n⚠️ WARNING: The collection exists, but it's empty.")
        print("   Please run 'python ingest.py' again to upload your book data.")

except Exception as e:
    print("\n❌ FAILED to connect or get collection info from Qdrant Cloud.")
    print("   This means there is likely an error in your QDRANT_URL or QDRANT_API_KEY.")
    print("   Please double-check your .env file and your Qdrant Cloud dashboard.")
    print(f"\n   Detailed Error: {e}")