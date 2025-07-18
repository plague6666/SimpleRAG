from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import os

# Load environment variables
dotenv_path = load_dotenv()
QDRANT_URI = os.getenv("QDRANT_URI", "https://your-qdrant-host")
QDRANT_API = os.getenv("QDRANT_API", "your-qdrant-api-key")
OPENROUTER_API_KEY = os.getenv("OPENROUTER", "your-openrouter-api-key")

qdrant_client = QdrantClient(url=QDRANT_URI, api_key=QDRANT_API, prefer_grpc=False)
embeddings = FastEmbedEmbeddings(max_length=384)
collection_name = "uploaded_pdfs_384"

# Ensure collection exists
try:
    qdrant_client.get_collection(collection_name)
except:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
vectorstore = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embeddings)