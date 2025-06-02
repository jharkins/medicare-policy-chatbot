import os
import openai
import qdrant_client
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

openai_client = openai.Client()

qdrant = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

EMBEDDING_MODEL = "text-embedding-3-small"  # 1 536-d vectors
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "docs")  # fallback name

response = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=openai_client.embeddings.create(
        input=["What is my maximum out of pocket?"],
        model=EMBEDDING_MODEL,
    )
    .data[0]
    .embedding,
)


for hit in response.points:
    print()
    pprint(hit.payload)
    print(f"score: {hit.score}")
    print()
