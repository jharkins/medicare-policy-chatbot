from qdrant_client import QdrantClient, models
from config import get_settings
from typing import List, Optional
import openai
import os


class HybridSearcher:
    def __init__(self):
        self.settings = get_settings()
        self.qdrant_client = QdrantClient(
            url=self.settings.qdrant_url, api_key=self.settings.qdrant_api_key
        )
        # Initialize OpenAI client for embeddings
        self.openai_client = openai.Client(api_key=self.settings.openai_api_key)
        self.embed_model = "text-embedding-3-small"  # Match what was used in embedding.py

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        response = self.openai_client.embeddings.create(
            model=self.embed_model,
            input=text
        )
        return response.data[0].embedding

    def search(self, text: str):
        # Use the actual collection name from env or default
        collection_name = os.getenv("QDRANT_COLLECTION", "docs")
        
        # Generate embedding for the query
        query_vector = self._get_embedding(text)
        
        search_result = self.qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=5,
        ).points

        return [point.payload for point in search_result]

    # def visual_grounding(self, text: str, limit: int = 5):
    #     search_result = self.qdrant_client.query_points(
    #         collection_name=self.settings.collection,
    #         query=models.FusionQuery(fusion=models.Fusion.RRF),
    #         prefetch=[
    #             models.Prefetch(
    #                 query=models.Document(
    #                     text=text, model=self.settings.embed_model_id
    #                 ),
    #                 using="fast-all-minilm-l6-v2",
    #             ),
    #             models.Prefetch(
    #                 query=models.Document(
    #                     text=text, model=self.settings.sparse_model_id
    #                 ),
    #                 using="fast-sparse-bm25",
    #             ),
    #         ],
    #         query_filter=None,
    #         limit=limit,
    #     )

    #     return search_result.points

    def visual_grounding(
        self,
        text: str,
        limit: int = 5,
        plan_hashes: Optional[List[str]] = None,
    ):
        """
        If plan_hashes is provided, only return points whose payload.origin.binary_hash
        is in that list.
        """
        # Use the actual collection name from env or default
        collection_name = os.getenv("QDRANT_COLLECTION", "docs")
        
        # Generate embedding for the query
        query_vector = self._get_embedding(text)
        
        # Build an optional Qdrant Filter
        qfilter = None
        if plan_hashes:
            qfilter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="origin.binary_hash",
                        match=models.MatchAny(any=plan_hashes),
                    )
                ]
            )

        resp = self.qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=qfilter,
            limit=limit,
        )
        return resp.points


if __name__ == "__main__":
    import pprint
    from dotenv import load_dotenv
    
    load_dotenv()
    searcher = HybridSearcher()
    results = searcher.search("What is my maximum out of pocket?")
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        if 'text' in result:
            print(f"Text: {result['text'][:200]}...")
        if 'origin' in result and isinstance(result['origin'], dict):
            print(f"Source: {result['origin'].get('filename', 'Unknown')}")
