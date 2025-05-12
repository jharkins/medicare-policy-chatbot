from qdrant_client import QdrantClient, models
from config import get_settings
from typing import List, Optional


class HybridSearcher:
    def __init__(self):
        self.settings = get_settings()
        self.qdrant_client = QdrantClient(
            url=self.settings.qdrant_url, api_key=self.settings.qdrant_api_key
        )
        # print(
        #     f"DEBUG: embed_model_id='{self.settings.embed_model_id}' TYPE: {type(self.settings.embed_model_id)}"
        # )  # DEBUG
        self.qdrant_client.set_model(self.settings.embed_model_id)  # dense
        self.qdrant_client.set_sparse_model(self.settings.sparse_model_id)  # sparse

    def search(self, text: str):
        search_result = self.qdrant_client.query_points(
            collection_name=self.settings.collection,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            prefetch=[
                models.Prefetch(
                    query=models.Document(
                        text=text, model=self.settings.embed_model_id
                    ),
                    using="fast-all-minilm-l6-v2",
                ),
                models.Prefetch(
                    query=models.Document(
                        text=text, model=self.settings.sparse_model_id
                    ),
                    using="fast-sparse-bm25",
                ),
            ],
            query_filter=None,
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
        # Build an optional Qdrant Filter
        qfilter = None
        if plan_hashes:
            # Flatten the payload path to a top-level field "binary_hash" if needed,
            # or point directly at origin.binary_hash if Qdrant supports nesting.
            qfilter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="origin.binary_hash",  # or just "binary_hash"
                        match=models.MatchAny(any=plan_hashes),
                    )
                ]
            )

        resp = self.qdrant_client.query_points(
            collection_name=self.settings.collection,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            prefetch=[
                models.Prefetch(
                    query=models.Document(
                        text=text, model=self.settings.embed_model_id
                    ),
                    using="fast-all-minilm-l6-v2",
                ),
                models.Prefetch(
                    query=models.Document(
                        text=text, model=self.settings.sparse_model_id
                    ),
                    using="fast-sparse-bm25",
                ),
            ],
            query_filter=qfilter,
            limit=limit,
        )
        return resp.points


if __name__ == "__main__":
    import pprint

    searcher = HybridSearcher()
    pprint.pprint(searcher.search("I want to call Aetna."))
