from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    qdrant_api_key: str
    qdrant_url: str
    embed_model_id: str
    sparse_model_id: str
    collection: str
    openai_api_key: str = ""  # Optional, only needed for embedding.py

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore
