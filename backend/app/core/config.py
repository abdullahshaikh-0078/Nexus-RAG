from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "NEXUS RAG"
    APP_ENV: str = "development"
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database Settings
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "nexus_rag_db"

    # Vector Store Settings
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "nexus_chunks"
    QDRANT_STORAGE_PATH: str = "./data/qdrant_db"

    # Embedding Settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Chunking Defaults
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # LLM Settings
    LLM_PROVIDER: str = "gemini"  # gemini, groq, or mock
    GEMINI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    LLM_MODEL_NAME: str = "gemini-1.5-flash"

    # FinanceBench Settings
    FINANCEBENCH_ROOT: str = r"C:\Abdullah files\datasets\financebench"
    FINANCEBENCH_PDF_DIR: str = r"C:\Abdullah files\datasets\financebench\pdfs"
    FINANCEBENCH_DATASET: str = r"C:\Abdullah files\datasets\financebench\data\financebench_open_source.jsonl"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )


settings = Settings()
