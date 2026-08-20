import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.endpoints import health, documents, chat, evaluation
from app.db.vectorstore import vector_store
from app.db.mongodb import mongo_db
from app.services.bm25_search import bm25_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info(f"Starting {settings.APP_NAME} ({settings.APP_ENV})...")

    # Initialize Qdrant, MongoDB connections, and hydrate BM25 index
    try:
        vector_store.initialize()
        await mongo_db.connect()
        bm25_service.hydrate_from_vector_store(vector_store)
        logger.info("Database connections & BM25 index initialized successfully.")
    except Exception as e:
        logger.warning(f"Error during startup initialization: {str(e)}")

    yield

    logger.info(f"Shutting down {settings.APP_NAME}...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade RAG engine for intelligent document search and synthesis.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production can lock down to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health.router, prefix=settings.API_V1_PREFIX, tags=["System Health"])
app.include_router(documents.router, prefix=f"{settings.API_V1_PREFIX}/documents", tags=["Document Management"])
app.include_router(chat.router, prefix=f"{settings.API_V1_PREFIX}/chat", tags=["Chat & RAG Query"])
app.include_router(chat.router, prefix=f"{settings.API_V1_PREFIX}/chats", tags=["Chat Sessions"])
app.include_router(evaluation.router, prefix=f"{settings.API_V1_PREFIX}/evaluation", tags=["Evaluation Engine"])


@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "version": "1.0.0",
        "health_check": f"{settings.API_V1_PREFIX}/health",
        "docs_url": "/docs",
    }
