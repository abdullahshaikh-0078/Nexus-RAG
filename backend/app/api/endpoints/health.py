from fastapi import APIRouter
from app.core.config import settings
from app.db.mongodb import mongo_db
from app.db.vectorstore import vector_store
from app.services.embedder import embedding_service
from app.models.schemas import SystemHealthResponse, ServiceHealth

router = APIRouter()


@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """
    Verifies backend infrastructure and dependency health.
    Checks MongoDB, Qdrant vector database, embedding model, and LLM setup.
    """
    mongo_status = await mongo_db.check_health()
    qdrant_status = vector_store.check_health()

    # Embedding health check
    try:
        model_name = embedding_service.model_name
        embed_health = ServiceHealth(status="healthy", details=f"Model: {model_name}")
    except Exception as e:
        embed_health = ServiceHealth(status="unhealthy", details=str(e))

    # LLM config check
    llm_provider = settings.LLM_PROVIDER
    has_key = bool(settings.GEMINI_API_KEY or settings.GROQ_API_KEY)
    llm_details = f"Provider: {llm_provider} | Model: {settings.LLM_MODEL_NAME} | Key Present: {has_key}"
    llm_health = ServiceHealth(status="healthy", details=llm_details)

    services = {
        "mongodb": ServiceHealth(**mongo_status),
        "qdrant": ServiceHealth(**qdrant_status),
        "embeddings": embed_health,
        "llm": llm_health,
    }

    # Calculate overall status
    is_unhealthy = any(s.status == "unhealthy" for s in services.values())
    is_degraded = any(s.status == "degraded" for s in services.values())

    if is_unhealthy:
        overall = "unhealthy"
    elif is_degraded:
        overall = "degraded"
    else:
        overall = "healthy"

    return SystemHealthResponse(
        status=overall,
        app_name=settings.APP_NAME,
        environment=settings.APP_ENV,
        services=services,
    )
