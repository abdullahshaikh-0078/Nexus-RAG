import os
import sys
import pytest
import asyncio

# Ensure test database environment setting BEFORE app modules load
os.environ["MONGODB_DB_NAME"] = "nexus_rag_test_db"

from app.core.config import settings
settings.MONGODB_DB_NAME = "nexus_rag_test_db"

from app.db.mongodb import mongo_db


@pytest.fixture(autouse=True, scope="session")
def setup_test_database():
    """
    Session-wide autouse fixture to isolate test runs:
    1. Sets MONGODB_DB_NAME to 'nexus_rag_test_db'.
    2. Ensures tests never pollute primary development database 'nexus_rag_db'.
    3. Drops test database upon test session teardown.
    """
    os.environ["MONGODB_DB_NAME"] = "nexus_rag_test_db"
    settings.MONGODB_DB_NAME = "nexus_rag_test_db"
    
    yield

    # Teardown: Drop isolated test database
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_cleanup_test_db())
        else:
            loop.run_until_complete(_cleanup_test_db())
    except Exception:
        pass


async def _cleanup_test_db():
    if mongo_db.client is not None:
        try:
            await mongo_db.client.drop_database("nexus_rag_test_db")
        except Exception:
            pass
