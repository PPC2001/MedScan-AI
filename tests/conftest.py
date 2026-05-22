"""
pytest configuration and shared fixtures.
"""

import asyncio
import uuid
from datetime import date
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from medscan.api.dependencies import get_session
from medscan.api.main import app
from medscan.config import get_settings

settings = get_settings()

# Use an in-memory SQLite for tests (fast, no Docker required)
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_medscan.db"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    async_session = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


@pytest_asyncio.fixture
async def client(test_session) -> AsyncGenerator[AsyncClient, None]:
    """Async test client with DB session override."""
    async def override_session():
        yield test_session

    app.dependency_overrides[get_session] = override_session

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"X-API-Key": settings.api_key},
    ) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_patient_data():
    return {
        "name": "Test Patient",
        "date_of_birth": "1980-05-15",
        "gender": "Male",
        "mrn": f"TEST-{uuid.uuid4().hex[:6].upper()}",
        "blood_type": "A+",
    }
