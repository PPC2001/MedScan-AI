"""Tests for the FastAPI endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    """Health endpoint returns 200."""
    response = await client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_create_patient(client: AsyncClient, sample_patient_data: dict) -> None:
    """Create a patient and verify response."""
    response = await client.post("/patients/", json=sample_patient_data)
    assert response.status_code == 201
    data = response.json()
    assert data["mrn"] == sample_patient_data["mrn"]
    assert data["name"] == sample_patient_data["name"]
    assert "id" in data


@pytest.mark.asyncio
async def test_create_patient_duplicate_mrn(
    client: AsyncClient, sample_patient_data: dict
) -> None:
    """Creating patient with duplicate MRN returns 409."""
    # Create first
    await client.post("/patients/", json=sample_patient_data)
    # Try to create again
    response = await client.post("/patients/", json=sample_patient_data)
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_get_patient(client: AsyncClient, sample_patient_data: dict) -> None:
    """Create and retrieve a patient."""
    create_resp = await client.post("/patients/", json=sample_patient_data)
    patient_id = create_resp.json()["id"]

    get_resp = await client.get(f"/patients/{patient_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == patient_id


@pytest.mark.asyncio
async def test_get_patient_not_found(client: AsyncClient) -> None:
    """Getting nonexistent patient returns 404."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = await client.get(f"/patients/{fake_id}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_unauthorized_request() -> None:
    """Request without API key returns 401."""
    from httpx import ASGITransport, AsyncClient
    from medscan.api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        response = await c.get("/patients/")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_patients(client: AsyncClient, sample_patient_data: dict) -> None:
    """List patients returns array."""
    await client.post("/patients/", json=sample_patient_data)
    response = await client.get("/patients/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
