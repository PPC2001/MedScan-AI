"""Patients CRUD router."""

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from medscan.api.dependencies import AuthDep, SessionDep
from medscan.models.patient import Patient, PatientCreate, PatientRead, PatientUpdate

router = APIRouter()


@router.post("/", response_model=PatientRead, status_code=status.HTTP_201_CREATED)
async def create_patient(
    patient_in: PatientCreate, session: SessionDep, _auth: AuthDep
) -> PatientRead:
    """Create a new patient record."""
    # Check for duplicate MRN
    existing = await session.execute(
        select(Patient).where(Patient.mrn == patient_in.mrn)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Patient with MRN '{patient_in.mrn}' already exists",
        )

    patient = Patient(**patient_in.model_dump())
    session.add(patient)
    await session.commit()
    await session.refresh(patient)
    return patient


@router.get("/{patient_id}", response_model=PatientRead)
async def get_patient(
    patient_id: uuid.UUID, session: SessionDep, _auth: AuthDep
) -> PatientRead:
    """Get a patient by UUID."""
    result = await session.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.get("/mrn/{mrn}", response_model=PatientRead)
async def get_patient_by_mrn(
    mrn: str, session: SessionDep, _auth: AuthDep
) -> PatientRead:
    """Get a patient by Medical Record Number."""
    result = await session.execute(select(Patient).where(Patient.mrn == mrn))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.get("/", response_model=list[PatientRead])
async def list_patients(
    session: SessionDep,
    _auth: AuthDep,
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
) -> list[PatientRead]:
    """List patients with optional name search."""
    query = select(Patient).offset(offset).limit(limit)
    if search:
        query = query.where(Patient.name.ilike(f"%{search}%"))
    result = await session.execute(query)
    return list(result.scalars().all())


@router.patch("/{patient_id}", response_model=PatientRead)
async def update_patient(
    patient_id: uuid.UUID,
    updates: PatientUpdate,
    session: SessionDep,
    _auth: AuthDep,
) -> PatientRead:
    """Update patient fields."""
    result = await session.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    for field, value in updates.model_dump(exclude_none=True).items():
        setattr(patient, field, value)
    patient.updated_at = datetime.utcnow()

    await session.commit()
    await session.refresh(patient)
    return patient


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(
    patient_id: uuid.UUID, session: SessionDep, _auth: AuthDep
) -> None:
    """Delete a patient and all associated records."""
    result = await session.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    await session.delete(patient)
    await session.commit()
