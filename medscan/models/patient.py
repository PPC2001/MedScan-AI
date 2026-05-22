"""Patient model — core identity record."""

import uuid
from datetime import date, datetime

from sqlmodel import Field, SQLModel


class PatientBase(SQLModel):
    name: str = Field(index=True, description="Full patient name")
    date_of_birth: date
    gender: str = Field(max_length=20)
    mrn: str = Field(
        unique=True, index=True, description="Medical Record Number"
    )
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    blood_type: str | None = Field(default=None, max_length=5)
    allergies: str | None = None  # comma-separated list
    notes: str | None = None


class Patient(PatientBase, table=True):
    __tablename__ = "patients"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        nullable=False,
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PatientCreate(PatientBase):
    pass


class PatientUpdate(SQLModel):
    name: str | None = None
    date_of_birth: date | None = None
    gender: str | None = None
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    blood_type: str | None = None
    allergies: str | None = None
    notes: str | None = None


class PatientRead(PatientBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
