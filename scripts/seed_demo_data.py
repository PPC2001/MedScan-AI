"""
Seed Demo Data — generates synthetic (HIPAA-safe) medical documents for testing.

Creates:
- 3 sample patients
- 5 synthetic lab reports (PDF-like text)
- 2 physician notes
- 1 prescription

Run: uv run python scripts/seed_demo_data.py
"""

import asyncio
import uuid
from datetime import date, datetime
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


SAMPLE_LAB_REPORT = """
CLINICAL LABORATORY REPORT
Patient: John Demo Doe
MRN: DEMO-001
Date of Birth: 1965-04-12
Physician: Dr. Sarah Smith
Collection Date: 2024-11-15
Report Date: 2024-11-15

COMPLETE BLOOD COUNT (CBC)
─────────────────────────────────────────
Test                    Result  Unit    Reference Range     Flag
─────────────────────────────────────────
White Blood Cells       11.2    K/µL    4.5-11.0           H
Red Blood Cells         4.1     M/µL    4.5-5.5            L
Hemoglobin              12.1    g/dL    13.5-17.5          L
Hematocrit              36.5    %       41-53              L
MCV                     89      fL      80-100
MCH                     29.5    pg      27-33
Platelets               220     K/µL    150-400

COMPREHENSIVE METABOLIC PANEL
─────────────────────────────────────────
Glucose                 186     mg/dL   70-100             H
BUN                     22      mg/dL   7-25
Creatinine              1.2     mg/dL   0.6-1.2
eGFR                    68      mL/min  >60
Sodium                  138     mEq/L   136-145
Potassium               4.1     mEq/L   3.5-5.0
Chloride                102     mEq/L   98-107
CO2                     24      mEq/L   22-29
Calcium                 9.4     mg/dL   8.5-10.2
ALT                     45      U/L     7-56
AST                     38      U/L     10-40              H
Albumin                 3.8     g/dL    3.5-5.0

HEMOGLOBIN A1C
─────────────────────────────────────────
HbA1c                   8.2     %       <5.7               H  ← CRITICAL
Estimated Average Glucose: 189 mg/dL

LIPID PANEL
─────────────────────────────────────────
Total Cholesterol       228     mg/dL   <200               H
HDL Cholesterol         38      mg/dL   >40                L
LDL Cholesterol         155     mg/dL   <100               H
Triglycerides           175     mg/dL   <150               H
Non-HDL Cholesterol     190     mg/dL   <130               H

COMMENTS:
Elevated HbA1c consistent with poorly controlled Type 2 Diabetes Mellitus.
Dyslipidemia pattern noted. Anemia of chronic disease pattern.
Recommend: Diabetes management review, statin therapy consideration.

Verified by: Lab Director, MD, FCAP
"""

SAMPLE_PHYSICIAN_NOTE = """
PROGRESS NOTE
─────────────────────────────────────────
Patient: Jane Demo Smith
MRN: DEMO-002
DOB: 1978-08-23
Date: 2024-11-20
Provider: Dr. Michael Chen, MD
Specialty: Internal Medicine

CHIEF COMPLAINT:
Follow-up for Type 2 Diabetes Mellitus and hypertension.

HISTORY OF PRESENT ILLNESS:
Mrs. Smith is a 46-year-old female presenting for 3-month follow-up.
She reports overall fatigue, polyuria, and polydipsia over the past 6 weeks.
Compliance with metformin is reported as good. She denies chest pain,
shortness of breath, or vision changes. Home BP readings averaging 145/90.

PAST MEDICAL HISTORY:
1. Type 2 Diabetes Mellitus (diagnosed 2019)
2. Essential Hypertension (diagnosed 2020)
3. Hyperlipidemia
4. Obesity (BMI 34.2)

MEDICATIONS:
1. Metformin 1000mg PO BID
2. Lisinopril 10mg PO daily
3. Atorvastatin 40mg PO nightly
4. Aspirin 81mg PO daily

ALLERGIES:
Penicillin - rash
Sulfa drugs - anaphylaxis

VITAL SIGNS:
Blood Pressure: 148/92 mmHg
Heart Rate: 82 bpm
Temperature: 98.4°F
SpO2: 98% on room air
Weight: 189 lbs
Height: 5'4"
BMI: 32.4

PHYSICAL EXAMINATION:
General: Alert, oriented, obese female in no acute distress
HEENT: Normocephalic, atraumatic
Cardiovascular: Regular rate and rhythm, no murmurs
Respiratory: Clear to auscultation bilaterally
Abdomen: Soft, non-tender, obese
Extremities: Mild bilateral pedal edema 1+, no ulcers
Neurological: Sensation intact bilateral feet

ASSESSMENT AND PLAN:
1. Type 2 Diabetes Mellitus - poorly controlled (HbA1c 8.2% last visit)
   - Increase metformin to 2000mg daily if tolerated
   - Add semaglutide 0.5mg subcutaneous weekly (Ozempic)
   - Referral to diabetes education program
   - Repeat HbA1c in 3 months

2. Hypertension - suboptimally controlled
   - Increase lisinopril to 20mg daily
   - Add amlodipine 5mg daily
   - DASH diet counseling

3. Hyperlipidemia
   - Continue atorvastatin 40mg
   - Repeat lipid panel in 6 months

4. Obesity
   - Weight loss counseling
   - Refer to nutrition services

Follow-up in 3 months or sooner if symptoms worsen.

Electronically signed: Dr. Michael Chen, MD
Date: 2024-11-20 14:32 EST
"""

SAMPLE_PRESCRIPTION = """
PRESCRIPTION
─────────────────────────────────────────
Prescriber: Dr. Michael Chen, MD
NPI: 1234567890
Address: 123 Medical Center Dr, Boston, MA 02115
Phone: (617) 555-0100
DEA: BC1234567

Patient Name: Jane Demo Smith
DOB: 08/23/1978
Date: 11/20/2024

Rx #1
Drug: Semaglutide (Ozempic) 0.5mg/dose
Sig: Inject 0.5mg subcutaneously once weekly
Dispense: 1 pen (2mg/1.5mL)
Refills: 5
Notes: Titrate to 1mg after 4 weeks if tolerated

Rx #2
Drug: Amlodipine 5mg tablets
Sig: Take 1 tablet by mouth daily
Dispense: 90 tablets
Refills: 3

Rx #3
Drug: Lisinopril 20mg tablets
Sig: Take 1 tablet by mouth daily
Dispense: 90 tablets
Refills: 3
Notes: Monitor potassium and creatinine in 2 weeks

Signature: ________________________________
Dr. Michael Chen, MD
"""


async def seed_database() -> None:
    """Create sample patients and documents in the database."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlmodel import SQLModel

    from medscan.config import get_settings
    from medscan.models import Patient, Document, DocumentStatus, DocumentType

    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    print("🏥 MedScan AI — Seeding demo data...")
    print(f"   Database: {settings.database_url.split('@')[-1]}")

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    # Create sample documents as text files
    upload_dir = settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    async with AsyncSessionLocal() as session:
        # Patient 1 — John Doe (diabetic)
        p1_id = uuid.uuid4()
        p1 = Patient(
            id=p1_id,
            name="John Demo Doe",
            date_of_birth=date(1965, 4, 12),
            gender="Male",
            mrn="DEMO-001",
            blood_type="O+",
            allergies="None known",
        )
        session.add(p1)

        # Patient 2 — Jane Smith
        p2_id = uuid.uuid4()
        p2 = Patient(
            id=p2_id,
            name="Jane Demo Smith",
            date_of_birth=date(1978, 8, 23),
            gender="Female",
            mrn="DEMO-002",
            blood_type="A+",
            allergies="Penicillin (rash), Sulfa drugs (anaphylaxis)",
        )
        session.add(p2)

        await session.commit()
        print(f"   ✅ Created patient: {p1.name} (MRN: {p1.mrn})")
        print(f"   ✅ Created patient: {p2.name} (MRN: {p2.mrn})")

        # Save sample documents to disk
        docs_to_create = [
            (p1_id, "lab_report_john_doe.txt", SAMPLE_LAB_REPORT, DocumentType.LAB_REPORT),
            (p2_id, "physician_note_jane_smith.txt", SAMPLE_PHYSICIAN_NOTE, DocumentType.PHYSICIAN_NOTE),
            (p2_id, "prescription_jane_smith.txt", SAMPLE_PRESCRIPTION, DocumentType.PRESCRIPTION),
        ]

        for patient_id, filename, content, doc_type in docs_to_create:
            doc_id = uuid.uuid4()
            file_path = upload_dir / f"{doc_id}.txt"
            file_path.write_text(content, encoding="utf-8")

            doc = Document(
                id=doc_id,
                patient_id=patient_id,
                filename=f"{doc_id}.txt",
                original_filename=filename,
                file_path=str(file_path),
                file_size_bytes=len(content.encode()),
                mime_type="text/plain",
                doc_type=doc_type,
                status=DocumentStatus.PENDING,
            )
            session.add(doc)
            print(f"   📄 Created document: {filename}")

        await session.commit()
        print("\n✅ Demo data seeded successfully!")
        print("\nNext steps:")
        print("  1. Start the API: uv run uvicorn medscan.api.main:app --reload")
        print("  2. Start the worker: uv run celery -A medscan.tasks.celery_app.celery_app worker")
        print("  3. Trigger processing via POST /documents/upload")
        print("  4. Open docs: http://localhost:8000/docs")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed_database())
