---
title: MedScan AI Backend
emoji: 🏥
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# MedScan AI 🏥

**Multimodal Clinical Document Intelligence Pipeline**

An end-to-end AI system that ingests messy, real-world medical documents (scanned PDFs, lab reports with images, handwritten physician notes, clinical photographs) and produces structured, queryable patient intelligence — with a multi-agent orchestration layer that reasons across text, tables, and images simultaneously.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                            │
│  Scanned PDFs → Lab Reports → Clinical Photos → HL7/FHIR    │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                  PROCESSING PIPELINE                          │
│  OCR (Surya) → Vision (GPT-4o) → NER → Table Extract        │
│  → Smart Chunk → Embed → Structured Extract (Instructor)     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│               ENRICHMENT & INDEXING                           │
│  pgvector (text) + Pinecone (images) + PostgreSQL (metadata) │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│          MULTI-AGENT QUERY LAYER (LangGraph)                 │
│                  SUPERVISOR ROUTER                            │
│       ┌──────────┬──────────┬──────────────┐                 │
│    Text RAG   Table QA   Visual Reasoning  │                 │
│       └──────────┴──────────┴──────────────┘                 │
│              Safety & Hallucination Guard                    │
└──────────────────────────────────────────────────────────────┘
```

## HuggingFace Tasks (12 Integrated)

| Task | Model | Use Case |
|---|---|---|
| Document QA | impira/layoutlm-document-qa | Core clinical query |
| Image-to-Text | Salesforce/blip-image-captioning-large | Clinical photo descriptions |
| Image Classification | google/vit-base-patch16-224 | Document type detection |
| Object Detection | microsoft/table-transformer-detection | Table/stamp detection |
| Token Classification | d4data/biomedical-ner-all | Medical NER |
| Table QA | google/tapas-large-finetuned-wtq | Lab value queries |
| Summarization | Falconsai/medical_summarization | Patient timelines |
| Zero-Shot Classification | facebook/bart-large-mnli | Document triage |
| Visual QA | Salesforce/blip-vqa-base | X-ray interpretation |
| Feature Extraction | sentence-transformers/all-MiniLM-L6-v2 | Embeddings |
| Text Classification | medicalai/ClinicalBERT | Urgency scoring |
| OCR (Image-to-Text) | Surya + TrOCR | Handwriting transcription |

---

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker + Docker Compose

### 1. Clone and setup environment

```bash
git clone <repo>
cd MedScan_AI
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Start infrastructure

```bash
docker compose up -d
# PostgreSQL (pgvector) on :5432
# Redis on :6379
# pgAdmin on :5050
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Initialize database

```bash
uv run alembic upgrade head
# Or let the app auto-create tables on startup
```

### 5. Seed demo data (optional)

```bash
uv run python scripts/seed_demo_data.py
```

### 6. Start the API server

```bash
uv run uvicorn medscan.api.main:app --reload --port 8000
```

### 7. Start the Celery worker (in a separate terminal)

```bash
uv run celery -A medscan.tasks.celery_app.celery_app worker \
  --loglevel=info -Q pipeline --concurrency=2
```

---

## API Usage

### Authentication
All endpoints require `X-API-Key` header (set `API_KEY` in `.env`).

### Upload a document

```bash
curl -X POST http://localhost:8000/documents/upload \
  -H "X-API-Key: medscan-dev-key-change-me" \
  -F "patient_id=<patient-uuid>" \
  -F "doc_type=lab_report" \
  -F "file=@/path/to/lab_report.pdf"
```

### Check processing status

```bash
curl http://localhost:8000/documents/<document-id>/status \
  -H "X-API-Key: medscan-dev-key-change-me"
```

### Query patient records

```bash
curl -X POST http://localhost:8000/query/ \
  -H "X-API-Key: medscan-dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What were the latest HbA1c and glucose levels?",
    "patient_id": "<patient-uuid>",
    "include_sources": true
  }'
```

### Interactive API docs
Open [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Running Tests

```bash
uv run pytest tests/ -v
```

## Running RAGAS Evaluation

```bash
uv run python scripts/run_evaluation.py
```

---

## Project Structure

```
medscan/
├── api/           # FastAPI application
├── agents/        # LangGraph multi-agent system
├── db/            # Database session management
├── evaluation/    # RAGAS evaluation harness
├── hf/            # 12 HuggingFace task wrappers
├── models/        # SQLModel + Pydantic schemas
├── pipeline/      # Document processing pipeline
├── tasks/         # Celery async tasks
└── vector_store/  # pgvector + Pinecone backends
```

---

## Environment Variables

See [`.env.example`](.env.example) for all required configuration.

Key variables:
| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | GPT-4o for vision and extraction |
| `ANTHROPIC_API_KEY` | Claude for reasoning and synthesis |
| `LANGCHAIN_API_KEY` | LangSmith observability |
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis for Celery broker |
| `VECTOR_STORE_BACKEND` | `pgvector` or `pinecone` |

---

## ⚠️ Medical Disclaimer

This system is for **research and development purposes only**. All AI-generated outputs must be reviewed by qualified healthcare professionals before any clinical use. This software is not a medical device and is not approved for clinical decision support.

---

## License

MIT
