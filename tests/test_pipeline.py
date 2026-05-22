"""Tests for the document processing pipeline."""

import pytest
from pathlib import Path
import tempfile


@pytest.mark.asyncio
async def test_chunker_basic() -> None:
    """Medical chunker produces valid chunks from text."""
    from medscan.pipeline.chunker import MedicalAwareChunker

    chunker = MedicalAwareChunker(chunk_size=100, chunk_overlap=20)
    text = """
MEDICATIONS:
Metformin 1000mg twice daily
Lisinopril 10mg daily

ASSESSMENT:
Type 2 Diabetes Mellitus poorly controlled.
Recommend increasing metformin dose.
"""
    chunks = chunker.chunk_text(text=text, page_number=1)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.content) > 0
        assert chunk.page_number == 1


@pytest.mark.asyncio
async def test_chunker_sections() -> None:
    """Chunker detects medical section headers."""
    from medscan.pipeline.chunker import MedicalAwareChunker

    chunker = MedicalAwareChunker(min_chunk_size=10)
    text = "MEDICATIONS:\nMetformin 1000mg\n\nASSESSMENT:\nDiabetes controlled."
    chunks = chunker.chunk_text(text=text)
    sections = [c.section for c in chunks if c.section]
    assert len(sections) > 0


@pytest.mark.asyncio
async def test_chunker_table() -> None:
    """Table chunking produces correct format."""
    from medscan.pipeline.chunker import MedicalAwareChunker

    chunker = MedicalAwareChunker()
    table = {
        "headers": ["Test", "Value", "Unit", "Flag"],
        "rows": [["HbA1c", "8.2", "%", "H"], ["Glucose", "186", "mg/dL", "H"]],
        "table_type": "lab_results",
    }
    chunk = chunker.chunk_table(table_data=table, page_number=3)
    assert chunk.chunk_type == "table"
    assert "HbA1c" in chunk.content
    assert "8.2" in chunk.content


def test_embedder_single_text() -> None:
    """Embedder produces 384-dim embedding."""
    from medscan.pipeline.embedder import DocumentEmbedder, EMBEDDING_DIM

    embedder = DocumentEmbedder()
    embedding = embedder.embed_single("Patient has Type 2 Diabetes")
    assert len(embedding) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in embedding)


def test_embedder_batch() -> None:
    """Batch embedding returns correct count."""
    from medscan.pipeline.embedder import DocumentEmbedder

    embedder = DocumentEmbedder()
    texts = ["First text", "Second text", "Third text"]
    embeddings = embedder.embed_texts(texts=texts)
    assert len(embeddings) == 3


def test_embedder_cosine_similarity() -> None:
    """Similar texts have higher cosine similarity than dissimilar ones."""
    from medscan.pipeline.embedder import DocumentEmbedder

    embedder = DocumentEmbedder()
    a = embedder.embed_single("Patient has diabetes mellitus")
    b = embedder.embed_single("Diabetes mellitus Type 2 diagnosis")
    c = embedder.embed_single("Weather forecast for tomorrow")

    sim_ab = DocumentEmbedder.cosine_similarity(a, b)
    sim_ac = DocumentEmbedder.cosine_similarity(a, c)

    assert sim_ab > sim_ac, "Similar medical texts should have higher similarity"


@pytest.mark.asyncio
async def test_ingestion_text_file() -> None:
    """Ingestion pipeline processes a text file successfully."""
    from medscan.pipeline.ingestion import DocumentIngestionPipeline

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("MEDICATIONS:\nMetformin 1000mg BID\n\nASSESSMENT:\nDiabetes Type 2.")
        temp_path = Path(f.name)

    try:
        pipeline = DocumentIngestionPipeline()
        result = pipeline.process(file_path=temp_path)

        assert result.success
        assert len(result.raw_text) > 0
        assert len(result.chunks) > 0
    finally:
        temp_path.unlink(missing_ok=True)
