"""Test the document ingestion pipeline directly and synchronously."""
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)

from medscan.pipeline.ingestion import DocumentIngestionPipeline

def main():
    print("Initializing pipeline...")
    pipeline = DocumentIngestionPipeline()
    
    file_path = Path("uploads/19981332-c0d4-4d06-87b6-239f597fb707.txt")
    print(f"File path exists: {file_path.exists()} ({file_path})")
    
    print("Running ingestion process...")
    result = pipeline.process(
        file_path=file_path,
        document_id="19981332-c0d4-4d06-87b6-239f597fb707"
    )
    
    print("Ingestion result:")
    print(f"  Success: {result.success}")
    print(f"  Error: {result.error}")
    print(f"  Urgency: {result.urgency_label} (Score: {result.urgency_score})")
    print(f"  Doc type: {result.doc_type_detected} (Confidence: {result.doc_type_confidence})")
    print(f"  Entities: {result.named_entities}")
    print(f"  Chunks count: {len(result.chunks)}")
    if result.chunks:
        print(f"  First chunk metadata keys: {list(result.chunks[0].metadata.keys())}")

if __name__ == "__main__":
    main()
