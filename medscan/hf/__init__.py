"""
HF package — 12 HuggingFace task wrappers for MedScan AI.

All pipelines use lazy model loading (loaded on first call, cached thereafter).
Device can be configured via settings.hf_device (cpu | cuda | mps).
"""

from medscan.config import get_settings

_settings = get_settings()
_device = _settings.hf_device


def get_doc_qa():
    from medscan.hf.doc_qa import DocumentQAPipeline
    return DocumentQAPipeline(device=_device)

def get_image_to_text():
    from medscan.hf.image_to_text import ImageToTextPipeline
    return ImageToTextPipeline(device=_device)

def get_image_classification():
    from medscan.hf.image_classification import ImageClassificationPipeline
    return ImageClassificationPipeline(device=_device)

def get_object_detection():
    from medscan.hf.object_detection import ObjectDetectionPipeline
    return ObjectDetectionPipeline(device=_device)

def get_token_classification():
    from medscan.hf.token_classification import TokenClassificationPipeline
    return TokenClassificationPipeline(device=_device)

def get_table_qa():
    from medscan.hf.table_qa import TableQAPipeline
    return TableQAPipeline(device=_device)

def get_summarization():
    from medscan.hf.summarization import SummarizationPipeline
    return SummarizationPipeline(device=_device)

def get_zero_shot():
    from medscan.hf.zero_shot import ZeroShotClassificationPipeline
    return ZeroShotClassificationPipeline(device=_device)

def get_vqa():
    from medscan.hf.vqa import VQAPipeline
    return VQAPipeline(device=_device)

def get_feature_extraction():
    from medscan.hf.feature_extraction import FeatureExtractionPipeline
    return FeatureExtractionPipeline(device=_device)

def get_text_classification():
    from medscan.hf.text_classification import TextClassificationPipeline
    return TextClassificationPipeline(device=_device)


__all__ = [
    "get_doc_qa",
    "get_image_to_text",
    "get_image_classification",
    "get_object_detection",
    "get_token_classification",
    "get_table_qa",
    "get_summarization",
    "get_zero_shot",
    "get_vqa",
    "get_feature_extraction",
    "get_text_classification",
]
