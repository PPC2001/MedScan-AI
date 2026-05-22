"""
Base class for all HuggingFace pipeline wrappers.

Provides lazy model loading with caching, device management,
error handling, and a uniform interface.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseHFPipeline(ABC):
    """
    Abstract base for all HuggingFace task wrappers.

    Subclasses declare `task` and `default_model`, then implement `run()`.
    Models are loaded lazily on first call and cached in-process.
    """

    task: str = ""
    default_model: str = ""

    def __init__(self, model: str | None = None, device: str = "cpu") -> None:
        self.model_name = model or self.default_model
        self.device = device
        self._pipeline: Any = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _load(self) -> Any:
        """Load the transformers pipeline (called once)."""
        from transformers import pipeline  # type: ignore[import]

        logger.info(
            "Loading HF pipeline task=%s model=%s device=%s",
            self.task,
            self.model_name,
            self.device,
        )
        return pipeline(
            task=self.task,
            model=self.model_name,
            device=self.device if self.device != "cpu" else -1,
        )

    @property
    def pipe(self) -> Any:
        """Return (cached) pipeline instance."""
        if self._pipeline is None:
            self._pipeline = self._load()
        return self._pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute inference and return structured result."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r}, device={self.device!r})"
