"""
LLM Provider Factory — returns the right LangChain chat model based on
the configured provider (Grok → OpenAI → Anthropic) with graceful fallback.

All agent nodes import `get_llm()` from here instead of hard-coding a provider.

Grok note:
  xAI's Grok exposes an OpenAI-compatible /v1 endpoint.
  We use `langchain_openai.ChatOpenAI` pointed at https://api.x.ai/v1
  with the XAI_API_KEY — no extra SDK needed.
"""

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from medscan.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def get_llm(
    *,
    max_tokens: int = 1500,
    temperature: float = 0.0,
    streaming: bool = False,
    provider: str | None = None,
) -> BaseChatModel:
    """
    Return a configured LangChain chat model for the active provider.

    Provider resolution order (unless `provider` is explicitly passed):
      1. Grok (xAI)  — XAI_API_KEY is set
      2. OpenAI      — OPENAI_API_KEY is set
      3. Anthropic   — ANTHROPIC_API_KEY is set

    Args:
        max_tokens: Maximum response tokens.
        temperature: Sampling temperature (0 = deterministic).
        streaming: Enable streaming responses.
        provider: Force a specific provider ('grok' | 'openai' | 'anthropic').

    Returns:
        A LangChain-compatible chat model instance.

    Raises:
        RuntimeError: If no LLM provider is configured.
    """
    resolved = provider or settings.active_llm_provider

    if resolved == "grok":
        return _make_grok(max_tokens=max_tokens, temperature=temperature, streaming=streaming)
    elif resolved == "openai":
        return _make_openai(max_tokens=max_tokens, temperature=temperature, streaming=streaming)
    elif resolved == "anthropic":
        return _make_anthropic(max_tokens=max_tokens, temperature=temperature, streaming=streaming)
    else:
        raise RuntimeError(
            "No LLM provider configured. Set one of: XAI_API_KEY (Grok), "
            "OPENAI_API_KEY, or ANTHROPIC_API_KEY in your .env file."
        )


def get_llm_or_none(**kwargs: Any) -> BaseChatModel | None:
    """
    Like get_llm() but returns None instead of raising if no provider is set.
    Use this in pipeline stages where LLM is optional.
    """
    try:
        return get_llm(**kwargs)
    except RuntimeError:
        logger.warning("No LLM provider configured — skipping LLM-dependent step.")
        return None


# ---------------------------------------------------------------------------
# Provider constructors
# ---------------------------------------------------------------------------

def _make_grok(max_tokens: int, temperature: float, streaming: bool) -> BaseChatModel:
    """
    Grok via xAI's OpenAI-compatible API.
    Uses langchain_openai.ChatOpenAI with a custom base_url and api_key.
    """
    from langchain_openai import ChatOpenAI  # type: ignore[import]

    logger.debug("Using Grok (%s) via xAI endpoint", settings.grok_model)
    return ChatOpenAI(
        model=settings.grok_model,
        api_key=settings.xai_api_key,          # type: ignore[arg-type]
        base_url=settings.grok_base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        streaming=streaming,
    )


def _make_openai(max_tokens: int, temperature: float, streaming: bool) -> BaseChatModel:
    """Standard OpenAI GPT-4o."""
    from langchain_openai import ChatOpenAI  # type: ignore[import]

    logger.debug("Using OpenAI (%s)", settings.openai_model)
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,        # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
        streaming=streaming,
    )


def _make_anthropic(max_tokens: int, temperature: float, streaming: bool) -> BaseChatModel:
    """Anthropic Claude."""
    from langchain_anthropic import ChatAnthropic  # type: ignore[import]

    logger.debug("Using Anthropic (%s)", settings.anthropic_model)
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,     # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
        streaming=streaming,
    )


def describe_active_provider() -> str:
    """Human-readable description of the active LLM configuration."""
    provider = settings.active_llm_provider
    if provider == "grok":
        return f"Grok ({settings.grok_model}) via xAI"
    elif provider == "openai":
        return f"OpenAI ({settings.openai_model})"
    elif provider == "anthropic":
        return f"Anthropic ({settings.anthropic_model})"
    else:
        return "No LLM — HuggingFace local models only"
