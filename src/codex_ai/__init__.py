"""codex-ai: Gemini-first and OpenAI provider helpers for Codex."""

from codex_ai.core import (
    ImageGenerationProvider,
    JsonGenerationProvider,
    LLMDispatcher,
    LLMMessage,
    LLMProviderError,
    LLMProviderProtocol,
    LLMRouter,
    PromptBuilder,
    PromptResult,
    SyncLLMDispatcher,
    TextGenerationProvider,
)

__all__ = [
    # Core
    "LLMDispatcher",
    "ImageGenerationProvider",
    "JsonGenerationProvider",
    "LLMMessage",
    "LLMProviderError",
    "LLMProviderProtocol",
    "LLMRouter",
    "PromptBuilder",
    "PromptResult",
    "SyncLLMDispatcher",
    "TextGenerationProvider",
    # Providers (lazy)
    "OpenAIProvider",
    "GeminiProvider",
]


def __getattr__(name: str) -> object:
    """Lazy-load provider classes to avoid mandatory SDK dependencies."""
    if name in ("OpenAIProvider", "GeminiProvider"):
        from codex_ai import providers

        return getattr(providers, name)
    raise AttributeError(f"module 'codex_ai' has no attribute {name!r}")
