"""codex-ai: Agnostic LLM abstraction layer (OpenAI, Gemini, Anthropic)."""

from codex_ai.core import (
    LLMDispatcher,
    LLMMessage,
    LLMProviderError,
    LLMProviderProtocol,
    LLMRouter,
    PromptBuilder,
    PromptResult,
    SyncLLMDispatcher,
)

__all__ = [
    # Core
    "LLMDispatcher",
    "LLMMessage",
    "LLMProviderError",
    "LLMProviderProtocol",
    "LLMRouter",
    "PromptBuilder",
    "PromptResult",
    "SyncLLMDispatcher",
    # Providers (lazy)
    "OpenAIProvider",
    "GeminiProvider",
    "AnthropicProvider",
    "OpenRouterProvider",
    "MultiLLMProvider",
]


def __getattr__(name: str) -> object:
    """Lazy-load provider classes to avoid mandatory SDK dependencies."""
    if name in ("OpenAIProvider", "GeminiProvider", "AnthropicProvider", "OpenRouterProvider", "MultiLLMProvider"):
        from codex_ai import providers

        return getattr(providers, name)
    raise AttributeError(f"module 'codex_ai' has no attribute {name!r}")
