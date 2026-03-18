"""
codex_ai.providers
==================
LLM provider implementations (OpenAI, Gemini, Anthropic, OpenRouter, Multi).

Providers are lazy-loaded to avoid mandatory dependency on all SDK packages.
"""

__all__ = [
    "OpenAIProvider",
    "GeminiProvider",
    "AnthropicProvider",
    "OpenRouterProvider",
    "MultiLLMProvider",
]


def __getattr__(name: str) -> object:
    if name == "OpenAIProvider":
        from .openai import OpenAIProvider

        return OpenAIProvider
    if name == "GeminiProvider":
        from .gemini import GeminiProvider

        return GeminiProvider
    if name == "MultiLLMProvider":
        from .multi import MultiLLMProvider

        return MultiLLMProvider
    if name == "AnthropicProvider":
        from .anthropic_ import AnthropicProvider

        return AnthropicProvider
    if name == "OpenRouterProvider":
        from .openrouter import OpenRouterProvider

        return OpenRouterProvider
    raise AttributeError(f"module 'codex_ai.providers' has no attribute {name!r}")
