"""
codex_ai.providers
==================
Provider adapters. Gemini is the primary API; OpenAI is text-only.

Providers are lazy-loaded to avoid mandatory dependency on all SDK packages.
"""

__all__ = [
    "OpenAIProvider",
    "GeminiProvider",
]


def __getattr__(name: str) -> object:
    if name == "OpenAIProvider":
        from .openai import OpenAIProvider

        return OpenAIProvider
    if name == "GeminiProvider":
        from .gemini import GeminiProvider

        return GeminiProvider
    raise AttributeError(f"module 'codex_ai.providers' has no attribute {name!r}")
