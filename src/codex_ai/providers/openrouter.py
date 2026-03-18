"""
codex_ai.providers.openrouter
==============================
OpenRouterProvider — LLM provider backed by OpenRouter.ai.

OpenRouter provides a single API for hundreds of models and is
OpenAI-compatible.
"""

from __future__ import annotations

from .openai import OpenAIProvider

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "google/gemini-2.0-flash-lite-preview-02-05:free"


class OpenRouterProvider(OpenAIProvider):
    """
    LLM provider using OpenRouter.

    Subclasses OpenAIProvider but changes the base_url.

    Args:
        api_key: OpenRouter API key.
        model: Model name. Defaults to a free Gemini model on OpenRouter.
        site_url: Optional URL of your site for OpenRouter ranking.
        site_name: Optional name of your site for OpenRouter ranking.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        site_url: str | None = None,
        site_name: str | None = None,
    ) -> None:
        super().__init__(api_key=api_key, model=model)

        # Re-initialize client with OpenRouter base URL
        from openai import AsyncOpenAI

        headers = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_name:
            headers["X-Title"] = site_name

        self._client = AsyncOpenAI(api_key=api_key, base_url=_OPENROUTER_BASE_URL, default_headers=headers)
