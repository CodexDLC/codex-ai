"""
codex_ai.providers.gemini
==========================
GeminiProvider — LLM provider backed by Google Gemini (google-genai).

Requires: ``pip install codex-ai[gemini]``
"""

from __future__ import annotations

from typing import Any, cast

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as e:
    if e.name in ("google", "google.genai"):
        raise ImportError(
            "GeminiProvider requires the 'google-genai' package. Install it with: pip install codex-ai[gemini]"
        ) from e
    raise

from codex_ai.core.exceptions import LLMProviderError
from codex_ai.core.protocol import PromptResult

_DEFAULT_MODEL = "gemini-2.5-flash-lite"


class GeminiProvider:
    """
    LLM provider using Google Gemini via the google-genai SDK.

    Implements LLMProviderProtocol.

    Args:
        api_key: Google AI API key.
        model: Gemini model name. Defaults to ``"gemini-2.5-flash-lite"``.

    Example:
        ```python
        provider = GeminiProvider(api_key="AIza...")
        result = PromptResult(
            messages=[LLMMessage(role="user", content="Hello!")],
            system="You are a helpful assistant.",
        )
        text = await provider.answer(result)
        ```
    """

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(api_version="v1alpha"),
        )
        self._model = model

    async def answer(self, prompt: PromptResult, **kw: Any) -> str:
        """
        Send prompt to Gemini and return response text.

        Args:
            prompt: PromptResult with messages list (Gemini contents format)
                    and optional system instruction string.
            **kw: Extra kwargs forwarded to ``generate_content``
                  (e.g., ``temperature`` via ``GenerateContentConfig``).

        Returns:
            Response text string, or a fallback error message on failure.
        """
        raw_messages: list[dict[str, Any]] = []
        for msg in prompt.messages:
            role = "model" if msg.role == "assistant" else "user" if msg.role == "user" else msg.role
            raw_messages.append({"role": role, "parts": [{"text": msg.content}]})

        # Приводим к Any, так как типы в google.genai SDK часто некорректны (Union перегружен)
        contents = cast(Any, raw_messages)

        model = prompt.model or kw.get("model") or self._model
        temperature = prompt.temperature or kw.get("temperature")
        max_tokens = prompt.max_tokens or kw.get("max_tokens")

        # Удаляем из kw те параметры, которые мы уже вытащили
        runtime_kw = kw.copy()
        runtime_kw.pop("model", None)
        runtime_kw.pop("temperature", None)
        runtime_kw.pop("max_tokens", None)

        config = genai_types.GenerateContentConfig(
            system_instruction=prompt.system or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
            **runtime_kw,
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response.text or ""
        except Exception as exc:
            raise LLMProviderError(f"Gemini error: {exc}") from exc
