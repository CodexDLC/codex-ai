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
_DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"


class GeminiProvider:
    """
    LLM provider using Google Gemini via the google-genai SDK.

    Implements LLMProviderProtocol.

    Args:
        api_key: Google AI API key.
        model: Gemini text model name. Defaults to ``"gemini-2.5-flash-lite"``.
        image_model: Gemini image model name. Defaults to ``"gemini-3.1-flash-image-preview"``.

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

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL, image_model: str = _DEFAULT_IMAGE_MODEL) -> None:
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(api_version="v1alpha"),
        )
        self._model = model
        self._image_model = image_model

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

        # Cast to Any — google.genai SDK types are often incorrect (Union is overloaded)
        contents = cast(Any, raw_messages)

        model = prompt.model or kw.get("model") or self._model
        temperature = prompt.temperature or kw.get("temperature")
        max_tokens = prompt.max_tokens or kw.get("max_tokens")

        # Remove params already extracted from kw to avoid duplicate kwargs
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

    async def generate_image_bytes(
        self,
        prompt: str,
        *,
        model: str | None = None,
        response_mime_type: str = "image/webp",
        **kwargs: Any,
    ) -> tuple[bytes, str]:
        """
        Generate an image with Gemini and return raw bytes plus content type.

        ``response_mime_type`` is treated as the requested MIME type. The actual
        MIME type returned by Gemini wins when present.
        """
        selected_model = model or self._image_model
        requested_mime = response_mime_type

        runtime_kw = kwargs.copy()
        runtime_kw.pop("model", None)
        runtime_kw.pop("response_mime_type", None)

        config = genai_types.GenerateContentConfig(
            response_modalities=[genai_types.Modality.IMAGE],
            response_mime_type=requested_mime,
            **runtime_kw,
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=selected_model,
                contents=prompt,
                config=config,
            )
            image = self._extract_first_inline_image(response, fallback_mime=requested_mime)
            if image is not None:
                return image

            detail = self._describe_non_image_response(response)
            raise LLMProviderError(f"Gemini image generation did not return image data{detail}")
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"Gemini image generation error: {exc}") from exc

    @staticmethod
    def _extract_first_inline_image(response: Any, *, fallback_mime: str) -> tuple[bytes, str] | None:
        for part in GeminiProvider._iter_response_parts(response):
            inline_data = getattr(part, "inline_data", None)
            data = getattr(inline_data, "data", None)
            if data is None:
                continue

            mime_type = getattr(inline_data, "mime_type", None) or fallback_mime
            return bytes(data), mime_type

        return None

    @staticmethod
    def _describe_non_image_response(response: Any) -> str:
        details: list[str] = []

        text = getattr(response, "text", None)
        if text:
            details.append(str(text))

        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback:
            details.append(f"prompt_feedback={prompt_feedback}")

        for candidate in getattr(response, "candidates", None) or []:
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason:
                details.append(f"finish_reason={finish_reason}")

            safety_ratings = getattr(candidate, "safety_ratings", None)
            if safety_ratings:
                details.append(f"safety_ratings={safety_ratings}")

        if not details:
            return ""

        return f": {'; '.join(details)}"

    @staticmethod
    def _iter_response_parts(response: Any) -> list[Any]:
        parts = getattr(response, "parts", None)
        if parts:
            return list(parts)

        collected: list[Any] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            candidate_parts = getattr(content, "parts", None)
            if candidate_parts:
                collected.extend(candidate_parts)

        return collected
