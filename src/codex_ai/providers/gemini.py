"""
codex_ai.providers.gemini
==========================
GeminiProvider — LLM provider backed by Google Gemini (google-genai).

Requires: ``pip install codex-ai[gemini]``
"""

from __future__ import annotations

import base64
import json
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

from pydantic import BaseModel, ValidationError

from codex_ai.core.exceptions import LLMProviderError
from codex_ai.core.protocol import PromptResult

_DEFAULT_MODEL = "gemini-2.5-flash-lite"
_DEFAULT_IMAGE_MODEL = "gemini-2.5-flash-image"
_DEFAULT_IMAGEN_MODEL = "imagen-3.0-generate-002"


class GeminiProvider:
    """
    LLM provider using Google Gemini via the google-genai SDK.

    Implements LLMProviderProtocol.

    Args:
        api_key: Google AI API key.
        model: Gemini text model name. Defaults to ``"gemini-2.5-flash-lite"``.
        image_model: Gemini image model name. Defaults to ``"gemini-2.5-flash-image"``.

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

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        image_model: str = _DEFAULT_IMAGE_MODEL,
        imagen_model: str = _DEFAULT_IMAGEN_MODEL,
    ) -> None:
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(api_version="v1alpha"),
        )
        self._model = model
        self._image_model = image_model
        self._imagen_model = imagen_model

    async def answer(self, prompt: PromptResult, **kw: Any) -> str:
        """
        Compatibility wrapper for the legacy text pipeline.
        """
        return await self.generate_text(prompt, **kw)

    async def generate_text(
        self,
        prompt: PromptResult | str,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Send prompt to Gemini and return response text.

        Args:
            prompt: PromptResult with messages list or a raw prompt string.
            model: Optional model override for this request.
            **kwargs: Extra kwargs forwarded to ``generate_content``
                  (e.g., ``temperature`` via ``GenerateContentConfig``).

        Returns:
            Response text string, or a fallback error message on failure.
        """
        contents = self._build_text_contents(prompt)
        selected_model = self._select_text_model(prompt, model=model, runtime_kwargs=kwargs)
        temperature = self._get_prompt_attr(prompt, "temperature") or kwargs.get("temperature")
        max_tokens = self._get_prompt_attr(prompt, "max_tokens") or kwargs.get("max_tokens")

        runtime_kw = kwargs.copy()
        self._strip_common_text_kwargs(runtime_kw)

        config = genai_types.GenerateContentConfig(
            system_instruction=self._get_prompt_attr(prompt, "system") or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
            **runtime_kw,
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=selected_model,
                contents=contents,
                config=config,
            )
            return response.text or ""
        except Exception as exc:
            raise LLMProviderError(f"Gemini error: {exc}") from exc

    async def generate_json(
        self,
        prompt: PromptResult | str,
        *,
        schema: type[BaseModel] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate JSON with Gemini-native response configuration and local validation.
        """
        contents = self._build_text_contents(prompt)
        selected_model = self._select_text_model(prompt, model=model, runtime_kwargs=kwargs)
        temperature = self._get_prompt_attr(prompt, "temperature") or kwargs.get("temperature")
        max_tokens = self._get_prompt_attr(prompt, "max_tokens") or kwargs.get("max_tokens")

        runtime_kw = kwargs.copy()
        self._strip_common_text_kwargs(runtime_kw)
        runtime_kw.pop("response_mime_type", None)
        runtime_kw.pop("response_schema", None)
        runtime_kw.pop("schema", None)

        config_kwargs: dict[str, Any] = {
            "system_instruction": self._get_prompt_attr(prompt, "system") or None,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
            **runtime_kw,
        }
        if schema is not None:
            config_kwargs["response_schema"] = schema

        config = genai_types.GenerateContentConfig(**config_kwargs)

        try:
            response = await self._client.aio.models.generate_content(
                model=selected_model,
                contents=contents,
                config=config,
            )
            text = response.text or ""
            if not text.strip():
                raise LLMProviderError("Gemini JSON generation returned an empty response")

            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMProviderError(f"Gemini JSON generation returned invalid JSON: {exc}") from exc

            if schema is None:
                return data

            try:
                return schema.model_validate(data)
            except ValidationError as exc:
                raise LLMProviderError(f"Gemini JSON generation failed schema validation: {exc}") from exc
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"Gemini JSON generation error: {exc}") from exc

    async def generate_image_bytes(
        self,
        prompt: str,
        *,
        model: str | None = None,
        response_mime_type: str = "image/webp",
        **kwargs: Any,
    ) -> tuple[bytes, str]:
        """
        Generate an image with Gemini image models and return bytes plus content type.

        This uses the Gemini ``generate_content`` image path with
        ``GenerateContentConfig.response_modalities=[IMAGE]``. Use it for Gemini
        image preview / flash-image / nano-banana style models.

        ``response_mime_type`` is a preferred/fallback MIME type only. Gemini's
        ``GenerateContentConfig.response_mime_type`` accepts text response MIME
        values, so image MIME values are not sent there. The actual MIME type
        returned in ``inline_data.mime_type`` wins when present.
        """
        selected_model = model or self._image_model
        requested_mime = response_mime_type

        runtime_kw = kwargs.copy()
        runtime_kw.pop("model", None)
        runtime_kw.pop("response_mime_type", None)

        config = genai_types.GenerateContentConfig(
            response_modalities=[genai_types.Modality.IMAGE],
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

    async def generate_imagen_bytes(
        self,
        prompt: str,
        *,
        model: str | None = None,
        response_mime_type: str = "image/jpeg",
        **kwargs: Any,
    ) -> tuple[bytes, str]:
        """
        Generate an image with Imagen models and return bytes plus content type.

        This uses the Imagen ``generate_images`` SDK path and passes
        ``response_mime_type`` as ``GenerateImagesConfig.output_mime_type``.
        Use it for ``imagen-*`` models, not Gemini flash-image / nano-banana
        models.
        """
        selected_model = model or self._imagen_model
        requested_mime = response_mime_type

        runtime_kw = kwargs.copy()
        runtime_kw.pop("model", None)
        runtime_kw.pop("response_mime_type", None)
        runtime_kw.pop("output_mime_type", None)

        config = genai_types.GenerateImagesConfig(
            output_mime_type=requested_mime,
            **runtime_kw,
        )

        try:
            response = await self._client.aio.models.generate_images(
                model=selected_model,
                prompt=prompt,
                config=config,
            )
            image = self._extract_first_imagen_image(response, fallback_mime=requested_mime)
            if image is not None:
                return image

            detail = self._describe_imagen_non_image_response(response)
            raise LLMProviderError(f"Gemini Imagen generation did not return image data{detail}")
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"Gemini Imagen generation error: {exc}") from exc

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
    def _extract_first_imagen_image(response: Any, *, fallback_mime: str) -> tuple[bytes, str] | None:
        for generated_image in getattr(response, "generated_images", None) or []:
            image = getattr(generated_image, "image", None)
            if image is None:
                continue

            data = getattr(image, "image_bytes", None)
            if data is None:
                data = getattr(image, "data", None)
            if data is None:
                continue

            image_bytes = base64.b64decode(data) if isinstance(data, str) else bytes(data)

            mime_type = getattr(image, "mime_type", None) or getattr(generated_image, "mime_type", None)
            mime_type = mime_type or fallback_mime
            return image_bytes, mime_type

        return None

    @staticmethod
    def _describe_imagen_non_image_response(response: Any) -> str:
        details: list[str] = []

        for generated_image in getattr(response, "generated_images", None) or []:
            rai_reason = getattr(generated_image, "rai_filtered_reason", None)
            if rai_reason:
                details.append(f"rai_filtered_reason={rai_reason}")

            safety_attributes = getattr(generated_image, "safety_attributes", None)
            if safety_attributes:
                details.append(f"safety_attributes={safety_attributes}")

        if not details:
            return ""

        return f": {'; '.join(details)}"

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

    @staticmethod
    def _build_text_contents(prompt: PromptResult | str) -> Any:
        if isinstance(prompt, str):
            return prompt

        raw_messages: list[dict[str, Any]] = []
        for msg in prompt.messages:
            role = "model" if msg.role == "assistant" else "user" if msg.role == "user" else msg.role
            raw_messages.append({"role": role, "parts": [{"text": msg.content}]})

        return cast(Any, raw_messages)

    def _select_text_model(
        self,
        prompt: PromptResult | str,
        *,
        model: str | None,
        runtime_kwargs: dict[str, Any],
    ) -> str:
        prompt_model = self._get_prompt_attr(prompt, "model")
        return prompt_model or model or runtime_kwargs.get("model") or self._model

    @staticmethod
    def _get_prompt_attr(prompt: PromptResult | str, name: str) -> Any:
        if isinstance(prompt, str):
            return None
        return getattr(prompt, name)

    @staticmethod
    def _strip_common_text_kwargs(runtime_kw: dict[str, Any]) -> None:
        runtime_kw.pop("model", None)
        runtime_kw.pop("temperature", None)
        runtime_kw.pop("max_tokens", None)
