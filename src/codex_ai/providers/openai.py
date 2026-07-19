"""
codex_ai.providers.openai
=========================
Modern OpenAI adapter backed by the Responses API.

Requires: ``pip install codex-ai[openai]``
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, cast

from pydantic import BaseModel

try:
    from openai import AsyncOpenAI, OpenAIError
except ImportError as e:
    if e.name == "openai":
        raise ImportError(
            "OpenAIProvider requires the 'openai' package. Install it with: pip install codex-ai[openai]"
        ) from e
    raise

from codex_ai.core.exceptions import LLMProviderError
from codex_ai.core.protocol import PromptResult

_DEFAULT_MODEL = "gpt-5.6-luna"
_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_MAX_RETRIES = 2


class _ResponsesResource(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...

    async def parse(self, **kwargs: Any) -> Any: ...


class _OpenAIClientPort(Protocol):
    responses: _ResponsesResource


@dataclass(frozen=True, slots=True)
class _RequestDefaults:
    model: str
    store: bool
    reasoning_effort: str | None


class _OpenAIRequestMapper:
    """Map package-level prompt contracts to one Responses API request."""

    def __init__(self, defaults: _RequestDefaults) -> None:
        self._defaults = defaults

    def build(
        self,
        prompt: PromptResult | str,
        *,
        model: str | None,
        runtime_options: dict[str, Any],
    ) -> dict[str, Any]:
        if "max_tokens" in runtime_options:
            raise TypeError("OpenAI Responses API uses 'max_output_tokens'; 'max_tokens' is not supported")

        request = runtime_options.copy()
        request["model"] = model or self._prompt_value(prompt, "model") or self._defaults.model
        request["input"] = self._build_input(prompt)
        request.setdefault("store", self._defaults.store)

        if self._defaults.reasoning_effort is not None:
            request.setdefault("reasoning", {"effort": self._defaults.reasoning_effort})

        instructions = self._prompt_value(prompt, "system")
        if instructions:
            request.setdefault("instructions", instructions)

        temperature = self._prompt_value(prompt, "temperature")
        if temperature is not None:
            request.setdefault("temperature", temperature)

        max_tokens = self._prompt_value(prompt, "max_tokens")
        if max_tokens is not None:
            request.setdefault("max_output_tokens", max_tokens)

        return request

    @staticmethod
    def _build_input(prompt: PromptResult | str) -> str | list[dict[str, str]]:
        if isinstance(prompt, str):
            return prompt
        return [{"role": message.role, "content": message.content} for message in prompt.messages]

    @staticmethod
    def _prompt_value(prompt: PromptResult | str, name: str) -> Any:
        if isinstance(prompt, str):
            return None
        return getattr(prompt, name)


class OpenAIProvider:
    """
    Async OpenAI Responses API adapter.

    The provider exposes direct text, structured JSON, and streaming methods.
    Requests are not stored by default. The default model keeps the previous
    provider's cost-sensitive role while moving to the current GPT-5.6 family.

    Args:
        api_key: OpenAI API key. When omitted, the SDK reads ``OPENAI_API_KEY``.
        model: Default model used for requests.
        client: Optional injected client implementing the Responses API port.
        store: Whether OpenAI may store responses. Defaults to ``False``.
        reasoning_effort: Default GPT-5.6 reasoning effort. Defaults to ``"none"``.
        timeout: SDK request timeout in seconds.
        max_retries: SDK retry count for retryable failures.
        base_url: Optional OpenAI-compatible API base URL.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        *,
        client: _OpenAIClientPort | None = None,
        store: bool = False,
        reasoning_effort: str | None = "none",
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        base_url: str | None = None,
    ) -> None:
        self._client = client or self._build_client(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            base_url=base_url,
        )
        self._request_mapper = _OpenAIRequestMapper(
            _RequestDefaults(
                model=model,
                store=store,
                reasoning_effort=reasoning_effort,
            )
        )

    async def answer(self, prompt: PromptResult, **kwargs: Any) -> str:
        """Generate text for the router/dispatcher contract."""
        return await self.generate_text(prompt, **kwargs)

    async def generate_text(
        self,
        prompt: PromptResult | str,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate plain text through the Responses API."""
        request = self._request_mapper.build(
            prompt,
            model=model,
            runtime_options=kwargs,
        )
        try:
            response = await self._client.responses.create(**request)
        except OpenAIError as exc:
            raise self._provider_error(exc) from exc
        return getattr(response, "output_text", "") or ""

    async def generate_json(
        self,
        prompt: PromptResult | str,
        *,
        schema: type[BaseModel] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate structured JSON.

        A Pydantic schema uses the SDK's native structured-output parser.
        Without a schema, JSON mode is used and the returned text is decoded.
        """
        request = self._request_mapper.build(
            prompt,
            model=model,
            runtime_options=kwargs,
        )

        if schema is not None:
            request["text_format"] = schema
            try:
                response = await self._client.responses.parse(**request)
            except OpenAIError as exc:
                raise self._provider_error(exc) from exc

            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                raise LLMProviderError("OpenAI Responses API returned no parsed output")
            return parsed

        text_options = dict(request.get("text") or {})
        text_options["format"] = {"type": "json_object"}
        request["text"] = text_options

        try:
            response = await self._client.responses.create(**request)
        except OpenAIError as exc:
            raise self._provider_error(exc) from exc

        output_text = getattr(response, "output_text", "") or ""
        try:
            return json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise LLMProviderError(f"OpenAI returned invalid JSON: {exc}") from exc

    async def stream_text(
        self,
        prompt: PromptResult | str,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield typed text deltas from a streaming Responses API request."""
        request = self._request_mapper.build(
            prompt,
            model=model,
            runtime_options=kwargs,
        )
        request["stream"] = True

        try:
            stream = await self._client.responses.create(**request)
            async for event in stream:
                if getattr(event, "type", None) == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        yield delta
        except OpenAIError as exc:
            raise self._provider_error(exc) from exc

    @staticmethod
    def _build_client(
        *,
        api_key: str | None,
        timeout: float,
        max_retries: int,
        base_url: str | None,
    ) -> _OpenAIClientPort:
        client_options: dict[str, Any] = {
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if api_key is not None:
            client_options["api_key"] = api_key
        if base_url is not None:
            client_options["base_url"] = base_url
        return cast(_OpenAIClientPort, AsyncOpenAI(**client_options))

    @staticmethod
    def _provider_error(exc: OpenAIError) -> LLMProviderError:
        request_id = getattr(exc, "request_id", None)
        request_context = f" (request_id={request_id})" if request_id else ""
        return LLMProviderError(f"OpenAI Responses API error{request_context}: {exc}")
