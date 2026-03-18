"""
codex_ai.providers.openai
==========================
OpenAIProvider — LLM provider backed by OpenAI's Chat Completions API.

Requires: ``pip install codex-ai[openai]``
"""

from __future__ import annotations

from typing import Any, cast

try:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionMessageParam
except ImportError as e:
    if e.name == "openai":
        raise ImportError(
            "OpenAIProvider requires the 'openai' package. Install it with: pip install codex-ai[openai]"
        ) from e
    raise

from codex_ai.core.exceptions import LLMProviderError
from codex_ai.core.protocol import PromptResult

_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider:
    """
    LLM provider using OpenAI Chat Completions.

    Implements LLMProviderProtocol.

    Args:
        api_key: OpenAI API key.
        model: Model name. Defaults to ``"gpt-4o-mini"``.

    Example:
        ```python
        provider = OpenAIProvider(api_key="sk-...")
        result = PromptResult(
            messages=[LLMMessage(role="user", content="Hello!")],
            system="You are a helpful assistant.",
        )
        text = await provider.answer(result)
        ```
    """

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def answer(self, prompt: PromptResult, **kw: Any) -> str:
        """
        Send prompt to OpenAI and return response text.

        Args:
            prompt: PromptResult with messages list and optional system string.
            **kw: Extra kwargs forwarded to ``chat.completions.create``
                  (e.g., ``temperature``, ``max_tokens``).

        Returns:
            Response content string, or empty string if the model returned nothing.
        """
        raw_messages: list[dict[str, Any]] = [{"role": msg.role, "content": msg.content} for msg in prompt.messages]
        if prompt.system:
            if self._model.startswith(("o1", "o3")) or "gpt-4o" in self._model:
                raw_messages.insert(0, {"role": "developer", "content": prompt.system})
            else:
                raw_messages.insert(0, {"role": "system", "content": prompt.system})

        messages = cast(list[ChatCompletionMessageParam], raw_messages)

        model = prompt.model or kw.get("model") or self._model
        temperature = prompt.temperature or kw.get("temperature")
        max_tokens = prompt.max_tokens or kw.get("max_tokens")

        # Удаляем из kw те параметры, которые мы уже вытащили или которые нельзя дублировать
        runtime_kw = kw.copy()
        runtime_kw.pop("model", None)
        runtime_kw.pop("temperature", None)
        runtime_kw.pop("max_tokens", None)

        # Убираем параметры, которые могут быть None, если API их не поддерживает в таком виде
        if temperature is not None:
            runtime_kw["temperature"] = temperature
        if max_tokens is not None:
            runtime_kw["max_tokens"] = max_tokens

        try:
            completion = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                **runtime_kw,
            )
            content = completion.choices[0].message.content if completion.choices else None
            return content or ""
        except Exception as exc:
            raise LLMProviderError(f"OpenAI error: {exc}") from exc
