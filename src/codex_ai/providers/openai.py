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
        Send prompt to OpenAI and return response text.

        Args:
            prompt: PromptResult with messages list or a raw prompt string.
            model: Optional model override for this request.
            **kwargs: Extra kwargs forwarded to ``chat.completions.create``
                  (e.g., ``temperature``, ``max_tokens``).

        Returns:
            Response content string, or empty string if the model returned nothing.
        """
        raw_messages = self._build_messages(prompt)
        selected_model = self._select_model(prompt, model=model, runtime_kwargs=kwargs)

        system = self._get_prompt_attr(prompt, "system")
        if system:
            if selected_model.startswith(("o1", "o3")) or "gpt-4o" in selected_model:
                raw_messages.insert(0, {"role": "developer", "content": system})
            else:
                raw_messages.insert(0, {"role": "system", "content": system})

        messages = cast(list[ChatCompletionMessageParam], raw_messages)

        temperature = self._get_prompt_attr(prompt, "temperature") or kwargs.get("temperature")
        max_tokens = self._get_prompt_attr(prompt, "max_tokens") or kwargs.get("max_tokens")

        # Remove params already extracted from kw or that cannot be duplicated
        runtime_kw = kwargs.copy()
        runtime_kw.pop("model", None)
        runtime_kw.pop("temperature", None)
        runtime_kw.pop("max_tokens", None)

        # Strip None-valued params — the API does not accept them as keyword arguments
        if temperature is not None:
            runtime_kw["temperature"] = temperature
        if max_tokens is not None:
            runtime_kw["max_tokens"] = max_tokens

        try:
            completion = await self._client.chat.completions.create(
                model=selected_model,
                messages=messages,
                **runtime_kw,
            )
            content = completion.choices[0].message.content if completion.choices else None
            return content or ""
        except Exception as exc:
            raise LLMProviderError(f"OpenAI error: {exc}") from exc

    @staticmethod
    def _build_messages(prompt: PromptResult | str) -> list[dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return [{"role": msg.role, "content": msg.content} for msg in prompt.messages]

    def _select_model(
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
