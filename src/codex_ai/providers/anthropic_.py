"""
codex_ai.providers.anthropic_
==============================
AnthropicProvider — LLM provider backed by Anthropic Claude API.

Requires: ``pip install codex-ai[anthropic]``
"""

from __future__ import annotations

from typing import Any, cast

try:
    from anthropic import AsyncAnthropic
    from anthropic.types import MessageParam
except ImportError as e:
    if e.name == "anthropic":
        raise ImportError(
            "AnthropicProvider requires the 'anthropic' package. Install it with: pip install codex-ai[anthropic]"
        ) from e
    raise

from codex_ai.core.exceptions import LLMProviderError
from codex_ai.core.protocol import PromptResult

_DEFAULT_MODEL = "claude-3-5-sonnet-latest"


class AnthropicProvider:
    """
    LLM provider using Anthropic Claude.

    Implements LLMProviderProtocol.

    Args:
        api_key: Anthropic API key.
        model: Model name. Defaults to ``"claude-3-5-sonnet-latest"``.
    """

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def answer(self, prompt: PromptResult, **kw: Any) -> str:
        """
        Send prompt to Anthropic and return response text.
        """
        raw_messages: list[dict[str, Any]] = [{"role": msg.role, "content": msg.content} for msg in prompt.messages]
        messages = cast(list[MessageParam], raw_messages)

        # Anthropic uses a separate 'system' parameter
        system = prompt.system or None

        model = prompt.model or kw.get("model") or self._model
        max_tokens = prompt.max_tokens or kw.get("max_tokens") or 1024
        temperature = prompt.temperature or kw.get("temperature")

        # Cleanup kw
        runtime_kw = kw.copy()
        runtime_kw.pop("model", None)
        runtime_kw.pop("max_tokens", None)
        runtime_kw.pop("temperature", None)

        # Avoid passing None values to typed API arguments
        if temperature is not None:
            runtime_kw["temperature"] = temperature
        if system is not None:
            runtime_kw["system"] = system

        try:
            message = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                **runtime_kw,
            )

            if not message.content:
                return ""

            # Message content has different types. The safest way is to extract text if hasattr
            content_block = message.content[0]
            if hasattr(content_block, "text"):
                return str(content_block.text)

            return ""
        except Exception as exc:
            raise LLMProviderError(f"Anthropic error: {exc}") from exc
