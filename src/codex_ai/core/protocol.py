"""
codex_ai.core.protocol
=======================
Core types and contracts for the LLM abstraction layer.

PromptResult — frozen DTO returned by every prompt builder.
LLMProviderProtocol — adapter contract for LLM backends (OpenAI, Gemini, etc.).
PromptBuilder — type alias for async builder functions registered via LLMRouter.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, runtime_checkable

from codex_core.core.base_dto import BaseDTO
from pydantic import ConfigDict


class LLMMessage(BaseDTO):
    """
    Strict contract for single message exchanged with LLM.
    """

    role: Literal["user", "assistant", "system"]
    content: str


class PromptResult(BaseDTO):
    """
    Frozen DTO produced by a prompt builder.

    Passed directly to the LLM provider's ``answer()`` method.

    Attributes:
        messages: Provider-specific message list (OpenAI ChatCompletionMessageParam format
                  or Gemini contents). Each provider interprets this field as needed.
        system: Optional system/developer instruction (top-level string, used by Gemini
                and OpenAI o-series models that accept a dedicated system field).

    Example:
        ```python
        result = PromptResult(
            messages=[LLMMessage(role="user", content="Hello!")],
            system="You are a helpful assistant.",
        )
        ```
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    messages: list[LLMMessage]
    system: str = ""
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """
    Adapter contract for LLM backends.

    Implement this protocol to add a new provider (OpenAI, Gemini, Anthropic, etc.).

    Example:
        ```python
        class MockProvider:
            async def answer(self, prompt: PromptResult, **kw: Any) -> str:
                return "mocked response"
        ```
    """

    async def answer(self, prompt: PromptResult, **kw: Any) -> str:
        """
        Send prompt to the LLM and return the response text.

        Args:
            prompt: Frozen DTO with messages and system instruction.
            **kw: Extra provider-specific kwargs (temperature, max_tokens, etc.).

        Returns:
            Response text from the LLM.
        """
        ...


# Callable registered via @LLMRouter.prompt(mode)
PromptBuilder = Callable[..., Awaitable[PromptResult]]
