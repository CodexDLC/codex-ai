"""
codex_ai.core.dispatcher
=========================
LLMDispatcher — orchestrates prompt building and LLM provider calls.

Registers routers, selects the correct builder by mode,
and delegates response generation to the provider.
"""

from __future__ import annotations

import logging
from typing import Any

from .protocol import LLMProviderProtocol, PromptResult
from .router import LLMRouter

log = logging.getLogger(__name__)


class LLMDispatcher:
    """
    Orchestrates prompt building and LLM response generation.

    Connects one or more LLMRouters, selects the builder by mode,
    calls it with provided kwargs, then passes the result to the provider.

    Args:
        provider: LLM backend implementing LLMProviderProtocol.

    Example:
        ```python
        from codex_ai.core import LLMDispatcher, LLMRouter, PromptResult
        from codex_ai.providers import OpenAIProvider

        router = LLMRouter()

        @router.prompt("chat")
        async def build_chat(text: str, **kw) -> PromptResult:
            return PromptResult(messages=[{"role": "user", "content": text}])

        provider = OpenAIProvider(api_key="sk-...")
        dispatcher = LLMDispatcher(provider=provider)
        dispatcher.include_router(router)

        response = await dispatcher.process("chat", text="Hello!")
        ```
    """

    def __init__(self, provider: LLMProviderProtocol) -> None:
        self._provider = provider
        self._builders: dict[str, Any] = {}

    def include_router(self, router: LLMRouter) -> None:
        """
        Register all builders from a router.

        Args:
            router: LLMRouter with decorated builder functions.
        """
        self._builders.update(router.builders)
        log.debug(f"LLMDispatcher | included router modes={list(router.builders.keys())}")

    async def process(self, mode: str, **kw: Any) -> str:
        """
        Build a prompt for the given mode and obtain an LLM response.

        Args:
            mode: Identifier matching a registered builder (e.g., ``"chat"``).
            **kw: Arguments forwarded to the builder function.

        Returns:
            Response text from the LLM provider.

        Raises:
            KeyError: If no builder is registered for the given mode.
        """
        builder = self._builders.get(mode)
        if builder is None:
            raise KeyError(f"LLMDispatcher | no builder registered for mode='{mode}'")

        log.debug(f"LLMDispatcher | calling builder '{builder.__name__}' for mode='{mode}'")
        prompt: PromptResult = await builder(**kw)
        return await self._provider.answer(prompt, **kw)
