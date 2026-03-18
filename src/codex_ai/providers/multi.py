"""
codex_ai.providers.multi
=========================
MultiLLMProvider — aggregator for multiple LLM backends.

Allows switching between OpenAI, Gemini, etc. on the fly by passing
the ``provider`` argument to the dispatcher.
"""

from __future__ import annotations

import logging
from typing import Any

from codex_ai.core.protocol import LLMProviderProtocol, PromptResult

log = logging.getLogger(__name__)


class MultiLLMProvider:
    """
    Orchestrates multiple LLM providers.

    Implements LLMProviderProtocol by delegating to one of registered providers.

    Args:
        providers: Map of {name: provider_instance}.
        default: Name of the default provider to use.

    Example:
        ```python
        multi = MultiLLMProvider(
            providers={
                "openai": OpenAIProvider(api_key="..."),
                "gemini": GeminiProvider(api_key="...")
            },
            default="openai"
        )
        dispatcher = LLMDispatcher(provider=multi)

        # Will use Gemini
        await dispatcher.process("chat", provider="gemini")
        ```
    """

    def __init__(
        self, providers: dict[str, LLMProviderProtocol], default: str, failover_list: list[str] | None = None
    ) -> None:
        self._providers = providers
        self._default = default
        self._failover = failover_list or []

        if default not in providers:
            raise KeyError(f"MultiLLMProvider | default provider '{default}' not found in registry")

    async def answer(self, prompt: PromptResult, **kw: Any) -> str:
        """
        Delegate request to a specific provider with failover support.

        Choice priority:
        1. Explicit ``provider`` in kw.
        2. Inferred from ``model`` name.
        3. Default provider.

        If the primary provider fails (LLMProviderError), it tries providers
        from ``failover_list`` in order.
        """
        primary_name = kw.get("provider")

        # Infer from model name if not specified
        if not primary_name:
            model = prompt.model or kw.get("model")
            if model:
                if model.startswith(("gpt-", "o1-", "o3-")):
                    primary_name = "openai"
                elif model.startswith("gemini-"):
                    primary_name = "gemini"
                elif model.startswith("claude-"):
                    primary_name = "anthropic"
                elif "/" in model:  # OpenRouter typical format 'vendor/model'
                    primary_name = "openrouter"

        # Fallback to default
        if not primary_name:
            primary_name = self._default

        # Building the list of providers to try
        tried_names: set[str] = set()
        providers_to_try = [primary_name] + self._failover

        last_exc: Exception | None = None

        # Remove 'provider' from kw once
        runtime_kw = kw.copy()
        runtime_kw.pop("provider", None)

        from codex_ai.core.exceptions import LLMProviderError

        for name in providers_to_try:
            if name in tried_names:
                continue

            provider = self._providers.get(name)
            if not provider:
                log.warning(f"MultiLLMProvider | provider '{name}' not found in registry. Skipping.")
                continue

            try:
                tried_names.add(name)
                log.debug(f"MultiLLMProvider | trying provider '{name}'")
                return await provider.answer(prompt, **runtime_kw)
            except LLMProviderError as exc:
                last_exc = exc
                log.error(f"MultiLLMProvider | provider '{name}' failed: {exc}. Trying next if available.")
                continue

        if last_exc:
            raise last_exc

        raise LLMProviderError("MultiLLMProvider | no valid providers available for this request")
