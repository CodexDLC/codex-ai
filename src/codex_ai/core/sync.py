"""
codex_ai.core.sync
===================
SyncLLMDispatcher — synchronous wrapper around LLMDispatcher.

For use in environments without a running event loop (Django WSGI sync views,
scripts, CLI tools). NOT suitable for use inside a running event loop
(ARQ workers, async views, Telegram bot handlers — use LLMDispatcher directly).

In Django production it is strongly recommended to offload LLM calls
to ARQ background tasks rather than blocking a sync view.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .dispatcher import LLMDispatcher


class SyncLLMDispatcher:
    """
    Synchronous wrapper for LLMDispatcher.

    Runs the async ``process()`` call using the current event loop
    or a new one if no loop exists.

    WARNING: Do NOT use inside a running event loop (async def context).
    In those cases use LLMDispatcher directly with ``await``.

    Args:
        dispatcher: Configured LLMDispatcher with routers and provider.

    Example:
        ```python
        # Django sync view (WSGI)
        from codex_ai.core import LLMDispatcher, LLMRouter, SyncLLMDispatcher
        from codex_ai.providers import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-...")
        dispatcher = LLMDispatcher(provider=provider)
        dispatcher.include_router(my_router)

        sync_dispatcher = SyncLLMDispatcher(dispatcher)

        def my_django_view(request):
            response = sync_dispatcher.process("summarize", text=request.POST["text"])
            return JsonResponse({"response": response})
        ```
    """

    def __init__(self, dispatcher: LLMDispatcher) -> None:
        self._dispatcher = dispatcher

    def process(self, mode: str, **kw: Any) -> str:
        """
        Build a prompt and obtain an LLM response synchronously.

        Args:
            mode: Identifier matching a registered builder.
            **kw: Arguments forwarded to the builder function.

        Returns:
            Response text from the LLM provider.
        """
        return asyncio.run(self._dispatcher.process(mode, **kw))
