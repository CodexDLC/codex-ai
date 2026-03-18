"""
codex_ai.core.router
=====================
LLMRouter — registry for prompt builder functions.

Register builders via the ``@router.prompt(mode)`` decorator,
then connect to LLMDispatcher via ``include_router()``.
"""

from __future__ import annotations

from collections.abc import Callable

from .protocol import PromptBuilder


class LLMRouter:
    """
    Router for grouping prompt builder functions by mode.

    Example:
        ```python
        router = LLMRouter()

        @router.prompt("chat")
        async def build_chat(text: str, history: list, **kw) -> PromptResult:
            return PromptResult(messages=[{"role": "user", "content": text}])

        @router.prompt("summarize")
        async def build_summarize(text: str, **kw) -> PromptResult:
            return PromptResult(
                messages=[{"role": "user", "content": f"Summarize: {text}"}],
                system="Be concise.",
            )
        ```
    """

    def __init__(self) -> None:
        # {mode: builder_func}
        self._builders: dict[str, PromptBuilder] = {}

    def prompt(self, mode: str) -> Callable[[PromptBuilder], PromptBuilder]:
        """
        Decorator for registering a prompt builder for a given mode.

        Args:
            mode: Identifier string (e.g., ``"chat"``, ``"summarize"``).

        Returns:
            Decorator returning the original builder unchanged.
        """

        def decorator(fn: PromptBuilder) -> PromptBuilder:
            self._builders[mode] = fn
            return fn

        return decorator

    @property
    def builders(self) -> dict[str, PromptBuilder]:
        """Registered builders (read-only view)."""
        return self._builders
