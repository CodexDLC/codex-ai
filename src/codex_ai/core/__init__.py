"""
codex_ai.core
=============
Core legacy text router, dispatcher, and provider compatibility contracts.
"""

from .dispatcher import LLMDispatcher
from .exceptions import LLMProviderError
from .protocol import (
    ImageGenerationProvider,
    ImagenGenerationProvider,
    JsonGenerationProvider,
    LLMMessage,
    LLMProviderProtocol,
    PromptBuilder,
    PromptResult,
    TextGenerationProvider,
)
from .router import LLMRouter
from .sync import SyncLLMDispatcher

__all__ = [
    "LLMDispatcher",
    "LLMProviderError",
    "ImageGenerationProvider",
    "ImagenGenerationProvider",
    "JsonGenerationProvider",
    "LLMMessage",
    "LLMProviderProtocol",
    "PromptBuilder",
    "PromptResult",
    "TextGenerationProvider",
    "LLMRouter",
    "SyncLLMDispatcher",
]
