"""
codex_ai.core.protocol
=======================
Core types and contracts for the LLM abstraction layer.

PromptResult — frozen DTO returned by every prompt builder.
LLMProviderProtocol — adapter contract for LLM backends (OpenAI, Gemini, etc.).
TextGenerationProvider — optional adapter contract for direct text generation.
JsonGenerationProvider — optional adapter contract for direct JSON generation.
ImageGenerationProvider — optional adapter contract for binary image generation.
PromptBuilder — type alias for async builder functions registered via LLMRouter.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    # mypy complains about Any base class if we don't cheat or provide types.
    from pydantic import BaseModel as BaseDTO
else:
    from codex_core.core.base_dto import BaseDTO


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


@runtime_checkable
class TextGenerationProvider(Protocol):
    """
    Optional adapter contract for direct provider text generation.
    """

    async def generate_text(
        self,
        prompt: PromptResult | str,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate plain text from a prompt DTO or a raw prompt string.
        """
        ...


@runtime_checkable
class JsonGenerationProvider(Protocol):
    """
    Optional adapter contract for provider-native JSON generation.
    """

    async def generate_json(
        self,
        prompt: PromptResult | str,
        *,
        schema: type[BaseModel] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate JSON and validate it locally when a Pydantic schema is provided.
        """
        ...


@runtime_checkable
class ImageGenerationProvider(Protocol):
    """
    Optional adapter contract for providers that can generate image bytes.

    This is intentionally separate from ``LLMProviderProtocol.answer()`` because
    image generation returns binary parts, not text.
    """

    async def generate_image_bytes(
        self,
        prompt: str,
        *,
        model: str | None = None,
        response_mime_type: str = "image/webp",
        **kwargs: Any,
    ) -> tuple[bytes, str]:
        """
        Generate an image and return raw bytes plus the actual content type.

        Args:
            prompt: Plain image-generation prompt.
            model: Optional image model override.
            response_mime_type: Requested/preferred image MIME type.
            **kwargs: Extra provider-specific kwargs.

        Returns:
            Tuple of ``(image_bytes, content_type)``.
        """
        ...


# Callable registered via @LLMRouter.prompt(mode)
PromptBuilder = Callable[..., Awaitable[PromptResult]]
