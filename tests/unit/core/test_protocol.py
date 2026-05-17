import pytest
from pydantic import ValidationError

from codex_ai.core.protocol import (
    ImageGenerationProvider,
    ImagenGenerationProvider,
    JsonGenerationProvider,
    LLMMessage,
    LLMProviderProtocol,
    PromptResult,
    TextGenerationProvider,
)

# ---------------------------------------------------------------------------
# LLMMessage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["user", "assistant", "system"])
def test_llm_message_valid_roles(role):
    msg = LLMMessage(role=role, content="hello")
    assert msg.role == role
    assert msg.content == "hello"


def test_llm_message_invalid_role_raises():
    with pytest.raises(ValidationError):
        LLMMessage(role="invalid", content="hello")


def test_llm_message_empty_content_allowed():
    msg = LLMMessage(role="user", content="")
    assert msg.content == ""


# ---------------------------------------------------------------------------
# PromptResult
# ---------------------------------------------------------------------------


def test_prompt_result_minimal_defaults(simple_prompt):
    assert simple_prompt.system == ""
    assert simple_prompt.model is None
    assert simple_prompt.temperature is None
    assert simple_prompt.max_tokens is None


def test_prompt_result_all_fields(full_prompt):
    assert full_prompt.system == "You are helpful"
    assert full_prompt.model == "gpt-4o-mini"
    assert full_prompt.temperature == 0.5
    assert full_prompt.max_tokens == 256
    assert len(full_prompt.messages) == 3


def test_prompt_result_is_frozen(simple_prompt):
    with pytest.raises((ValidationError, TypeError)):
        simple_prompt.system = "new value"


def test_prompt_result_empty_messages_allowed():
    result = PromptResult(messages=[])
    assert result.messages == []


# ---------------------------------------------------------------------------
# LLMProviderProtocol structural checks
# ---------------------------------------------------------------------------


def test_mock_provider_satisfies_protocol(mock_provider):
    assert isinstance(mock_provider, LLMProviderProtocol)


def test_object_without_answer_fails_protocol_check():
    assert not isinstance(object(), LLMProviderProtocol)


def test_class_without_answer_fails_protocol_check():
    class NoAnswer:
        pass

    assert not isinstance(NoAnswer(), LLMProviderProtocol)


def test_runtime_checkable_does_not_verify_async():
    # @runtime_checkable only checks method existence, not coroutine type.
    class SyncAnswer:
        def answer(self, prompt, **kw):
            return "sync"

    # This is the documented limitation: sync method still passes the check.
    assert isinstance(SyncAnswer(), LLMProviderProtocol)


# ---------------------------------------------------------------------------
# ImageGenerationProvider structural checks
# ---------------------------------------------------------------------------


def test_image_generation_provider_structural_check():
    class MockImageProvider:
        async def generate_image_bytes(
            self,
            prompt: str,
            *,
            model: str | None = None,
            response_mime_type: str = "image/webp",
            image_config: dict | None = None,
            **kwargs,
        ) -> tuple[bytes, str]:
            return b"image", response_mime_type

    assert isinstance(MockImageProvider(), ImageGenerationProvider)


def test_object_without_generate_image_bytes_fails_image_provider_check():
    assert not isinstance(object(), ImageGenerationProvider)


def test_imagen_generation_provider_structural_check():
    class MockImagenProvider:
        async def generate_imagen_bytes(
            self,
            prompt: str,
            *,
            model: str | None = None,
            response_mime_type: str = "image/jpeg",
            **kwargs,
        ) -> tuple[bytes, str]:
            return b"image", response_mime_type

    assert isinstance(MockImagenProvider(), ImagenGenerationProvider)


def test_object_without_generate_imagen_bytes_fails_imagen_provider_check():
    assert not isinstance(object(), ImagenGenerationProvider)


def test_text_generation_provider_structural_check():
    class MockTextProvider:
        async def generate_text(self, prompt: PromptResult | str, *, model: str | None = None, **kwargs) -> str:
            return "text"

    assert isinstance(MockTextProvider(), TextGenerationProvider)


def test_json_generation_provider_structural_check():
    class MockJsonProvider:
        async def generate_json(
            self,
            prompt: PromptResult | str,
            *,
            schema=None,
            model: str | None = None,
            **kwargs,
        ):
            return {}

    assert isinstance(MockJsonProvider(), JsonGenerationProvider)
