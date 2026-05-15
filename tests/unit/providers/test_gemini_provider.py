"""Tests for GeminiProvider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codex_ai.core.exceptions import LLMProviderError
from codex_ai.core.protocol import ImageGenerationProvider, LLMMessage, LLMProviderProtocol, PromptResult
from codex_ai.providers.gemini import GeminiProvider


def _make_provider() -> tuple[GeminiProvider, AsyncMock, MagicMock]:
    """Create a GeminiProvider with mocked client and genai_types."""
    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.HttpOptions.return_value = MagicMock()
        mock_types.GenerateContentConfig.return_value = MagicMock()
        provider = GeminiProvider(api_key="test-key")

    mock_generate = AsyncMock(return_value=MagicMock(text="gemini response"))
    provider._client = MagicMock()
    provider._client.aio.models.generate_content = mock_generate

    return provider, mock_generate, mock_types


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_gemini_provider_satisfies_protocol():
    with patch("codex_ai.providers.gemini.genai_types"):
        provider = GeminiProvider(api_key="x")
    assert isinstance(provider, LLMProviderProtocol)


def test_gemini_provider_satisfies_image_generation_protocol():
    with patch("codex_ai.providers.gemini.genai_types"):
        provider = GeminiProvider(api_key="x")
    assert isinstance(provider, ImageGenerationProvider)


# ---------------------------------------------------------------------------
# Role mapping
# ---------------------------------------------------------------------------


async def test_gemini_maps_assistant_role_to_model():
    provider, mock_generate, _ = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="assistant", content="Hi")])

    with patch("codex_ai.providers.gemini.genai_types"):
        await provider.answer(prompt)

    _, kwargs = mock_generate.call_args
    contents = kwargs["contents"]
    assert contents[0]["role"] == "model"
    assert contents[0]["parts"] == [{"text": "Hi"}]


async def test_gemini_maps_user_role_to_user():
    provider, mock_generate, _ = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hello")])

    with patch("codex_ai.providers.gemini.genai_types"):
        await provider.answer(prompt)

    _, kwargs = mock_generate.call_args
    contents = kwargs["contents"]
    assert contents[0]["role"] == "user"


async def test_gemini_maps_system_role_unchanged():
    provider, mock_generate, _ = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="system", content="Instruction")])

    with patch("codex_ai.providers.gemini.genai_types"):
        await provider.answer(prompt)

    _, kwargs = mock_generate.call_args
    assert kwargs["contents"][0]["role"] == "system"


async def test_gemini_multi_message_order_preserved():
    provider, mock_generate, _ = _make_provider()
    prompt = PromptResult(
        messages=[
            LLMMessage(role="user", content="Hi"),
            LLMMessage(role="assistant", content="Hello"),
            LLMMessage(role="user", content="Bye"),
        ]
    )

    with patch("codex_ai.providers.gemini.genai_types"):
        await provider.answer(prompt)

    _, kwargs = mock_generate.call_args
    contents = kwargs["contents"]
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"
    assert contents[2]["role"] == "user"


# ---------------------------------------------------------------------------
# System instruction
# ---------------------------------------------------------------------------


async def test_gemini_system_instruction_passed_to_config():
    provider, mock_generate, _ = _make_provider()
    prompt = PromptResult(
        messages=[LLMMessage(role="user", content="Hi")],
        system="Be concise",
    )

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateContentConfig.return_value = MagicMock()
        await provider.answer(prompt)
        kwargs = mock_types.GenerateContentConfig.call_args.kwargs
        assert kwargs["system_instruction"] == "Be concise"


async def test_gemini_system_instruction_none_when_empty():
    provider, mock_generate, _ = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")], system="")

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateContentConfig.return_value = MagicMock()
        await provider.answer(prompt)
        kwargs = mock_types.GenerateContentConfig.call_args.kwargs
        # "" becomes None via `prompt.system or None`
        assert kwargs["system_instruction"] is None


# ---------------------------------------------------------------------------
# Config parameters
# ---------------------------------------------------------------------------


async def test_gemini_temperature_in_config():
    provider, mock_generate, _ = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")], temperature=0.8)

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateContentConfig.return_value = MagicMock()
        await provider.answer(prompt)
        kwargs = mock_types.GenerateContentConfig.call_args.kwargs
        assert kwargs["temperature"] == 0.8


async def test_gemini_max_output_tokens_from_max_tokens():
    provider, mock_generate, _ = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")], max_tokens=512)

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateContentConfig.return_value = MagicMock()
        await provider.answer(prompt)
        kwargs = mock_types.GenerateContentConfig.call_args.kwargs
        assert kwargs["max_output_tokens"] == 512


async def test_gemini_model_from_prompt():
    provider, mock_generate, _ = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")], model="gemini-1.5-pro")

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateContentConfig.return_value = MagicMock()
        await provider.answer(prompt)

    _, kwargs = mock_generate.call_args
    assert kwargs["model"] == "gemini-1.5-pro"


async def test_gemini_model_falls_back_to_instance_default():
    provider, mock_generate, _ = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateContentConfig.return_value = MagicMock()
        await provider.answer(prompt)

    _, kwargs = mock_generate.call_args
    assert kwargs["model"] == "gemini-2.5-flash-lite"


async def test_gemini_answer_uses_text_model_not_image_model():
    provider, mock_generate, _ = _make_provider()
    provider._model = "text-model"
    provider._image_model = "image-model"
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateContentConfig.return_value = MagicMock()
        await provider.answer(prompt)

    _, kwargs = mock_generate.call_args
    assert kwargs["model"] == "text-model"


# ---------------------------------------------------------------------------
# Return value handling
# ---------------------------------------------------------------------------


async def test_gemini_returns_response_text():
    provider, mock_generate, _ = _make_provider()
    mock_generate.return_value = MagicMock(text="Deep thoughts")
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateContentConfig.return_value = MagicMock()
        result = await provider.answer(prompt)

    assert result == "Deep thoughts"


async def test_gemini_returns_empty_string_when_text_none():
    provider, mock_generate, _ = _make_provider()
    mock_generate.return_value = MagicMock(text=None)
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateContentConfig.return_value = MagicMock()
        result = await provider.answer(prompt)

    assert result == ""


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


async def test_gemini_raises_llm_provider_error_on_exception():
    provider, mock_generate, _ = _make_provider()
    mock_generate.side_effect = Exception("quota exceeded")
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateContentConfig.return_value = MagicMock()
        with pytest.raises(LLMProviderError, match="Gemini error"):
            await provider.answer(prompt)


async def test_gemini_provider_error_chains_original():
    provider, mock_generate, _ = _make_provider()
    original = TimeoutError("deadline")
    mock_generate.side_effect = original
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateContentConfig.return_value = MagicMock()
        with pytest.raises(LLMProviderError) as exc_info:
            await provider.answer(prompt)

    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


def _image_response(data: bytes | bytearray = b"image", mime_type: str | None = "image/png") -> SimpleNamespace:
    inline_data = SimpleNamespace(data=data, mime_type=mime_type)
    part = SimpleNamespace(inline_data=inline_data, text=None)
    return SimpleNamespace(parts=[part], text=None)


async def test_gemini_generate_image_bytes_uses_image_model_not_text_model():
    provider, mock_generate, _ = _make_provider()
    provider._model = "text-model"
    provider._image_model = "image-model"
    mock_generate.return_value = _image_response()

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.Modality.IMAGE = "IMAGE"
        mock_types.GenerateContentConfig.return_value = MagicMock()
        await provider.generate_image_bytes("draw a castle")

    _, kwargs = mock_generate.call_args
    assert kwargs["model"] == "image-model"
    assert kwargs["contents"] == "draw a castle"


async def test_gemini_generate_image_bytes_model_override_wins():
    provider, mock_generate, _ = _make_provider()
    provider._image_model = "image-model"
    mock_generate.return_value = _image_response()

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.Modality.IMAGE = "IMAGE"
        mock_types.GenerateContentConfig.return_value = MagicMock()
        await provider.generate_image_bytes("draw a castle", model="explicit-image-model")

    _, kwargs = mock_generate.call_args
    assert kwargs["model"] == "explicit-image-model"


async def test_gemini_generate_image_bytes_config_requests_image_modality_and_mime():
    provider, mock_generate, _ = _make_provider()
    mock_generate.return_value = _image_response()

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.Modality.IMAGE = "IMAGE"
        mock_types.GenerateContentConfig.return_value = MagicMock()
        await provider.generate_image_bytes("draw a castle", response_mime_type="image/webp", seed=7)

    config_kwargs = mock_types.GenerateContentConfig.call_args.kwargs
    assert config_kwargs["response_modalities"] == ["IMAGE"]
    assert config_kwargs["response_mime_type"] == "image/webp"
    assert config_kwargs["seed"] == 7


async def test_gemini_generate_image_bytes_returns_inline_image_bytes_and_actual_mime():
    provider, mock_generate, _ = _make_provider()
    mock_generate.return_value = _image_response(data=b"png-bytes", mime_type="image/png")

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.Modality.IMAGE = "IMAGE"
        mock_types.GenerateContentConfig.return_value = MagicMock()
        result = await provider.generate_image_bytes("draw a castle", response_mime_type="image/webp")

    assert result == (b"png-bytes", "image/png")


async def test_gemini_generate_image_bytes_falls_back_to_requested_mime_when_missing():
    provider, mock_generate, _ = _make_provider()
    mock_generate.return_value = _image_response(data=bytearray(b"webp-bytes"), mime_type=None)

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.Modality.IMAGE = "IMAGE"
        mock_types.GenerateContentConfig.return_value = MagicMock()
        result = await provider.generate_image_bytes("draw a castle", response_mime_type="image/webp")

    assert result == (b"webp-bytes", "image/webp")


async def test_gemini_generate_image_bytes_raises_when_no_image_parts():
    provider, mock_generate, _ = _make_provider()
    mock_generate.return_value = SimpleNamespace(parts=[SimpleNamespace(text="no image", inline_data=None)], text=None)

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.Modality.IMAGE = "IMAGE"
        mock_types.GenerateContentConfig.return_value = MagicMock()
        with pytest.raises(LLMProviderError, match="did not return image data"):
            await provider.generate_image_bytes("draw a castle")


async def test_gemini_generate_image_bytes_includes_text_diagnostic_for_text_only_response():
    provider, mock_generate, _ = _make_provider()
    mock_generate.return_value = SimpleNamespace(parts=[], text="blocked by safety policy")

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.Modality.IMAGE = "IMAGE"
        mock_types.GenerateContentConfig.return_value = MagicMock()
        with pytest.raises(LLMProviderError, match="blocked by safety policy"):
            await provider.generate_image_bytes("draw a castle")


async def test_gemini_generate_image_bytes_includes_block_diagnostic():
    provider, mock_generate, _ = _make_provider()
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[]),
        finish_reason="SAFETY",
        safety_ratings=["blocked"],
    )
    mock_generate.return_value = SimpleNamespace(parts=[], text=None, candidates=[candidate])

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.Modality.IMAGE = "IMAGE"
        mock_types.GenerateContentConfig.return_value = MagicMock()
        with pytest.raises(LLMProviderError, match="finish_reason=SAFETY"):
            await provider.generate_image_bytes("draw a castle")


async def test_gemini_generate_image_bytes_wraps_sdk_errors():
    provider, mock_generate, _ = _make_provider()
    original = RuntimeError("quota exceeded")
    mock_generate.side_effect = original

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.Modality.IMAGE = "IMAGE"
        mock_types.GenerateContentConfig.return_value = MagicMock()
        with pytest.raises(LLMProviderError, match="Gemini image generation error") as exc_info:
            await provider.generate_image_bytes("draw a castle")

    assert exc_info.value.__cause__ is original
