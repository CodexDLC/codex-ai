"""Tests for GeminiProvider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from codex_ai.core.exceptions import LLMProviderError
from codex_ai.core.protocol import (
    ImageGenerationProvider,
    ImagenGenerationProvider,
    JsonGenerationProvider,
    LLMMessage,
    LLMProviderProtocol,
    PromptResult,
    TextGenerationProvider,
)
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


def test_gemini_provider_satisfies_imagen_generation_protocol():
    with patch("codex_ai.providers.gemini.genai_types"):
        provider = GeminiProvider(api_key="x")
    assert isinstance(provider, ImagenGenerationProvider)


def test_gemini_provider_satisfies_text_generation_protocol():
    with patch("codex_ai.providers.gemini.genai_types"):
        provider = GeminiProvider(api_key="x")
    assert isinstance(provider, TextGenerationProvider)


def test_gemini_provider_satisfies_json_generation_protocol():
    with patch("codex_ai.providers.gemini.genai_types"):
        provider = GeminiProvider(api_key="x")
    assert isinstance(provider, JsonGenerationProvider)


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


async def test_gemini_generate_text_accepts_raw_prompt_string():
    provider, mock_generate, _ = _make_provider()
    mock_generate.return_value = MagicMock(text="raw response")

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateContentConfig.return_value = MagicMock()
        result = await provider.generate_text("hello", model="gemini-text", temperature=0.1)

    _, kwargs = mock_generate.call_args
    assert kwargs["contents"] == "hello"
    assert kwargs["model"] == "gemini-text"
    assert result == "raw response"


class LootItem(BaseModel):
    name: str
    power: int


async def test_gemini_generate_json_returns_dict_without_schema():
    provider, mock_generate, _ = _make_provider()
    mock_generate.return_value = MagicMock(text='{"name": "sword", "power": 7}')

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateContentConfig.return_value = MagicMock()
        result = await provider.generate_json("make loot")

    config_kwargs = mock_types.GenerateContentConfig.call_args.kwargs
    assert config_kwargs["response_mime_type"] == "application/json"
    assert "response_schema" not in config_kwargs
    assert result == {"name": "sword", "power": 7}


async def test_gemini_generate_json_validates_schema():
    provider, mock_generate, _ = _make_provider()
    mock_generate.return_value = MagicMock(text='{"name": "sword", "power": 7}')

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateContentConfig.return_value = MagicMock()
        result = await provider.generate_json("make loot", schema=LootItem)

    config_kwargs = mock_types.GenerateContentConfig.call_args.kwargs
    assert config_kwargs["response_schema"] is LootItem
    assert result == LootItem(name="sword", power=7)


async def test_gemini_generate_json_raises_on_invalid_json():
    provider, mock_generate, _ = _make_provider()
    mock_generate.return_value = MagicMock(text="not json")

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateContentConfig.return_value = MagicMock()
        with pytest.raises(LLMProviderError, match="invalid JSON"):
            await provider.generate_json("make loot")


async def test_gemini_generate_json_raises_on_schema_validation_error():
    provider, mock_generate, _ = _make_provider()
    mock_generate.return_value = MagicMock(text='{"name": "sword"}')

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateContentConfig.return_value = MagicMock()
        with pytest.raises(LLMProviderError, match="schema validation"):
            await provider.generate_json("make loot", schema=LootItem)


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


def _imagen_response(data: bytes | str = b"image", mime_type: str | None = "image/jpeg") -> SimpleNamespace:
    image = SimpleNamespace(image_bytes=data, mime_type=mime_type)
    generated_image = SimpleNamespace(image=image)
    return SimpleNamespace(generated_images=[generated_image])


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


async def test_gemini_generate_image_bytes_default_image_model_matches_api_id():
    provider, mock_generate, _ = _make_provider()
    mock_generate.return_value = _image_response()

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.Modality.IMAGE = "IMAGE"
        mock_types.GenerateContentConfig.return_value = MagicMock()
        await provider.generate_image_bytes("draw a castle")

    _, kwargs = mock_generate.call_args
    assert kwargs["model"] == "gemini-2.5-flash-image"


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


async def test_gemini_generate_image_bytes_config_requests_image_modality_not_text_mime():
    provider, mock_generate, _ = _make_provider()
    mock_generate.return_value = _image_response()

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.Modality.IMAGE = "IMAGE"
        mock_types.GenerateContentConfig.return_value = MagicMock()
        await provider.generate_image_bytes(
            "draw a castle",
            response_mime_type="image/png",
            image_config={"aspect_ratio": "1:1", "image_size": "4K"},
            seed=7,
        )

    config_kwargs = mock_types.GenerateContentConfig.call_args.kwargs
    assert config_kwargs["response_modalities"] == ["IMAGE"]
    assert "response_mime_type" not in config_kwargs
    assert config_kwargs["image_config"] == {"aspect_ratio": "1:1", "image_size": "4K"}
    assert config_kwargs["seed"] == 7


async def test_gemini_generate_image_bytes_falls_back_from_4k_to_2k_when_config_rejected():
    provider, mock_generate, _ = _make_provider()
    mock_generate.side_effect = [
        ValueError("unsupported image_size"),
        _image_response(data=b"png", mime_type="image/png"),
    ]

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.Modality.IMAGE = "IMAGE"
        mock_types.GenerateContentConfig.side_effect = lambda **kwargs: kwargs
        result = await provider.generate_image_bytes(
            "draw a castle",
            response_mime_type="image/png",
            image_config={"aspect_ratio": "1:1", "image_size": "4K"},
        )

    assert result == (b"png", "image/png")
    first_config = mock_generate.call_args_list[0].kwargs["config"]
    second_config = mock_generate.call_args_list[1].kwargs["config"]
    assert first_config["image_config"] == {"aspect_ratio": "1:1", "image_size": "4K"}
    assert second_config["image_config"] == {"aspect_ratio": "1:1", "image_size": "2K"}


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


async def test_gemini_generate_imagen_bytes_uses_imagen_model_not_gemini_image_model():
    provider, _, _ = _make_provider()
    provider._image_model = "gemini-image-model"
    provider._imagen_model = "imagen-model"
    mock_generate_images = AsyncMock(return_value=_imagen_response())
    provider._client.aio.models.generate_images = mock_generate_images

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateImagesConfig.return_value = MagicMock()
        await provider.generate_imagen_bytes("draw a castle")

    _, kwargs = mock_generate_images.call_args
    assert kwargs["model"] == "imagen-model"
    assert kwargs["prompt"] == "draw a castle"


async def test_gemini_generate_imagen_bytes_model_override_wins():
    provider, _, _ = _make_provider()
    provider._imagen_model = "imagen-model"
    mock_generate_images = AsyncMock(return_value=_imagen_response())
    provider._client.aio.models.generate_images = mock_generate_images

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateImagesConfig.return_value = MagicMock()
        await provider.generate_imagen_bytes("draw a castle", model="explicit-imagen-model")

    _, kwargs = mock_generate_images.call_args
    assert kwargs["model"] == "explicit-imagen-model"


async def test_gemini_generate_imagen_bytes_config_sets_output_mime_type():
    provider, _, _ = _make_provider()
    provider._client.aio.models.generate_images = AsyncMock(return_value=_imagen_response())

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateImagesConfig.return_value = MagicMock()
        await provider.generate_imagen_bytes("draw a castle", response_mime_type="image/jpeg", seed=7)

    config_kwargs = mock_types.GenerateImagesConfig.call_args.kwargs
    assert config_kwargs["output_mime_type"] == "image/jpeg"
    assert config_kwargs["seed"] == 7


async def test_gemini_generate_imagen_bytes_returns_image_bytes_and_actual_mime():
    provider, _, _ = _make_provider()
    provider._client.aio.models.generate_images = AsyncMock(
        return_value=_imagen_response(data=b"jpeg-bytes", mime_type="image/jpeg")
    )

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateImagesConfig.return_value = MagicMock()
        result = await provider.generate_imagen_bytes("draw a castle", response_mime_type="image/png")

    assert result == (b"jpeg-bytes", "image/jpeg")


async def test_gemini_generate_imagen_bytes_decodes_base64_image_bytes():
    provider, _, _ = _make_provider()
    provider._client.aio.models.generate_images = AsyncMock(
        return_value=_imagen_response(data="anBlZy1ieXRlcw==", mime_type="image/jpeg")
    )

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateImagesConfig.return_value = MagicMock()
        result = await provider.generate_imagen_bytes("draw a castle")

    assert result == (b"jpeg-bytes", "image/jpeg")


async def test_gemini_generate_imagen_bytes_falls_back_to_requested_mime_when_missing():
    provider, _, _ = _make_provider()
    provider._client.aio.models.generate_images = AsyncMock(
        return_value=_imagen_response(data=b"image-bytes", mime_type=None)
    )

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateImagesConfig.return_value = MagicMock()
        result = await provider.generate_imagen_bytes("draw a castle", response_mime_type="image/png")

    assert result == (b"image-bytes", "image/png")


async def test_gemini_generate_imagen_bytes_raises_when_no_image_data():
    provider, _, _ = _make_provider()
    generated_image = SimpleNamespace(image=SimpleNamespace(image_bytes=None, mime_type=None))
    provider._client.aio.models.generate_images = AsyncMock(
        return_value=SimpleNamespace(generated_images=[generated_image])
    )

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateImagesConfig.return_value = MagicMock()
        with pytest.raises(LLMProviderError, match="Imagen generation did not return image data"):
            await provider.generate_imagen_bytes("draw a castle")


async def test_gemini_generate_imagen_bytes_wraps_sdk_errors():
    provider, _, _ = _make_provider()
    original = RuntimeError("quota exceeded")
    provider._client.aio.models.generate_images = AsyncMock(side_effect=original)

    with patch("codex_ai.providers.gemini.genai_types") as mock_types:
        mock_types.GenerateImagesConfig.return_value = MagicMock()
        with pytest.raises(LLMProviderError, match="Gemini Imagen generation error") as exc_info:
            await provider.generate_imagen_bytes("draw a castle")

    assert exc_info.value.__cause__ is original
