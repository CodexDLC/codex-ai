"""Tests for GeminiProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codex_ai.core.exceptions import LLMProviderError
from codex_ai.core.protocol import LLMMessage, LLMProviderProtocol, PromptResult
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
