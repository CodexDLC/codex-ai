"""Tests for AnthropicProvider."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from codex_ai.core.exceptions import LLMProviderError
from codex_ai.core.protocol import LLMMessage, LLMProviderProtocol, PromptResult
from codex_ai.providers.anthropic_ import AnthropicProvider


def _make_message(text: str | None = "hello from anthropic", has_text_attr: bool = True):
    mock = MagicMock()
    if text is None:
        mock.content = []
    else:
        block = MagicMock(spec=["text"] if has_text_attr else [])
        if has_text_attr:
            block.text = text
        mock.content = [block]
    return mock


def _make_provider(model: str = "claude-3-5-sonnet-latest") -> tuple[AnthropicProvider, AsyncMock]:
    provider = AnthropicProvider(api_key="test-key", model=model)
    mock_create = AsyncMock(return_value=_make_message())
    provider._client = MagicMock()
    provider._client.messages.create = mock_create
    return provider, mock_create


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_anthropic_provider_satisfies_protocol():
    provider = AnthropicProvider(api_key="x")
    assert isinstance(provider, LLMProviderProtocol)


# ---------------------------------------------------------------------------
# Message format conversion
# ---------------------------------------------------------------------------


async def test_anthropic_builds_message_list():
    provider, mock_create = _make_provider()
    prompt = PromptResult(
        messages=[
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi"),
        ]
    )
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    messages = kwargs["messages"]
    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "Hello"}
    assert messages[1] == {"role": "assistant", "content": "Hi"}


# ---------------------------------------------------------------------------
# System parameter handling
# ---------------------------------------------------------------------------


async def test_anthropic_system_passed_as_kwarg_when_present():
    provider, mock_create = _make_provider()
    prompt = PromptResult(
        messages=[LLMMessage(role="user", content="Hi")],
        system="You are helpful",
    )
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    assert kwargs["system"] == "You are helpful"


async def test_anthropic_system_not_passed_when_empty():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")], system="")
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    assert "system" not in kwargs


# ---------------------------------------------------------------------------
# Parameter resolution: prompt > kw > default
# ---------------------------------------------------------------------------


async def test_anthropic_max_tokens_default_1024():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    assert kwargs["max_tokens"] == 1024


async def test_anthropic_max_tokens_from_prompt():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")], max_tokens=512)
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    assert kwargs["max_tokens"] == 512


async def test_anthropic_max_tokens_from_kw():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    await provider.answer(prompt, max_tokens=2048)

    _, kwargs = mock_create.call_args
    assert kwargs["max_tokens"] == 2048


async def test_anthropic_temperature_from_prompt():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")], temperature=0.8)
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    assert kwargs["temperature"] == 0.8


async def test_anthropic_temperature_from_kw():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    await provider.answer(prompt, temperature=0.2)

    _, kwargs = mock_create.call_args
    assert kwargs["temperature"] == 0.2


async def test_anthropic_temperature_not_passed_when_none():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    assert "temperature" not in kwargs


async def test_anthropic_model_from_prompt():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")], model="claude-3-opus-20240229")
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    assert kwargs["model"] == "claude-3-opus-20240229"


async def test_anthropic_model_from_kw():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    await provider.answer(prompt, model="claude-3-haiku-20240307")

    _, kwargs = mock_create.call_args
    assert kwargs["model"] == "claude-3-haiku-20240307"


async def test_anthropic_model_falls_back_to_instance_default():
    provider, mock_create = _make_provider(model="claude-3-5-sonnet-latest")
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    assert kwargs["model"] == "claude-3-5-sonnet-latest"


# ---------------------------------------------------------------------------
# Return value handling
# ---------------------------------------------------------------------------


async def test_anthropic_returns_text_from_content_block():
    provider, mock_create = _make_provider()
    mock_create.return_value = _make_message("The answer is 42")
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    result = await provider.answer(prompt)
    assert result == "The answer is 42"


async def test_anthropic_returns_empty_string_when_content_empty():
    provider, mock_create = _make_provider()
    mock_create.return_value = _make_message(text=None)
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    result = await provider.answer(prompt)
    assert result == ""


async def test_anthropic_returns_empty_string_when_block_has_no_text():
    provider, mock_create = _make_provider()
    mock_create.return_value = _make_message("ignored", has_text_attr=False)
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    result = await provider.answer(prompt)
    assert result == ""


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


async def test_anthropic_raises_llm_provider_error_on_exception():
    provider, mock_create = _make_provider()
    mock_create.side_effect = Exception("api error")
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])

    with pytest.raises(LLMProviderError, match="Anthropic error"):
        await provider.answer(prompt)


async def test_anthropic_provider_error_chains_original():
    provider, mock_create = _make_provider()
    original = ConnectionError("refused")
    mock_create.side_effect = original
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])

    with pytest.raises(LLMProviderError) as exc_info:
        await provider.answer(prompt)

    assert exc_info.value.__cause__ is original
