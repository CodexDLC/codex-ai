"""Tests for OpenAIProvider."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codex_ai.core.exceptions import LLMProviderError
from codex_ai.core.protocol import LLMMessage, LLMProviderProtocol, PromptResult
from codex_ai.providers.openai import OpenAIProvider


def _make_completion(content: str | None = "hello from openai"):
    mock = MagicMock()
    if content is None:
        mock.choices = []
    else:
        choice = MagicMock()
        choice.message.content = content
        mock.choices = [choice]
    return mock


def _make_provider(model: str = "gpt-4o-mini") -> tuple[OpenAIProvider, AsyncMock]:
    provider = OpenAIProvider(api_key="test-key", model=model)
    mock_create = AsyncMock(return_value=_make_completion())
    provider._client = MagicMock()
    provider._client.chat.completions.create = mock_create
    return provider, mock_create


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_openai_provider_satisfies_protocol():
    provider = OpenAIProvider(api_key="x")
    assert isinstance(provider, LLMProviderProtocol)


# ---------------------------------------------------------------------------
# Message format conversion
# ---------------------------------------------------------------------------


async def test_openai_builds_user_messages():
    provider, mock_create = _make_provider()
    prompt = PromptResult(
        messages=[
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi"),
            LLMMessage(role="user", content="Bye"),
        ]
    )
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    messages = kwargs["messages"]
    # system was empty, so no prepended message
    assert len(messages) == 3
    assert messages[0] == {"role": "user", "content": "Hello"}
    assert messages[1] == {"role": "assistant", "content": "Hi"}
    assert messages[2] == {"role": "user", "content": "Bye"}


# ---------------------------------------------------------------------------
# system role branching: developer vs system
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected_role",
    [
        ("gpt-4o-mini", "developer"),   # contains "gpt-4o"
        ("gpt-4o", "developer"),
        ("o1-mini", "developer"),       # startswith "o1"
        ("o3", "developer"),            # startswith "o3"
        ("gpt-3.5-turbo", "system"),    # plain system role
        ("gpt-4", "system"),
    ],
)
async def test_openai_system_role_by_model(model, expected_role):
    provider, mock_create = _make_provider(model=model)
    prompt = PromptResult(
        messages=[LLMMessage(role="user", content="Hi")],
        system="You are helpful",
    )
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    first_msg = kwargs["messages"][0]
    assert first_msg["role"] == expected_role
    assert first_msg["content"] == "You are helpful"


async def test_openai_no_system_message_when_system_empty():
    provider, mock_create = _make_provider()
    prompt = PromptResult(
        messages=[LLMMessage(role="user", content="Hi")],
        system="",
    )
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    assert len(kwargs["messages"]) == 1
    assert kwargs["messages"][0]["role"] == "user"


# ---------------------------------------------------------------------------
# Parameter resolution: prompt > kw > instance default
# ---------------------------------------------------------------------------


async def test_openai_model_from_prompt():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")], model="gpt-4")
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    assert kwargs["model"] == "gpt-4"


async def test_openai_model_from_kw():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    await provider.answer(prompt, model="gpt-3.5-turbo")

    _, kwargs = mock_create.call_args
    assert kwargs["model"] == "gpt-3.5-turbo"


async def test_openai_model_falls_back_to_instance_default():
    provider, mock_create = _make_provider(model="gpt-4o-mini")
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    assert kwargs["model"] == "gpt-4o-mini"


async def test_openai_temperature_from_prompt():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")], temperature=0.7)
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    assert kwargs["temperature"] == 0.7


async def test_openai_temperature_from_kw():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    await provider.answer(prompt, temperature=0.3)

    _, kwargs = mock_create.call_args
    assert kwargs["temperature"] == 0.3


async def test_openai_temperature_not_passed_when_none():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    assert "temperature" not in kwargs


async def test_openai_max_tokens_from_prompt():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")], max_tokens=512)
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    assert kwargs["max_tokens"] == 512


async def test_openai_max_tokens_from_kw():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    await provider.answer(prompt, max_tokens=256)

    _, kwargs = mock_create.call_args
    assert kwargs["max_tokens"] == 256


async def test_openai_max_tokens_not_passed_when_none():
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    await provider.answer(prompt)

    _, kwargs = mock_create.call_args
    assert "max_tokens" not in kwargs


async def test_openai_known_kwargs_not_duplicated_in_create():
    """model/temperature/max_tokens from kw must not appear twice in create()."""
    provider, mock_create = _make_provider()
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    await provider.answer(prompt, model="gpt-4", temperature=0.5, max_tokens=100)

    _, kwargs = mock_create.call_args
    # Each appears exactly once (as direct kwarg, not inside **runtime_kw)
    assert kwargs["model"] == "gpt-4"
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 100


# ---------------------------------------------------------------------------
# Return value handling
# ---------------------------------------------------------------------------


async def test_openai_returns_content_string():
    provider, mock_create = _make_provider()
    mock_create.return_value = _make_completion("Great answer!")
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    result = await provider.answer(prompt)
    assert result == "Great answer!"


async def test_openai_returns_empty_string_when_choices_empty():
    provider, mock_create = _make_provider()
    mock_create.return_value = _make_completion(content=None)
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    result = await provider.answer(prompt)
    assert result == ""


async def test_openai_returns_empty_string_when_content_none():
    provider, mock_create = _make_provider()
    completion = MagicMock()
    choice = MagicMock()
    choice.message.content = None
    completion.choices = [choice]
    mock_create.return_value = completion

    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])
    result = await provider.answer(prompt)
    assert result == ""


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


async def test_openai_raises_llm_provider_error_on_exception():
    provider, mock_create = _make_provider()
    mock_create.side_effect = Exception("network failure")
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])

    with pytest.raises(LLMProviderError, match="OpenAI error"):
        await provider.answer(prompt)


async def test_openai_provider_error_chains_original():
    provider, mock_create = _make_provider()
    original = RuntimeError("timeout")
    mock_create.side_effect = original
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])

    with pytest.raises(LLMProviderError) as exc_info:
        await provider.answer(prompt)

    assert exc_info.value.__cause__ is original
