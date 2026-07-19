"""Tests for the Responses API based OpenAIProvider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import OpenAIError
from pydantic import BaseModel

from codex_ai.core.exceptions import LLMProviderError
from codex_ai.core.protocol import (
    JsonGenerationProvider,
    LLMMessage,
    LLMProviderProtocol,
    PromptResult,
    TextGenerationProvider,
)
from codex_ai.providers.openai import OpenAIProvider


class LootItem(BaseModel):
    name: str
    rarity: str


def _response(text: str = "hello from openai"):
    return SimpleNamespace(output_text=text)


def _make_provider(
    model: str = "gpt-5.6-luna",
    *,
    store: bool = False,
) -> tuple[OpenAIProvider, AsyncMock, AsyncMock]:
    create = AsyncMock(return_value=_response())
    parse = AsyncMock()
    client = SimpleNamespace(responses=SimpleNamespace(create=create, parse=parse))
    provider = OpenAIProvider(client=client, model=model, store=store)
    return provider, create, parse


def test_openai_provider_satisfies_supported_protocols():
    provider, _, _ = _make_provider()

    assert isinstance(provider, LLMProviderProtocol)
    assert isinstance(provider, TextGenerationProvider)
    assert isinstance(provider, JsonGenerationProvider)


def test_openai_provider_builds_sdk_client_with_explicit_runtime_settings(monkeypatch):
    client = MagicMock()
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr("codex_ai.providers.openai.AsyncOpenAI", client_factory)

    provider = OpenAIProvider(
        api_key="test-key",  # pragma: allowlist secret
        model="gpt-5.6-luna",
        timeout=30.0,
        max_retries=4,
        base_url="https://example.test/v1",
    )

    client_factory.assert_called_once_with(
        api_key="test-key",  # pragma: allowlist secret
        timeout=30.0,
        max_retries=4,
        base_url="https://example.test/v1",
    )
    assert provider._client is client


async def test_generate_text_uses_responses_api_with_private_storage_default():
    provider, create, _ = _make_provider()

    result = await provider.generate_text("hello")

    create.assert_awaited_once_with(
        model="gpt-5.6-luna",
        input="hello",
        reasoning={"effort": "none"},
        store=False,
    )
    assert result == "hello from openai"


async def test_generate_text_maps_prompt_result_to_input_and_instructions():
    provider, create, _ = _make_provider()
    prompt = PromptResult(
        messages=[
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi"),
            LLMMessage(role="user", content="Bye"),
        ],
        system="Follow the product rules.",
    )

    await provider.generate_text(prompt)

    kwargs = create.await_args.kwargs
    assert kwargs["input"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Bye"},
    ]
    assert kwargs["instructions"] == "Follow the product rules."


async def test_explicit_model_override_wins_over_prompt_and_instance_defaults():
    provider, create, _ = _make_provider(model="instance-model")
    prompt = PromptResult(
        messages=[LLMMessage(role="user", content="Hi")],
        model="prompt-model",
    )

    await provider.generate_text(prompt, model="request-model")

    assert create.await_args.kwargs["model"] == "request-model"


async def test_prompt_model_wins_over_instance_default():
    provider, create, _ = _make_provider(model="instance-model")
    prompt = PromptResult(
        messages=[LLMMessage(role="user", content="Hi")],
        model="prompt-model",
    )

    await provider.generate_text(prompt)

    assert create.await_args.kwargs["model"] == "prompt-model"


async def test_prompt_sampling_and_token_limit_are_mapped_to_responses_fields():
    provider, create, _ = _make_provider()
    prompt = PromptResult(
        messages=[LLMMessage(role="user", content="Hi")],
        temperature=0.0,
        max_tokens=512,
    )

    await provider.generate_text(prompt)

    kwargs = create.await_args.kwargs
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_output_tokens"] == 512


async def test_explicit_response_options_override_prompt_defaults():
    provider, create, _ = _make_provider()
    prompt = PromptResult(
        messages=[LLMMessage(role="user", content="Hi")],
        temperature=0.7,
        max_tokens=512,
    )

    await provider.generate_text(
        prompt,
        temperature=0.2,
        max_output_tokens=128,
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
        previous_response_id="resp_previous",
        store=True,
    )

    kwargs = create.await_args.kwargs
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_output_tokens"] == 128
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["text"] == {"verbosity": "low"}
    assert kwargs["previous_response_id"] == "resp_previous"
    assert kwargs["store"] is True


async def test_legacy_max_tokens_request_option_is_rejected():
    provider, _, _ = _make_provider()

    with pytest.raises(TypeError, match="max_output_tokens"):
        await provider.generate_text("hello", max_tokens=100)


async def test_answer_uses_responses_api_and_returns_output_text():
    provider, create, _ = _make_provider()
    create.return_value = _response("Great answer!")
    prompt = PromptResult(messages=[LLMMessage(role="user", content="Hi")])

    result = await provider.answer(prompt)

    assert result == "Great answer!"


async def test_generate_text_returns_empty_string_when_output_text_is_empty():
    provider, create, _ = _make_provider()
    create.return_value = _response("")

    result = await provider.generate_text("hello")

    assert result == ""


async def test_generate_json_uses_sdk_structured_output_parsing():
    provider, _, parse = _make_provider()
    parsed = LootItem(name="Moon blade", rarity="epic")
    parse.return_value = SimpleNamespace(output_parsed=parsed)

    result = await provider.generate_json(
        "Create one loot item.",
        schema=LootItem,
        instructions="Return game data.",
    )

    parse.assert_awaited_once_with(
        model="gpt-5.6-luna",
        input="Create one loot item.",
        instructions="Return game data.",
        reasoning={"effort": "none"},
        store=False,
        text_format=LootItem,
    )
    assert result == parsed


async def test_generate_json_without_schema_uses_json_mode_and_decodes_output():
    provider, create, _ = _make_provider()
    create.return_value = _response('{"name":"Moon blade","rarity":"epic"}')

    result = await provider.generate_json("Create one loot item.")

    assert create.await_args.kwargs["text"] == {"format": {"type": "json_object"}}
    assert result == {"name": "Moon blade", "rarity": "epic"}


async def test_generate_json_rejects_missing_parsed_output():
    provider, _, parse = _make_provider()
    parse.return_value = SimpleNamespace(output_parsed=None)

    with pytest.raises(LLMProviderError, match="no parsed output"):
        await provider.generate_json("Create one loot item.", schema=LootItem)


async def test_stream_text_yields_only_output_text_deltas():
    async def events():
        yield SimpleNamespace(type="response.created")
        yield SimpleNamespace(type="response.output_text.delta", delta="Hello")
        yield SimpleNamespace(type="response.output_text.delta", delta=" world")
        yield SimpleNamespace(type="response.completed")

    provider, create, _ = _make_provider()
    create.return_value = events()

    chunks = [chunk async for chunk in provider.stream_text("hello")]

    assert chunks == ["Hello", " world"]
    assert create.await_args.kwargs["stream"] is True


async def test_openai_sdk_errors_are_wrapped_with_provider_context():
    provider, create, _ = _make_provider()
    original = OpenAIError("network failure")
    create.side_effect = original

    with pytest.raises(LLMProviderError, match="OpenAI Responses API error") as exc_info:
        await provider.generate_text("hello")

    assert exc_info.value.__cause__ is original


def test_provider_source_does_not_reference_chat_completions():
    from pathlib import Path

    source = Path("src/codex_ai/providers/openai.py").read_text(encoding="utf-8")

    assert "chat.completions" not in source
