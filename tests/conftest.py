"""Shared test fixtures."""

import pytest

from codex_ai.core.protocol import LLMMessage, PromptResult


class MockProvider:
    """Minimal mock satisfying LLMProviderProtocol — no external deps."""

    def __init__(self, return_value: str = "mock response") -> None:
        self.return_value = return_value
        self.calls: list[tuple[PromptResult, dict]] = []

    async def answer(self, prompt: PromptResult, **kw) -> str:
        self.calls.append((prompt, kw))
        return self.return_value


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def mock_provider_factory():
    def _make(return_value: str = "mock response") -> MockProvider:
        return MockProvider(return_value=return_value)

    return _make


@pytest.fixture
def simple_prompt():
    return PromptResult(messages=[LLMMessage(role="user", content="Hello")])


@pytest.fixture
def full_prompt():
    return PromptResult(
        messages=[
            LLMMessage(role="user", content="Hi"),
            LLMMessage(role="assistant", content="Hello!"),
            LLMMessage(role="user", content="How are you?"),
        ],
        system="You are helpful",
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=256,
    )
