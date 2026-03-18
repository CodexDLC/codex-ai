import pytest

from codex_ai.core.exceptions import LLMProviderError


def test_llm_provider_error_is_exception_subclass():
    assert issubclass(LLMProviderError, Exception)


def test_llm_provider_error_message_preserved():
    exc = LLMProviderError("something went wrong")
    assert "something went wrong" in str(exc)


def test_llm_provider_error_can_chain():
    cause = ValueError("root cause")
    try:
        raise LLMProviderError("wrapper") from cause
    except LLMProviderError as exc:
        assert exc.__cause__ is cause


def test_llm_provider_error_can_be_raised_and_caught():
    with pytest.raises(LLMProviderError, match="test error"):
        raise LLMProviderError("test error")
