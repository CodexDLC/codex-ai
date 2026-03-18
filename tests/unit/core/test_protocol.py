import pytest
from pydantic import ValidationError

from codex_ai.core.protocol import LLMMessage, LLMProviderProtocol, PromptResult

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
