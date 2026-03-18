from unittest.mock import patch

import pytest

from codex_ai.core.dispatcher import LLMDispatcher
from codex_ai.core.protocol import PromptResult
from codex_ai.core.router import LLMRouter
from codex_ai.core.sync import SyncLLMDispatcher


def _make_sync_dispatcher(mock_provider, return_prompt: PromptResult):
    router = LLMRouter()

    @router.prompt("chat")
    async def builder(**kw) -> PromptResult:
        return return_prompt

    dispatcher = LLMDispatcher(provider=mock_provider)
    dispatcher.include_router(router)
    return SyncLLMDispatcher(dispatcher)


def test_sync_dispatcher_returns_string(mock_provider, simple_prompt):
    sync = _make_sync_dispatcher(mock_provider, simple_prompt)
    result = sync.process("chat")
    assert isinstance(result, str)
    assert result == "mock response"


def test_sync_dispatcher_forwards_kwargs(mock_provider, simple_prompt):
    sync = _make_sync_dispatcher(mock_provider, simple_prompt)
    sync.process("chat", temperature=0.5)

    _, kw = mock_provider.calls[0]
    assert kw["temperature"] == 0.5


def test_sync_dispatcher_raises_key_error_for_unknown_mode(mock_provider):
    dispatcher = LLMDispatcher(provider=mock_provider)
    sync = SyncLLMDispatcher(dispatcher)

    with pytest.raises(KeyError):
        sync.process("nonexistent")


def test_sync_dispatcher_uses_asyncio_run(mock_provider, simple_prompt):
    sync = _make_sync_dispatcher(mock_provider, simple_prompt)

    with patch("codex_ai.core.sync.asyncio.run") as mock_run:
        mock_run.return_value = "patched"
        result = sync.process("chat")

    mock_run.assert_called_once()
    assert result == "patched"
