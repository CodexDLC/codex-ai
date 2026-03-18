import pytest

from codex_ai.core.dispatcher import LLMDispatcher
from codex_ai.core.protocol import LLMMessage, PromptResult
from codex_ai.core.router import LLMRouter


def make_router(mode: str, result: PromptResult):
    """Helper: create a router with a single builder that returns `result`."""
    router = LLMRouter()

    @router.prompt(mode)
    async def builder(**kw) -> PromptResult:
        builder.last_kw = kw
        return result

    return router, builder


async def test_dispatcher_process_returns_provider_response(mock_provider, simple_prompt):
    router, _ = make_router("chat", simple_prompt)
    dispatcher = LLMDispatcher(provider=mock_provider)
    dispatcher.include_router(router)

    result = await dispatcher.process("chat")
    assert result == "mock response"


async def test_dispatcher_process_calls_provider_with_prompt(mock_provider, simple_prompt):
    router, _ = make_router("chat", simple_prompt)
    dispatcher = LLMDispatcher(provider=mock_provider)
    dispatcher.include_router(router)

    await dispatcher.process("chat")

    assert len(mock_provider.calls) == 1
    received_prompt, _ = mock_provider.calls[0]
    assert received_prompt is simple_prompt


async def test_dispatcher_unknown_mode_raises_key_error(mock_provider):
    dispatcher = LLMDispatcher(provider=mock_provider)

    with pytest.raises(KeyError, match="nonexistent"):
        await dispatcher.process("nonexistent")


async def test_dispatcher_kwargs_forwarded_to_builder(mock_provider, simple_prompt):
    router, builder = make_router("chat", simple_prompt)
    dispatcher = LLMDispatcher(provider=mock_provider)
    dispatcher.include_router(router)

    await dispatcher.process("chat", text="Hello", temperature=0.9)

    assert builder.last_kw["text"] == "Hello"
    assert builder.last_kw["temperature"] == 0.9


async def test_dispatcher_kwargs_forwarded_to_provider(mock_provider, simple_prompt):
    router, _ = make_router("chat", simple_prompt)
    dispatcher = LLMDispatcher(provider=mock_provider)
    dispatcher.include_router(router)

    await dispatcher.process("chat", temperature=0.7)

    _, kw = mock_provider.calls[0]
    assert kw["temperature"] == 0.7


async def test_dispatcher_include_router_merges_builders(mock_provider, simple_prompt):
    router_a = LLMRouter()

    @router_a.prompt("chat")
    async def build_a(**kw) -> PromptResult:
        return simple_prompt

    router_b = LLMRouter()

    @router_b.prompt("summarize")
    async def build_b(**kw) -> PromptResult:
        return simple_prompt

    dispatcher = LLMDispatcher(provider=mock_provider)
    dispatcher.include_router(router_a)
    dispatcher.include_router(router_b)

    await dispatcher.process("chat")
    await dispatcher.process("summarize")
    assert len(mock_provider.calls) == 2


async def test_dispatcher_include_router_overwrites_duplicate_mode(mock_provider, simple_prompt):
    router_a = LLMRouter()
    router_b = LLMRouter()
    called = []

    @router_a.prompt("chat")
    async def build_a(**kw) -> PromptResult:
        called.append("a")
        return simple_prompt

    @router_b.prompt("chat")
    async def build_b(**kw) -> PromptResult:
        called.append("b")
        return simple_prompt

    dispatcher = LLMDispatcher(provider=mock_provider)
    dispatcher.include_router(router_a)
    dispatcher.include_router(router_b)

    await dispatcher.process("chat")
    assert called == ["b"]
