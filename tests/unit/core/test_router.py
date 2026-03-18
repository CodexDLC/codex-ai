from codex_ai.core.protocol import PromptResult
from codex_ai.core.router import LLMRouter


def test_router_empty_on_init():
    router = LLMRouter()
    assert router.builders == {}


def test_router_registers_builder():
    router = LLMRouter()

    @router.prompt("chat")
    async def build_chat(**kw) -> PromptResult:
        return PromptResult(messages=[])

    assert "chat" in router.builders


def test_router_builder_is_original_function():
    router = LLMRouter()

    async def build_chat(**kw) -> PromptResult:
        return PromptResult(messages=[])

    result = router.prompt("chat")(build_chat)
    assert result is build_chat
    assert router.builders["chat"] is build_chat


def test_router_multiple_modes():
    router = LLMRouter()

    @router.prompt("chat")
    async def build_chat(**kw) -> PromptResult:
        return PromptResult(messages=[])

    @router.prompt("summarize")
    async def build_summarize(**kw) -> PromptResult:
        return PromptResult(messages=[])

    @router.prompt("translate")
    async def build_translate(**kw) -> PromptResult:
        return PromptResult(messages=[])

    assert set(router.builders.keys()) == {"chat", "summarize", "translate"}


def test_router_overwrite_same_mode():
    router = LLMRouter()

    @router.prompt("chat")
    async def first(**kw) -> PromptResult:
        return PromptResult(messages=[])

    @router.prompt("chat")
    async def second(**kw) -> PromptResult:
        return PromptResult(messages=[])

    assert router.builders["chat"] is second


def test_router_builders_property_is_dict():
    router = LLMRouter()
    assert isinstance(router.builders, dict)
