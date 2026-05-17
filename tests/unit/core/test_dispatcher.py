import pytest

from codex_ai.core.dispatcher import LLMDispatcher
from codex_ai.core.protocol import PromptResult
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


class ImageProvider:
    def __init__(self) -> None:
        self.calls = []

    async def answer(self, prompt: PromptResult, **kw) -> str:
        msg = "text fallback should not be used for image generation"
        raise AssertionError(msg)

    async def generate_image_bytes(
        self,
        prompt: str,
        *,
        model: str | None = None,
        response_mime_type: str = "image/webp",
        image_config: dict | None = None,
        **kwargs,
    ) -> tuple[bytes, str]:
        self.calls.append((prompt, model, response_mime_type, image_config, kwargs))
        return b"image-bytes", "image/png"

    async def generate_imagen_bytes(
        self,
        prompt: str,
        *,
        model: str | None = None,
        response_mime_type: str = "image/jpeg",
        **kwargs,
    ) -> tuple[bytes, str]:
        self.calls.append((prompt, model, response_mime_type, kwargs))
        return b"imagen-bytes", "image/jpeg"

    async def generate_text(self, prompt: PromptResult | str, *, model: str | None = None, **kwargs) -> str:
        self.calls.append((prompt, model, kwargs))
        return "direct text"

    async def generate_json(self, prompt: PromptResult | str, *, schema=None, model: str | None = None, **kwargs):
        self.calls.append((prompt, schema, model, kwargs))
        return {"ok": True}


async def test_dispatcher_generate_image_bytes_delegates_to_image_provider():
    provider = ImageProvider()
    dispatcher = LLMDispatcher(provider=provider)

    result = await dispatcher.generate_image_bytes(
        "draw a castle",
        model="gemini-image",
        response_mime_type="image/webp",
        image_config={"aspect_ratio": "1:1", "image_size": "4K"},
        seed=123,
    )

    assert result == (b"image-bytes", "image/png")
    assert provider.calls == [
        ("draw a castle", "gemini-image", "image/webp", {"aspect_ratio": "1:1", "image_size": "4K"}, {"seed": 123})
    ]


async def test_dispatcher_generate_image_bytes_raises_for_unsupported_provider(mock_provider):
    dispatcher = LLMDispatcher(provider=mock_provider)

    with pytest.raises(
        TypeError,
        match=r"Provider MockProvider does not support image generation; expected generate_image_bytes\(\.\.\.\)",
    ):
        await dispatcher.generate_image_bytes("draw a castle")

    assert mock_provider.calls == []


async def test_dispatcher_generate_imagen_bytes_delegates_to_imagen_provider():
    provider = ImageProvider()
    dispatcher = LLMDispatcher(provider=provider)

    result = await dispatcher.generate_imagen_bytes(
        "draw a castle",
        model="imagen-model",
        response_mime_type="image/jpeg",
        seed=123,
    )

    assert result == (b"imagen-bytes", "image/jpeg")
    assert provider.calls == [("draw a castle", "imagen-model", "image/jpeg", {"seed": 123})]


async def test_dispatcher_generate_imagen_bytes_raises_for_unsupported_provider(mock_provider):
    dispatcher = LLMDispatcher(provider=mock_provider)

    with pytest.raises(
        TypeError,
        match=r"Provider MockProvider does not support Imagen generation; expected generate_imagen_bytes\(\.\.\.\)",
    ):
        await dispatcher.generate_imagen_bytes("draw a castle")

    assert mock_provider.calls == []


async def test_dispatcher_generate_text_delegates_to_text_provider():
    provider = ImageProvider()
    dispatcher = LLMDispatcher(provider=provider)

    result = await dispatcher.generate_text("hello", model="gemini-text", temperature=0.2)

    assert result == "direct text"
    assert provider.calls == [("hello", "gemini-text", {"temperature": 0.2})]


async def test_dispatcher_generate_json_delegates_to_json_provider():
    provider = ImageProvider()
    dispatcher = LLMDispatcher(provider=provider)

    result = await dispatcher.generate_json("json please", schema=None, model="gemini-text")

    assert result == {"ok": True}
    assert provider.calls == [("json please", None, "gemini-text", {})]
