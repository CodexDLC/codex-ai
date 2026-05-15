"""
Integration tests for LLM providers.

Required env vars (set in .env or shell):
    OPENAI_API_KEY     — for OpenAI tests
    GOOGLE_API_KEY     — for Gemini tests

Run:
    pytest tests/integration/ -v --no-cov
"""

import os

import pytest

from codex_ai.core.dispatcher import LLMDispatcher
from codex_ai.core.protocol import LLMMessage, PromptResult
from codex_ai.core.router import LLMRouter


def _simple_prompt(text: str = "Say 'ok' and nothing else.") -> PromptResult:
    return PromptResult(messages=[LLMMessage(role="user", content=text)])


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_provider_real_call():
    from codex_ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
    result = await provider.answer(_simple_prompt(), max_tokens=10)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_dispatcher_end_to_end():
    from codex_ai.providers.openai import OpenAIProvider

    router = LLMRouter()

    @router.prompt("ping")
    async def build_ping(**kw) -> PromptResult:
        return _simple_prompt(kw.get("text", "Say 'pong'."))

    provider = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
    dispatcher = LLMDispatcher(provider=provider)
    dispatcher.include_router(router)

    result = await dispatcher.process("ping", text="Say 'pong'.")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
async def test_gemini_provider_real_call():
    from codex_ai.providers.gemini import GeminiProvider

    provider = GeminiProvider(api_key=os.environ["GOOGLE_API_KEY"])
    result = await provider.answer(_simple_prompt(), max_tokens=10)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
async def test_gemini_with_system_instruction():
    from codex_ai.providers.gemini import GeminiProvider

    provider = GeminiProvider(api_key=os.environ["GOOGLE_API_KEY"])
    prompt = PromptResult(
        messages=[LLMMessage(role="user", content="Hello")],
        system="Respond only with 'hi'.",
        max_tokens=10,
    )
    result = await provider.answer(prompt)
    assert isinstance(result, str)
    assert len(result) > 0
