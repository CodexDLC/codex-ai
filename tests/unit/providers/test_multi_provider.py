"""Tests for MultiLLMProvider."""
import pytest

from codex_ai.core.exceptions import LLMProviderError
from codex_ai.core.protocol import LLMMessage, LLMProviderProtocol, PromptResult
from codex_ai.providers.multi import MultiLLMProvider


class _MockProvider:
    def __init__(self, return_value: str = "ok", raises: Exception | None = None) -> None:
        self.return_value = return_value
        self.raises = raises
        self.calls: list[dict] = []

    async def answer(self, prompt: PromptResult, **kw) -> str:
        self.calls.append(kw)
        if self.raises:
            raise self.raises
        return self.return_value


def _prompt(model: str | None = None) -> PromptResult:
    return PromptResult(messages=[LLMMessage(role="user", content="Hi")], model=model)


def _multi(default: str = "openai", failover: list[str] | None = None, **extra_providers):
    providers = {
        "openai": _MockProvider("openai response"),
        "gemini": _MockProvider("gemini response"),
        "anthropic": _MockProvider("anthropic response"),
        "openrouter": _MockProvider("openrouter response"),
        **extra_providers,
    }
    return providers, MultiLLMProvider(providers=providers, default=default, failover_list=failover)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_multi_raises_key_error_when_default_not_in_providers():
    with pytest.raises(KeyError, match="missing"):
        MultiLLMProvider(providers={"openai": _MockProvider()}, default="missing")


def test_multi_accepts_valid_default():
    _, multi = _multi(default="openai")
    assert multi is not None


def test_multi_failover_defaults_to_empty_list():
    _, multi = _multi()
    assert multi._failover == []


def test_multi_satisfies_protocol():
    _, multi = _multi()
    assert isinstance(multi, LLMProviderProtocol)


# ---------------------------------------------------------------------------
# Explicit provider kwarg
# ---------------------------------------------------------------------------


async def test_multi_uses_explicit_provider_kwarg():
    providers, multi = _multi()
    await multi.answer(_prompt(), provider="gemini")
    assert len(providers["gemini"].calls) == 1
    assert len(providers["openai"].calls) == 0


async def test_multi_strips_provider_from_kw_before_forwarding():
    providers, multi = _multi()
    await multi.answer(_prompt(), provider="openai")
    kw = providers["openai"].calls[0]
    assert "provider" not in kw


# ---------------------------------------------------------------------------
# Provider inference from model name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-4o", "openai"),
        ("gpt-3.5-turbo", "openai"),
        ("o1-mini", "openai"),
        ("o3", "openai"),
        ("gemini-2.0-flash", "gemini"),
        ("gemini-1.5-pro", "gemini"),
        ("claude-3-5-sonnet-latest", "anthropic"),
        ("claude-3-opus-20240229", "anthropic"),
        ("google/gemini-2.0", "openrouter"),
        ("meta-llama/llama-3", "openrouter"),
    ],
)
async def test_multi_infers_provider_from_model(model, expected):
    providers, multi = _multi()
    await multi.answer(_prompt(model=model))
    assert len(providers[expected].calls) == 1


async def test_multi_falls_back_to_default_when_model_unknown():
    providers, multi = _multi(default="gemini")
    await multi.answer(_prompt(model="unknown-model-xyz"))
    assert len(providers["gemini"].calls) == 1


async def test_multi_kw_model_used_when_prompt_model_none():
    providers, multi = _multi()
    await multi.answer(_prompt(model=None), model="gemini-1.5-flash")
    assert len(providers["gemini"].calls) == 1


async def test_multi_explicit_provider_wins_over_model_inference():
    providers, multi = _multi()
    await multi.answer(_prompt(model="gemini-2.0"), provider="openai")
    assert len(providers["openai"].calls) == 1
    assert len(providers["gemini"].calls) == 0


# ---------------------------------------------------------------------------
# Failover behavior
# ---------------------------------------------------------------------------


async def test_multi_failover_tries_next_on_llm_provider_error():
    providers = {
        "openai": _MockProvider(raises=LLMProviderError("openai failed")),
        "gemini": _MockProvider("gemini ok"),
    }
    multi = MultiLLMProvider(providers=providers, default="openai", failover_list=["gemini"])
    result = await multi.answer(_prompt())
    assert result == "gemini ok"
    assert len(providers["gemini"].calls) == 1


async def test_multi_returns_first_success_without_calling_failover():
    providers = {
        "openai": _MockProvider("success"),
        "gemini": _MockProvider("backup"),
    }
    multi = MultiLLMProvider(providers=providers, default="openai", failover_list=["gemini"])
    result = await multi.answer(_prompt())
    assert result == "success"
    assert len(providers["gemini"].calls) == 0


async def test_multi_raises_last_exception_when_all_fail():
    err1 = LLMProviderError("first failed")
    err2 = LLMProviderError("second failed")
    providers = {
        "openai": _MockProvider(raises=err1),
        "gemini": _MockProvider(raises=err2),
    }
    multi = MultiLLMProvider(providers=providers, default="openai", failover_list=["gemini"])

    with pytest.raises(LLMProviderError) as exc_info:
        await multi.answer(_prompt())

    assert exc_info.value is err2


async def test_multi_deduplication_same_name_not_called_twice():
    providers = {
        "openai": _MockProvider(raises=LLMProviderError("fail")),
        "gemini": _MockProvider("ok"),
    }
    multi = MultiLLMProvider(
        providers=providers,
        default="openai",
        failover_list=["openai", "gemini"],  # "openai" repeated
    )
    result = await multi.answer(_prompt())
    assert result == "ok"
    # openai was tried once, not twice
    assert len(providers["openai"].calls) == 1


async def test_multi_skips_unknown_provider_name_in_failover():
    providers = {
        "openai": _MockProvider(raises=LLMProviderError("fail")),
        "gemini": _MockProvider("ok"),
    }
    multi = MultiLLMProvider(
        providers=providers,
        default="openai",
        failover_list=["unknown_provider", "gemini"],
    )
    result = await multi.answer(_prompt())
    assert result == "ok"


async def test_multi_raises_when_no_valid_providers():
    providers = {"openai": _MockProvider()}
    multi = MultiLLMProvider(providers=providers, default="openai")
    # Override with unknown name — simulate all providers missing
    multi._providers = {}

    with pytest.raises(LLMProviderError, match="no valid providers"):
        await multi.answer(_prompt())


async def test_multi_failover_multiple_steps():
    err = LLMProviderError("fail")
    providers = {
        "a": _MockProvider(raises=err),
        "b": _MockProvider(raises=err),
        "c": _MockProvider("c ok"),
    }
    multi = MultiLLMProvider(providers=providers, default="a", failover_list=["b", "c"])
    result = await multi.answer(_prompt())
    assert result == "c ok"
    assert len(providers["a"].calls) == 1
    assert len(providers["b"].calls) == 1
    assert len(providers["c"].calls) == 1
