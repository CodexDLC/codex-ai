"""Tests for the top-level public API and lazy-loader __getattr__ paths."""

import pytest


def test_lazy_import_openai_provider():
    from codex_ai import OpenAIProvider

    assert OpenAIProvider.__name__ == "OpenAIProvider"


def test_lazy_import_gemini_provider():
    from codex_ai import GeminiProvider

    assert GeminiProvider.__name__ == "GeminiProvider"


def test_lazy_import_anthropic_provider():
    from codex_ai import AnthropicProvider

    assert AnthropicProvider.__name__ == "AnthropicProvider"


def test_lazy_import_openrouter_provider():
    from codex_ai import OpenRouterProvider

    assert OpenRouterProvider.__name__ == "OpenRouterProvider"


def test_lazy_import_multi_provider():
    from codex_ai import MultiLLMProvider

    assert MultiLLMProvider.__name__ == "MultiLLMProvider"


def test_top_level_core_exports():
    import codex_ai

    assert hasattr(codex_ai, "LLMDispatcher")
    assert hasattr(codex_ai, "LLMRouter")
    assert hasattr(codex_ai, "LLMMessage")
    assert hasattr(codex_ai, "PromptResult")
    assert hasattr(codex_ai, "LLMProviderError")
    assert hasattr(codex_ai, "SyncLLMDispatcher")


def test_top_level_unknown_attr_raises():
    import codex_ai

    with pytest.raises(AttributeError):
        _ = codex_ai.NonExistentClass


def test_providers_module_unknown_attr_raises():
    import codex_ai.providers as providers

    with pytest.raises(AttributeError):
        _ = providers.NonExistentProvider
