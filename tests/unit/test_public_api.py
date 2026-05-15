"""Tests for the top-level public API and lazy-loader __getattr__ paths."""

import pytest


def test_lazy_import_openai_provider():
    from codex_ai import OpenAIProvider

    assert OpenAIProvider.__name__ == "OpenAIProvider"


def test_lazy_import_gemini_provider():
    from codex_ai import GeminiProvider

    assert GeminiProvider.__name__ == "GeminiProvider"


def test_top_level_core_exports():
    import codex_ai

    assert hasattr(codex_ai, "LLMDispatcher")
    assert hasattr(codex_ai, "ImageGenerationProvider")
    assert hasattr(codex_ai, "JsonGenerationProvider")
    assert hasattr(codex_ai, "LLMRouter")
    assert hasattr(codex_ai, "LLMMessage")
    assert hasattr(codex_ai, "PromptResult")
    assert hasattr(codex_ai, "LLMProviderError")
    assert hasattr(codex_ai, "SyncLLMDispatcher")
    assert hasattr(codex_ai, "TextGenerationProvider")


def test_top_level_unknown_attr_raises():
    import codex_ai

    with pytest.raises(AttributeError):
        _ = codex_ai.NonExistentClass


def test_providers_module_unknown_attr_raises():
    import codex_ai.providers as providers

    with pytest.raises(AttributeError):
        _ = providers.NonExistentProvider
