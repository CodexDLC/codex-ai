"""Tests for OpenRouterProvider."""

from unittest.mock import patch

from codex_ai.core.protocol import LLMProviderProtocol
from codex_ai.providers.openai import OpenAIProvider
from codex_ai.providers.openrouter import _DEFAULT_MODEL, _OPENROUTER_BASE_URL, OpenRouterProvider

# ---------------------------------------------------------------------------
# Inheritance and protocol
# ---------------------------------------------------------------------------


def test_openrouter_is_subclass_of_openai_provider():
    assert issubclass(OpenRouterProvider, OpenAIProvider)


def test_openrouter_satisfies_llm_provider_protocol():
    with patch("codex_ai.providers.openai.AsyncOpenAI"), patch("openai.AsyncOpenAI"):
        provider = OpenRouterProvider(api_key="dummy-key-for-test")  # pragma: allowlist secret
    assert isinstance(provider, LLMProviderProtocol)


# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------


def test_openrouter_default_model_is_free_gemini():
    assert "free" in _DEFAULT_MODEL or "gemini" in _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Constructor: base_url and headers
# ---------------------------------------------------------------------------


def test_openrouter_uses_openrouter_base_url():
    with patch("codex_ai.providers.openai.AsyncOpenAI"), patch("openai.AsyncOpenAI") as mock_cls:
        OpenRouterProvider(api_key="dummy-key-for-test")  # pragma: allowlist secret

    mock_cls.assert_called_once()
    _, kwargs = mock_cls.call_args
    assert kwargs["base_url"] == _OPENROUTER_BASE_URL


def test_openrouter_no_headers_when_none():
    with patch("codex_ai.providers.openai.AsyncOpenAI"), patch("openai.AsyncOpenAI") as mock_cls:
        OpenRouterProvider(api_key="dummy-key-for-test", site_url=None, site_name=None)  # pragma: allowlist secret

    _, kwargs = mock_cls.call_args
    assert kwargs["default_headers"] == {}


def test_openrouter_site_url_sets_http_referer():
    with patch("codex_ai.providers.openai.AsyncOpenAI"), patch("openai.AsyncOpenAI") as mock_cls:
        OpenRouterProvider(api_key="dummy-key-for-test", site_url="https://example.com")  # pragma: allowlist secret

    _, kwargs = mock_cls.call_args
    assert kwargs["default_headers"]["HTTP-Referer"] == "https://example.com"


def test_openrouter_site_name_sets_x_title():
    with patch("codex_ai.providers.openai.AsyncOpenAI"), patch("openai.AsyncOpenAI") as mock_cls:
        OpenRouterProvider(api_key="dummy-key-for-test", site_name="My App")  # pragma: allowlist secret

    _, kwargs = mock_cls.call_args
    assert kwargs["default_headers"]["X-Title"] == "My App"


def test_openrouter_both_headers_set():
    with patch("codex_ai.providers.openai.AsyncOpenAI"), patch("openai.AsyncOpenAI") as mock_cls:
        OpenRouterProvider(
            api_key="dummy-key-for-test", site_url="https://x.com", site_name="X"
        )  # pragma: allowlist secret

    _, kwargs = mock_cls.call_args
    assert kwargs["default_headers"]["HTTP-Referer"] == "https://x.com"
    assert kwargs["default_headers"]["X-Title"] == "X"


def test_openrouter_api_key_forwarded():
    with patch("codex_ai.providers.openai.AsyncOpenAI"), patch("openai.AsyncOpenAI") as mock_cls:
        OpenRouterProvider(api_key="dummy-secret-key")  # pragma: allowlist secret

    _, kwargs = mock_cls.call_args
    assert kwargs["api_key"] == "dummy-secret-key"
