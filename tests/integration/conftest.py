"""Integration test fixtures — requires real API keys via env vars."""
import os

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


def skip_if_no_key(env_var: str, provider: str):
    """Return a pytest.mark.skipif for missing API key."""
    return pytest.mark.skipif(
        not os.getenv(env_var),
        reason=f"{provider} integration test requires {env_var} env var",
    )


openai_key = skip_if_no_key("OPENAI_API_KEY", "OpenAI")
anthropic_key = skip_if_no_key("ANTHROPIC_API_KEY", "Anthropic")
gemini_key = skip_if_no_key("GOOGLE_API_KEY", "Gemini")
