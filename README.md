# codex-ai <!-- Type: LANDING -->

[![PyPI version](https://img.shields.io/pypi/v/codex-ai.svg)](https://pypi.org/project/codex-ai/)
[![Python](https://img.shields.io/pypi/pyversions/codex-ai.svg)](https://pypi.org/project/codex-ai/)
[![CI](https://github.com/codexdlc/codex-ai/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/codexdlc/codex-ai/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/codexdlc/codex-ai/blob/main/LICENSE)

Agnostic LLM abstraction layer for the Codex ecosystem. It gives you one async prompt pipeline for OpenAI, Gemini, Anthropic, and OpenRouter, with routing and provider failover built in.

---

## Install

```bash
# Core only
pip install codex-ai

# Provider extras
pip install "codex-ai[openai]"
pip install "codex-ai[gemini]"
pip install "codex-ai[anthropic]"
pip install "codex-ai[openai,gemini,anthropic]"
```

Requires Python 3.12 or newer.

## Quick Start

Define a prompt once and dispatch it through any provider:

```python
from codex_ai import LLMDispatcher, LLMMessage, LLMRouter, PromptResult, OpenAIProvider, GeminiProvider
from codex_ai.providers import MultiLLMProvider

router = LLMRouter()

@router.prompt("chat")
async def build_chat(text: str, **kw) -> PromptResult:
    return PromptResult(
        messages=[LLMMessage(role="user", content=text)],
        system="You are a helpful assistant.",
    )

# Multiple providers with automatic failover: OpenAI → Gemini
provider = MultiLLMProvider(
    providers={
        "openai": OpenAIProvider(api_key="sk-..."),
        "gemini": GeminiProvider(api_key="AIza..."),
    },
    default="openai",
    failover_list=["gemini"],  # if OpenAI fails — try Gemini
)

dispatcher = LLMDispatcher(provider=provider)
dispatcher.include_router(router)

# Uses default (OpenAI), falls back to Gemini on error
response = await dispatcher.process("chat", text="Hello!")

# Or pick a provider explicitly
response = await dispatcher.process("chat", text="Hello!", provider="gemini")
```

## Modules

| Module | Extra | Description |
| :--- | :--- | :--- |
| `codex_ai.core` | — | Dispatcher, router, protocol types, sync wrapper, and shared exception contract |
| `codex_ai.providers.openai` | `[openai]` | OpenAI Chat Completions provider |
| `codex_ai.providers.gemini` | `[gemini]` | Google Gemini provider via `google-genai` |
| `codex_ai.providers.anthropic_` | `[anthropic]` | Anthropic Claude provider |
| `codex_ai.providers.openrouter` | `[openai]` | OpenRouter provider built on the OpenAI-compatible SDK |
| `codex_ai.providers.multi` | — | Multi-provider dispatcher with failover and model-based inference |

## Development

```bash
uv sync --extra dev
uv run pytest
uv run mypy src/
uv run pre-commit run --all-files
uv build --no-sources
```

## Documentation

Full docs with architecture, API reference, and data flow diagrams:

**[codexdlc.github.io/codex-ai](https://codexdlc.github.io/codex-ai/)**

## Part of the Codex ecosystem

| Package | Role |
| :--- | :--- |
| [codex-core](https://github.com/codexdlc/codex-core) | Foundation — immutable DTOs, PII masking, env settings |
| [codex-platform](https://github.com/codexdlc/codex-platform) | Infrastructure — Redis, Streams, ARQ workers, Notifications |
| **codex-ai** | LLM layer — unified async interface for OpenAI, Gemini, Anthropic |
| [codex-services](https://github.com/codexdlc/codex-services) | Business logic — Booking engine, CRM, Calendar |

Each library is **fully standalone** — install only what your project needs.
Together they form the backbone of **[codex-bot](https://github.com/codexdlc/codex-bot)**
(Telegram AI-agent infrastructure built on aiogram) and
**[codex-django](https://github.com/codexdlc/codex-django)** (Django integration layer).
