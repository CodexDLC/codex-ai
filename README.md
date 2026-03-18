# codex-ai <!-- Type: LANDING -->

[![PyPI version](https://img.shields.io/pypi/v/codex-ai.svg)](https://pypi.org/project/codex-ai/)
[![Python](https://img.shields.io/pypi/pyversions/codex-ai.svg)](https://pypi.org/project/codex-ai/)
[![CI](https://github.com/codexdlc/codex-ai/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/codexdlc/codex-ai/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/codexdlc/codex-ai/blob/main/LICENSE)

Agnostic LLM abstraction layer for the **Codex platform**. Unified async interface for OpenAI, Google Gemini, Anthropic, and OpenRouter with prompt routing and failover support.

---

## Modules

| Module | Extra | Description |
| :--- | :--- | :--- |
| `core` | — | Dispatcher, Router, PromptResult — prompt orchestration layer |
| `providers` | `[openai]` `[gemini]` `[anthropic]` | Provider implementations for each LLM vendor |

## Install

```bash
pip install "codex-ai[openai,gemini]"
```

## Quick Start

Define prompts once — route to any provider:

```python
from codex_ai import LLMDispatcher, LLMRouter, PromptResult, OpenAIProvider, GeminiProvider
from codex_ai.providers import MultiLLMProvider

router = LLMRouter()

@router.prompt("chat")
async def build_chat(text: str, **kw) -> PromptResult:
    return PromptResult(
        messages=[{"role": "user", "content": text}],
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

## Documentation

Full docs with architecture, API reference, and data flow diagrams:

**[codexdlc.github.io/codex-ai](https://codexdlc.github.io/codex-ai/)**

## Part of the Codex ecosystem

[codex-core](https://github.com/codexdlc/codex-core) · **codex-ai** · [codex-platform](https://github.com/codexdlc/codex-platform) · [codex-bot](https://github.com/codexdlc/codex-bot)
