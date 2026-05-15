# codex-ai <!-- Type: LANDING -->

[![PyPI version](https://img.shields.io/pypi/v/codex-ai.svg)](https://pypi.org/project/codex-ai/)
[![Python](https://img.shields.io/pypi/pyversions/codex-ai.svg)](https://pypi.org/project/codex-ai/)
[![CI](https://github.com/codexdlc/codex-ai/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/codexdlc/codex-ai/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/codexdlc/codex-ai/blob/main/LICENSE)

Gemini-first and OpenAI provider helpers for the Codex ecosystem. The library keeps the legacy prompt router for text generation, and exposes direct provider methods for practical Gemini workflows.

## Install

```bash
pip install codex-ai
pip install "codex-ai[gemini]"
pip install "codex-ai[openai]"
pip install "codex-ai[openai,gemini]"
```

Requires Python 3.12 or newer.

## Gemini Direct API

```python
from pydantic import BaseModel

from codex_ai import GeminiProvider


class LootItem(BaseModel):
    name: str
    power: int


gemini = GeminiProvider(api_key="AIza...")

text = await gemini.generate_text("Write one short tavern rumor.")
loot = await gemini.generate_json("Create one loot item.", schema=LootItem)
image_bytes, content_type = await gemini.generate_image_bytes(
    "A fantasy clan banner, game icon style.",
    response_mime_type="image/webp",
)
```

`answer(prompt)` remains available as a compatibility wrapper for text generation.

## Router Pipeline

```python
from codex_ai import GeminiProvider, LLMDispatcher, LLMMessage, LLMRouter, PromptResult

router = LLMRouter()


@router.prompt("chat")
async def build_chat(text: str, **kw) -> PromptResult:
    return PromptResult(
        messages=[LLMMessage(role="user", content=text)],
        system="You are a helpful assistant.",
    )


dispatcher = LLMDispatcher(provider=GeminiProvider(api_key="AIza..."))
dispatcher.include_router(router)

response = await dispatcher.process("chat", text="Hello!")
```

## Modules

| Module | Extra | Description |
| :--- | :--- | :--- |
| `codex_ai.core` | - | Dispatcher, router, protocol types, sync wrapper, and shared exception contract |
| `codex_ai.providers.gemini` | `[gemini]` | Google Gemini text, JSON, and image generation via `google-genai` |
| `codex_ai.providers.openai` | `[openai]` | OpenAI Chat Completions text provider |

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
