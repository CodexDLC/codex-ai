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
    "Square tactical dark fantasy ruined capital city map, no labels.",
    model="gemini-3-pro-image-preview",
    response_mime_type="image/png",
    image_config={"aspect_ratio": "1:1", "image_size": "4K"},
)

imagen_bytes, imagen_content_type = await gemini.generate_imagen_bytes(
    "A fantasy clan banner, game icon style.",
    response_mime_type="image/jpeg",
)
```

`answer(prompt)` remains available as a compatibility wrapper for text generation.

`generate_image_bytes()` targets Gemini image models through `generate_content` and treats
`response_mime_type` as a preferred/fallback MIME type. It does not pass image MIME values
to Gemini's text `response_mime_type` config field. Pass Gemini image controls such as
`aspect_ratio` and `image_size` with `image_config`; if a `4K` request is rejected,
the Gemini provider retries once with `2K`. Use `generate_imagen_bytes()` for
Imagen models; that path uses `generate_images` and passes the requested MIME as
`output_mime_type`.

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
| `codex_ai.providers.gemini` | `[gemini]` | Google Gemini text, JSON, Gemini image, and Imagen generation via pinned `google-genai` |
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
