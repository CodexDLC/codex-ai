# Providers

## Purpose

`codex_ai.providers` contains concrete provider adapters for the APIs currently supported by the library: Gemini and OpenAI.

Gemini is the primary target and exposes direct methods for text, JSON, and image generation:

```python
await gemini.generate_text(...)
await gemini.generate_json(...)
await gemini.generate_image_bytes(...)
```

OpenAI is kept as a text provider with the same `generate_text(...)` convenience.

## Architecture

```
PromptResult/String
      │
      ├── GeminiProvider.generate_text(...)        -> str
      ├── GeminiProvider.generate_json(...)        -> dict | BaseModel
      ├── GeminiProvider.generate_image_bytes(...) -> tuple[bytes, str]
      └── OpenAIProvider.generate_text(...)        -> str
```

The legacy router pipeline remains available:

```
LLMRouter builder -> PromptResult -> LLMDispatcher.process() -> provider.answer() -> str
```

`answer()` is a compatibility wrapper for text generation.

## Key Components

| Component | Class | SDK | Default Model |
|-----------|-------|-----|---------------|
| `gemini.py` | `GeminiProvider` | `google-genai` | `gemini-2.5-flash-lite` |
| `openai.py` | `OpenAIProvider` | `openai` | `gpt-4o-mini` |

## Key Design Decisions

- Gemini-specific capabilities are represented directly instead of being hidden behind a broad universal abstraction.
- JSON generation uses provider-native JSON configuration and still validates locally with `json.loads` and optional Pydantic models.
- Image generation returns raw bytes plus the actual MIME type returned by Gemini.
- Anthropic, OpenRouter, and multi-provider failover are not active APIs in this alpha line.
