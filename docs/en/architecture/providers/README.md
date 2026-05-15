# Providers

## Purpose

`codex_ai.providers` contains concrete provider adapters for the APIs currently supported by the library: Gemini and OpenAI.

Gemini is the primary target and exposes direct methods for text, JSON, and image generation:

```python
await gemini.generate_text(...)
await gemini.generate_json(...)
await gemini.generate_image_bytes(...)
await gemini.generate_imagen_bytes(...)
```

OpenAI is kept as a text provider with the same `generate_text(...)` convenience.

## Architecture

```
PromptResult/String
      │
      ├── GeminiProvider.generate_text(...)        -> str
      ├── GeminiProvider.generate_json(...)        -> dict | BaseModel
      ├── GeminiProvider.generate_image_bytes(...) -> tuple[bytes, str]
      ├── GeminiProvider.generate_imagen_bytes(...) -> tuple[bytes, str]
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
- Gemini image generation and Imagen generation are separate explicit methods because they use different SDK calls.
- `generate_image_bytes()` uses Gemini `generate_content` with image modality. Its `response_mime_type` is only a preferred/fallback MIME type.
- `generate_imagen_bytes()` uses Imagen `generate_images` and passes the requested MIME as `output_mime_type`.
- Anthropic, OpenRouter, and multi-provider failover are not active APIs in this alpha line.
