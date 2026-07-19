# Providers

## Purpose

`codex_ai.providers` contains concrete SDK adapters, not a broad interchangeable provider framework. Gemini is the product focus; OpenAI is a separate modern Responses API adapter.

Gemini is the primary target and exposes direct methods for text, JSON, and image generation:

```python
await gemini.generate_text(...)
await gemini.generate_json(...)
await gemini.generate_image_bytes(...)
await gemini.generate_imagen_bytes(...)
```

OpenAI exposes direct text, structured JSON, and typed streaming helpers:

```python
await openai.generate_text(...)
await openai.generate_json(..., schema=MyModel)
async for chunk in openai.stream_text(...):
    ...
```

## Architecture

```
PromptResult/String
      │
      ├── GeminiProvider.generate_text(...)        -> str
      ├── GeminiProvider.generate_json(...)        -> dict | BaseModel
      ├── GeminiProvider.generate_image_bytes(...) -> tuple[bytes, str]
      ├── GeminiProvider.generate_imagen_bytes(...) -> tuple[bytes, str]
      ├── OpenAIProvider.generate_text(...)        -> str
      ├── OpenAIProvider.generate_json(...)        -> dict | BaseModel
      └── OpenAIProvider.stream_text(...)          -> AsyncIterator[str]
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
| `openai.py` | `OpenAIProvider` | `openai` 2.x | `gpt-5.6-luna` |

## Key Design Decisions

- Gemini-specific capabilities are represented directly instead of being hidden behind a broad universal abstraction.
- JSON generation uses provider-native JSON configuration and still validates locally with `json.loads` and optional Pydantic models.
- OpenAI uses the Responses API exclusively; the old Chat Completions path is not retained.
- OpenAI requests default to `store=False` and `reasoning={"effort": "none"}`.
- OpenAI Pydantic schemas use the SDK's native `responses.parse()` structured-output path.
- The OpenAI SDK client is injectable behind a small Responses API port for deterministic tests and custom transports.
- Gemini image generation and Imagen generation are separate explicit methods because they use different SDK calls.
- `generate_image_bytes()` uses Gemini `generate_content` with image modality. Its `response_mime_type` is only a preferred/fallback MIME type, and Gemini image controls are passed through `image_config`.
- `generate_image_bytes()` can also accept `input_images=[ImageInput(...)]` for image reference/edit workflows; without `input_images`, the prompt-only path is unchanged.
- `generate_image_bytes()` retries a rejected `image_config={"image_size": "4K"}` request once as `2K`.
- `generate_imagen_bytes()` uses Imagen `generate_images` and passes the requested MIME as `output_mime_type`.
- Anthropic, OpenRouter, and multi-provider failover are not active APIs in this line.
