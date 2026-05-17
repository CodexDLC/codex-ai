# Провайдеры

## Назначение

`codex_ai.providers` содержит конкретные SDK-адаптеры, а не широкий interchangeable provider framework. Фокус продукта — Gemini; OpenAI сохранен только для текстовой генерации.

Gemini является основным направлением и дает прямые методы для текста, JSON и картинок:

```python
await gemini.generate_text(...)
await gemini.generate_json(...)
await gemini.generate_image_bytes(...)
await gemini.generate_imagen_bytes(...)
```

OpenAI оставлен как text-only адаптер с `generate_text(...)`.

## Архитектура

```
PromptResult/String
      │
      ├── GeminiProvider.generate_text(...)        -> str
      ├── GeminiProvider.generate_json(...)        -> dict | BaseModel
      ├── GeminiProvider.generate_image_bytes(...) -> tuple[bytes, str]
      ├── GeminiProvider.generate_imagen_bytes(...) -> tuple[bytes, str]
      └── OpenAIProvider.generate_text(...)        -> str
```

Старый router pipeline остается доступным:

```
LLMRouter builder -> PromptResult -> LLMDispatcher.process() -> provider.answer() -> str
```

`answer()` является compatibility-wrapper для текстовой генерации.

## Ключевые компоненты

| Компонент | Класс | SDK | Модель по умолчанию |
|-----------|-------|-----|---------------------|
| `gemini.py` | `GeminiProvider` | `google-genai` | `gemini-2.5-flash-lite` |
| `openai.py` | `OpenAIProvider` | `openai` | `gpt-4o-mini` |

## Ключевые решения

- Возможности Gemini представлены напрямую, без широкой универсальной абстракции.
- JSON generation использует native JSON config провайдера и локальную проверку через `json.loads` и optional Pydantic schema.
- Gemini image generation и Imagen generation разведены в отдельные явные методы, потому что они используют разные SDK calls.
- `generate_image_bytes()` использует Gemini `generate_content` с image modality. Его `response_mime_type` является только preferred/fallback MIME type, а Gemini image controls передаются через `image_config`.
- `generate_image_bytes()` один раз повторяет отклоненный `image_config={"image_size": "4K"}` запрос как `2K`.
- `generate_imagen_bytes()` использует Imagen `generate_images` и передает requested MIME как `output_mime_type`.
- Anthropic, OpenRouter и multi-provider failover не являются активными API в этой линейке.
