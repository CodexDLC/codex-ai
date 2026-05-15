# Провайдеры

## Назначение

`codex_ai.providers` содержит адаптеры для API, которые сейчас реально поддерживаются библиотекой: Gemini и OpenAI.

Gemini является основным направлением и дает прямые методы для текста, JSON и картинок:

```python
await gemini.generate_text(...)
await gemini.generate_json(...)
await gemini.generate_image_bytes(...)
```

OpenAI оставлен как текстовый провайдер с `generate_text(...)`.

## Архитектура

```
PromptResult/String
      │
      ├── GeminiProvider.generate_text(...)        -> str
      ├── GeminiProvider.generate_json(...)        -> dict | BaseModel
      ├── GeminiProvider.generate_image_bytes(...) -> tuple[bytes, str]
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
- Image generation возвращает raw bytes и фактический MIME type от Gemini.
- Anthropic, OpenRouter и multi-provider failover не являются активными API в этой alpha-линейке.
