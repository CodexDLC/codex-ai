# Провайдеры — Поток данных

## Одиночный провайдер

```
LLMDispatcher
    │
    └── OpenAIProvider.answer(prompt)
            │
            ├── Конвертация LLMMessage → формат OpenAI messages
            ├── Вставка system-сообщения если есть
            ├── Извлечение model / temperature / max_tokens из prompt или kw
            └── AsyncOpenAI.chat.completions.create(...)
                        │
                        ▼
                   текст ответа: str
```

## MultiLLMProvider — Failover

```
LLMDispatcher
    │
    └── MultiLLMProvider.answer(prompt, provider="openai")
            │
            ├── Определить цель: явный / префикс модели / по умолчанию
            ├── Попытка с primary-провайдером
            │       ├── Успех → вернуть текст
            │       └── LLMProviderError → залогировать, попробовать следующий
            ├── Попытка с failover_list[0]
            │       ├── Успех → вернуть текст
            │       └── LLMProviderError → попробовать следующий
            └── Все упали → поднять последний LLMProviderError
```

## Трансляция формата сообщений

| Поле | OpenAI | Gemini | Anthropic |
|------|--------|--------|-----------|
| `role: user` | `role: "user"` | `role: "user"` | `role: "user"` |
| `role: assistant` | `role: "assistant"` | `role: "model"` | `role: "assistant"` |
| `system` | сообщение `developer`/`system` | `system_instruction` в конфиге | top-level параметр `system` |
