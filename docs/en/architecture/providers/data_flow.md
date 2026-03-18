# Providers — Data Flow

## Single Provider

```
LLMDispatcher
    │
    └── OpenAIProvider.answer(prompt)
            │
            ├── Convert LLMMessage list → OpenAI messages format
            ├── Insert system message if present
            ├── Extract model / temperature / max_tokens from prompt or kw
            └── AsyncOpenAI.chat.completions.create(...)
                        │
                        ▼
                   response text: str
```

## MultiLLMProvider Failover

```
LLMDispatcher
    │
    └── MultiLLMProvider.answer(prompt, provider="openai")
            │
            ├── Determine target: explicit / model-prefix / default
            ├── Try primary provider
            │       ├── Success → return text
            │       └── LLMProviderError → log error, try next
            ├── Try failover_list[0]
            │       ├── Success → return text
            │       └── LLMProviderError → try next
            └── All failed → raise last LLMProviderError
```

## Message Format Translation

| Field | OpenAI | Gemini | Anthropic |
|-------|--------|--------|-----------|
| `role: user` | `role: "user"` | `role: "user"` | `role: "user"` |
| `role: assistant` | `role: "assistant"` | `role: "model"` | `role: "assistant"` |
| `system` | inserted as `developer`/`system` message | `system_instruction` config | top-level `system` param |
