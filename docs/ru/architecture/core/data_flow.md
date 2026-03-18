# Core — Поток данных

## Жизненный цикл запроса

```
1. Приложение вызывает:
   dispatcher.process("chat", text="Привет!")

2. LLMDispatcher ищет билдер:
   builder = self._builders["chat"]  # зарегистрирован через @router.prompt

3. Билдер строит PromptResult:
   prompt = await builder(text="Привет!")
   # → PromptResult(messages=[LLMMessage(role="user", content="Привет!")])

4. Диспетчер вызывает провайдер:
   response = await self._provider.answer(prompt, **kw)

5. Провайдер возвращает текст:
   "Привет! Чем могу помочь?"
```

## Взаимодействие компонентов

```
Приложение
    │
    ▼
LLMDispatcher.process(mode, **kw)
    │
    ├── LLMRouter.builders[mode] ──► PromptBuilder(**kw)
    │                                       │
    │                               PromptResult (frozen)
    │                                       │
    └── LLMProviderProtocol.answer(prompt) ◄┘
                │
                ▼
           Ответ: str
```

## Поля PromptResult

| Поле | Тип | Описание |
|------|-----|----------|
| `messages` | `list[LLMMessage]` | Упорядоченная история сообщений |
| `system` | `str` | Системная инструкция (опционально) |
| `model` | `str \| None` | Переопределение модели для запроса |
| `temperature` | `float \| None` | Температура сэмплинга |
| `max_tokens` | `int \| None` | Лимит токенов |
