# Core

## Назначение

`codex_ai.core` — legacy-слой текстовой оркестрации. Он сохраняет существующие workflow на `LLMRouter`/`LLMDispatcher`, пока активная API-поверхность переехала в прямые методы провайдеров.

## Зачем это модуль

Старые Codex-интеграции используют mode-based prompt builders. Этот модуль сохраняет такую форму, но больше не является основной абстракцией для новой работы:

| Потребность | Текущая роль |
|-------------|--------------|
| Сохранить зарегистрированные prompt builders | `LLMRouter` связывает modes с билдерами |
| Не переписывать существующие текстовые вызовы | `LLMDispatcher.process()` продолжает вызывать `provider.answer()` |
| Поддержать sync-only контексты | `SyncLLMDispatcher` остается для CLI/WSGI кода |

Для новой Gemini-интеграции лучше использовать прямые методы `GeminiProvider.generate_text()`, `generate_json()`, `generate_image_bytes()` и `generate_imagen_bytes()`.

Сохраненный текстовый pipeline:

```
@router.prompt("mode") → PromptResult (frozen DTO) → LLMProviderProtocol.answer()
```

## Архитектура

```
┌─────────────────────────────────────────────┐
│              Ваше приложение                │
│                                             │
│   @router.prompt("chat")                    │
│   async def build_chat(...) -> PromptResult │
└────────────────────┬────────────────────────┘
                     │ include_router(router)
                     ▼
          ┌──────────────────────┐
          │    LLMDispatcher     │  legacy text builder → provider
          │                      │
          │  .process(mode, **kw)│
          └──────┬───────────────┘
                 │
      ┌──────────┴──────────┐
      │                     │
      ▼                     ▼
┌──────────┐       ┌────────────────────┐
│ LLMRouter│       │ LLMProviderProtocol│
│          │       │                    │
│ builders │       │ .answer(prompt, **kw) → str
│ {mode:fn}│       └────────────────────┘
└──────────┘
```

## Ключевые компоненты

| Компонент | Класс | Роль |
|-----------|-------|------|
| `protocol.py` | `PromptResult` | Frozen DTO для legacy text prompt builders |
| `protocol.py` | `LLMProviderProtocol` | Структурный текстовый compatibility-протокол для `answer()` |
| `protocol.py` | `PromptBuilder` | Type alias для `async def (...) -> PromptResult` |
| `router.py` | `LLMRouter` | Реестр: связывает строки `mode` с функциями-билдерами через декоратор |
| `dispatcher.py` | `LLMDispatcher` | Выполняет legacy text prompts и делегирует прямые provider convenience methods |
| `sync.py` | `SyncLLMDispatcher` | Оборачивает `LLMDispatcher` через `asyncio.run()` для WSGI/CLI |
| `exceptions.py` | `LLMProviderError` | Базовое исключение, поднимаемое всеми реализациями провайдеров |

## Ключевые решения

- **Direct provider APIs first** — возможности Gemini раскрываются явными методами, а не проталкиваются через универсальный provider interface.
- **Frozen DTO (`PromptResult`)** — сохранен для legacy builders и не может быть изменён downstream.
- **`@runtime_checkable` Protocol** — оставлен для runtime-проверок на compatibility boundaries.
- **Mode-based dispatch — legacy text infrastructure** — `dispatcher.process("chat", ...)` находит зарегистрированный билдер для существующих flows.
- **Все логи на `DEBUG`** — диспетчер пишет только debug-сообщения. Уровень логирования в продакшне управляет видимостью без изменений кода.
- **`SyncLLMDispatcher` только для Django** — использует `asyncio.run()`, создающий новый event loop. Никогда не вызывать из async-контекста (ARQ, async views, боты) — используйте `LLMDispatcher` напрямую.

## См. также

- [Поток данных](data_flow.md) — пошаговый жизненный цикл запроса
- [Обзор провайдеров](../providers/README.md) — доступные бэкенды
- [API Reference: Core](../../../en/api/core/protocol.md) — полная документация классов
