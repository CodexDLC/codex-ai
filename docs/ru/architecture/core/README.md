# Core

## Назначение

`codex_ai.core` — оркестрационный слой для работы с LLM. Разделяет построение промпта и выбор провайдера: логика промптов описывается один раз и может быть направлена к любому бэкенду без изменений.

## Зачем это модуль

Прямая работа с LLM SDK по всей кодовой базе создаёт три повторяющихся проблемы:

| Проблема | Что ломается |
|----------|-------------|
| Логика промптов разбросана по call site-ам | Трудно тестировать, ревьювить и переиспользовать |
| Вызовы SDK провайдера связаны с бизнес-кодом | Смена провайдера требует правок в каждом вызове |
| Нет единого контракта для async/sync контекстов | Разная обвязка для Django views, ARQ workers, ботов |

`core` решает это через единый pipeline:

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
          │    LLMDispatcher     │  оркестрирует builder → provider
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
| `protocol.py` | `PromptResult` | Frozen DTO — иммутабельный промпт, передаваемый провайдерам |
| `protocol.py` | `LLMProviderProtocol` | Структурный протокол — любой класс с `async answer()` подходит |
| `protocol.py` | `PromptBuilder` | Type alias для `async def (...) -> PromptResult` |
| `router.py` | `LLMRouter` | Реестр: связывает строки `mode` с функциями-билдерами через декоратор |
| `dispatcher.py` | `LLMDispatcher` | Связывает router + provider; единая точка входа для выполнения промптов |
| `sync.py` | `SyncLLMDispatcher` | Оборачивает `LLMDispatcher` через `asyncio.run()` для WSGI/CLI |
| `exceptions.py` | `LLMProviderError` | Базовое исключение, поднимаемое всеми реализациями провайдеров |

## Ключевые решения

- **Frozen DTO (`PromptResult`)** — строится один раз в билдере, не может быть изменён downstream. Исключает случайный shared state между запросами.
- **`@runtime_checkable` Protocol** — `isinstance(obj, LLMProviderProtocol)` работает в runtime. Наследование от базового класса не требуется.
- **Mode-based dispatch** — `dispatcher.process("chat", ...)` находит зарегистрированный билдер. Добавление нового типа промпта не затрагивает существующий код.
- **Все логи на `DEBUG`** — диспетчер пишет только debug-сообщения. Уровень логирования в продакшне управляет видимостью без изменений кода.
- **`SyncLLMDispatcher` только для Django** — использует `asyncio.run()`, создающий новый event loop. Никогда не вызывать из async-контекста (ARQ, async views, боты) — используйте `LLMDispatcher` напрямую.

## См. также

- [Поток данных](data_flow.md) — пошаговый жизненный цикл запроса
- [Обзор провайдеров](../providers/README.md) — доступные бэкенды
- [API Reference: Core](../../../en/api/core/protocol.md) — полная документация классов
