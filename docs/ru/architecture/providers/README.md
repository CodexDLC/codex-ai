# Провайдеры

## Назначение

`codex_ai.providers` содержит конкретные реализации `LLMProviderProtocol`. Каждый провайдер оборачивает SDK вендора, транслирует `PromptResult` в формат вендора и поднимает `LLMProviderError` при ошибке.

## Зачем это модуль

| Проблема | Что ломается |
|----------|-------------|
| У каждого SDK разная поверхность API | Бизнес-код должен знать об `AsyncOpenAI`, `genai.Client`, `AsyncAnthropic` |
| System-сообщения работают по-разному у вендоров | OpenAI использует роль `developer`/`system`, Gemini — `system_instruction`, Anthropic — top-level параметр `system` |
| Production-системы нуждаются в fallback при ошибках API | Система с одним провайдером падает при недоступности вендора |

`providers` решает это через единый интерфейс `answer(prompt, **kw) → str` и `MultiLLMProvider` с автоматическим failover.

## Архитектура

```
                    LLMProviderProtocol
                   .answer(prompt, **kw)
                          │
        ┌─────────────────┼──────────────────────┐
        │                 │                      │
        ▼                 ▼                      ▼
┌──────────────┐ ┌──────────────┐   ┌────────────────────┐
│OpenAIProvider│ │GeminiProvider│   │  MultiLLMProvider  │
│              │ │              │   │                    │
│ AsyncOpenAI  │ │ genai.Client │   │ {name: provider}   │
│ gpt-4o-mini  │ │ gemini-2.5   │   │ default + failover │
└──────────────┘ └──────────────┘   └──────────┬─────────┘
                                               │
                        ┌──────────────────────┼───────────────────┐
                        ▼                      ▼                   ▼
               OpenAIProvider        GeminiProvider       AnthropicProvider
               (основной)            (failover[0])        (failover[1])
```

**OpenRouterProvider** расширяет `OpenAIProvider` с переопределённым `base_url` — полная совместимость с OpenAI SDK, сотни моделей.

## Ключевые компоненты

| Компонент | Класс | SDK | Модель по умолчанию |
|-----------|-------|-----|---------------------|
| `openai.py` | `OpenAIProvider` | `openai` | `gpt-4o-mini` |
| `gemini.py` | `GeminiProvider` | `google-genai` | `gemini-2.5-flash-lite` |
| `anthropic_.py` | `AnthropicProvider` | `anthropic` | `claude-3-5-sonnet-latest` |
| `openrouter.py` | `OpenRouterProvider` | `openai` (remapped) | бесплатная Gemini через OpenRouter |
| `multi.py` | `MultiLLMProvider` | — | настраиваемый |

## Ключевые решения

- **Lazy imports** — классы провайдеров не импортируются при загрузке пакета. `from codex_ai import OpenAIProvider` импортирует SDK `openai` только при выполнении этой строки. Установка `codex-ai[gemini]` не тянет OpenAI SDK.
- **Поля `PromptResult` имеют приоритет над `**kw`** — если `prompt.model` задан, он перекрывает любой `model=` из kwargs. Намерение билдера приоритетнее call-site переопределений.
- **Инференс провайдера в `MultiLLMProvider`** — префикс имени модели автоматически выбирает провайдер: `gpt-*` → openai, `gemini-*` → gemini, `claude-*` → anthropic, `vendor/model` → openrouter. Явный `provider=` в kwargs всегда побеждает.
- **Failover только на `LLMProviderError`** — каждый провайдер оборачивает SDK-исключения в `LLMProviderError`. Только они запускают failover-цепочку; неожиданные исключения пробрасываются немедленно.
- **Кастомные провайдеры не требуют регистрации** — реализуйте `async def answer(self, prompt: PromptResult, **kw) -> str` и передайте экземпляр напрямую в `LLMDispatcher`.

## См. также

- [Поток данных](data_flow.md) — трансляция формата сообщений по вендорам
- [Обзор Core](../core/README.md) — как провайдеры подключаются к диспетчеру
- [API Reference: Провайдеры](../../../en/api/providers/openai.md) — полная документация классов
