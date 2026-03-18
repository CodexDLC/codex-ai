# Providers

## Purpose

`codex_ai.providers` contains concrete implementations of `LLMProviderProtocol`. Each provider wraps a vendor SDK, translates `PromptResult` into the vendor's wire format, and raises `LLMProviderError` on failure.

## Why It's a Module

| Problem | What breaks |
|---------|-------------|
| Each vendor SDK has a different API surface | Business code must know about `AsyncOpenAI`, `genai.Client`, `AsyncAnthropic` |
| System messages work differently across vendors | OpenAI uses `developer`/`system` role, Gemini uses `system_instruction`, Anthropic uses a top-level `system` param |
| Production systems need fallback on API errors | A single-provider setup goes down when the vendor has an outage |

`providers` solves this with a uniform `answer(prompt, **kw) → str` interface and `MultiLLMProvider` for automatic failover.

## Architecture

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
               (primary)             (failover[0])        (failover[1])
```

**OpenRouterProvider** subclasses `OpenAIProvider` with a remapped `base_url` — full OpenAI SDK compatibility, hundreds of models.

## Key Components

| Component | Class | SDK | Default Model |
|-----------|-------|-----|---------------|
| `openai.py` | `OpenAIProvider` | `openai` | `gpt-4o-mini` |
| `gemini.py` | `GeminiProvider` | `google-genai` | `gemini-2.5-flash-lite` |
| `anthropic_.py` | `AnthropicProvider` | `anthropic` | `claude-3-5-sonnet-latest` |
| `openrouter.py` | `OpenRouterProvider` | `openai` (remapped) | free Gemini via OpenRouter |
| `multi.py` | `MultiLLMProvider` | — | configurable |

## Key Design Decisions

- **Lazy imports** — provider classes are not imported at package load time. `from codex_ai import OpenAIProvider` only imports the `openai` SDK when that line executes. Installing `codex-ai[gemini]` does not pull the OpenAI SDK.
- **`PromptResult` fields take priority over `**kw`** — if `prompt.model` is set, it overrides any `model=` passed in kwargs. Builder intent wins over call-site overrides.
- **`MultiLLMProvider` provider inference** — model name prefix auto-selects provider: `gpt-*` → openai, `gemini-*` → gemini, `claude-*` → anthropic, `vendor/model` → openrouter. Explicit `provider=` kwarg always wins.
- **Failover on `LLMProviderError` only** — each provider wraps SDK exceptions in `LLMProviderError`. Only these trigger the failover chain; unexpected exceptions propagate immediately.
- **Custom providers need no registration** — implement `async def answer(self, prompt: PromptResult, **kw) -> str` and pass the instance directly to `LLMDispatcher`.

## See Also

- [Data Flow](data_flow.md) — message format translation per vendor
- [Core Overview](../core/README.md) — how providers plug into the dispatcher
- [API Reference: Providers](../../../en/api/providers/openai.md) — full class documentation
