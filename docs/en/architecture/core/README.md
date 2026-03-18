# Core

## Purpose

`codex_ai.core` is the orchestration layer for LLM interactions. It decouples prompt construction from provider selection — prompt logic is defined once and can be routed to any backend without changes.

## Why It's a Module

Working directly with LLM SDKs across a codebase creates three recurring problems:

| Problem | What breaks |
|---------|-------------|
| Prompt logic scattered across call sites | Hard to test, review, or reuse |
| Provider SDK calls coupled to business code | Switching providers requires touching every call |
| No unified contract for async/sync contexts | Different wiring for Django views, ARQ workers, bots |

`core` solves these by introducing a clean pipeline:

```
@router.prompt("mode") → PromptResult (frozen DTO) → LLMProviderProtocol.answer()
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              Your Application               │
│                                             │
│   @router.prompt("chat")                    │
│   async def build_chat(...) -> PromptResult │
└────────────────────┬────────────────────────┘
                     │ include_router(router)
                     ▼
          ┌──────────────────────┐
          │    LLMDispatcher     │  orchestrates builder → provider
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

## Key Components

| Component | Class | Role |
|-----------|-------|------|
| `protocol.py` | `PromptResult` | Frozen DTO — immutable prompt passed to providers |
| `protocol.py` | `LLMProviderProtocol` | Structural protocol — any class with `async answer()` qualifies |
| `protocol.py` | `PromptBuilder` | Type alias for `async def (...) -> PromptResult` |
| `router.py` | `LLMRouter` | Registry: maps `mode` strings to builder functions via decorator |
| `dispatcher.py` | `LLMDispatcher` | Wires router + provider; single entry point for prompt execution |
| `sync.py` | `SyncLLMDispatcher` | Wraps `LLMDispatcher` with `asyncio.run()` for WSGI/CLI contexts |
| `exceptions.py` | `LLMProviderError` | Base exception raised by all provider implementations |

## Key Design Decisions

- **Frozen DTO (`PromptResult`)** — built once by the builder, cannot be mutated downstream. Prevents accidental state sharing between requests.
- **`@runtime_checkable` Protocol** — `isinstance(obj, LLMProviderProtocol)` works at runtime. No inheritance required from any base class.
- **Mode-based dispatch** — `dispatcher.process("chat", ...)` maps to a registered builder. Adding a new prompt type never touches existing code.
- **All logs at `DEBUG`** — dispatcher emits only debug-level messages. Production log level controls visibility without code changes.
- **`SyncLLMDispatcher` for Django only** — uses `asyncio.run()` which creates a new event loop. Never call from inside an async context (ARQ, async views, bots) — use `LLMDispatcher` directly.

## See Also

- [Data Flow](data_flow.md) — step-by-step request lifecycle
- [Providers Overview](../providers/README.md) — available backends
- [API Reference: Core](../../../en/api/core/protocol.md) — full class documentation
