# Core

## Purpose

`codex_ai.core` is the legacy text orchestration layer. It keeps existing `LLMRouter`/`LLMDispatcher` prompt-builder workflows working while the active API surface moves to direct provider methods.

## Why It's a Module

Older Codex integrations use mode-based prompt builders. This module keeps that shape stable without making it the primary abstraction for new work:

| Need | Current role |
|------|--------------|
| Keep registered prompt builders working | `LLMRouter` maps modes to builders |
| Run existing text flows without rewriting callers | `LLMDispatcher.process()` still calls `provider.answer()` |
| Bridge sync-only contexts | `SyncLLMDispatcher` remains available for CLI/WSGI code |

For new Gemini work, prefer `GeminiProvider.generate_text()`, `generate_json()`,
`generate_image_bytes()`, and `generate_imagen_bytes()` directly.

The retained text pipeline is:

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

## Key Components

| Component | Class | Role |
|-----------|-------|------|
| `protocol.py` | `PromptResult` | Frozen DTO used by legacy text prompt builders |
| `protocol.py` | `LLMProviderProtocol` | Structural text compatibility protocol for `answer()` |
| `protocol.py` | `PromptBuilder` | Type alias for `async def (...) -> PromptResult` |
| `router.py` | `LLMRouter` | Registry: maps `mode` strings to builder functions via decorator |
| `dispatcher.py` | `LLMDispatcher` | Runs legacy text prompts and delegates direct provider convenience methods |
| `sync.py` | `SyncLLMDispatcher` | Wraps `LLMDispatcher` with `asyncio.run()` for WSGI/CLI contexts |
| `exceptions.py` | `LLMProviderError` | Base exception raised by all provider implementations |

## Key Design Decisions

- **Direct provider APIs first** — Gemini capabilities are exposed as explicit methods instead of being forced through a universal provider interface.
- **Frozen DTO (`PromptResult`)** — kept for legacy builders and cannot be mutated downstream.
- **`@runtime_checkable` Protocol** — retained for runtime checks at compatibility boundaries.
- **Mode-based dispatch is legacy text infrastructure** — `dispatcher.process("chat", ...)` maps to a registered builder for existing flows.
- **All logs at `DEBUG`** — dispatcher emits only debug-level messages. Production log level controls visibility without code changes.
- **`SyncLLMDispatcher` for Django only** — uses `asyncio.run()` which creates a new event loop. Never call from inside an async context (ARQ, async views, bots) — use `LLMDispatcher` directly.

## See Also

- [Data Flow](data_flow.md) — step-by-step request lifecycle
- [Providers Overview](../providers/README.md) — available backends
- [API Reference: Core](../../../en/api/core/protocol.md) — full class documentation
