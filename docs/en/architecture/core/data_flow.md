# Core — Data Flow

## Legacy Text Request Lifecycle

This flow is retained for existing `LLMRouter` integrations. New Gemini work should use direct provider methods.

```
1. Application calls:
   dispatcher.process("chat", text="Hello!")

2. LLMDispatcher looks up builder:
   builder = self._builders["chat"]  # registered via @router.prompt

3. Builder constructs PromptResult:
   prompt = await builder(text="Hello!")
   # → PromptResult(messages=[LLMMessage(role="user", content="Hello!")])

4. Dispatcher calls the text compatibility method:
   response = await self._provider.answer(prompt, **kw)

5. Provider returns text:
   "Hi there! How can I help?"
```

## Legacy Component Interactions

```
Application
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
           Response: str
```

## PromptResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `list[LLMMessage]` | Ordered message history |
| `system` | `str` | System instruction (optional) |
| `model` | `str \| None` | Override model for this request |
| `temperature` | `float \| None` | Sampling temperature override |
| `max_tokens` | `int \| None` | Token limit override |
