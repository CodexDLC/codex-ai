# Existing router integrations

This is a fragment for an async application handler; `gemini_key` comes from application configuration.

```python
from codex_ai import GeminiProvider, LLMDispatcher, LLMMessage, LLMRouter, PromptResult

router = LLMRouter()

@router.prompt("chat")
async def build_chat(text: str, **kwargs) -> PromptResult:
    return PromptResult(messages=[LLMMessage(role="user", content=text)])

dispatcher = LLMDispatcher(provider=GeminiProvider(api_key=gemini_key))
dispatcher.include_router(router)
reply = await dispatcher.process("chat", text="Hello")
```

`process(mode, **kwargs)` looks up a registered builder, awaits it, then calls the provider's text-compatible `answer()` method. An unknown mode raises `KeyError`. Keep prompt builder names unique across included routers because later registrations replace earlier ones. The dispatcher also delegates direct `generate_text()`, `generate_json()`, and supported image methods without routing; prefer direct provider calls for new code.
