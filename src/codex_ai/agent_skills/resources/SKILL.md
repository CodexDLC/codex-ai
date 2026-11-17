---
name: codex-ai
description: Use the codex-ai Python package for direct Gemini or OpenAI generation, and maintain integrations that already use its legacy router.
---

# codex-ai consumer guide

Use this skill when writing application code with the installed `codex-ai` package. Install the needed optional extra: `codex-ai[gemini]` for `GeminiProvider`, `codex-ai[openai]` for `OpenAIProvider`. Python 3.12 or newer is required. Keep credentials in the application's secret management; pass them as provider configuration, never commit them.

For new work, call provider methods directly. `generate_text()` returns text, `generate_json()` returns a model-validated object when given a Pydantic schema, and image methods return `(bytes, MIME type)`. Read [references/providers.md](references/providers.md) for small examples and the distinct Gemini image and Imagen contracts.

The `LLMRouter` and `LLMDispatcher.process()` path remains for integrations with registered prompt builders. Read [references/legacy-router.md](references/legacy-router.md) only when changing those integrations. For install, update, status, delete, or ownership questions, read [references/installation.md](references/installation.md). Provider exceptions are surfaced as `LLMProviderError`; handle them at the application boundary where retries or user feedback can be decided.

The installed skill and its `.manifest.json` are managed content; change guidance in the `codex-ai` source repository, then use the official update command. Keep application policy in an application-owned skill. Check the installed version with `importlib.metadata.version("codex-ai")` and locate its import with `codex_ai.__file__` before relying on model names or SDK-specific options. Consult the version-matched package documentation and actual installed signatures for version-specific detail.
