# Source map and contracts

- `src/codex_ai/providers/gemini.py` implements direct Gemini text, structured JSON, Gemini image generation, and separate Imagen generation. Image edits accept `ImageInput` values.
- `src/codex_ai/providers/openai.py` implements OpenAI Responses text, JSON, and streaming. It does not expose an image-generation method.
- `src/codex_ai/core/protocol.py` defines shared DTOs and runtime-checkable provider capabilities; `core/exceptions.py` defines provider errors.
- `src/codex_ai/core/router.py` registers prompt builders. `core/dispatcher.py` handles the legacy `process()` flow and delegates supported direct calls. `core/sync.py` wraps that flow for synchronous callers.
- `src/codex_ai/__init__.py` lazily loads providers so optional SDK dependencies remain optional.
- `src/codex_ai/agent_skills/` owns the optional offline consumer installer and packaged skill. It must touch only its managed destination and marked `AGENTS.md` block, preserve unrelated content, and reject modified managed files or unsafe paths.

When provider behavior changes, test the direct public call and error path. When a public signature or capability changes, update README, library skill examples, and public exports or protocols as needed. Do not use live provider calls in unit tests.
