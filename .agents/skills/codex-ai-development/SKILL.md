---
name: codex-ai-development
description: Maintain this repository's provider contracts, tests, package guidance, and offline consumer skill delivery.
---

# codex-ai development

This repository publishes `codex-ai`, a Gemini-first Python library with an OpenAI Responses adapter. Read [references/architecture.md](references/architecture.md) when changing provider behavior or shared contracts. For external application usage, use the packaged `codex-ai` consumer skill under `src/codex_ai/agent_skills/resources/`; keep application-specific rules out of that package.

Before architecture or cross-module changes, follow the graphify rules in the repository `AGENTS.md`, including the workspace graph report and update after code changes. Keep the public exports in `src/codex_ai/__init__.py` and provider protocols in `src/codex_ai/core/protocol.py` aligned with implementation. Preserve the documented legacy text router unless a task explicitly changes that contract.

Use the relevant optional extra when verifying a provider. Run focused unit tests for changed behavior and the applicable quality checks. Integration tests can require API credentials; do not inspect or expose local secrets. For changes to packaged skill delivery, verify a built wheel and install/update/status/delete in an isolated project.

The repository source skill is authored here; any installed consumer copy and its manifest belong to the installer. Keep source and installed ownership separate. Typical checks from the repository root are `python -m pytest tests/unit -q`, `python -m ruff check src tests`, and `python -m mypy src`. Verify package artifact contents when changing delivery; inspect failures rather than accepting a silent test run.
