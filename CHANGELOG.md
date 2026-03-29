# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.0] - 2026-03-29
- **Test**: Added comprehensive unit and integration test suites for core modules and LLM providers.
- **Build**: Standardized packaging on Python 3.12+, aligned classifiers, and declared compatibility with `codex-core` from `0.2.2` up to `<0.4.0`.
- **Fix**: Adjusted type casting and prevented passing `None` to SDK kwargs in Anthropic, Gemini, and OpenAI providers.
- **CI**: Added locked `uv`-based quality checks, artifact validation, tag-based PyPI publishing, and versioned docs deployment.
- **Docs**: Brought the landing README and release notes in line with the Codex ecosystem documentation standard.
- Initial split from monolithic `codex_tools` repository.
