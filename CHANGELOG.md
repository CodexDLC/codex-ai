# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- **Test**: Added comprehensive unit and integration test suites for core modules and LLM providers.
- **Build**: Lowered minimum Python version requirement from 3.12 to 3.10 and updated classifiers.
- **Fix**: Adjusted type casting and prevented passing `None` to SDK kwargs in Anthropic, Gemini, and OpenAI providers.
- **CI**: Configured `pytest-cov` with 90% fail-under threshold.
- Initial split from monolithic `codex_tools` repository.
