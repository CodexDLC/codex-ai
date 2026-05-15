# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.2.1] - 2026-05-15

### Added
- Added explicit `GeminiProvider.generate_imagen_bytes()` for Imagen models through `generate_images`.
- Added `ImagenGenerationProvider` and dispatcher delegation for explicit Imagen generation.

### Fixed
- Stopped passing image MIME values such as `image/webp` into Gemini `GenerateContentConfig.response_mime_type` on the Gemini image path.
- Pinned `google-genai` to `1.68.0` to avoid accidental SDK API drift in the alpha image-generation contract.

## [0.2.0] - 2026-05-15

### Added
- Added Gemini-first direct provider methods: `generate_text()`, `generate_json()`, and `generate_image_bytes()`.
- Added OpenAI `generate_text()` as the direct text generation API.
- Added text, JSON, and image generation provider protocols.
- Added `py.typed` marker for PEP 561 compliance — downstream consumers now benefit from full type inference when using mypy or pyright.

### Changed
- Refocused the active provider surface on `GeminiProvider` and `OpenAIProvider`.
- Kept `answer()` and `LLMDispatcher.process()` as text compatibility wrappers.
- Updated docs, README, integration tests, and CI gate for the Gemini/OpenAI provider line.
- Translated all Russian inline comments to English in `providers/gemini.py` and `providers/openai.py`.

### Removed
- Removed active Anthropic, OpenRouter, and Multi provider modules from this alpha line.

## [0.1.0] - 2026-03-29
- **Test**: Added comprehensive unit and integration test suites for core modules and LLM providers.
- **Build**: Standardized packaging on Python 3.12+, aligned classifiers, and declared compatibility with `codex-core` from `0.2.2` up to `<0.4.0`.
- **Fix**: Adjusted type casting and prevented passing `None` to SDK kwargs in Anthropic, Gemini, and OpenAI providers.
- **CI**: Added locked `uv`-based quality checks, artifact validation, tag-based PyPI publishing, and versioned docs deployment.
- **Docs**: Brought the landing README and release notes in line with the Codex ecosystem documentation standard.
- Initial split from monolithic `codex_tools` repository.
