# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.2.6] - 2026-10-09

### Added

- Added a project development skill and an optional packaged consumer skill for `codex-ai` integrations.
- Added an offline `python -m codex_ai.agent_skills` installer with install, update, status, and delete commands, ownership checks, path validation, and rollback for ordinary I/O failures.
- Added English and Russian guides for installing and maintaining the optional agent skill.

### Changed

- Widened the supported `codex-core` range to include the additive `0.5.x` agent-skill release.
- Refreshed the development and documentation lockfile to use patched versions of AnyIO, cryptography, pip, pyasn1, urllib3, virtualenv, Material for MkDocs, and PyMdown Extensions.

## [0.2.5] - 2026-07-19

### Added
- Added `ImageInput` and optional `input_images` support to Gemini `generate_image_bytes()` for image reference/edit workflows.
- Added OpenAI structured output generation through `generate_json()` with native Pydantic parsing.
- Added typed OpenAI Responses API text streaming through `stream_text()`.

### Changed
- Replaced the OpenAI Chat Completions implementation with the Responses API.
- Updated the OpenAI SDK requirement to `openai>=2.0,<3.0`.
- Changed the OpenAI default model to `gpt-5.6-luna` with `reasoning={"effort": "none"}` and `store=False` defaults.

## [0.2.4] - 2026-05-21

### Changed
- Widened the supported `codex-core` range to include the `0.4.x` series.
- Clarified the Gemini-first provider focus in the package documentation.

## [0.2.3] - 2026-05-17

### Added
- Added explicit Gemini image `image_config` support for aspect ratio and image size controls.
- Added a Gemini image retry from `image_size="4K"` to `image_size="2K"` when the initial 4K request is rejected.

### Changed
- Updated the Gemini extra to `google-genai==2.3.0` for the SDK image configuration contract.

## [0.2.2] - 2026-05-15

### Fixed
- Changed the Gemini image default model to the official API model id `gemini-2.5-flash-image`.

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
