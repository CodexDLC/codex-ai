# Agent skill installation

`codex-ai` 0.2.6 and newer ships an optional, offline usage guide for coding agents working in a consumer project. Installing the Python package does not write to that project. Run the installer only when you want its project-level skill:

```bash
python -m codex_ai.agent_skills install --project /path/to/project
python -m codex_ai.agent_skills status --project /path/to/project
```

On Windows PowerShell, quote a path containing spaces, for example `--project "D:\My Projects\game"`. Run the command with the Python environment where `codex-ai` is installed. The installer works offline and does not need the `[gemini]` or `[openai]` SDK extras. Install the relevant extra separately before using a provider.

The installer creates `.agents/skills/codex-ai/` under the target project and adds one marked instruction block to its `AGENTS.md`. It records the package version and SHA-256 hashes of the files it owns in `.agents/skills/codex-ai/.manifest.json`. Existing instructions and unrelated files are preserved. The installed skill and manifest are managed content; put application-specific guidance in a separate project-owned skill.

## Refresh or remove the skill

After upgrading the Python package, check and refresh the installed guide:

```bash
python -m codex_ai.agent_skills status --project /path/to/project
python -m codex_ai.agent_skills update --project /path/to/project
```

`status` reports whether the installed files match the current package and exits with a nonzero status when absent, stale, or inconsistent. `update` requires an existing managed installation. Upgrading the Python package alone does not update the copied skill.

To remove the guide and its marked `AGENTS.md` block:

```bash
python -m codex_ai.agent_skills delete --project /path/to/project
```

Update and delete refuse to change managed files that were edited or deleted locally. They also refuse corrupt ownership records, altered instruction blocks, unsafe links, and collisions with unrelated files. Restore or back up local edits before retrying; do not change the manifest to claim them. Unrelated files beside the skill remain in place. Ordinary file-operation errors trigger rollback, with recovery backups retained if rollback itself fails.
