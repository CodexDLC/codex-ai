# Offline skill delivery

The opt-in command `python -m codex_ai.agent_skills install --project PATH` copies this packaged guide to `PATH/.agents/skills/codex-ai/` and appends one marked instruction block to `PATH/AGENTS.md`. It does not alter provider runtime or install optional SDKs.

Use `status` to check whether the destination matches its manifest and this installed package. After upgrading `codex-ai`, run `update` to refresh the owned files. Use `delete` to remove only owned files and the marked block. Each command requires `--project PATH`.

The installer records the package version and SHA-256 hashes of owned files in `.manifest.json`. It refuses malformed ownership data, changed managed files, unsafe paths, and collisions with unrelated files. Preserve such changes manually before asking for an update or deletion; never edit the manifest to claim ownership of a local modification. If another project skill is needed, create it separately outside `.agents/skills/codex-ai/`.
