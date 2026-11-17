"""Offline skill installation and ownership boundaries."""

import json
from pathlib import Path

import pytest

from codex_ai.agent_skills import fs_transaction
from codex_ai.agent_skills.cli import main
from codex_ai.agent_skills.manifest_rules import ManifestError, build_manifest, parse_manifest
from codex_ai.agent_skills.orchestrator import OrchestratorError, check_status, perform_delete, perform_install

pytestmark = pytest.mark.unit


def test_lifecycle_preserves_unrelated_project_content(tmp_path: Path) -> None:
    agents = tmp_path / "AGENTS.md"
    original = b"# User rules\r\nKeep this line.\r\n"
    agents.write_bytes(original)
    perform_install(tmp_path)
    skill = tmp_path / ".agents/skills/codex-ai"
    (skill / "personal.txt").write_text("personal", encoding="utf-8")
    assert check_status(tmp_path) == (True, "Installed and up to date.")
    perform_install(tmp_path, update=True)
    assert check_status(tmp_path)[0]
    perform_delete(tmp_path)
    assert agents.read_bytes() == original
    assert (skill / "personal.txt").read_text(encoding="utf-8") == "personal"
    assert not (skill / "SKILL.md").exists()


def test_modified_managed_content_blocks_update_and_delete(tmp_path: Path) -> None:
    perform_install(tmp_path)
    skill = tmp_path / ".agents/skills/codex-ai"
    managed = skill / "SKILL.md"
    managed.write_bytes(managed.read_bytes() + b"\nlocal change\n")
    agents = (tmp_path / "AGENTS.md").read_bytes()
    with pytest.raises(OrchestratorError, match="modified managed content"):
        perform_install(tmp_path, update=True)
    with pytest.raises(OrchestratorError, match="modified managed content"):
        perform_delete(tmp_path)
    assert managed.read_bytes().endswith(b"local change\n")
    assert (tmp_path / "AGENTS.md").read_bytes() == agents


def test_manifest_rejects_unsafe_paths_and_duplicate_keys() -> None:
    for name in ("../escape", "CON.txt", "reference//file", "a\\b", "skill.md"):
        with pytest.raises(ManifestError):
            build_manifest("1", {"SKILL.md": b"ok", name: b"bad"})
    raw = build_manifest("1", {"SKILL.md": b"ok"})
    duplicate = raw.replace(b'"version": "1.0",', b'"version": "1.0", "version": "1.0",')
    with pytest.raises(ManifestError):
        parse_manifest(duplicate)
    assert json.loads(raw)["skill"] == "codex-ai"


def test_symlink_destination_rejected(tmp_path: Path) -> None:
    destination = tmp_path / ".agents"
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        destination.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable")
    with pytest.raises(fs_transaction.FSError, match="Symlink or reparse"):
        perform_install(tmp_path)
    assert not (outside / "skills").exists()


def test_write_error_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Existing\n", encoding="utf-8")
    original_replace = fs_transaction.os.replace
    count = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("injected write failure")
        original_replace(source, destination)

    monkeypatch.setattr(fs_transaction.os, "replace", fail_second_replace)
    with pytest.raises(fs_transaction.FSError, match="rolled back"):
        perform_install(tmp_path)
    assert agents.read_text(encoding="utf-8") == "# Existing\n"
    assert not (tmp_path / ".agents/skills/codex-ai/SKILL.md").exists()
    assert not list(tmp_path.glob(".codex-ai-tx-*"))


def test_cli_lifecycle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--project", str(tmp_path)]
    main(["install", *args])
    assert "Installed successfully" in capsys.readouterr().out
    main(["status", *args])
    assert "up to date" in capsys.readouterr().out
    main(["update", *args])
    assert "Updated successfully" in capsys.readouterr().out
    main(["delete", *args])
    assert "Deleted successfully" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exited:
        main(["status", *args])
    assert exited.value.code == 1
    assert "Not installed" in capsys.readouterr().out


def test_cli_invalid_project_and_duplicate_install(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["install", "--project", str(tmp_path / "missing")])
    assert exited.value.code == 1
    assert "does not exist" in capsys.readouterr().err
    perform_install(tmp_path)
    main(["install", "--project", str(tmp_path)])
    assert "Installed successfully" in capsys.readouterr().out


def test_manifest_and_instruction_corruption_block_changes(tmp_path: Path) -> None:
    with pytest.raises(OrchestratorError, match="Not installed"):
        perform_install(tmp_path, update=True)
    perform_install(tmp_path)
    manifest = tmp_path / ".agents/skills/codex-ai/.manifest.json"
    original = manifest.read_bytes()
    manifest.write_bytes(b"{bad json")
    assert not check_status(tmp_path)[0]
    with pytest.raises(OrchestratorError, match="Corrupt ownership manifest"):
        perform_delete(tmp_path)
    manifest.write_bytes(original)
    agents = tmp_path / "AGENTS.md"
    agents.write_bytes(agents.read_bytes().replace(b"Use [codex-ai]", b"Ignore [codex-ai]"))
    with pytest.raises(OrchestratorError, match="Corrupt AGENTS.md"):
        perform_install(tmp_path, update=True)


def test_unowned_destination_and_new_file_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skill = tmp_path / ".agents/skills/codex-ai"
    skill.mkdir(parents=True)
    (skill / "personal.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(OrchestratorError, match="without ownership manifest"):
        perform_install(tmp_path)
    (skill / "personal.txt").unlink()
    skill.rmdir()
    perform_install(tmp_path)
    (skill / "new.txt").write_text("keep", encoding="utf-8")
    import codex_ai.agent_skills.orchestrator as orchestrator

    original_payload = orchestrator.get_payload_files
    monkeypatch.setattr(orchestrator, "get_payload_files", lambda: {**original_payload(), "new.txt": b"package"})
    with pytest.raises(OrchestratorError, match="colliding"):
        perform_install(tmp_path, update=True)
    assert (skill / "new.txt").read_text(encoding="utf-8") == "keep"


def test_version_drift_requires_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    perform_install(tmp_path)
    import codex_ai.agent_skills.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "get_package_version", lambda: "next-version")
    assert "Needs update" in check_status(tmp_path)[1]
    with pytest.raises(OrchestratorError, match="Use update"):
        perform_install(tmp_path)
    perform_install(tmp_path, update=True)
    assert check_status(tmp_path)[0]


def test_orphan_instruction_and_incomplete_installation(tmp_path: Path) -> None:
    from codex_ai.agent_skills.manifest_rules import update_agents_md_bytes

    agents = tmp_path / "AGENTS.md"
    agents.write_bytes(update_agents_md_bytes(b"# My rules\n"))
    assert "Orphan" in check_status(tmp_path)[1]
    with pytest.raises(OrchestratorError, match="Orphan"):
        perform_install(tmp_path)
    with pytest.raises(OrchestratorError, match="No ownership manifest"):
        perform_delete(tmp_path)
    agents.unlink()
    perform_install(tmp_path)
    agents.unlink()
    assert "instruction missing" in check_status(tmp_path)[1]
    with pytest.raises(OrchestratorError, match="instruction missing"):
        perform_install(tmp_path, update=True)
    with pytest.raises(OrchestratorError, match="instruction missing"):
        perform_delete(tmp_path)


def test_missing_owned_file_blocks_update_and_delete(tmp_path: Path) -> None:
    perform_install(tmp_path)
    managed = tmp_path / ".agents/skills/codex-ai/SKILL.md"
    managed.unlink()
    assert "Missing file" in check_status(tmp_path)[1]
    with pytest.raises(OrchestratorError, match="Missing file"):
        perform_install(tmp_path, update=True)
    with pytest.raises(OrchestratorError, match="Missing file"):
        perform_delete(tmp_path)


def test_invalid_ancestor_and_corrupt_markers(tmp_path: Path) -> None:
    (tmp_path / ".agents").write_text("not a directory", encoding="utf-8")
    with pytest.raises(fs_transaction.FSError, match="Non-directory ancestor"):
        perform_install(tmp_path)
    (tmp_path / ".agents").unlink()
    agents = tmp_path / "AGENTS.md"
    agents.write_bytes(b"<!-- codex-ai:skill:start -->\n")
    assert "Corrupt AGENTS.md" in check_status(tmp_path)[1]
    with pytest.raises(OrchestratorError, match="Corrupt AGENTS.md"):
        perform_install(tmp_path)


def test_cli_reports_ownership_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / ".agents/skills/codex-ai").mkdir(parents=True)
    with pytest.raises(SystemExit) as exited:
        main(["install", "--project", str(tmp_path)])
    assert exited.value.code == 1
    assert "without ownership manifest" in capsys.readouterr().err


def test_failed_rollback_retains_recovery_backup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Preserve me\n", encoding="utf-8")
    original_replace = fs_transaction.os.replace
    original_copy = fs_transaction.shutil.copy2
    replacement_count = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal replacement_count
        replacement_count += 1
        if replacement_count == 2:
            raise OSError("injected write failure")
        original_replace(source, destination)

    def fail_restore(source: Path, destination: Path) -> str:
        if source.name.startswith("backup_"):
            raise OSError("injected recovery failure")
        return str(original_copy(source, destination))

    monkeypatch.setattr(fs_transaction.os, "replace", fail_second_replace)
    monkeypatch.setattr(fs_transaction.shutil, "copy2", fail_restore)
    with pytest.raises(fs_transaction.FSError, match="Backups retained at"):
        perform_install(tmp_path)
    backups = list(tmp_path.glob(".codex-ai-tx-*/backup/backup_*"))
    assert backups and any(b"# Preserve me" in path.read_bytes() for path in backups)
