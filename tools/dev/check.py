"""Quality gate for codex-ai."""

import sys
from pathlib import Path

from codex_core.dev.check_runner import BaseCheckRunner


class CheckRunner(BaseCheckRunner):
    """Thin launcher; project policy lives in pyproject.toml."""

    def check_quality(self) -> bool:
        self.print_step("Running Quality Hooks (pre-commit: Ruff, Format, Bandit)")
        success, out = self.run_command(f'"{sys.executable}" -m pre_commit run --all-files', capture_output=True)
        if not success:
            self.print_error(f"Pre-commit failed:\n{out}")
            return False
        self.print_success("Quality hooks passed.")
        return True

    def check_security(self) -> bool:
        self.print_step("Security Audit (pip-audit)")
        success, out = self.run_command(
            f'"{sys.executable}" -m pip_audit {self.AUDIT_FLAGS}',
            capture_output=True,
        )
        if not success:
            self.print_error(f"Security audit failed:\n{out}")
        else:
            self.print_success("Security audit passed.")
        return success

    def run_tests(self, marker: str = "unit") -> bool:
        self.print_step(f"Running {marker.capitalize()} Tests")
        no_cov = " --no-cov" if marker == "integration" else ""
        success, _ = self.run_command(
            f'"{sys.executable}" -m pytest {self.tests_dir} -m {marker} -v --tb=short{no_cov}'
        )
        if success:
            self.print_success(f"{marker.capitalize()} tests passed.")
        else:
            self.print_error(f"{marker.capitalize()} tests failed.")
        return success


if __name__ == "__main__":
    CheckRunner(Path(__file__).parent.parent.parent).main()
