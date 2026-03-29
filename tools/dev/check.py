"""Quality gate for codex-ai."""

import sys
from pathlib import Path

from codex_core.dev.check_runner import BaseCheckRunner


class CheckRunner(BaseCheckRunner):
    PROJECT_NAME = "codex-ai"
    INTEGRATION_REQUIRES = "API keys (OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY)"
    # CVE-2026-4539: pygments — no fix available yet (latest version)
    AUDIT_FLAGS = "--skip-editable --ignore-vuln CVE-2026-4539"

    def run_tests(self, marker: str = "unit") -> bool:
        if marker == "integration":
            # --no-cov: integration tests may be skipped when keys are absent.
            # Running coverage on 0 tests would fail the threshold gate.
            self.print_step("Running Integration Tests")
            success, _ = self.run_command(
                f'"{sys.executable}" -m pytest {self.tests_dir} -m integration -v --tb=short --no-cov'
            )
            if success:
                self.print_success("Integration tests passed.")
            else:
                self.print_error("Integration tests failed.")
            return success
        return super().run_tests(marker)


if __name__ == "__main__":
    CheckRunner(Path(__file__).parent.parent.parent).main()
