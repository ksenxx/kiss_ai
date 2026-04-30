"""Integration tests for install URL correctness in README and scripts."""

import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

RAW_URL_PATTERN = re.compile(
    r"https://raw\.githubusercontent\.com/ksenxx/kiss_ai/main/scripts/install\.sh"
)
BLOB_URL_PATTERN = re.compile(
    r"https://github\.com/ksenxx/kiss_ai/blob/main/scripts/install\.sh"
)


class TestInstallUrl(unittest.TestCase):
    """Verify install URLs use raw.githubusercontent.com, not github.com/blob."""

    def test_readme_uses_raw_url(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text()
        self.assertTrue(
            RAW_URL_PATTERN.search(readme),
            "README.md should contain raw.githubusercontent.com install URL",
        )
        self.assertFalse(
            BLOB_URL_PATTERN.search(readme),
            "README.md must not contain github.com/blob install URL (returns HTML)",
        )

    def test_vscode_readme_uses_raw_url(self) -> None:
        readme = (REPO_ROOT / "src" / "kiss" / "agents" / "vscode" / "README.md").read_text()
        self.assertTrue(
            RAW_URL_PATTERN.search(readme),
            "vscode README.md should contain raw.githubusercontent.com install URL",
        )
        self.assertFalse(
            BLOB_URL_PATTERN.search(readme),
            "vscode README.md must not contain github.com/blob install URL",
        )

    def test_install_script_has_shebang(self) -> None:
        script = (REPO_ROOT / "scripts" / "install.sh").read_text()
        self.assertTrue(
            script.startswith("#!/bin/bash"),
            "scripts/install.sh must start with #!/bin/bash shebang",
        )

    def test_curl_raw_url_returns_shell_not_html(self) -> None:
        """Confirm the raw URL returns actual shell script content, not HTML."""
        result = subprocess.run(
            [
                "curl",
                "-fsSL",
                "--max-time",
                "10",
                "https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, f"curl failed: {result.stderr}")
        self.assertNotIn("<!DOCTYPE html>", result.stdout)
        # The script should contain shell commands, not HTML
        self.assertIn("git clone", result.stdout)

    def test_website_uses_raw_url(self) -> None:
        """Confirm kisssorcar.github.io install instruction uses raw URL."""
        result = subprocess.run(
            [
                "curl",
                "-fsSL",
                "--max-time",
                "10",
                "https://kisssorcar.github.io/",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, f"curl failed: {result.stderr}")
        self.assertTrue(
            RAW_URL_PATTERN.search(result.stdout),
            "kisssorcar.github.io should contain raw.githubusercontent.com install URL",
        )
        self.assertFalse(
            BLOB_URL_PATTERN.search(result.stdout),
            "kisssorcar.github.io must not contain github.com/blob install URL",
        )

    def test_blob_url_returns_html(self) -> None:
        """Confirm the blob URL returns HTML (proving the old URL was broken)."""
        result = subprocess.run(
            [
                "curl",
                "-sSL",
                "--max-time",
                "10",
                "https://github.com/ksenxx/kiss_ai/blob/main/scripts/install.sh",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        # The blob URL returns HTML, which is the bug
        self.assertIn("<!DOCTYPE html>", result.stdout)


if __name__ == "__main__":
    unittest.main()
