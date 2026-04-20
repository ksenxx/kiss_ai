"""Tests for DependencyInstaller.ts behavior.

Validates that the dependency installer handles Playwright failures
gracefully and doesn't show unnecessary warnings to users.
"""

import re
import unittest
from pathlib import Path

VSCODE_SRC = Path(__file__).resolve().parents[3] / "agents" / "vscode" / "src"
INSTALLER_SOURCE = (VSCODE_SRC / "DependencyInstaller.ts").read_text()


class TestFastPathPlaywrightFailure(unittest.TestCase):
    """The fast-path Playwright install runs in the background.

    If it fails, the extension should check whether Chromium is already
    available before alarming the user.  Playwright browsers are cached
    system-wide (not inside .venv), so a background update failure is
    usually benign.
    """

    def test_fast_path_checks_chromium_before_warning(self) -> None:
        """After fast-path playwright failure, chromium availability must be
        checked before showing a warning notification."""
        # The .catch handler for the fast-path playwright install should
        # verify chromium is actually missing before showing a warning.
        # Look for a chromium availability check inside or near the catch block.
        catch_match = re.search(
            r"\.catch\(\s*(?:async\s*)?\(?(?:\w+)?\)?\s*=>\s*\{[^}]*Fast-path Playwright",
            INSTALLER_SOURCE,
            re.DOTALL,
        )
        assert catch_match is not None, (
            "Expected a .catch handler with 'Fast-path Playwright' log message"
        )

        # The catch block should contain a chromium check before showing warning
        catch_block_start = catch_match.start()
        # Find the warning message after the catch
        warning_match = re.search(
            r"Chromium browser update failed in background",
            INSTALLER_SOURCE[catch_block_start:],
        )
        if warning_match:
            # If the warning still exists, there should be a chromium check before it
            chromium_check = re.search(
                r"isChromiumInstalled|chromiumAvailable|playwrightBrowsersPath|ms-playwright",
                INSTALLER_SOURCE[catch_block_start : catch_block_start + warning_match.start()],
            )
            assert chromium_check is not None, (
                "The fast-path .catch handler must check if Chromium is already "
                "installed before showing a warning notification. A transient "
                "background update failure should not alarm the user when "
                "Chromium is already cached."
            )


class TestCheckPythonVersionNotDestructive(unittest.TestCase):
    """checkPythonVersion should not cause .venv deletion on transient errors.

    The code deletes .venv when checkPythonVersion returns false, but the
    function returns false for ANY failure (timeout, spawn error), not just
    for genuinely old Python versions.  This causes unnecessary .venv
    recreation on every activation after a transient failure.
    """

    def test_version_check_distinguishes_old_from_error(self) -> None:
        """The .venv deletion logic should only trigger when Python is
        genuinely too old, not on transient errors like timeouts."""
        # Look for the code that deletes .venv based on checkPythonVersion.
        # It should use a more specific check than just "!checkPythonVersion()".
        # The fix should either:
        # a) Have checkPythonVersion return a more specific result (e.g. null for error)
        # b) Separate the "too old" check from the "error" case
        # c) Not delete .venv on error, only on confirmed "too old"

        # Find the .venv deletion block
        deletion_match = re.search(
            r"checkPythonVersion.*\n.*rmSync.*\.venv",
            INSTALLER_SOURCE,
            re.DOTALL,
        )
        if deletion_match is None:
            # If the deletion based on checkPythonVersion is gone, that's fine
            return

        # If the deletion still exists, checkPythonVersion should return
        # a type that distinguishes "too old" from "error"
        func_match = re.search(
            r"function checkPythonVersion\(.*?\):\s*(.+?)\s*\{",
            INSTALLER_SOURCE,
            re.DOTALL,
        )
        assert func_match is not None, "checkPythonVersion function not found"
        return_type = func_match.group(1).strip()
        # Should not be just 'boolean' - needs to distinguish error from too-old
        assert return_type != "boolean", (
            "checkPythonVersion should not return plain boolean when its result "
            "is used to delete .venv. It should distinguish 'too old' from "
            "'check failed' so that transient errors don't cause .venv deletion."
        )


class TestConcurrencyGuard(unittest.TestCase):
    """ensureDependencies should be protected against concurrent calls.

    The extension can be activated multiple times in rapid succession
    (e.g., window reload + workspace change).  Without a guard, concurrent
    calls can race: one deletes .venv while another is using it.
    """

    def test_has_concurrency_guard(self) -> None:
        """ensureDependencies should have a re-entry guard."""
        # Look for some form of concurrency protection:
        # - a lock/mutex variable
        # - a pending promise check
        # - an early return if already running
        guard_patterns = [
            r"pendingDeps",
            r"depsInProgress",
            r"isRunning",
            r"depLock",
            r"activeDeps",
            r"ensureDepsPromise",
            r"if\s*\(\s*\w+Running\b",
        ]
        has_guard = any(
            re.search(p, INSTALLER_SOURCE) for p in guard_patterns
        )
        assert has_guard, (
            "ensureDependencies should have a concurrency guard to prevent "
            "multiple simultaneous calls from interfering with each other. "
            "The extension can be activated multiple times in rapid succession."
        )


if __name__ == "__main__":
    unittest.main()
