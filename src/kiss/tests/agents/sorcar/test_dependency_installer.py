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
        catch_match = re.search(
            r"\.catch\(\s*(?:async\s*)?\(?(?:\w+)?\)?\s*=>\s*\{[^}]*Fast-path Playwright",
            INSTALLER_SOURCE,
            re.DOTALL,
        )
        assert catch_match is not None, (
            "Expected a .catch handler with 'Fast-path Playwright' log message"
        )

        catch_block_start = catch_match.start()
        warning_match = re.search(
            r"Chromium browser update failed in background",
            INSTALLER_SOURCE[catch_block_start:],
        )
        if warning_match:
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

        deletion_match = re.search(
            r"checkPythonVersion.*\n.*rmSync.*\.venv",
            INSTALLER_SOURCE,
            re.DOTALL,
        )
        if deletion_match is None:
            return

        func_match = re.search(
            r"function checkPythonVersion\(.*?\):\s*(.+?)\s*\{",
            INSTALLER_SOURCE,
            re.DOTALL,
        )
        assert func_match is not None, "checkPythonVersion function not found"
        return_type = func_match.group(1).strip()
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


class TestEarlyExitGuard(unittest.TestCase):
    """ensureDependenciesImpl should return immediately when all dependencies
    are fully installed and the daemon is running, avoiding heavyweight
    operations (Playwright download, daemon restart, CLI rewrite) on every
    VS Code activation."""

    def test_early_exit_checks_all_four_conditions(self) -> None:
        """The early-exit guard must check uv, .venv, Chromium, and daemon."""
        for condition in ["findUvPath()", ".venv", "isChromiumInstalled()", "isDaemonRunning()"]:
            assert condition in INSTALLER_SOURCE, (
                f"Early-exit guard must check {condition}"
            )

    def test_early_exit_checks_update_marker(self) -> None:
        """The early-exit guard must not trigger when the extension-updated
        marker exists, so that code changes from build-extension.sh are
        picked up via the normal restart path."""
        # Find the early-exit block (before the main uvPath / venvExists logic)
        guard_match = re.search(
            r"isDaemonRunning\(\).*?!fs\.existsSync\(updateMarker\)",
            INSTALLER_SOURCE,
            re.DOTALL,
        )
        assert guard_match is not None, (
            "Early-exit guard must check that the .extension-updated marker "
            "does NOT exist before returning early"
        )

    def test_early_exit_loads_api_keys(self) -> None:
        """The early-exit path must still call loadApiKeysFromShellRc so that
        API keys from ~/.zshrc are available when VS Code is launched from
        macOS Dock/Spotlight."""
        # Find the early-exit block and verify loadApiKeysFromShellRc is called
        nothing_match = re.search(
            r"nothing to do.*?loadApiKeysFromShellRc\(\)",
            INSTALLER_SOURCE,
            re.DOTALL,
        )
        assert nothing_match is not None, (
            "Early-exit path must call loadApiKeysFromShellRc() to populate "
            "API keys for macOS Dock launches"
        )

    def test_early_exit_returns_before_playwright_install(self) -> None:
        """The early-exit must happen before any playwright install command."""
        guard_pos = INSTALLER_SOURCE.find("nothing to do")
        assert guard_pos > 0, "Early-exit log message not found"
        playwright_pos = INSTALLER_SOURCE.find("playwright', 'install', 'chromium'")
        assert playwright_pos > 0, "Playwright install command not found"
        assert guard_pos < playwright_pos, (
            "Early-exit guard must appear before the Playwright install command"
        )

    def test_early_exit_returns_before_restart_daemon(self) -> None:
        """The early-exit must happen before restartKissWebDaemon."""
        guard_pos = INSTALLER_SOURCE.find("nothing to do")
        assert guard_pos > 0
        daemon_pos = INSTALLER_SOURCE.find("restartKissWebDaemon(kissProjectPath)")
        assert daemon_pos > 0, "restartKissWebDaemon call not found"
        assert guard_pos < daemon_pos, (
            "Early-exit guard must appear before the restartKissWebDaemon call"
        )


class TestIsDaemonRunning(unittest.TestCase):
    """isDaemonRunning function should exist and check port 8787."""

    def test_is_daemon_running_exists(self) -> None:
        """isDaemonRunning function must be defined."""
        assert "function isDaemonRunning()" in INSTALLER_SOURCE

    def test_checks_port_8787(self) -> None:
        """isDaemonRunning must probe port 8787."""
        func_match = re.search(
            r"function isDaemonRunning\(\).*?\n\}",
            INSTALLER_SOURCE,
            re.DOTALL,
        )
        assert func_match is not None
        assert "8787" in func_match.group(0), (
            "isDaemonRunning must check port 8787"
        )


if __name__ == "__main__":
    unittest.main()
