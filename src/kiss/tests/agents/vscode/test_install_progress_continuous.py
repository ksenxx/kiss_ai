# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: the "KISS Sorcar: Setting up" progress notification
must stay visible from the start of the dependency install until the
final "Installation complete" notification is shown.

Previously, ``vscode.window.withProgress`` ran ONLY around the
dependency installation (uv, git, node, code-cli, python env,
playwright) and CLOSED before the post-install finalization steps:

  - ``installCliScript`` — write the ``sorcar`` wrapper
  - ``restartKissWebDaemon`` — kick the kiss-web daemon
  - ``ensurePathInShellRc`` — persist PATH entries
  - ``ensureApiKeys`` — prompt for ANTHROPIC_API_KEY / OPENAI_API_KEY
    when neither Claude CLI nor any key is set

This created a visible UX gap: the "Installing dependencies..." toast
disappeared, the user saw NO notification for several seconds (or
much longer if ``ensureApiKeys`` resolved without prompting), and
then the post-install toast finally appeared.

The fix is to run those finalization steps INSIDE the same
``withProgress`` callback so the notification stays open continuously.
This static-source test verifies it.

The test pattern (regex on the TypeScript source) mirrors the
companion test ``test_restart_notification_sticky.py`` so a future
refactor that re-introduces the gap is caught immediately.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

VSCODE_TS_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode" / "src"


def _read(name: str) -> str:
    return (VSCODE_TS_DIR / name).read_text()


def _extract_balanced_block(src: str, open_idx: int) -> tuple[int, int]:
    """Return (body_start, body_end_exclusive) for the brace-matched
    block whose opening ``{`` is at ``open_idx``.  Comments and string
    literals inside the body are NOT specially handled — sufficient
    here because DependencyInstaller.ts has no unbalanced braces in
    strings or comments inside the regions we inspect."""
    assert src[open_idx] == "{", (
        f"_extract_balanced_block: expected '{{' at index {open_idx}, got {src[open_idx]!r}"
    )
    depth = 1
    i = open_idx + 1
    body_start = i
    while i < len(src) and depth > 0:
        ch = src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return body_start, i
        i += 1
    raise AssertionError("Unmatched '{' — could not find balanced close")


class TestInstallProgressIsContinuous(unittest.TestCase):
    """All post-install finalization runs INSIDE the slow-path
    ``withProgress`` callback so the user sees a single continuous
    final "Installation complete" toast.

    The finalization steps are factored into a ``runFinalization``
    helper so the same code is reused by the fast path (which has no
    progress notification, but still runs CLI install / daemon
    restart / shell-rc / api-keys) and the slow path (which calls
    the helper from inside the withProgress callback).  These tests
    therefore verify both that ``runFinalization`` is invoked inside
    the slow-path callback AND that the helper body contains the
    expected finalization functions — together that proves each
    function runs while the progress notification is still visible."""

    src: str = ""
    progress_body: str = ""
    finalization_body: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _read("DependencyInstaller.ts")

        # Locate the slow-path withProgress call by its title string.
        title_idx = cls.src.find("'KISS Sorcar: Setting up'")
        assert title_idx >= 0, (
            "Could not find 'KISS Sorcar: Setting up' title in "
            "DependencyInstaller.ts; test needs to be updated."
        )

        # The callback body opens at the next `async progress => {`
        # following the title.
        cb_match = re.search(
            r"async\s+progress\s*=>\s*\{",
            cls.src[title_idx:],
        )
        assert cb_match is not None, (
            "Could not find `async progress => {` after withProgress "
            "title; test needs to be updated."
        )
        cb_open = title_idx + cb_match.end() - 1  # index of the `{`
        body_start, body_end = _extract_balanced_block(cls.src, cb_open)
        cls.progress_body = cls.src[body_start:body_end]

        # Locate the runFinalization helper body.
        fin_match = re.search(
            r"async\s+function\s+runFinalization\s*\([^)]*\)"
            r"\s*:\s*Promise<boolean>\s*\{",
            cls.src,
        )
        assert fin_match is not None, (
            "Could not find `async function runFinalization(...)"
            ": Promise<boolean> {` in DependencyInstaller.ts.  The "
            "fix must keep finalization in a single helper so both "
            "the fast and slow paths share it."
        )
        fin_open = fin_match.end() - 1
        fin_start, fin_end = _extract_balanced_block(cls.src, fin_open)
        cls.finalization_body = cls.src[fin_start:fin_end]

    # ------------------------------------------------------------------
    # Sanity: we extracted the right block.
    # ------------------------------------------------------------------

    def test_progress_body_contains_install_steps(self) -> None:
        self.assertIn("Installing uv", self.progress_body)
        self.assertIn("Installing dependencies", self.progress_body)
        self.assertIn("playwright", self.progress_body.lower())

    # ------------------------------------------------------------------
    # The actual regression: finalization steps must live INSIDE the
    # slow-path withProgress callback.
    # ------------------------------------------------------------------

    def test_run_finalization_called_inside_progress(self) -> None:
        """The finalization helper must be invoked from inside the
        slow-path ``withProgress`` callback, otherwise the
        notification closes before the steps it performs run."""
        self.assertRegex(
            self.progress_body,
            r"\brunFinalization\s*\(",
            "runFinalization() is not called inside the withProgress "
            "callback — the 'Installing dependencies' notification "
            "will close before any finalization step runs.",
        )

    def test_install_cli_script_runs_inside_finalization(self) -> None:
        """``installCliScript`` writes the ``sorcar`` wrapper; this
        must happen while the progress notification is still visible
        so the user does not see a notification gap."""
        self.assertRegex(
            self.finalization_body,
            r"\binstallCliScript\s*\(",
            "installCliScript() is not called inside runFinalization() "
            "— moving it out of the helper would re-introduce the "
            "notification gap.",
        )

    def test_restart_kiss_web_daemon_runs_inside_finalization(self) -> None:
        """The kiss-web daemon restart must happen inside the
        finalization helper (and thus inside the progress
        notification)."""
        self.assertRegex(
            self.finalization_body,
            r"\brestartKissWebDaemon\s*\(",
            "restartKissWebDaemon() is not called inside "
            "runFinalization() — the user sees no progress while the "
            "daemon restarts.",
        )

    def test_ensure_path_in_shell_rc_runs_inside_finalization(self) -> None:
        """Persisting PATH entries to ~/.zshrc / ~/.bashrc must
        happen inside the finalization helper."""
        self.assertRegex(
            self.finalization_body,
            r"\bensurePathInShellRc\s*\(",
            "ensurePathInShellRc() is not called inside "
            "runFinalization() — the user sees no progress while the "
            "shell rc file is being updated.",
        )

    def test_ensure_api_keys_runs_inside_finalization(self) -> None:
        """``ensureApiKeys`` must happen inside the finalization helper."""
        self.assertRegex(
            self.finalization_body,
            r"\bensureApiKeys\s*\(",
            "ensureApiKeys() is not awaited inside runFinalization() "
            "— the 'Installing dependencies' notification will close "
            "before the user is prompted for an API key, creating a "
            "visible UX gap.",
        )

    # ------------------------------------------------------------------
    # Negative: the post-progress region must NOT re-call any of these
    # finalization functions.  Otherwise the gap would still exist.
    # ------------------------------------------------------------------



if __name__ == "__main__":
    unittest.main()
