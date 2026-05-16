"""Regression test: the post-install notification must NOT include a
"Restart VS Code" action button, and must NOT trigger
``workbench.action.reloadWindow`` from the post-install flow.

History: an earlier version of this extension showed a sticky
"Restart VS Code" prompt at the end of ``ensureDependencies``.  That
prompt is now removed: a window reload is not needed because the
``fs.watchFile`` watchers in ``extension.ts`` already auto-reload the
window when a fresh VSIX is installed, and the current Node host
already has the updated ``PATH``, API keys, venv, and daemon by the
time finalization runs.  The only thing a reload would refresh is
already-open integrated terminals — and a reload doesn't really fix
that either; opening a new terminal does.

This file used to enforce that the prompt was sticky.  It now
enforces the opposite: the prompt is gone.  Together with
``test_install_progress_continuous.py`` and the watcher tests, this
guarantees the no-restart contract.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

VSCODE_TS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "vscode" / "src"
)


def _read(name: str) -> str:
    return (VSCODE_TS_DIR / name).read_text()


class TestRestartNotificationSticky(unittest.TestCase):
    """The post-install restart notification must be sticky."""
def _strip_comments(src: str) -> str:
    """Strip line and block comments — needed because the source
    contains long explanatory comments that reference the historical
    'Restart VS Code' label, and those would create false positives
    for the assertions below."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", "", src)
    return src


class TestNoRestartPromptInDependencyInstaller(unittest.TestCase):
    post-install flow."""

    src: str = ""
    src_no_comments: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _read("DependencyInstaller.ts")
        cls.src_no_comments = _strip_comments(cls.src)

    def test_no_restart_vs_code_label(self) -> None:
        """No string literal in non-comment code may carry the
        ``'Restart VS Code'`` action label.  That label was the action
        button on the prompt we removed."""
        self.assertNotIn(
            "Restart VS Code",
            self.src_no_comments,
            "Found a 'Restart VS Code' label in DependencyInstaller.ts "
            "non-comment code.  The post-install restart prompt has "
            "been removed deliberately — a window reload is not "
            "required after dependency setup (the auto-reload "
            "watchers in extension.ts already handle the only case "
            "where new extension code needs to be loaded).",
        )

    def test_no_reload_window_call(self) -> None:
        """The dependency installer must not call
        ``workbench.action.reloadWindow``.  The only legitimate
        callers of that command in this extension are the auto-reload
        watchers in ``extension.ts`` (which fire when a freshly
        installed VSIX needs to be loaded)."""
        self.assertNotIn(
            "workbench.action.reloadWindow",
            self.src_no_comments,
            "DependencyInstaller.ts triggers a window reload from the "
            "post-install flow.  Reload should only be triggered "
            "automatically by the file watchers in extension.ts when "
            "the extension code itself changes — not as a "
            "post-install nag.",
        )

    def test_no_sticky_loop_calling_show_information_message(self) -> None:
        """The pre-fix shape was a ``for (;;)`` loop wrapping
        ``showInformationMessage`` so the user could not dismiss the
        prompt.  That loop must not be present."""
        # Find every for(;;) / while(true) and ensure no
        # showInformationMessage call sits inside it.
        for loop_match in re.finditer(
            r"(while\s*\(\s*true\s*\)|for\s*\(\s*;\s*;\s*\))\s*\{",
            self.src_no_comments,
        ):
            start = loop_match.end()
            depth = 1
            i = start
            while i < len(self.src_no_comments) and depth > 0:
                ch = self.src_no_comments[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                i += 1
            body = self.src_no_comments[start:i - 1]
            self.assertNotIn(
                "showInformationMessage(",
                body,
                "Found a `for (;;)` / `while (true)` loop wrapping "
                "a showInformationMessage() call — that was the "
                "shape of the sticky restart prompt we deliberately "
                "removed.",
            )

    def test_installation_complete_message_present(self) -> None:
        """The replacement notification — a single non-prompting
        ``showInformationMessage`` saying installation is complete —
        must still exist so the user knows setup finished."""
        self.assertRegex(
            self.src_no_comments,
            r"showInformationMessage\([^)]*Installation complete",
            "The non-prompting 'Installation complete' notification "
            "is missing.  Users should still see a clear indicator "
            "that dependency setup finished successfully — just "
            "without a forced reload.",
        )


class TestNoStraySorcarReloadOutsideWatchers(unittest.TestCase):
    """Sanity: the only ``workbench.action.reloadWindow`` call in the
    extension source must live in ``extension.ts`` and must be
    triggered by the auto-reload watchers, never from a notification
    handler."""

    def test_reload_only_in_watcher_path(self) -> None:
        src = _read("extension.ts")
        src_no_comments = _strip_comments(src)
        # extension.ts is allowed exactly one reloadWindow call (the
        # auto-reload trigger inside triggerReload()).
        n = src_no_comments.count("workbench.action.reloadWindow")
        self.assertEqual(
            n, 1,
            f"extension.ts contains {n} reloadWindow calls — expected "
            f"exactly 1 (the watcher-driven auto-reload).  More than "
            f"one suggests a notification-driven reload has been "
            f"re-introduced.",
        )


if __name__ == "__main__":
    unittest.main()
