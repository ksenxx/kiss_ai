"""Regression test: the post-install "Restart VS Code" notification
must stay visible until the user clicks the action button.

VS Code may auto-hide information notifications (depending on user
settings or notification-center state) and the user can dismiss the
toast with the close button.  The implementation in
``DependencyInstaller.ts`` therefore wraps ``showInformationMessage``
in a loop that re-shows the prompt whenever the API resolves to
``undefined`` (auto-hide or dismiss) and only exits when the user
clicks ``"Restart VS Code"``.

This test grep-audits the source so a future refactor cannot silently
drop the loop.
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

    src: str = ""
    block: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _read("DependencyInstaller.ts")
        # Extract the block from "if (apiKeysReady) {" up to the
        # matching "} else {" — that is the success-path notification
        # code we are auditing.
        m = re.search(
            r"if\s*\(\s*apiKeysReady\s*\)\s*\{(?P<body>.*?)\}\s*else\s*\{",
            cls.src,
            re.DOTALL,
        )
        assert m is not None, (
            "Could not locate `if (apiKeysReady) { ... } else {` block "
            "in DependencyInstaller.ts; test needs to be updated."
        )
        cls.block = m.group("body")

    def test_block_contains_restart_label(self) -> None:
        """Sanity check: we extracted the right block."""
        self.assertIn("'Restart VS Code'", self.block)
        self.assertIn(
            "workbench.action.reloadWindow",
            self.block,
            "Restart action must trigger reloadWindow",
        )

    def test_notification_is_in_a_loop(self) -> None:
        """The ``showInformationMessage`` call must live inside an
        unbounded loop so a dismissed/auto-hidden notification is
        re-shown.  We accept any of ``while (true)``,
        ``while(true)``, or ``for (;;)`` / ``for(;;)``."""
        loop_re = r"(while\s*\(\s*true\s*\)|for\s*\(\s*;\s*;\s*\))"
        loop_match = re.search(loop_re, self.block)
        self.assertIsNotNone(
            loop_match,
            "Restart notification is not wrapped in a loop — it can "
            "auto-hide before the user clicks 'Restart VS Code'.  "
            "Wrap showInformationMessage in `for (;;) { ... }` and "
            "re-show until choice === 'Restart VS Code'.",
        )
        # The showInformationMessage call (not just the identifier in
        # a comment) must come AFTER the loop opener.  We strip line
        # comments first so a documentation reference can't satisfy
        # the test.
        stripped = re.sub(r"//[^\n]*", "", self.block)
        stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
        # Re-find the loop in the stripped text since indices changed.
        s_loop = re.search(loop_re, stripped)
        self.assertIsNotNone(s_loop)
        assert s_loop is not None
        s_show = stripped.find("showInformationMessage(")
        self.assertGreaterEqual(
            s_show, 0,
            "No showInformationMessage(...) call found in success "
            "block.",
        )
        self.assertGreater(
            s_show, s_loop.start(),
            "showInformationMessage call must be inside the loop body, "
            "not before it.",
        )

    def test_loop_exits_only_on_restart_choice(self) -> None:
        """The only ``return`` / ``break`` inside the loop body must
        be guarded by a ``=== 'Restart VS Code'`` check, otherwise the
        loop would exit on the first auto-hide and the notification
        would not be sticky."""
        # Find the loop body — substring from the loop opener to the
        # matching close brace.  Simple brace counter.
        m = re.search(
            r"(while\s*\(\s*true\s*\)|for\s*\(\s*;\s*;\s*\))\s*\{",
            self.block,
        )
        self.assertIsNotNone(m)
        assert m is not None
        start = m.end()
        depth = 1
        i = start
        while i < len(self.block) and depth > 0:
            ch = self.block[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        loop_body = self.block[start:i - 1]

        # Every `return` or `break` in the loop body must be inside an
        # `if (... === 'Restart VS Code')` branch.  We verify by
        # checking that for each return/break occurrence, the nearest
        # preceding `if (...)` mentions the restart label.
        for kw in ("return", "break"):
            for match in re.finditer(rf"\b{kw}\b", loop_body):
                pos = match.start()
                preceding = loop_body[:pos]
                # Find the last `if (...)` before this position.
                if_match = None
                for cand in re.finditer(r"if\s*\([^)]*\)", preceding):
                    if_match = cand
                self.assertIsNotNone(
                    if_match,
                    f"`{kw}` in the loop body is not guarded by any "
                    f"if-check — the loop would exit before the user "
                    f"clicks 'Restart VS Code'.",
                )
                assert if_match is not None
                self.assertIn(
                    "'Restart VS Code'",
                    if_match.group(0),
                    f"`{kw}` is guarded by `{if_match.group(0)}` but "
                    f"that condition does not check for the "
                    f"'Restart VS Code' choice — the loop could exit "
                    f"on auto-hide / dismiss.",
                )

    def test_no_naked_then_chain_outside_loop(self) -> None:
        """The block must not contain a top-level
        ``showInformationMessage(...).then(...)`` outside the loop —
        that pattern was the pre-fix shape and is not sticky."""
        # The naive pattern: showInformationMessage(...).then( before
        # any loop construct.
        body = self.block
        loop_match = re.search(
            r"(while\s*\(\s*true\s*\)|for\s*\(\s*;\s*;\s*\))",
            body,
        )
        loop_idx = loop_match.start() if loop_match else len(body)
        head = body[:loop_idx]
        self.assertNotRegex(
            head,
            r"showInformationMessage\([^)]*\)\s*\.then\s*\(",
            "Found `showInformationMessage(...).then(...)` BEFORE the "
            "loop — that pattern auto-dismisses.  Move the call "
            "inside the loop and await it.",
        )


if __name__ == "__main__":
    unittest.main()
