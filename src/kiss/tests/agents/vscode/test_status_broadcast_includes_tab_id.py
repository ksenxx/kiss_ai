"""Regression test: every status broadcast must explicitly include ``tabId``.

Bug context: the frontend ``setReady(label, ev.tabId)`` handler only
updates the header when the incoming event's ``tabId`` matches the
currently active tab (or is ``undefined``).  Previously, the backend
emitted ``{"type": "status", "running": True/False}`` *without* a
``tabId`` field and relied entirely on the
``BaseBrowserPrinter._inject_task_id`` middleware to add the field from
the thread-local context.  When the thread-local was unset or held an
empty string, the frontend received a tabId of ``""``, which failed the
``ev.tabId === activeTabId`` check and left the timer ticking forever.

This test pins the post-fix invariant: every ``{"type": "status", ...}``
dictionary literal constructed inside the VS Code backend modules must
include a ``"tabId"`` key explicitly, so the frontend's
tabId-matching logic always works regardless of what middleware does.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"

# Backend modules that ever broadcast ``status`` events.  When new
# modules add a status broadcast, append them here.
SOURCES = [
    VSCODE_DIR / "task_runner.py",
    VSCODE_DIR / "web_server.py",
    VSCODE_DIR / "commands.py",
]


def _iter_status_dict_literals(
    path: Path,
) -> list[tuple[int, ast.Dict]]:
    """Return ``(lineno, dict_node)`` for every ``{"type": "status", ...}``.

    Walks the AST of *path* and yields each ``ast.Dict`` whose ``"type"``
    key is the constant string ``"status"``.
    """
    tree = ast.parse(path.read_text())
    out: list[tuple[int, ast.Dict]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values, strict=False):
            if (
                isinstance(k, ast.Constant)
                and k.value == "type"
                and isinstance(v, ast.Constant)
                and v.value == "status"
            ):
                out.append((node.lineno, node))
                break
    return out


class TestStatusBroadcastIncludesTabId(unittest.TestCase):
    """Every status-broadcast dict literal must include ``tabId``."""

    def test_all_status_dicts_have_tab_id_key(self) -> None:
        offenders: list[str] = []
        total = 0
        for path in SOURCES:
            for lineno, node in _iter_status_dict_literals(path):
                total += 1
                keys: set[str] = {
                    k.value
                    for k in node.keys
                    if isinstance(k, ast.Constant)
                    and isinstance(k.value, str)
                }
                if "tabId" not in keys:
                    offenders.append(f"{path.name}:{lineno} keys={sorted(keys)}")
        self.assertEqual(
            offenders,
            [],
            "Every '{\"type\": \"status\", ...}' broadcast must include "
            "an explicit 'tabId' key so the frontend's tab-id matching "
            "logic works without relying on _inject_task_id middleware. "
            f"Offending sites: {offenders}",
        )
        # Sanity: we are actually inspecting real status broadcasts —
        # not silently passing because the search returned nothing.
        self.assertGreaterEqual(
            total,
            4,
            "Expected at least four status broadcast sites across the "
            f"VS Code backend; found only {total}.  Did the broadcast "
            "API move?  Update SOURCES in this test.",
        )

    def test_task_runner_status_broadcasts_use_tab_id_variable(self) -> None:
        """The two broadcasts in ``_run_task`` must pass the local ``tab_id``.

        Both the ``running=True`` (pre-task) and ``running=False``
        (post-task in ``finally``) broadcasts must read the same
        ``tab_id`` local that was extracted at the top of ``_run_task``
        from ``cmd.get("tabId", "")``.  This guarantees that whichever
        tab initiated the task is the one the frontend's matching
        logic finally resolves against.
        """
        path = VSCODE_DIR / "task_runner.py"
        literals = _iter_status_dict_literals(path)
        self.assertEqual(
            len(literals),
            2,
            "task_runner.py should contain exactly the two status "
            "broadcasts wrapping _run_task; if this count changes, the "
            "regression test needs updating.",
        )
        for lineno, node in literals:
            tab_id_value: ast.AST | None = None
            for k, v in zip(node.keys, node.values, strict=False):
                if isinstance(k, ast.Constant) and k.value == "tabId":
                    tab_id_value = v
                    break
            self.assertIsNotNone(
                tab_id_value,
                f"task_runner.py:{lineno} status dict missing 'tabId'",
            )
            assert tab_id_value is not None  # for type-checker
            self.assertIsInstance(
                tab_id_value,
                ast.Name,
                f"task_runner.py:{lineno} 'tabId' value must be the local "
                "variable 'tab_id', not a constant or expression.",
            )
            assert isinstance(tab_id_value, ast.Name)  # narrow
            self.assertEqual(
                tab_id_value.id,
                "tab_id",
                f"task_runner.py:{lineno} 'tabId' must be set to the "
                "tab_id local extracted from cmd at the start of "
                "_run_task.",
            )


if __name__ == "__main__":
    unittest.main()
