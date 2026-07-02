# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 2: end-to-end tests for useful_tools.py and web_use_tool.py.

Bug 1 — vanished-worktree inconsistency in ``UsefulTools``:
    ``UsefulTools._spawn`` transparently falls back to the parent repo
    root when ``work_dir`` points at a torn-down
    ``.kiss-worktrees/kiss_wt-*`` directory (pinned by
    ``test_worktree_disappears_mid_run.py``).  But the sibling code
    paths did NOT apply the same fallback:

    * ``Read``/``Edit`` of an *existing* parent-repo file remapped the
      path into the deleted worktree and reported "File not found"
      (``Read`` even suggested "Did you mean: <the exact same path>?").
    * ``Write`` silently resurrected a zombie worktree directory and
      wrote the file there — the data never reached the user's checkout
      and could never be auto-committed/merged (the worktree is already
      torn down).
    * ``_bash_parent_repo_guard`` refused a command referencing the
      parent-repo path and told the model to "rewrite the command to
      use the worktree path" — a path that no longer exists — while
      ``_spawn`` itself would have run the command with cwd = parent
      repo.  A dead end for the model.

Bug 2 — accessible names containing double quotes break
    ``WebUseTool`` element resolution:
    Playwright's ``aria_snapshot()`` escapes double quotes inside
    accessible names (``- button "Say \\"hi\\" now"``).  The name
    extraction regex in ``_number_interactive_elements`` stopped at the
    first quote and recorded the name ``Say \\`` — so
    ``_resolve_locator``'s ``get_by_role(role, name=..., exact=True)``
    matched nothing and every click/type on such an element failed with
    "Element with ID N not found on page.".  Backslashes (escaped as
    ``\\\\``) were similarly corrupted.  Worse, when the name contains
    ``": "`` Playwright single-quote-wraps the whole YAML key
    (``- 'link "colon: \\"q\\""':``), and the role-line regex failed to
    match at all — the element was silently never numbered and thus
    invisible to the model as an interactive element.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.agents.sorcar.web_use_tool import _number_interactive_elements

# ---------------------------------------------------------------------------
# Bug 1 — vanished worktree: Read/Write/Edit/Bash-guard must fall back to
# the parent repo, consistent with UsefulTools._spawn.
# ---------------------------------------------------------------------------


@pytest.fixture()
def stale_worktree_setup(tmp_path):
    """A parent repo dir plus a work_dir pointing at a DELETED worktree."""
    repo = Path(os.path.realpath(tmp_path)) / "repo"
    repo.mkdir()
    (repo / "f.txt").write_text("hello parent\n")
    stale_wt = repo / ".kiss-worktrees" / "kiss_wt-gone"
    # NOTE: stale_wt is intentionally never created (worktree torn down).
    tools = UsefulTools(work_dir=str(stale_wt))
    return repo, stale_wt, tools


class TestVanishedWorktreeFallback:
    """Read/Write/Edit/Bash must not dead-end when the worktree is gone."""

    def test_read_parent_path_after_worktree_vanished(self, stale_worktree_setup):
        """Read of an existing parent-repo file must return its content."""
        repo, _stale_wt, tools = stale_worktree_setup
        result = tools.Read(str(repo / "f.txt"))
        assert result == "hello parent\n", result

    def test_write_parent_path_does_not_resurrect_zombie_worktree(
        self, stale_worktree_setup
    ):
        """Write must land in the parent repo, not a resurrected zombie dir."""
        repo, stale_wt, tools = stale_worktree_setup
        result = tools.Write(str(repo / "new.txt"), "data")
        assert result.startswith("Successfully wrote"), result
        assert (repo / "new.txt").exists(), (
            "Write did not land in the parent repo"
        )
        assert not stale_wt.exists(), (
            "Write resurrected the torn-down worktree directory"
        )

    def test_edit_parent_path_after_worktree_vanished(self, stale_worktree_setup):
        """Edit of an existing parent-repo file must apply the edit."""
        repo, _stale_wt, tools = stale_worktree_setup
        result = tools.Edit(str(repo / "f.txt"), "hello", "bye")
        assert result.startswith("Successfully replaced"), result
        assert (repo / "f.txt").read_text() == "bye parent\n"

    def test_bash_guard_allows_parent_path_after_worktree_vanished(
        self, stale_worktree_setup
    ):
        """Bash must run a parent-repo-path command (consistent with _spawn).

        ``_spawn`` already falls back to running with cwd = parent repo
        when the worktree vanished, so refusing the command and telling
        the model to use the (nonexistent) worktree path is a dead end.
        """
        repo, _stale_wt, tools = stale_worktree_setup
        result = tools.Bash(f"cat {repo}/f.txt", "read file")
        assert "hello parent" in result, result

    def test_live_worktree_still_remaps_and_guards(self, tmp_path):
        """Regression: with a LIVE worktree the remap and guard still apply."""
        repo = Path(os.path.realpath(tmp_path)) / "repo"
        repo.mkdir()
        (repo / "f.txt").write_text("main content\n")
        wt = repo / ".kiss-worktrees" / "kiss_wt-live"
        wt.mkdir(parents=True)
        (wt / "f.txt").write_text("worktree content\n")
        tools = UsefulTools(work_dir=str(wt))
        # Read of the parent-repo path must observe the worktree copy.
        assert tools.Read(str(repo / "f.txt")) == "worktree content\n"
        # Write via parent path must land in the worktree.
        tools.Write(str(repo / "n.txt"), "x")
        assert (wt / "n.txt").exists()
        assert not (repo / "n.txt").exists()
        # Bash referencing the parent-repo path must still be refused.
        out = tools.Bash(f"echo hi > {repo}/f.txt", "write file")
        assert "parent-repo path" in out, out
        assert (repo / "f.txt").read_text() == "main content\n"


# ---------------------------------------------------------------------------
# Bug 2 — aria snapshot name extraction: escaped quotes/backslashes and
# single-quote-wrapped YAML keys.
# ---------------------------------------------------------------------------


class TestAriaSnapshotNameExtraction:
    """_number_interactive_elements must decode Playwright's YAML escaping."""

    def test_name_with_escaped_double_quotes(self):
        """``<button>Say "hi" now</button>`` snapshots as
        ``- button "Say \\"hi\\" now"`` — the recorded name must be the
        real accessible name so get_by_role(name=..., exact=True) matches.
        """
        snapshot = '- button "Say \\"hi\\" now"'
        numbered, elements = _number_interactive_elements(snapshot)
        assert len(elements) == 1, elements
        assert elements[0]["role"] == "button"
        assert elements[0]["name"] == 'Say "hi" now', elements[0]["name"]
        assert numbered.startswith("- [1] button"), numbered

    def test_name_with_escaped_backslash(self):
        """``back\\slash x`` snapshots as ``- button "back\\\\slash x"``."""
        snapshot = '- button "back\\\\slash x"'
        _numbered, elements = _number_interactive_elements(snapshot)
        assert len(elements) == 1, elements
        assert elements[0]["name"] == "back\\slash x", elements[0]["name"]

    def test_single_quote_wrapped_role_line_is_numbered(self):
        """Names containing ``": "`` make Playwright single-quote-wrap the
        whole YAML key: ``- 'link "colon: \\"q\\""':``.  The element must
        still be numbered (visible to the model) with the decoded name.
        """
        snapshot = '- \'link "colon: \\"q\\""\':\n  - /url: "#"'
        _numbered, elements = _number_interactive_elements(snapshot)
        assert len(elements) == 1, elements
        assert elements[0]["role"] == "link"
        assert elements[0]["name"] == 'colon: "q"', elements[0]["name"]

    def test_plain_names_unchanged(self):
        """Regression: plain unescaped names keep working."""
        snapshot = '- link "plain":\n  - /url: "#"\n- button "Submit"'
        numbered, elements = _number_interactive_elements(snapshot)
        assert [e["name"] for e in elements] == ["plain", "Submit"]
        assert "- [1] link" in numbered
        assert "- [2] button" in numbered

    def test_end_to_end_click_element_with_quoted_name(self):
        """Full-stack check: a real Chromium page with a quoted-name button
        must be numbered and resolvable (clickable) via WebUseTool's own
        locator logic."""
        from playwright.sync_api import sync_playwright

        from kiss.agents.sorcar.web_use_tool import WebUseTool

        tool = WebUseTool(headless=True, user_data_dir=None)
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_content(
                    '<button onclick="this.textContent=\'done\'">'
                    'Say "hi" now</button>'
                )
                # Wire the tool to the live page without launching its
                # own browser (still exercises the real logic layer).
                tool._playwright = p
                tool._browser = browser
                tool._context = page.context
                tool._page = page
                tree = tool._get_ax_tree()
                assert "[1] button" in tree, tree
                locator = tool._resolve_locator(1)
                assert locator.count() == 1
                locator.click()
                assert page.locator("button").inner_text() == "done"
                browser.close()
        finally:
            tool._playwright = None
            tool._browser = None
            tool._context = None
            tool._page = None
            tool.close()
