# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: ``WebUseTool.screenshot`` must respect worktree isolation.

Bug (bughunt round 9): the file tools (Read/Write/Edit) remap absolute
paths that point into the parent repo onto the equivalent path inside
the agent's live ``.kiss-worktrees/kiss_wt-*`` worktree, so nothing the
agent writes can leak into the user's main checkout.
``WebUseTool.screenshot`` did NOT apply the same remap: a model that
passed an absolute parent-repo path (ignoring the ``Work dir:`` hint)
saved the PNG straight into the user's main working tree — dirtying the
checkout outside the isolated worktree.

Uses a real headless Chromium via Playwright (no mocks).
"""

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool


@pytest.fixture
def fake_worktree(tmp_path):
    """A parent-repo directory with a live kiss worktree inside it."""
    repo = tmp_path / "repo"
    wt = repo / ".kiss-worktrees" / "kiss_wt-shot"
    wt.mkdir(parents=True)
    return repo, wt


def test_screenshot_parent_repo_path_remaps_into_worktree(fake_worktree):
    """An absolute parent-repo path is redirected into the live worktree."""
    repo, wt = fake_worktree
    tool = WebUseTool(headless=True, user_data_dir=None, work_dir=str(wt))
    try:
        tool.go_to_url("data:text/html,<h1>worktree isolation</h1>")
        result = tool.screenshot(str(repo / "shots" / "page.png"))
    finally:
        tool.close()

    assert result.startswith("Screenshot saved to "), result
    remapped = wt / "shots" / "page.png"
    leaked = repo / "shots" / "page.png"
    assert remapped.is_file(), (
        "screenshot must land inside the active worktree, got: " + result
    )
    assert not leaked.exists(), (
        "screenshot leaked into the parent repo's working tree: " + result
    )


def test_screenshot_relative_path_stays_anchored_in_work_dir(fake_worktree):
    """Relative paths keep anchoring at work_dir (regression guard)."""
    repo, wt = fake_worktree
    tool = WebUseTool(headless=True, user_data_dir=None, work_dir=str(wt))
    try:
        tool.go_to_url("data:text/html,<h1>relative</h1>")
        result = tool.screenshot("rel_shot.png")
    finally:
        tool.close()

    assert result.startswith("Screenshot saved to "), result
    assert (wt / "rel_shot.png").is_file()


def test_screenshot_outside_repo_absolute_path_untouched(tmp_path, fake_worktree):
    """Absolute paths outside the parent repo are not remapped."""
    repo, wt = fake_worktree
    elsewhere = tmp_path / "elsewhere"
    tool = WebUseTool(headless=True, user_data_dir=None, work_dir=str(wt))
    try:
        tool.go_to_url("data:text/html,<h1>outside</h1>")
        result = tool.screenshot(str(elsewhere / "out.png"))
    finally:
        tool.close()

    assert result.startswith("Screenshot saved to "), result
    assert (elsewhere / "out.png").is_file()
