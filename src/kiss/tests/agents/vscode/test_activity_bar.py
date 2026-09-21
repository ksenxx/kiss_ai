# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` module fixture is imported from
#   kiss.tests.server.test_explorer_scm_commands and is intentionally
#   shadowed by test parameters of the same name)
"""End-to-end browser tests for the remote webapp's activity bar.

The task-history panel of the remote webapp carries a VS Code-like
activity bar on its left edge with three views:

* **Tasks** — the history panel contents;
* **Explorer** — the workspace folder tree (``listDir`` →
  ``dirListing``); a folder click lists it in place, a file click opens
  the file as a content tab (``openFile`` → ``fileContent``);
* **Source Control** — the working tree's changes (``gitStatus``) and
  a graph of the recent commits (``gitLog``), each expandable to the
  files it modified.

Every test drives a REAL headless Chromium (Playwright) against a REAL
:class:`RemoteAccessServer` over ``wss://`` whose work dir is a REAL
git repository (see ``build_repo``) — no mocks.
"""

from __future__ import annotations

import json
import os

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from kiss.tests.server.test_explorer_scm_commands import (
    harness,  # noqa: F401  (module fixture used by param name)
)


@pytest.fixture(scope="module")
def browser():
    """One shared headless Chromium for every test in this module."""
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


def _open_page(browser, harness, width: int = 1400):
    """Open the remote page in desktop mode and record sent WS frames.

    Returns ``(context, page, sent_frames)``; *sent_frames* is a live
    list of the JSON frames the page sent over its WebSocket.
    """
    context = browser.new_context(
        ignore_https_errors=True, viewport={"width": width, "height": 900},
    )
    page = context.new_page()
    sent_frames: list[dict] = []

    def _on_ws(ws) -> None:
        def _on_sent(payload) -> None:
            try:
                sent_frames.append(json.loads(payload))
            except Exception:
                pass

        ws.on("framesent", _on_sent)

    page.on("websocket", _on_ws)
    for attempt in range(3):
        page.goto(harness.base_url + "/")
        try:
            page.wait_for_selector("#task-input", state="visible", timeout=30000)
            break
        except PlaywrightTimeoutError:
            # Under full-suite parallel load the page's first WS
            # connect/auth round-trip can stall past the wait while the
            # element stays hidden; a fresh navigation re-establishes
            # the socket.  Persistent invisibility still fails.
            if attempt == 2:
                raise
    page.wait_for_selector("body.remote-desktop", state="attached")
    # The workspace is known once the config reply landed.
    page.wait_for_function(
        "document.getElementById('meta-workdir').textContent.length > 1",
        timeout=30000,
    )
    return context, page, sent_frames


def _explorer_row_sel(suffix: str, cls: str = "") -> str:
    """Selector for the Explorer row whose path ends with *suffix*.

    *suffix* is written with ``/``; the daemon reports native paths, so
    on Windows the separators become ``\\`` (escaped for the CSS
    attribute string).  *cls* is an optional extra class such as
    ``.is-dir``.
    """
    native = suffix.replace("/", os.sep).replace("\\", "\\\\")
    return f".explorer-row{cls}[data-explorer-path$='{native}']"


def _focus_ends_with(suffix: str) -> str:
    """JS expression: the focused Explorer row's path ends with *suffix*
    (written with ``/``; the daemon reports native separators)."""
    return (
        "document.activeElement.dataset.explorerPath.endsWith("
        f"{json.dumps(suffix.replace('/', os.sep))})"
    )


def _explorer_row(page, name: str):
    return page.locator(_explorer_row_sel("/" + name))


def _sent(frames: list[dict], kind: str) -> list[dict]:
    return [f for f in frames if f.get("type") == kind]


def test_activity_bar_shows_three_views_with_tasks_first(browser, harness):
    """The bar sits flush with the panel's left edge and offers Tasks,
    Explorer and Source Control; Tasks (the history panel) is up."""
    context, page, _ = _open_page(browser, harness)
    try:
        bar = page.locator("#activity-bar")
        assert bar.is_visible()
        labels = page.eval_on_selector_all(
            "#activity-bar .activity-btn",
            "els => els.map(e => e.getAttribute('aria-label'))",
        )
        assert labels == ["Tasks", "Explorer", "Source Control"]
        box = bar.bounding_box()
        assert box is not None
        sidebar_box = page.locator("#sidebar").bounding_box()
        assert sidebar_box is not None
        assert box["x"] == pytest.approx(sidebar_box["x"], abs=1)
        assert box["y"] == pytest.approx(sidebar_box["y"], abs=1)
        assert box["height"] == pytest.approx(sidebar_box["height"], abs=1)
        assert box["width"] == pytest.approx(40, abs=1)
        # The bar is the leftmost thing in the panel: the views start
        # to its right.
        views_box = page.locator("#sidebar-views").bounding_box()
        assert views_box is not None
        assert views_box["x"] >= box["x"] + box["width"]
        assert page.locator("#activity-tasks").get_attribute("aria-selected") == "true"
        assert page.locator("#sidebar-tab-history-panel").is_visible()
        assert page.locator("#history-list").is_visible()
        assert page.locator("#sidebar-explorer-panel").is_hidden()
        assert page.locator("#sidebar-scm-panel").is_hidden()
    finally:
        context.close()


def test_activity_bar_is_a_remote_only_surface(browser, harness):
    """Without body.remote-chat (the VS Code webview) the bar is not
    rendered and the history panel keeps the whole width."""
    context, page, _ = _open_page(browser, harness)
    try:
        display = page.evaluate(
            """() => {
              document.body.classList.remove('remote-chat');
              return getComputedStyle(
                document.getElementById('activity-bar')).display;
            }"""
        )
        assert display == "none"
        assert page.locator("#sidebar-tab-history-panel").is_visible()
    finally:
        context.close()


def test_switching_views_shows_one_panel_at_a_time(browser, harness):
    context, page, _ = _open_page(browser, harness)
    try:
        page.click("#activity-explorer")
        assert page.locator("#sidebar-explorer-panel").is_visible()
        assert page.locator("#sidebar-tab-history-panel").is_hidden()
        assert page.locator("#sidebar-scm-panel").is_hidden()
        assert page.locator("#activity-explorer").get_attribute("aria-selected") == "true"
        assert page.locator("#activity-tasks").get_attribute("aria-selected") == "false"
        page.click("#activity-scm")
        assert page.locator("#sidebar-scm-panel").is_visible()
        assert page.locator("#sidebar-explorer-panel").is_hidden()
        page.click("#activity-tasks")
        assert page.locator("#sidebar-tab-history-panel").is_visible()
        assert page.locator("#history-list").is_visible()
        assert page.locator("#sidebar-scm-panel").is_hidden()
        assert page.locator("#activity-tasks").get_attribute("aria-selected") == "true"
    finally:
        context.close()


def test_explorer_lists_the_workspace_and_expands_folders(browser, harness):
    context, page, frames = _open_page(browser, harness)
    try:
        page.click("#activity-explorer")
        page.wait_for_selector(".explorer-row.is-file", timeout=15000)
        root = page.locator(".explorer-row[aria-level='1']")
        assert root.inner_text().strip() == "repo"
        assert "expanded" in (root.get_attribute("class") or "")
        names = page.eval_on_selector_all(
            ".explorer-row[aria-level='2'] .explorer-name",
            "els => els.map(e => e.textContent)",
        )
        assert names == [
            "dir", "feature.txt", "main-only.txt", "README.md",
            "untracked.txt",
        ]
        first = _sent(frames, "listDir")[0]
        assert first["path"] == str(harness.work_dir)
        assert first["workDir"] == str(harness.work_dir)
        # Expand a folder: its children are listed in place, one level
        # deeper; a second click collapses them again.
        assert _explorer_row(page, "dir/nested.py").count() == 0
        _explorer_row(page, "dir").click()
        page.wait_for_selector(
            _explorer_row_sel("/dir/nested.py"),
            timeout=15000,
        )
        nested = _explorer_row(page, "dir/nested.py")
        assert nested.get_attribute("aria-level") == "3"
        assert nested.is_visible()
        assert _explorer_row(page, "dir").get_attribute("aria-expanded") == "true"
        _explorer_row(page, "dir").click()
        assert nested.is_hidden()
        assert _explorer_row(page, "dir").get_attribute("aria-expanded") == "false"
        # Re-expanding does not ask the daemon again.
        n_before = len(_sent(frames, "listDir"))
        _explorer_row(page, "dir").click()
        assert nested.is_visible()
        assert len(_sent(frames, "listDir")) == n_before
    finally:
        context.close()


def test_explorer_file_click_opens_a_content_tab(browser, harness):
    context, page, frames = _open_page(browser, harness)
    try:
        page.click("#activity-explorer")
        page.wait_for_selector(".explorer-row.is-file", timeout=15000)
        tabs_before = page.locator(".chat-tab").count()
        _explorer_row(page, "feature.txt").click()
        page.wait_for_function(
            f"document.querySelectorAll('.chat-tab').length === {tabs_before + 1}",
            timeout=15000,
        )
        titles = page.eval_on_selector_all(
            ".chat-tab", "els => els.map(e => e.textContent)",
        )
        assert any("feature.txt" in t for t in titles)
        opened = _sent(frames, "openFile")
        assert len(opened) == 1
        assert opened[0]["path"] == str(harness.work_dir.resolve() / "feature.txt")
        assert opened[0]["workDir"] == str(harness.work_dir)
        # The file's own text is on screen in the content view (Monaco
        # or its <pre> fallback) — a sentinel that appears nowhere else
        # on the page.
        page.wait_for_function(
            "document.querySelector('.content-tab-view') && "
            "document.querySelector('.content-tab-view').innerText"
            ".includes('feature-file-sentinel-7c1e')",
            timeout=15000,
        )
        # The Explorer still shows the same workspace while the content
        # tab is active, and the docked panel stays open.
        assert page.locator(".explorer-row[aria-level='1']").inner_text().strip() == "repo"
        assert page.locator("#sidebar").evaluate("el => el.classList.contains('open')")
        # No chat tab was closed or created on the daemon by the open.
        assert not _sent(frames, "closeTab")
        assert not _sent(frames, "newChat")
    finally:
        context.close()


def test_source_control_shows_changes_and_commit_graph(browser, harness):
    context, page, frames = _open_page(browser, harness)
    try:
        page.click("#activity-scm")
        page.wait_for_selector("#scm-changes .scm-row", timeout=15000)
        page.wait_for_selector("#scm-graph .scm-commit", timeout=15000)
        assert page.locator("#scm-branch").inner_text() == "main"
        assert page.locator("#scm-changes-count").inner_text() == "4"
        group_hdrs = page.eval_on_selector_all(
            "#scm-changes .scm-group-hdr", "els => els.map(e => e.textContent)",
        )
        assert group_hdrs == ["Staged Changes (1)", "Changes (3)"]
        rows = page.eval_on_selector_all(
            "#scm-changes .scm-row",
            "els => els.map(e => [e.querySelector('.scm-path').textContent,"
            " e.querySelector('.scm-dir').textContent,"
            " e.querySelector('.scm-status').textContent])",
        )
        assert rows[0] == ["nested.py", "dir", "M"]
        assert sorted(r[2] for r in rows[1:]) == ["D", "M", "U"]
        deleted = page.locator("#scm-changes .scm-row.is-deleted")
        assert deleted.count() == 1
        assert deleted.get_attribute("data-scm-status") == "D"
        # The graph: the uncommitted-changes row, then the five commits
        # newest first; the merge commit carries the branch and tag.
        subjects = page.eval_on_selector_all(
            "#scm-graph .scm-commit .scm-commit-subject",
            "els => els.map(e => e.textContent)",
        )
        assert subjects[0] == "Uncommitted changes"
        assert subjects[1] == "Merge branch 'feature'"
        assert subjects[-1] == "first: add a.txt, README.md"
        assert len(subjects) == 6
        assert page.locator("#scm-graph-count").inner_text() == "5"
        head_refs = page.eval_on_selector_all(
            "#scm-graph .scm-commit .scm-ref.is-head",
            "els => els.map(e => e.textContent)",
        )
        assert head_refs == ["main"]
        tag_refs = page.eval_on_selector_all(
            "#scm-graph .scm-commit .scm-ref.is-tag",
            "els => els.map(e => e.textContent)",
        )
        assert tag_refs == ["v1"]
        # The merge forks the graph into two lanes: every row's SVG cell
        # is two lanes wide and the merge row draws two outgoing lines.
        widths = page.eval_on_selector_all(
            "#scm-graph .scm-graph-cell",
            "els => els.map(e => Number(e.getAttribute('width')))",
        )
        assert set(widths) == {28}
        n_paths = page.eval_on_selector_all(
            "#scm-graph .scm-commit-wrap",
            "els => els.map(e => e.querySelectorAll('path').length)",
        )
        # The uncommitted row is a tip: one line down to HEAD.  The
        # merge row: one line in from above, two out (one per parent).
        assert n_paths[0] == 1
        assert n_paths[1] == 3
        # Both requests carried the same token and the workspace.
        st = _sent(frames, "gitStatus")
        lg = _sent(frames, "gitLog")
        assert st and lg
        assert st[-1]["token"] == lg[-1]["token"]
        assert st[-1]["workDir"] == str(harness.work_dir)
        assert lg[-1]["limit"] == 50
    finally:
        context.close()


def test_commit_click_lists_modified_files_and_opens_them(browser, harness):
    context, page, frames = _open_page(browser, harness)
    try:
        page.click("#activity-scm")
        page.wait_for_selector("#scm-graph .scm-commit", timeout=15000)
        second = page.locator(
            f"#scm-graph .scm-commit[data-scm-sha='{harness.shas['second']}']"
        )
        assert second.get_attribute("aria-expanded") == "false"
        second.click()
        assert second.get_attribute("aria-expanded") == "true"
        wrap = second.locator("xpath=..")
        files = wrap.locator(".scm-commit-files .scm-row")
        assert files.count() == 2
        listed = files.evaluate_all(
            "els => els.map(e => [e.querySelector('.scm-path').textContent,"
            " e.querySelector('.scm-status').textContent, e.title])",
        )
        assert listed == [
            ["b.txt", "R", "a.txt \u2192 b.txt"],
            ["nested.py", "A", "dir/nested.py"],
        ]
        # A merge commit lists what it brought in against its first
        # parent (VS Code's graph does the same).
        merge = page.locator(
            f"#scm-graph .scm-commit[data-scm-sha='{harness.shas['merge']}']"
        )
        merge.click()
        merge_files = merge.locator("xpath=..").locator(".scm-commit-files .scm-row")
        assert merge_files.evaluate_all(
            "els => els.map(e => [e.querySelector('.scm-path').textContent,"
            " e.querySelector('.scm-status').textContent])",
        ) == [["feature.txt", "A"]]
        # Clicking a listed file opens it as a content tab.
        tabs_before = page.locator(".chat-tab").count()
        files.nth(1).click()
        page.wait_for_function(
            f"document.querySelectorAll('.chat-tab').length === {tabs_before + 1}",
            timeout=15000,
        )
        opened = _sent(frames, "openFile")
        assert opened[-1]["path"] == str(harness.work_dir.resolve() / "dir" / "nested.py")
        page.wait_for_function(
            "document.querySelector('.content-tab-view') && "
            "Array.from(document.querySelectorAll('.content-tab-view'))"
            ".some(v => v.innerText.includes('nested-sentinel-4f2a'))",
            timeout=15000,
        )
        # A deleted file in the Changes list is not openable.
        n_open = len(opened)
        page.locator("#scm-changes .scm-row.is-deleted").click()
        page.wait_for_timeout(300)
        assert len(_sent(frames, "openFile")) == n_open
        # Collapse again.
        second.click()
        assert second.get_attribute("aria-expanded") == "false"
        assert files.first.is_hidden()
    finally:
        context.close()


def test_uncommitted_row_lists_the_working_tree_changes(browser, harness):
    context, page, _ = _open_page(browser, harness)
    try:
        page.click("#activity-scm")
        page.wait_for_selector("#scm-graph .scm-commit.is-worktree", timeout=15000)
        row = page.locator("#scm-graph .scm-commit.is-worktree")
        assert row.locator(".scm-commit-meta").inner_text() == "4 files"
        row.click()
        paths = row.locator("xpath=..").locator(".scm-commit-files .scm-row").evaluate_all(
            "els => els.map(e => e.title)",
        )
        assert sorted(paths) == [
            "README.md", "b.txt (deleted)", "dir/nested.py", "untracked.txt",
        ]
    finally:
        context.close()


def test_view_choice_survives_a_reload(browser, harness):
    context, page, _ = _open_page(browser, harness)
    try:
        page.click("#activity-scm")
        page.wait_for_selector("#scm-graph .scm-commit", timeout=15000)
        page.reload()
        page.wait_for_selector("#task-input", state="visible", timeout=30000)
        page.wait_for_selector("#scm-graph .scm-commit", timeout=15000)
        assert page.locator("#activity-scm").get_attribute("aria-selected") == "true"
        assert page.locator("#sidebar-scm-panel").is_visible()
        assert page.locator("#sidebar-tab-history-panel").is_hidden()
    finally:
        context.close()


def test_refresh_relists_folders_keeping_them_expanded(browser, harness):
    context, page, _ = _open_page(browser, harness)
    extra = harness.work_dir / "dir" / "later.txt"
    try:
        page.click("#activity-explorer")
        page.wait_for_selector(".explorer-row.is-file", timeout=15000)
        _explorer_row(page, "dir").click()
        page.wait_for_selector(
            _explorer_row_sel("/dir/nested.py"), timeout=15000,
        )
        extra.write_text("later\n")
        page.click("#explorer-refresh")
        page.wait_for_selector(
            _explorer_row_sel("/dir/later.txt"), timeout=15000,
        )
        assert _explorer_row(page, "dir").get_attribute("aria-expanded") == "true"
        assert _explorer_row(page, "dir/nested.py").is_visible()
        # Source Control picks the new file up as untracked on refresh.
        page.click("#activity-scm")
        page.wait_for_function(
            "document.getElementById('scm-changes-count').textContent === '5'",
            timeout=15000,
        )
        extra.unlink()
        page.click("#scm-refresh")
        page.wait_for_function(
            "document.getElementById('scm-changes-count').textContent === '4'",
            timeout=15000,
        )
        # And the Explorer drops the vanished file on its next refresh.
        page.click("#activity-explorer")
        page.click("#explorer-refresh")
        page.wait_for_function(
            "sel => !document.querySelector(sel)",
            arg=_explorer_row_sel("/dir/later.txt"), timeout=15000,
        )
        assert _explorer_row(page, "dir/nested.py").is_visible()
    finally:
        if extra.exists():
            extra.unlink()
        context.close()


def test_symlink_cycle_is_not_expandable(browser, harness):
    """A folder that links back to one of its ancestors shows as a
    folder but refuses to expand (no endless tree, no request)."""
    context, page, frames = _open_page(browser, harness)
    link = harness.work_dir / "dir" / "up"
    link.symlink_to(harness.work_dir, target_is_directory=True)
    try:
        page.click("#activity-explorer")
        page.wait_for_selector(".explorer-row.is-file", timeout=15000)
        _explorer_row(page, "dir").click()
        page.wait_for_selector(
            _explorer_row_sel("/dir/up"), timeout=15000,
        )
        up = _explorer_row(page, "dir/up")
        assert "is-dir" in (up.get_attribute("class") or "")
        n_list = len(_sent(frames, "listDir"))
        up.click()
        page.wait_for_selector(".explorer-note", timeout=15000)
        note = page.locator(
            _explorer_row_sel("/dir/up") + " + .explorer-kids .explorer-note"
        )
        assert note.inner_text() == "(symbolic link cycle)"
        page.wait_for_timeout(300)
        assert len(_sent(frames, "listDir")) == n_list
        assert page.locator(
            _explorer_row_sel("/dir/up/dir")
        ).count() == 0
    finally:
        link.unlink()
        context.close()


def test_keyboard_model_roving_tabindex_and_arrows(browser, harness):
    """One tab stop per composite: the selected activity tab and one
    tree row; arrows move within, Right/Left step into/out of folders."""
    context, page, frames = _open_page(browser, harness)
    try:
        stops = page.eval_on_selector_all(
            "#activity-bar .activity-btn", "els => els.map(e => e.tabIndex)",
        )
        assert stops == [0, -1, -1]
        # The composer re-grabs focus for ~300ms after a tab activation
        # (focusInputWithRetry); wait it out before driving the bar.
        page.wait_for_timeout(500)
        page.focus("#activity-tasks")
        assert page.evaluate("document.activeElement.id") == "activity-tasks"
        page.keyboard.press("ArrowDown")
        assert page.locator("#activity-explorer").get_attribute("aria-selected") == "true"
        assert page.evaluate("document.activeElement.id") == "activity-explorer"
        stops = page.eval_on_selector_all(
            "#activity-bar .activity-btn", "els => els.map(e => e.tabIndex)",
        )
        assert stops == [-1, 0, -1]
        page.wait_for_selector(".explorer-row.is-file", timeout=15000)
        # Exactly one row is in the tab order: the first (the root).
        row_stops = page.eval_on_selector_all(
            ".explorer-row", "els => els.map(e => e.tabIndex)",
        )
        assert row_stops.count(0) == 1
        assert row_stops[0] == 0
        page.focus(".explorer-row[aria-level='1']")
        page.keyboard.press("ArrowDown")  # -> dir
        assert page.evaluate(
            _focus_ends_with("/dir")
        )
        page.keyboard.press("ArrowRight")  # open dir
        page.wait_for_selector(
            _explorer_row_sel("/dir/nested.py"), timeout=15000,
        )
        assert _explorer_row(page, "dir").get_attribute("aria-expanded") == "true"
        page.keyboard.press("ArrowRight")  # step into first child
        assert page.evaluate(
            _focus_ends_with("/dir/nested.py")
        )
        page.keyboard.press("ArrowLeft")  # back to the parent folder
        assert page.evaluate(
            _focus_ends_with("/dir")
        )
        row_stops = page.eval_on_selector_all(
            ".explorer-row", "els => els.map(e => e.tabIndex)",
        )
        assert row_stops.count(0) == 1
        page.keyboard.press("ArrowLeft")  # collapse
        assert _explorer_row(page, "dir").get_attribute("aria-expanded") == "false"
        page.keyboard.press("End")
        assert page.evaluate(
            _focus_ends_with("/untracked.txt")
        )
        n_open = len(_sent(frames, "openFile"))
        page.keyboard.press("Enter")
        page.wait_for_function(
            "document.querySelectorAll('.chat-tab').length >= 2", timeout=15000,
        )
        assert len(_sent(frames, "openFile")) == n_open + 1
        # Source Control rows: one tab stop as well.
        page.click("#activity-scm")
        page.wait_for_selector("#scm-graph .scm-commit", timeout=15000)
        scm_stops = page.eval_on_selector_all(
            "#scm-body .scm-commit, #scm-body .scm-row",
            "els => els.map(e => e.tabIndex)",
        )
        assert scm_stops.count(0) == 1
    finally:
        context.close()


def test_orphaned_content_tab_keeps_browsing_its_folder(browser, harness):
    """A file opened from a chat keeps the Explorer on that chat's
    folder even after the chat tab is closed."""
    context, page, frames = _open_page(browser, harness)
    try:
        page.click("#activity-explorer")
        page.wait_for_selector(".explorer-row.is-file", timeout=15000)
        _explorer_row(page, "README.md").click()
        page.wait_for_function(
            "document.querySelectorAll('.chat-tab').length === 2", timeout=15000,
        )
        # Close the chat tab (the first strip) while the content tab shows.
        page.evaluate(
            """() => {
              const chat = document.querySelector('.chat-tab:not(.active)');
              chat.querySelector('.chat-tab-close').click();
            }"""
        )
        page.wait_for_function(
            "document.querySelectorAll('.chat-tab').length === 1", timeout=15000,
        )
        page.wait_for_timeout(300)
        assert page.locator(".explorer-row[aria-level='1']").inner_text().strip() == "repo"
        assert _explorer_row(page, "README.md").is_visible()
    finally:
        context.close()


def test_hidden_views_catch_up_when_shown(browser, harness):
    """Task news while Tasks is selected marks the other views dirty:
    the Explorer / Source Control view reloads the moment it is shown
    instead of showing what it listed before."""
    context, page, frames = _open_page(browser, harness)
    extra = harness.work_dir / "arrived-later.txt"
    try:
        # Prime both data views, then go back to Tasks.
        page.click("#activity-explorer")
        page.wait_for_selector(".explorer-row.is-file", timeout=15000)
        page.click("#activity-scm")
        page.wait_for_selector("#scm-changes .scm-row", timeout=15000)
        page.click("#activity-tasks")
        n_list = len(_sent(frames, "listDir"))
        n_status = len(_sent(frames, "gitStatus"))
        extra.write_text("late\n")
        # The daemon's task news (a tasks_updated broadcast, here
        # delivered through the page's own message channel exactly as
        # the WebSocket layer would) arrives while Tasks is up: no
        # request goes out for the hidden views...
        page.evaluate(
            "window.dispatchEvent(new MessageEvent('message', "
            "{data: {type: 'tasks_updated'}}))"
        )
        page.wait_for_timeout(700)
        assert len(_sent(frames, "listDir")) == n_list
        assert len(_sent(frames, "gitStatus")) == n_status
        # ...but each view reloads as soon as it is shown.
        page.click("#activity-explorer")
        page.wait_for_selector(
            _explorer_row_sel("/arrived-later.txt"), timeout=15000,
        )
        page.click("#activity-scm")
        page.wait_for_function(
            "document.getElementById('scm-changes-count').textContent === '5'",
            timeout=15000,
        )
    finally:
        if extra.exists():
            extra.unlink()
        context.close()


def test_phone_drawer_refresh_button_is_clickable(browser, harness):
    """On the phone layout the drawer's close button must not sit on
    top of the Explorer / Source Control refresh buttons."""
    context, page, frames = _open_page_mobile(browser, harness)
    try:
        page.click("#menu-btn")
        page.wait_for_selector("#sidebar.open", timeout=15000)
        page.click("#activity-explorer")
        page.wait_for_selector(".explorer-row.is-file", timeout=15000)
        n_list = len(_sent(frames, "listDir"))
        hit = page.evaluate(
            """() => {
              const b = document.getElementById('explorer-refresh')
                .getBoundingClientRect();
              const el = document.elementFromPoint(
                b.x + b.width / 2, b.y + b.height / 2);
              return el && el.closest('button') ? el.closest('button').id : '';
            }"""
        )
        assert hit == "explorer-refresh"
        page.locator("#explorer-refresh").click()
        page.wait_for_timeout(300)
        assert page.locator("#sidebar").evaluate("el => el.classList.contains('open')")
        assert len(_sent(frames, "listDir")) == n_list + 1
        page.click("#activity-scm")
        hit = page.evaluate(
            """() => {
              const b = document.getElementById('scm-refresh')
                .getBoundingClientRect();
              const el = document.elementFromPoint(
                b.x + b.width / 2, b.y + b.height / 2);
              return el && el.closest('button') ? el.closest('button').id : '';
            }"""
        )
        assert hit == "scm-refresh"
        # Opening a file from the drawer closes it so the tab shows.
        page.click("#activity-explorer")
        _explorer_row(page, "README.md").click()
        page.wait_for_function(
            "!document.getElementById('sidebar').classList.contains('open')",
            timeout=15000,
        )
    finally:
        context.close()


def _open_page_mobile(browser, harness):
    """Open the remote page at a phone width (drawer layout)."""
    context = browser.new_context(
        ignore_https_errors=True, viewport={"width": 390, "height": 844},
    )
    page = context.new_page()
    sent_frames: list[dict] = []

    def _on_ws(ws) -> None:
        def _on_sent(payload) -> None:
            try:
                sent_frames.append(json.loads(payload))
            except Exception:
                pass

        ws.on("framesent", _on_sent)

    page.on("websocket", _on_ws)
    page.goto(harness.base_url + "/")
    page.wait_for_selector("#task-input", state="visible", timeout=30000)
    page.wait_for_function(
        "!document.body.classList.contains('remote-desktop')", timeout=15000,
    )
    # The config reply (workspace) has landed once the model name shows.
    page.wait_for_function(
        "document.getElementById('cfg-work-dir') !== null", timeout=15000,
    )
    page.wait_for_timeout(500)
    return context, page, sent_frames


def test_views_report_a_plain_folder_without_git(browser, harness):
    """A workspace that is no git repository still browses in the
    Explorer; Source Control says so instead of failing silently."""
    context, page, _ = _open_page(browser, harness)
    try:
        # Re-point the workspace through the settings panel, the way a
        # user does: closing the panel saves the form and re-scopes.
        _set_work_dir(page, str(harness.plain_dir))
        page.click("#activity-explorer")
        page.wait_for_selector(
            _explorer_row_sel("/plain/only.txt"), timeout=15000,
        )
        assert page.locator(".explorer-row[aria-level='1']").inner_text().strip() == "plain"
        page.click("#activity-scm")
        page.wait_for_function(
            "document.querySelector('#scm-changes .sidebar-empty') && "
            "document.querySelector('#scm-changes .sidebar-empty').textContent"
            ".startsWith('Not a git repository')",
            timeout=15000,
        )
        assert page.locator("#scm-branch").inner_text() == ""
        # The graph is filled by the separate gitLog reply, which may land
        # after gitStatus (each runs its own git subprocess); until then
        # the pane reads "Loading...".
        page.wait_for_function(
            "document.querySelector('#scm-graph .sidebar-empty') && "
            "document.querySelector('#scm-graph .sidebar-empty').textContent"
            ".startsWith('Not a git repository')",
            timeout=15000,
        )
        # Back to the repository: both views follow the workspace.
        _set_work_dir(page, str(harness.work_dir))
        page.wait_for_selector("#scm-graph .scm-commit", timeout=15000)
        assert page.locator("#scm-branch").inner_text() == "main"
        page.click("#activity-explorer")
        page.wait_for_function(
            "document.querySelector(\".explorer-row[aria-level='1']\") && "
            "document.querySelector(\".explorer-row[aria-level='1']\")"
            ".textContent.trim() === 'repo'",
            timeout=15000,
        )
    finally:
        # Leave the shared server on the repository for the other tests.
        try:
            _set_work_dir(page, str(harness.work_dir))
        except Exception:
            pass
        context.close()


def _set_work_dir(page, work_dir: str) -> None:
    """Change the workspace through the settings panel and wait for the
    docked task-info panel to show it."""
    page.click("#more-btn")
    page.click("#settings-btn")
    page.wait_for_selector("#settings-panel.open", timeout=15000)
    page.wait_for_function(
        "document.getElementById('cfg-work-dir').value.length > 0", timeout=15000,
    )
    page.fill("#cfg-work-dir", work_dir)
    page.click("#settings-panel-close")
    page.wait_for_function(
        "wd => document.getElementById('meta-workdir').textContent === wd",
        arg=work_dir,
        timeout=15000,
    )
