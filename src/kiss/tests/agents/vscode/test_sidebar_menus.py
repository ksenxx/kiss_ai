# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (module fixtures imported from
#   kiss.tests.server.test_scm_worktrees_and_actions are intentionally
#   shadowed by test parameters of the same name)
"""Browser tests for the remote sidebar's VS Code-style context menus.

Right-clicking an Explorer row shows VS Code's Explorer menu (New
File..., New Folder..., Open to the Side, Select for Compare, Find in
Folder..., Cut / Copy / Paste, Copy Path / Copy Relative Path,
Rename..., Delete) and the entries act on the real file system through
``fsAction``; right-clicking a commit of the Source Control graph shows
VS Code's history-item menu (Open Changes, Checkout (Detached), Create
Branch..., Create Tag..., Cherry Pick, Compare with..., Copy Commit
Hash / Copy Commit Message) driving ``gitShow`` / ``gitAction``.  The
Source Control view lists every worktree's changes, the Explorer header
carries a folder picker, a clicked PDF opens in a viewer tab.

Every test drives a REAL headless Chromium against a REAL
:class:`RemoteAccessServer` whose work dir is a REAL git repository
with a REAL linked worktree — no mocks.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.agents.vscode.test_activity_bar import (
    _explorer_row,
    _explorer_row_sel,
    _sent,
)
from kiss.tests.conftest import (
    TRANSIENT_REPLACE_READ_ERRORS,
    goto_retrying_network_change,
)
from kiss.tests.server.test_scm_worktrees_and_actions import (
    harness,  # noqa: F401  (module fixture used by param name)
    worktree,  # noqa: F401
)


@pytest.fixture(scope="module")
def browser():
    """One shared headless Chromium for every test in this module."""
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


def _css(value: str) -> str:
    """Escape *value* for a quoted CSS attribute selector (Windows paths
    carry backslashes, which CSS would read as escapes)."""
    return value.replace("\\", "\\\\")


def _row_at(path: str, cls: str = "") -> str:
    """Selector for the Explorer row whose path is exactly *path*."""
    return f".explorer-row{cls}[data-explorer-path='{_css(path)}']"


def _click_root_button(page, root: str, action: str) -> None:
    """Hover *root*'s Explorer row and click its ``set`` / ``remove`` button.

    The buttons are shown by the row's ``:hover`` rule.  A tree
    re-render after the hover (a late explorer refresh from the daemon)
    replaces the row and the mouse, not having moved, leaves the new
    row un-hovered, so the hover is repeated until the button of the
    current row is visible.
    """
    row = page.locator(_row_at(root, ".is-root"))
    button = page.locator(f"{_row_at(root, '.is-root')} .explorer-root-{action}")
    for _ in range(50):
        row.hover()
        if button.is_visible():
            break
        page.wait_for_timeout(100)
    button.click()


def _open_page(browser, harness):
    """Open the remote page in desktop mode with clipboard access and
    record the WS frames the client sends."""
    context = browser.new_context(
        ignore_https_errors=True,
        viewport={"width": 1400, "height": 900},
        permissions=["clipboard-read", "clipboard-write"],
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
    goto_retrying_network_change(page, harness.base_url + "/")
    page.wait_for_selector("#task-input", state="visible", timeout=30000)
    page.wait_for_selector("body.remote-desktop", state="attached")
    page.wait_for_function(
        "document.getElementById('meta-workdir').textContent.length > 1",
        timeout=30000,
    )
    return context, page, sent_frames


def _menu_labels(page) -> list[str]:
    page.wait_for_selector("#sidebar-context-menu:not([hidden])", timeout=10000)
    labels = page.eval_on_selector_all(
        "#sidebar-context-menu .tree-ctx-item .tree-ctx-label",
        "els => els.map(e => e.textContent)",
    )
    return [str(label) for label in labels]


def _menu_item(page, label: str):
    return page.locator(
        "#sidebar-context-menu .tree-ctx-item", has_text=label,
    ).first


def _settle(page) -> None:
    """Let focusInputWithRetry's 100 / 300 ms re-focus timers (started
    by the tab activation on load) run out, so a menu's keyboard focus
    is not stolen mid-test."""
    page.wait_for_timeout(400)


def _open_explorer(page):
    page.click("#activity-explorer")
    page.wait_for_selector(".explorer-row.is-file", timeout=15000)
    _settle(page)


def _open_scm(page):
    page.click("#activity-scm")
    page.wait_for_selector("#scm-graph .scm-commit", timeout=15000)
    _settle(page)


def _toast(page, toast_id: str):
    """Locator of the in-webview confirm / prompt toast *toast_id* that
    main.js shows in place of the browser's confirm() / prompt() (the
    VS Code webview sandbox never displays those)."""
    return page.locator(f".kiss-notification[data-notification-id='{toast_id}']")


def _answer_prompt(page, toast_id: str, text: str | None) -> None:
    """Type *text* into the prompt toast *toast_id* and submit it with
    Enter, or cancel the prompt with Escape when *text* is ``None``;
    return once the toast is gone."""
    toast = _toast(page, toast_id)
    inp = toast.locator(".kiss-notification-input")
    inp.wait_for(timeout=5000)
    if text is None:
        inp.press("Escape")
    else:
        inp.fill(text)
        inp.press("Enter")
    toast.wait_for(state="detached", timeout=5000)


def _answer_confirm(page, toast_id: str, accept: bool) -> str:
    """Press the confirm toast's verb button (*accept*) or its cancel
    button, wait for the toast to close and return the question it
    asked."""
    toast = _toast(page, toast_id)
    toast.wait_for(timeout=5000)
    message = toast.locator(".kiss-notification-message").text_content() or ""
    if accept:
        toast.locator(".kiss-notification-action").first.click()
    else:
        toast.locator("[data-dialog-cancel]").click()
    toast.wait_for(state="detached", timeout=5000)
    return message


def _wait_content(page, text: str) -> None:
    """Wait until an open content tab holds *text*: in a Monaco model
    (Monaco only renders the visible lines) or in a rendered view."""
    page.wait_for_function(
        """text => (window.monaco && monaco.editor.getModels()
             .some(m => m.getValue().includes(text))) ||
           Array.from(document.querySelectorAll('.content-tab-view'))
             .some(v => v.innerText.includes(text))""",
        arg=text,
        timeout=20000,
    )


def _wait_tab_count(page, n: int) -> None:
    page.wait_for_function(
        f"document.querySelectorAll('.chat-tab').length === {n}", timeout=15000,
    )


def _wait_explorer_root(page, name: str) -> None:
    """Wait until the Explorer's root row (aria-level 1) is named *name*."""
    page.wait_for_function(
        """name => {
             const row = document.querySelector('.explorer-row[aria-level="1"]');
             return !!row && row.textContent.trim() === name;
           }""",
        arg=name,
        timeout=15000,
    )


def test_explorer_file_menu_matches_vscode(browser, harness, worktree):
    context, page, frames = _open_page(browser, harness)
    try:
        _open_explorer(page)
        _explorer_row(page, "feature.txt").click(button="right")
        labels = _menu_labels(page)
        assert labels == [
            "Open to the Side",
            "Select for Compare",
            "Cut",
            "Copy",
            "Paste",
            "Copy Path",
            "Copy Relative Path",
            "Rename...",
            "Delete",
        ]
        # Separators sit between the groups, as in VS Code.
        assert page.locator("#sidebar-context-menu .tree-ctx-sep").count() == 4
        # Paste is disabled until something was cut or copied.
        assert "disabled" in (_menu_item(page, "Paste").get_attribute("class") or "")
        keys = page.eval_on_selector_all(
            "#sidebar-context-menu .tree-ctx-key",
            "els => els.map(e => e.textContent)",
        )
        # Copy Path renders the platform-correct VS Code chord: Shift+Alt+C
        # on Windows, Ctrl+Alt+C on Linux, Alt+Cmd+C on macOS.  Headless
        # Playwright reports the real host platform, so derive it.
        platform = page.evaluate("() => navigator.platform || ''")
        is_mac = bool(re.search(r"Mac|iPhone|iPad", platform))
        is_linux = bool(re.search(r"Linux|X11", platform)) and not is_mac
        if is_mac:
            copy_path_key = "\u2325\u2318C"
            cut_key = "\u2318X"
        elif is_linux:
            copy_path_key = "Ctrl+Alt+C"
            cut_key = "Ctrl+X"
        else:
            copy_path_key = "Shift+Alt+C"
            cut_key = "Ctrl+X"
        assert "F2" in keys and cut_key in keys and copy_path_key in keys
        # The row under the pointer is highlighted while the menu is up.
        assert _explorer_row(page, "feature.txt").evaluate(
            "el => el.classList.contains('ctx-active')",
        )
        # Escape closes it and drops the highlight.
        page.keyboard.press("Escape")
        page.wait_for_selector("#sidebar-context-menu", state="hidden")
        assert not _explorer_row(page, "feature.txt").evaluate(
            "el => el.classList.contains('ctx-active')",
        )
        # Copy Path / Copy Relative Path put the paths on the clipboard.
        _explorer_row(page, "feature.txt").click(button="right")
        _menu_item(page, "Copy Path").click()
        page.wait_for_function(
            "navigator.clipboard.readText().then(t => window.__clip = t) && true",
        )
        page.wait_for_function(
            f"window.__clip === {json.dumps(str(harness.work_dir / 'feature.txt'))}",
            timeout=5000,
        )
        _explorer_row(page, "feature.txt").click(button="right")
        _menu_item(page, "Copy Relative Path").click()
        page.wait_for_function(
            "navigator.clipboard.readText().then(t => window.__clip2 = t) && true",
        )
        page.wait_for_function("window.__clip2 === 'feature.txt'", timeout=5000)
    finally:
        context.close()


def test_explorer_folder_menu_and_root_guards(browser, harness, worktree):
    context, page, frames = _open_page(browser, harness)
    try:
        _open_explorer(page)
        _explorer_row(page, "dir").click(button="right")
        labels = _menu_labels(page)
        assert labels == [
            "New File...",
            "New Folder...",
            "Find in Folder...",
            "Cut",
            "Copy",
            "Paste",
            "Copy Path",
            "Copy Relative Path",
            "Rename...",
            "Delete",
        ]
        page.keyboard.press("Escape")
        # The workspace root cannot be cut or copied and, like VS Code's
        # root folder menu, offers no Rename / Delete: it ends with the
        # top-level folder items instead (the working directory itself
        # cannot be switched to or removed).
        page.locator(".explorer-row[aria-level='1']").click(button="right")
        root_labels = _menu_labels(page)
        assert "Rename..." not in root_labels and "Delete" not in root_labels
        assert root_labels[-3:] == [
            "Add Folder to Explorer...",
            "Set as Working Directory",
            "Remove Folder from Explorer",
        ]
        for label in ("Cut", "Copy", "Set as Working Directory", "Remove Folder from Explorer"):
            assert "disabled" in (_menu_item(page, label).get_attribute("class") or ""), label
        # Keyboard: Down moves through enabled items, Enter runs one.
        page.keyboard.press("ArrowDown")
        focused = page.evaluate("document.activeElement.textContent")
        assert focused.startswith("New Folder...")
        page.keyboard.press("Escape")
    finally:
        context.close()


def test_new_file_rename_and_delete_act_on_disk(browser, harness, worktree):
    context, page, frames = _open_page(browser, harness)
    try:
        _open_explorer(page)
        tabs_before = page.locator(".chat-tab").count()
        _explorer_row(page, "dir").click(button="right")
        _menu_item(page, "New File...").click()
        inp = page.locator(".explorer-row.is-editing .explorer-input")
        inp.wait_for(timeout=5000)
        inp.fill("fresh.py")
        inp.press("Enter")
        page.wait_for_selector(
            _explorer_row_sel("/dir/fresh.py"), timeout=15000,
        )
        assert (harness.work_dir / "dir" / "fresh.py").is_file()
        # Like VS Code, the new file opened in an editor.
        _wait_tab_count(page, tabs_before + 1)
        sent = [f for f in _sent(frames, "fsAction") if f["action"] == "newFile"]
        assert sent and sent[0]["path"] == str(harness.work_dir.resolve() / "dir")
        assert sent[0]["name"] == "fresh.py"

        # New Folder... via the inline box, Escape cancels first.
        _explorer_row(page, "dir").click(button="right")
        _menu_item(page, "New Folder...").click()
        inp = page.locator(".explorer-row.is-editing .explorer-input")
        inp.wait_for(timeout=5000)
        inp.fill("nope")
        inp.press("Escape")
        page.wait_for_selector(".explorer-row.is-editing", state="detached")
        assert not (harness.work_dir / "dir" / "nope").exists()
        _explorer_row(page, "dir").click(button="right")
        _menu_item(page, "New Folder...").click()
        inp = page.locator(".explorer-row.is-editing .explorer-input")
        inp.wait_for(timeout=5000)
        inp.fill("made")
        inp.press("Enter")
        page.wait_for_selector(
            _explorer_row_sel("/dir/made", ".is-dir"), timeout=15000,
        )
        assert (harness.work_dir / "dir" / "made").is_dir()

        # Rename... pre-selects the stem and renames on Enter; the
        # editor tab opened for the file follows the new name.
        _explorer_row(page, "fresh.py").click(button="right")
        _menu_item(page, "Rename...").click()
        inp = page.locator(".explorer-row.is-editing .explorer-input")
        inp.wait_for(timeout=5000)
        assert inp.input_value() == "fresh.py"
        # The client focuses the box and selects the stem in a deferred
        # (setTimeout 0) step, so poll for it instead of reading once.
        page.wait_for_function(
            "() => { const el = document.querySelector("
            "'.explorer-row.is-editing .explorer-input');"
            " return el && el.selectionStart === 0 && el.selectionEnd === 5; }",
            timeout=5000,
        )
        inp.fill("renamed.py")
        inp.press("Enter")
        page.wait_for_selector(
            _explorer_row_sel("/dir/renamed.py"), timeout=15000,
        )
        assert (harness.work_dir / "dir" / "renamed.py").is_file()
        assert not (harness.work_dir / "dir" / "fresh.py").exists()
        page.wait_for_function(
            "Array.from(document.querySelectorAll('.chat-tab'))"
            ".some(t => t.textContent.includes('renamed.py'))",
            timeout=15000,
        )

        # F2 is the Rename shortcut on a focused row.
        row = _explorer_row(page, "renamed.py")
        row.focus()
        row.press("F2")
        inp = page.locator(".explorer-row.is-editing .explorer-input")
        inp.wait_for(timeout=5000)
        inp.press("Escape")

        # Delete asks first; "Keep" keeps the file.
        _explorer_row(page, "renamed.py").click(button="right")
        _menu_item(page, "Delete").click()
        message = _answer_confirm(page, "fs-delete", accept=False)
        assert "renamed.py" in message
        page.wait_for_timeout(300)
        assert (harness.work_dir / "dir" / "renamed.py").is_file()
        _explorer_row(page, "renamed.py").click(button="right")
        _menu_item(page, "Delete").click()
        _answer_confirm(page, "fs-delete", accept=True)
        page.wait_for_selector(
            _explorer_row_sel("/dir/renamed.py"),
            state="detached",
            timeout=15000,
        )
        assert not (harness.work_dir / "dir" / "renamed.py").exists()
        # The editor tab of the deleted file closed with it.
        _wait_tab_count(page, tabs_before)
        _explorer_row(page, "made").click(button="right")
        _menu_item(page, "Delete").click()
        _answer_confirm(page, "fs-delete", accept=True)
        page.wait_for_selector(
            _explorer_row_sel("/dir/made"),
            state="detached",
            timeout=15000,
        )
    finally:
        context.close()


def test_copy_paste_cut_and_conflict_prompt(browser, harness, worktree):
    context, page, frames = _open_page(browser, harness)
    try:
        _open_explorer(page)
        _explorer_row(page, "main-only.txt").click(button="right")
        _menu_item(page, "Copy").click()
        page.wait_for_selector("#sidebar-context-menu", state="hidden")
        # Paste into the same folder: "main-only copy.txt", like VS Code.
        page.locator(".explorer-row[aria-level='1']").click(button="right")
        assert "disabled" not in (_menu_item(page, "Paste").get_attribute("class") or "")
        _menu_item(page, "Paste").click()
        page.wait_for_selector(
            _explorer_row_sel("/main-only copy.txt"), timeout=15000,
        )
        assert (harness.work_dir / "main-only copy.txt").read_text() == "m\n"
        # Paste into dir/, then again: the second paste collides and
        # "Keep existing" on the replace question changes nothing.
        _explorer_row(page, "dir").click(button="right")
        _menu_item(page, "Paste").click()
        page.wait_for_selector(
            _explorer_row_sel("/dir/main-only.txt"), timeout=15000,
        )
        (harness.work_dir / "dir" / "main-only.txt").write_text("keep\n")
        _explorer_row(page, "dir").click(button="right")
        _menu_item(page, "Paste").click()
        message = _answer_confirm(page, "fs-overwrite", accept=False)
        assert "already exists" in message
        page.wait_for_timeout(300)
        assert (harness.work_dir / "dir" / "main-only.txt").read_text() == "keep\n"
        # "Replace" replaces it.
        _explorer_row(page, "dir").click(button="right")
        _menu_item(page, "Paste").click()
        _answer_confirm(page, "fs-overwrite", accept=True)
        # The server renames the old file aside and copies the new one in;
        # a read landing inside that window finds no file or, on Windows,
        # a copy still holding the target exclusively.  Neither is torn.
        for _ in range(50):
            try:
                if (harness.work_dir / "dir" / "main-only.txt").read_text() == "m\n":
                    break
            except TRANSIENT_REPLACE_READ_ERRORS:
                pass
            page.wait_for_timeout(100)
        assert (harness.work_dir / "dir" / "main-only.txt").read_text() == "m\n"
        # Cut + Paste moves.
        page.locator(_explorer_row_sel("/main-only copy.txt")).click(
            button="right",
        )
        _menu_item(page, "Cut").click()
        _explorer_row(page, "dir").click(button="right")
        _menu_item(page, "Paste").click()
        page.wait_for_selector(
            _explorer_row_sel("/dir/main-only copy.txt"), timeout=15000,
        )
        assert not (harness.work_dir / "main-only copy.txt").exists()
        assert (harness.work_dir / "dir" / "main-only copy.txt").exists()
        # The clipboard is spent after a move: Paste is disabled again.
        _explorer_row(page, "dir").click(button="right")
        assert "disabled" in (_menu_item(page, "Paste").get_attribute("class") or "")
        page.keyboard.press("Escape")
        for name in ("dir/main-only.txt", "dir/main-only copy.txt"):
            (harness.work_dir / name).unlink()
    finally:
        context.close()


def test_find_in_folder_and_compare_open_result_tabs(browser, harness, worktree):
    context, page, frames = _open_page(browser, harness)
    try:
        _open_explorer(page)
        tabs_before = page.locator(".chat-tab").count()
        page.locator(".explorer-row[aria-level='1']").click(button="right")
        _menu_item(page, "Find in Folder...").click()
        _answer_prompt(page, "find-in-folder", "sentinel")
        _wait_tab_count(page, tabs_before + 1)
        _wait_content(page, "feature.txt:1:feature-file-sentinel-7c1e")
        titles = page.eval_on_selector_all(
            ".chat-tab", "els => els.map(e => e.textContent)",
        )
        assert any("Search: sentinel" in t for t in titles)
        # Select for Compare + Compare with Selected -> a diff tab.
        _explorer_row(page, "feature.txt").click(button="right")
        _menu_item(page, "Select for Compare").click()
        _explorer_row(page, "main-only.txt").click(button="right")
        labels = _menu_labels(page)
        assert "Compare with Selected" in labels
        _menu_item(page, "Compare with Selected").click()
        _wait_tab_count(page, tabs_before + 2)
        _wait_content(page, "-feature-file-sentinel-7c1e")
        titles = page.eval_on_selector_all(
            ".chat-tab", "els => els.map(e => e.textContent)",
        )
        assert any("feature.txt \u2194 main-only.txt" in t for t in titles)
    finally:
        context.close()


def test_open_to_the_side_keeps_the_current_tab(browser, harness, worktree):
    context, page, frames = _open_page(browser, harness)
    try:
        _open_explorer(page)
        active_before = page.evaluate(
            "document.querySelector('.chat-tab.active').textContent",
        )
        tabs_before = page.locator(".chat-tab").count()
        _explorer_row(page, "feature.txt").click(button="right")
        _menu_item(page, "Open to the Side").click()
        _wait_tab_count(page, tabs_before + 1)
        assert page.evaluate(
            "document.querySelector('.chat-tab.active').textContent",
        ) == active_before
        opened = [f for f in _sent(frames, "openFile") if f.get("background")]
        assert opened and opened[0]["path"].endswith(os.sep + "feature.txt")
    finally:
        context.close()


def test_source_control_lists_every_worktree(browser, harness, worktree):
    context, page, frames = _open_page(browser, harness)
    try:
        _open_scm(page)
        page.wait_for_selector("#scm-changes .scm-worktree-hdr", timeout=15000)
        names = page.eval_on_selector_all(
            "#scm-changes .scm-worktree-hdr .scm-worktree-name",
            "els => els.map(e => e.textContent)",
        )
        assert names == ["repo", "wt-task"]
        branches = page.eval_on_selector_all(
            "#scm-changes .scm-worktree-hdr .scm-worktree-branch",
            "els => els.map(e => e.textContent)",
        )
        assert branches == ["main", "kiss/wt-task"]
        # The worktree's rows point at ITS copy of the files.
        wt_rows = page.locator(
            f"#scm-changes .scm-row[data-scm-worktree='{_css(str(worktree))}']",
        )
        assert wt_rows.count() >= 2
        paths = wt_rows.evaluate_all("els => els.map(e => e.dataset.scmPath)")
        assert str(worktree / "README.md") in paths
        assert str(worktree / "wt-untracked.txt") in paths
        # The graph shows the worktree's own commit with its branch, a
        # worktree badge, and a second "Uncommitted changes" row for it.
        wt_commit = page.locator("#scm-graph .scm-commit", has_text="worktree: add task.txt")
        assert wt_commit.count() == 1
        assert wt_commit.locator(".scm-ref.is-worktree").inner_text() == "wt-task"
        assert wt_commit.locator(".scm-ref", has_text="kiss/wt-task").count() == 1
        subjects = page.eval_on_selector_all(
            "#scm-graph .scm-commit.is-worktree .scm-commit-subject",
            "els => els.map(e => e.textContent)",
        )
        assert subjects == ["Uncommitted changes", "Uncommitted changes (wt-task)"]
        # Clicking the worktree's changed README opens the worktree copy.
        tabs_before = page.locator(".chat-tab").count()
        wt_rows.filter(has_text="wt-untracked.txt").first.click()
        _wait_tab_count(page, tabs_before + 1)
        _wait_content(page, "worktree-untracked-sentinel-5e2d")
    finally:
        context.close()


def test_commit_menu_matches_vscode_and_open_changes(browser, harness, worktree):
    context, page, frames = _open_page(browser, harness)
    try:
        _open_scm(page)
        second = page.locator("#scm-graph .scm-commit", has_text="second: rename")
        second.click(button="right")
        labels = _menu_labels(page)
        assert labels == [
            "Open Changes",
            "Checkout (Detached)",
            "Create Branch...",
            "Create Tag...",
            "Cherry Pick",
            "Compare with...",
            "Copy Commit Hash",
            "Copy Commit Message",
        ]
        assert page.locator("#sidebar-context-menu .tree-ctx-sep").count() == 6
        # The uncommitted row has no VS Code commit menu.
        page.keyboard.press("Escape")
        page.locator("#scm-graph .scm-commit.is-worktree").first.click(button="right")
        page.wait_for_timeout(300)
        assert page.locator("#sidebar-context-menu:not([hidden])").count() == 0
        # Open Changes -> a read-only tab with the commit's patch.
        tabs_before = page.locator(".chat-tab").count()
        second.click(button="right")
        _menu_item(page, "Open Changes").click()
        _wait_tab_count(page, tabs_before + 1)
        _wait_content(page, "nested-sentinel-4f2a")
        titles = page.eval_on_selector_all(
            ".chat-tab", "els => els.map(e => e.textContent)",
        )
        assert any(
            harness.shas["second"][:7] + " - second: rename" in t for t in titles
        )
        assert not page.locator(".content-tab-view .content-save-bar").count()
        shows = _sent(frames, "gitShow")
        assert shows and shows[0]["sha"] == harness.shas["second"]
        # Copy Commit Hash / Message.
        second.click(button="right")
        _menu_item(page, "Copy Commit Hash").click()
        page.wait_for_function(
            "navigator.clipboard.readText().then(t => window.__clip = t) && true",
        )
        page.wait_for_function(
            f"window.__clip === {json.dumps(harness.shas['second'])}", timeout=5000,
        )
        second.click(button="right")
        _menu_item(page, "Copy Commit Message").click()
        page.wait_for_function(
            "navigator.clipboard.readText().then(t => window.__clip2 = t) && true",
        )
        page.wait_for_function(
            "window.__clip2 === 'second: rename a.txt -> b.txt'", timeout=5000,
        )
        # A file of an expanded commit: Open Changes / Open File.
        second.click()
        file_row = page.locator("#scm-graph .scm-commit-files .scm-row", has_text="nested.py")
        file_row.first.click(button="right")
        assert _menu_labels(page) == ["Open Changes", "Open File"]
        _menu_item(page, "Open File").click()
        _wait_tab_count(page, tabs_before + 2)
        _wait_content(page, "x = 1  # nested-sentinel-4f2a")
        titles = page.eval_on_selector_all(
            ".chat-tab", "els => els.map(e => e.textContent)",
        )
        assert any(
            "nested.py (" + harness.shas["second"][:7] + ")" in t for t in titles
        )
    finally:
        context.close()


def test_commit_actions_run_git_and_refresh(browser, harness, worktree):
    context, page, frames = _open_page(browser, harness)
    try:
        _open_scm(page)
        first = page.locator("#scm-graph .scm-commit", has_text="first: add a.txt")
        # Create Tag... (name, then optional message) -> the tag shows
        # up on the commit after the refresh.
        first.click(button="right")
        _menu_item(page, "Create Tag...").click()
        _answer_prompt(page, "git-create-tag", "ui-tag")
        _answer_prompt(page, "git-create-tag-message", "ui-tag")
        page.wait_for_selector(
            "#scm-graph .scm-commit .scm-ref.is-tag:text-is('ui-tag')", timeout=15000,
        )
        actions = _sent(frames, "gitAction")
        assert actions[-1]["action"] == "createTag"
        assert actions[-1]["name"] == "ui-tag" and actions[-1]["message"] == "ui-tag"
        # A cancelled prompt sends nothing.
        first.click(button="right")
        _menu_item(page, "Create Branch...").click()
        _answer_prompt(page, "git-create-branch", None)
        page.wait_for_timeout(300)
        assert len(_sent(frames, "gitAction")) == len(actions)
        # Checkout (Detached) fails on the dirty main checkout: the
        # daemon's git error is shown as a notification.
        first.click(button="right")
        _menu_item(page, "Checkout (Detached)").click()
        page.wait_for_function(
            "document.body.innerText.includes('would be overwritten') || "
            "document.body.innerText.includes('local changes')",
            timeout=15000,
        )
        # Compare with... -> a diff tab against the typed revision.
        tabs_before = page.locator(".chat-tab").count()
        first.click(button="right")
        _menu_item(page, "Compare with...").click()
        _answer_prompt(page, "git-compare-with", "v1")
        _wait_tab_count(page, tabs_before + 1)
        _wait_content(page, "feature-file-sentinel-7c1e")
        titles = page.eval_on_selector_all(
            ".chat-tab", "els => els.map(e => e.textContent)",
        )
        assert any("v1 \u2194 " + harness.shas["first"][:7] in t for t in titles)
    finally:
        context.close()


def _selected_names(page) -> list[str]:
    """Base names of the selected Explorer rows, in tree order."""
    names = page.eval_on_selector_all(
        ".explorer-row.is-selected",
        "els => els.map(e => e.dataset.explorerPath.split(/[\\\\/]/).pop())",
    )
    return [str(n) for n in names]


def _read_clipboard(page, slot: str) -> str:
    """Return the clipboard text with its line breaks normalized to LF.

    Chromium hands multi-line text to the Windows clipboard as CRLF, so
    the ``'\\n'``-joined paths come back with a ``\\r`` per line there.
    """
    page.wait_for_function(
        f"navigator.clipboard.readText().then(t => window.{slot} = t) && true",
    )
    page.wait_for_function(f"typeof window.{slot} === 'string'", timeout=5000)
    return str(page.evaluate(f"window.{slot}")).replace("\r\n", "\n")


def test_explorer_multi_select_and_multi_target_menu(browser, harness, worktree):
    """Ctrl/Cmd-click toggles rows, Shift-click selects a range, the
    arrow keys move the selection (Shift extends it), Ctrl+A selects
    every visible row and Escape clears; the context menu of a selected
    row acts on the whole selection (Compare Selected for two files,
    Copy Path joins the paths, Delete removes them all)."""
    folder = harness.work_dir / "multi"
    folder.mkdir()
    for name in ("a.txt", "b.txt", "c.txt", "d.txt"):
        (folder / name).write_text(name + " multi-sentinel\n")
    context, page, frames = _open_page(browser, harness)
    try:
        _open_explorer(page)
        page.click("#explorer-refresh")
        page.wait_for_selector(_explorer_row_sel("/multi", ".is-dir"), timeout=15000)
        _explorer_row(page, "multi").click()
        page.wait_for_selector(_explorer_row_sel("/multi/d.txt"), timeout=15000)
        assert page.get_attribute("#explorer-tree", "aria-multiselectable") == "true"
        tabs_before = page.locator(".chat-tab").count()

        # A plain click selects the row alone and opens the file.
        _explorer_row(page, "a.txt").click()
        _wait_tab_count(page, tabs_before + 1)
        assert _selected_names(page) == ["a.txt"]
        assert _explorer_row(page, "a.txt").get_attribute("aria-selected") == "true"
        # Ctrl-click adds a row without opening it; Shift-click selects
        # from the anchor (the last row clicked) to the target.
        # ControlOrMeta: on macOS Ctrl-click is the context-menu gesture
        # (Chromium fires ``contextmenu``), so Cmd-click is the toggle.
        _explorer_row(page, "c.txt").click(modifiers=["ControlOrMeta"])
        assert _selected_names(page) == ["a.txt", "c.txt"]
        _explorer_row(page, "d.txt").click(modifiers=["Shift"])
        assert _selected_names(page) == ["a.txt", "c.txt", "d.txt"]
        page.wait_for_timeout(300)
        assert page.locator(".chat-tab").count() == tabs_before + 1
        # Ctrl-click on a selected row deselects it.
        _explorer_row(page, "a.txt").click(modifiers=["ControlOrMeta"])
        assert _selected_names(page) == ["c.txt", "d.txt"]

        # The menu of a selected row is the multi-selection menu.
        _explorer_row(page, "c.txt").click(button="right")
        assert _menu_labels(page) == [
            "Open to the Side",
            "Compare Selected",
            "Cut",
            "Copy",
            "Copy Path",
            "Copy Relative Path",
            "Delete",
        ]
        _menu_item(page, "Copy Path").click()
        clip = _read_clipboard(page, "__multiPaths")
        assert clip.split("\n") == [str(folder / "c.txt"), str(folder / "d.txt")]
        _explorer_row(page, "d.txt").click(button="right")
        _menu_item(page, "Copy Relative Path").click()
        rel = _read_clipboard(page, "__multiRel")
        assert rel.split("\n") == [
            os.path.join("multi", "c.txt"), os.path.join("multi", "d.txt"),
        ]
        # Compare Selected diffs the two files in a result tab.
        _explorer_row(page, "c.txt").click(button="right")
        _menu_item(page, "Compare Selected").click()
        _wait_tab_count(page, tabs_before + 2)
        _wait_content(page, "-c.txt multi-sentinel")
        titles = page.eval_on_selector_all(
            ".chat-tab", "els => els.map(e => e.textContent)",
        )
        assert any("c.txt \u2194 d.txt" in t for t in titles)
        compares = [f for f in _sent(frames, "fsAction") if f["action"] == "compare"]
        assert compares[-1]["path"] == str(folder / "c.txt")
        assert compares[-1]["dest"] == str(folder / "d.txt")

        # Right-clicking a row outside the selection selects it alone:
        # the single-row menu (with Rename...) comes up.
        _explorer_row(page, "b.txt").click(button="right")
        assert "Rename..." in _menu_labels(page)
        assert _selected_names(page) == ["b.txt"]
        page.keyboard.press("Escape")
        page.wait_for_selector("#sidebar-context-menu", state="hidden")

        # Arrow keys move the selection; Shift+arrow extends it.
        row_b = _explorer_row(page, "b.txt")
        row_b.focus()
        row_b.press("ArrowDown")
        assert _selected_names(page) == ["c.txt"]
        page.keyboard.press("Shift+ArrowDown")
        assert _selected_names(page) == ["c.txt", "d.txt"]
        page.keyboard.press("Shift+ArrowUp")
        assert _selected_names(page) == ["c.txt"]
        # Left goes up to the folder and selects it; Right steps into
        # the first child and selects that.
        page.keyboard.press("ArrowLeft")
        assert _selected_names(page) == ["multi"]
        page.keyboard.press("ArrowRight")
        assert _selected_names(page) == ["a.txt"]
        # Ctrl+A selects every visible row, Escape clears the selection.
        page.keyboard.press("ControlOrMeta+a")
        selected = _selected_names(page)
        assert {"multi", "a.txt", "b.txt", "c.txt", "d.txt"} <= set(selected)
        assert len(selected) == page.locator(".explorer-row").count()
        page.keyboard.press("Escape")
        assert _selected_names(page) == []

        # Three files, the Delete key on one of them: one confirmation
        # for all three, then every one is gone (b.txt survives).
        _explorer_row(page, "a.txt").click(modifiers=["ControlOrMeta"])
        _explorer_row(page, "c.txt").click(modifiers=["ControlOrMeta"])
        _explorer_row(page, "d.txt").click(modifiers=["ControlOrMeta"])
        assert _selected_names(page) == ["a.txt", "c.txt", "d.txt"]
        _explorer_row(page, "d.txt").press("Delete")
        message = _answer_confirm(page, "fs-delete", accept=True)
        assert message.startswith("Delete 3 items?")
        for name in ("a.txt", "c.txt", "d.txt"):
            page.wait_for_selector(
                _explorer_row_sel("/multi/" + name), state="detached", timeout=15000,
            )
            assert not (folder / name).exists()
        assert (folder / "b.txt").is_file()
        # The editor tab of a deleted file closed with it.
        _wait_tab_count(page, tabs_before + 1)

        # A folder and a file inside it: the file is not sent separately
        # (the folder's delete takes it along), so one delete goes out.
        deletes_before = len(
            [f for f in _sent(frames, "fsAction") if f["action"] == "delete"],
        )
        _explorer_row(page, "multi").click(modifiers=["ControlOrMeta"])
        _explorer_row(page, "b.txt").click(modifiers=["ControlOrMeta"])
        assert _selected_names(page) == ["multi", "b.txt"]
        # Opened on the child, the menu is still the FOLDER's single-row
        # menu (the selection reduces to the folder).
        _explorer_row(page, "b.txt").click(button="right")
        assert _menu_labels(page)[:2] == ["New File...", "New Folder..."]
        page.keyboard.press("Escape")
        page.wait_for_selector("#sidebar-context-menu", state="hidden")
        assert _selected_names(page) == ["multi", "b.txt"]
        _explorer_row(page, "multi").click(button="right")
        _menu_item(page, "Delete").click()
        message = _answer_confirm(page, "fs-delete", accept=True)
        assert message.startswith("Delete 'multi'?")
        page.wait_for_selector(
            _explorer_row_sel("/multi", ".is-dir"), state="detached", timeout=15000,
        )
        assert not folder.exists()
        deletes = [f for f in _sent(frames, "fsAction") if f["action"] == "delete"]
        assert len(deletes) == deletes_before + 1
        assert deletes[-1]["path"] == str(folder)
    finally:
        context.close()
        if folder.exists():
            import shutil

            shutil.rmtree(folder, ignore_errors=True)


def test_explorer_multi_copy_pastes_every_entry(browser, harness, worktree):
    """Copy with two rows selected, then Paste into a folder: both files
    are copied there (one fsAction each)."""
    src = harness.work_dir / "multi-src"
    dest = harness.work_dir / "multi-dest"
    src.mkdir()
    dest.mkdir()
    (src / "one.txt").write_text("one\n")
    (src / "two.txt").write_text("two\n")
    context, page, frames = _open_page(browser, harness)
    try:
        _open_explorer(page)
        page.click("#explorer-refresh")
        page.wait_for_selector(_explorer_row_sel("/multi-src", ".is-dir"), timeout=15000)
        _explorer_row(page, "multi-src").click()
        page.wait_for_selector(_explorer_row_sel("/multi-src/two.txt"), timeout=15000)
        _explorer_row(page, "one.txt").click(modifiers=["ControlOrMeta"])
        _explorer_row(page, "two.txt").click(modifiers=["Shift"])
        assert _selected_names(page) == ["one.txt", "two.txt"]
        _explorer_row(page, "one.txt").click(button="right")
        _menu_item(page, "Copy").click()
        page.wait_for_selector("#sidebar-context-menu", state="hidden")
        _explorer_row(page, "multi-dest").click(button="right")
        _menu_item(page, "Paste").click()
        page.wait_for_selector(_explorer_row_sel("/multi-dest/one.txt"), timeout=15000)
        page.wait_for_selector(_explorer_row_sel("/multi-dest/two.txt"), timeout=15000)
        assert (dest / "one.txt").read_text() == "one\n"
        assert (dest / "two.txt").read_text() == "two\n"
        copies = [f for f in _sent(frames, "fsAction") if f["action"] == "copy"]
        assert sorted(c["path"] for c in copies[-2:]) == [
            str(src / "one.txt"), str(src / "two.txt"),
        ]
        assert {c["dest"] for c in copies[-2:]} == {str(dest)}
    finally:
        context.close()
        import shutil

        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(dest, ignore_errors=True)


def test_changes_panel_folds_each_worktree(browser, harness, worktree):
    """Each worktree header of the Changes list folds its rows (click,
    Enter, Left/Right arrows); the fold survives a refresh."""
    context, page, frames = _open_page(browser, harness)
    try:
        _open_scm(page)
        page.wait_for_selector("#scm-changes .scm-worktree-hdr", timeout=15000)
        headers = page.locator("#scm-changes .scm-worktree-hdr")
        assert headers.count() == 2
        wt_hdr = headers.filter(has_text="wt-task")
        wt_body = wt_hdr.locator("xpath=following-sibling::*[1]")
        assert wt_body.get_attribute("class") == "scm-worktree-body"
        assert wt_hdr.get_attribute("aria-expanded") == "true"
        assert wt_hdr.get_attribute("role") == "treeitem"
        wt_row = wt_body.locator(".scm-row", has_text="wt-untracked.txt")
        assert wt_row.is_visible()
        # Click folds the worktree; the main checkout stays open.
        wt_hdr.click()
        assert wt_hdr.get_attribute("aria-expanded") == "false"
        assert not wt_row.is_visible()
        main_hdr = headers.filter(has_text="repo")
        assert main_hdr.get_attribute("aria-expanded") == "true"
        assert page.locator(
            "#scm-changes .scm-row[data-scm-worktree]", has_text="README.md",
        ).first.is_visible()
        # A refresh re-renders the list with the fold kept.
        statuses_before = len(_sent(frames, "gitStatus"))
        page.click("#scm-refresh")
        for _ in range(100):
            if len(_sent(frames, "gitStatus")) > statuses_before:
                break
            page.wait_for_timeout(100)
        assert len(_sent(frames, "gitStatus")) > statuses_before
        # The reply re-renders the list; its header keeps the fold.
        page.wait_for_function(
            """() => {
                 const hdr = Array.from(document.querySelectorAll(
                   '#scm-changes .scm-worktree-hdr')).find(
                   h => h.textContent.includes('wt-task'));
                 return !!hdr && hdr.getAttribute('aria-expanded') === 'false';
               }""",
            timeout=15000,
        )
        page.wait_for_timeout(500)
        wt_hdr = page.locator("#scm-changes .scm-worktree-hdr", has_text="wt-task")
        assert wt_hdr.get_attribute("aria-expanded") == "false"
        assert not page.locator(
            "#scm-changes .scm-row", has_text="wt-untracked.txt",
        ).first.is_visible()
        # Keyboard: Right opens, Left folds, Enter toggles.
        wt_hdr.focus()
        wt_hdr.press("ArrowRight")
        assert wt_hdr.get_attribute("aria-expanded") == "true"
        assert page.locator(
            "#scm-changes .scm-row", has_text="wt-untracked.txt",
        ).first.is_visible()
        wt_hdr.press("ArrowLeft")
        assert wt_hdr.get_attribute("aria-expanded") == "false"
        wt_hdr.press("Enter")
        assert wt_hdr.get_attribute("aria-expanded") == "true"
        # Down from the header lands on its first row, up from a row
        # of the second worktree reaches the header.
        wt_hdr.press("ArrowDown")
        page.wait_for_function(
            "document.activeElement && document.activeElement.classList"
            ".contains('scm-row')",
        )
    finally:
        context.close()


# Monaco 0.52 keeps a disposed diff editor in getDiffEditors(); the one
# on screen is the one whose container is still attached and visible.
_VISIBLE_DIFF = """(() => monaco.editor.getDiffEditors().find(d =>
    d.getContainerDomNode().isConnected
    && d.getContainerDomNode().offsetParent !== null))()"""


def _diff_tab_texts(page) -> dict:
    """The original / modified texts of the diff editor on screen."""
    texts: dict = page.evaluate(
        """() => {
             const ed = """ + _VISIBLE_DIFF + """;
             if (!ed) return null;
             const m = ed.getModel();
             return {original: m.original.getValue(),
                     modified: m.modified.getValue(),
                     readOnly: ed.getModifiedEditor().getOption(
                       monaco.editor.EditorOption.readOnly)};
           }""",
    )
    return texts


def _wait_diff_tab(page, modified_text: str) -> dict:
    page.wait_for_function(
        """text => {
             if (!window.monaco) return false;
             const ed = """ + _VISIBLE_DIFF + """;
             const m = ed ? ed.getModel() : null;
             return !!m && m.modified.getValue().includes(text);
           }""",
        arg=modified_text,
        timeout=20000,
    )
    return _diff_tab_texts(page)


def test_graph_file_click_opens_a_diff_editor(browser, harness, worktree):
    """Clicking a file under a commit of the Graph opens VS Code's diff
    editor: the parent's version on the left, the commit's on the
    right, titled "name (parent ↔ sha)"; a file of the "Uncommitted
    changes" row diffs HEAD against the working tree."""
    context, page, frames = _open_page(browser, harness)
    try:
        _open_scm(page)
        tabs_before = page.locator(".chat-tab").count()
        second = page.locator("#scm-graph .scm-commit", has_text="second: rename")
        second.click()
        files = page.locator("#scm-graph .scm-commit-files .scm-row")
        files.filter(has_text="nested.py").first.click()
        _wait_tab_count(page, tabs_before + 1)
        diff = _wait_diff_tab(page, "x = 1  # nested-sentinel-4f2a")
        assert diff["original"] == ""
        assert diff["modified"] == "x = 1  # nested-sentinel-4f2a\n"
        assert diff["readOnly"] is True
        assert page.locator(".content-tab-view .monaco-diff-editor").first.is_visible()
        first7 = harness.shas["first"][:7]
        second7 = harness.shas["second"][:7]
        titles = page.eval_on_selector_all(
            ".chat-tab", "els => els.map(e => e.textContent)",
        )
        assert any(
            f"nested.py ({first7} \u2194 {second7})" in t for t in titles
        )
        shows = _sent(frames, "gitShow")
        assert shows[-1]["mode"] == "diff"
        assert shows[-1]["sha"] == harness.shas["second"]
        assert shows[-1]["path"] == "dir/nested.py"
        # The renamed file: the left side is the parent's a.txt.
        files.filter(has_text="b.txt").first.click()
        _wait_tab_count(page, tabs_before + 2)
        page.wait_for_function(
            f"Array.from(document.querySelectorAll('.chat-tab'))"
            f".some(t => t.textContent.includes('b.txt ({first7} \u2194 {second7})'))",
            timeout=15000,
        )
        diff = _wait_diff_tab(page, "a\n")
        assert diff["original"] == "a\n" and diff["modified"] == "a\n"
        assert _sent(frames, "gitShow")[-1]["origPath"] == "a.txt"
        # Clicking the same file again brings its tab back, no new tab.
        files.filter(has_text="nested.py").first.click()
        page.wait_for_timeout(500)
        assert page.locator(".chat-tab").count() == tabs_before + 2
        # The menu's Open Changes on a file of a commit is the same diff.
        files.filter(has_text="nested.py").first.click(button="right")
        assert _menu_labels(page) == ["Open Changes", "Open File"]
        page.keyboard.press("Escape")
        # A file of the main checkout's "Uncommitted changes" row:
        # HEAD on the left, the working tree on the right.
        rows = page.locator("#scm-graph .scm-commit.is-worktree")
        rows.first.click()
        wt_files = rows.first.locator("xpath=following-sibling::*[1]").locator(".scm-row")
        wt_files.filter(has_text="README.md").first.click()
        _wait_tab_count(page, tabs_before + 3)
        diff = _wait_diff_tab(page, "# readme changed")
        assert diff["original"] == "# readme\n"
        assert diff["modified"] == "# readme changed\n"
        titles = page.eval_on_selector_all(
            ".chat-tab", "els => els.map(e => e.textContent)",
        )
        assert any("README.md (Working Tree)" in t for t in titles)
        last = _sent(frames, "gitShow")[-1]
        assert last["sha"] == "" and last["mode"] == "diff"
        assert last["workDir"] == str(harness.work_dir)
        # Its context menu offers Open Changes ahead of Open File.
        wt_files.filter(has_text="README.md").first.click(button="right")
        assert _menu_labels(page)[:2] == ["Open Changes", "Open File"]
        page.keyboard.press("Escape")
        # The linked worktree's row diffs ITS HEAD against ITS file.
        rows.nth(1).click()
        lt_files = rows.nth(1).locator("xpath=following-sibling::*[1]").locator(".scm-row")
        lt_files.filter(has_text="README.md").first.click()
        _wait_tab_count(page, tabs_before + 4)
        diff = _wait_diff_tab(page, "# readme edited in the worktree")
        assert diff["original"] == "# readme\n"
        last = _sent(frames, "gitShow")[-1]
        assert last["workDir"] == str(worktree)
        # A deleted file: the right side is empty.
        wt_files.filter(has_text="b.txt").first.click()
        _wait_tab_count(page, tabs_before + 5)
        page.wait_for_function(
            "Array.from(document.querySelectorAll('.chat-tab'))"
            ".some(t => t.textContent.includes('b.txt (Working Tree)'))",
            timeout=15000,
        )
        page.wait_for_function(
            """() => {
                 const ed = """ + _VISIBLE_DIFF + """;
                 const m = ed ? ed.getModel() : null;
                 return !!m && m.original.getValue() === 'a\\n'
                   && m.modified.getValue() === '';
               }""",
            timeout=15000,
        )
    finally:
        context.close()


def test_graph_file_diff_without_monaco_is_a_unified_diff(browser, harness, worktree):
    """With the Monaco CDN unreachable the diff tab shows the two sides
    as a plain unified diff (LCS over lines) instead of a diff editor."""
    context = browser.new_context(ignore_https_errors=True, viewport={"width": 1400, "height": 900})
    context.route("https://cdn.jsdelivr.net/**", lambda route: route.abort())
    page = context.new_page()
    try:
        goto_retrying_network_change(page, harness.base_url + "/")
        page.wait_for_selector("#task-input", state="visible", timeout=30000)
        page.wait_for_function(
            "document.getElementById('meta-workdir').textContent.length > 1",
            timeout=30000,
        )
        _open_scm(page)
        rows = page.locator("#scm-graph .scm-commit.is-worktree")
        rows.first.click()
        wt_files = rows.first.locator("xpath=following-sibling::*[1]").locator(".scm-row")
        wt_files.filter(has_text="README.md").first.click()
        page.wait_for_selector(".content-tab-view .content-code-fallback", timeout=30000)
        text = page.locator(".content-tab-view .content-code-fallback").evaluate(
            "el => el.textContent",
        )
        assert text == "-# readme\n+# readme changed\n "
        assert page.locator(".content-menubar").count() == 0
    finally:
        context.close()


def test_folder_picker_changes_the_workspace(browser, harness, worktree):
    context, page, frames = _open_page(browser, harness)
    try:
        _open_explorer(page)
        assert page.locator("#explorer-pick-folder").is_visible()
        page.click("#explorer-pick-folder")
        page.wait_for_selector("#folder-picker:not([hidden])", timeout=5000)
        inp = page.locator("#folder-picker .folder-picker-input")
        page.wait_for_function(
            f"document.querySelector('.folder-picker-input').value === "
            f"{json.dumps(str(harness.work_dir.resolve()))}",
            timeout=15000,
        )
        # Only folders are listed; a click highlights one (its path goes
        # into the box), a double-click steps into it.
        page.wait_for_selector("#folder-picker .folder-picker-item", timeout=15000)
        items = page.eval_on_selector_all(
            "#folder-picker .folder-picker-item", "els => els.map(e => e.textContent)",
        )
        assert items == ["dir"]
        dir_item = page.locator("#folder-picker .folder-picker-item", has_text="dir")
        dir_item.click()
        assert dir_item.get_attribute("aria-selected") == "true"
        assert inp.input_value() == str(harness.work_dir.resolve() / "dir")
        dir_item.dblclick()
        page.wait_for_function(
            f"document.querySelector('.folder-picker-input').value === "
            f"{json.dumps(str(harness.work_dir.resolve() / 'dir'))}",
            timeout=15000,
        )
        subdirs = sorted(
            p.name for p in (harness.work_dir / "dir").iterdir() if p.is_dir()
        )
        if subdirs:
            page.wait_for_selector("#folder-picker .folder-picker-item", timeout=15000)
            assert page.eval_on_selector_all(
                "#folder-picker .folder-picker-item", "els => els.map(e => e.textContent)",
            ) == subdirs
        else:
            page.wait_for_selector(
                "#folder-picker .explorer-note:text-is('(no subfolders)')", timeout=15000,
            )
        # Up goes to the parent; typing a path + Enter navigates too.
        page.click("#folder-picker .folder-picker-up")
        page.wait_for_function(
            f"document.querySelector('.folder-picker-input').value === "
            f"{json.dumps(str(harness.work_dir.resolve()))}",
            timeout=15000,
        )
        inp.fill(str(harness.plain_dir))
        inp.press("Enter")
        page.wait_for_function(
            f"document.querySelector('.folder-picker-input').value === "
            f"{json.dumps(str(harness.plain_dir))}",
            timeout=15000,
        )
        # A bad path shows the daemon's error and keeps the dialog.
        inp.fill(str(harness.plain_dir / "nope"))
        inp.press("Enter")
        page.wait_for_function(
            "document.querySelector('.folder-picker-note').textContent.includes('not found')",
            timeout=15000,
        )
        inp.fill(str(harness.plain_dir))
        inp.press("Enter")
        page.wait_for_function(
            f"document.querySelector('.folder-picker-input').value === "
            f"{json.dumps(str(harness.plain_dir))}",
            timeout=15000,
        )
        page.click("#folder-picker .folder-picker-select")
        page.wait_for_selector("#folder-picker", state="hidden")
        # The Explorer now shows the picked folder as its root.
        _wait_explorer_root(page, "plain")
        page.wait_for_selector(
            _explorer_row_sel("/plain/only.txt"), timeout=15000,
        )
        # The daemon was told once, with setWorkDir (it persists the
        # global value itself; no saveConfig); the Source Control view
        # reports no repository.
        assert _sent(frames, "setWorkDir")[-1]["workDir"] == str(harness.plain_dir)
        assert not _sent(frames, "saveConfig")
        page.click("#activity-scm")
        page.wait_for_function(
            "document.getElementById('scm-changes').innerText.includes('Not a git repository')",
            timeout=15000,
        )
        # Escape closes a reopened picker without changing anything.
        page.click("#activity-explorer")
        page.click("#explorer-pick-folder")
        page.wait_for_selector("#folder-picker:not([hidden])", timeout=5000)
        page.keyboard.press("Escape")
        page.wait_for_selector("#folder-picker", state="hidden")
        assert _sent(frames, "setWorkDir")[-1]["workDir"] == str(harness.plain_dir)
        assert not _sent(frames, "saveConfig")
        # Highlighting a folder in the list and pressing Select picks it:
        # back to the repo (restoring the saved workspace for the other
        # tests too).
        page.click("#explorer-pick-folder")
        page.wait_for_selector("#folder-picker:not([hidden])", timeout=5000)
        inp.fill(harness.tmpdir)
        inp.press("Enter")
        repo_item = page.locator("#folder-picker .folder-picker-item").filter(
            has_text=re.compile(r"^repo$"),
        )
        repo_item.wait_for(timeout=15000)
        repo_item.click()
        assert inp.input_value() == str(harness.work_dir)
        page.click("#folder-picker .folder-picker-select")
        page.wait_for_selector("#folder-picker", state="hidden")
        _wait_explorer_root(page, "repo")
        assert _sent(frames, "setWorkDir")[-1]["workDir"] == str(harness.work_dir)
        # "Add Folder to Explorer..." offers the folders opened so far
        # (the daemon's recent_work_dirs) minus the ones already shown:
        # the plain folder opened above, not the repo.  One click adds it.
        page.click("#explorer-add-folder")
        page.wait_for_selector("#folder-picker:not([hidden])", timeout=5000)
        page.wait_for_selector(
            "#folder-picker .folder-picker-recent:not([hidden]) .workdir-item",
            timeout=15000,
        )
        recent = page.eval_on_selector_all(
            "#folder-picker .folder-picker-recent-list .workdir-item",
            "els => els.map(e => e.dataset.path)",
        )
        assert str(harness.plain_dir.resolve()) in recent
        assert str(harness.work_dir.resolve()) not in recent
        plain = json.dumps(str(harness.plain_dir.resolve()))
        page.click(f"#folder-picker .workdir-item[data-path={plain}]")
        page.wait_for_selector("#folder-picker", state="hidden")
        page.wait_for_selector(_explorer_row_sel("/plain/only.txt"), timeout=15000)
        roots = page.eval_on_selector_all(
            '.explorer-row[aria-level="1"]', "els => els.map(e => e.dataset.explorerPath)",
        )
        assert roots == [str(harness.work_dir.resolve()), str(harness.plain_dir.resolve())]
        assert _sent(frames, "setWorkDir")[-1]["workDir"] == str(harness.work_dir)
        assert not _sent(frames, "saveConfig")
    finally:
        context.close()


_PDF_VIEWER = ".content-tab-view .pdf-viewer"
_PDF_PAGE = _PDF_VIEWER + " .pdf-page"

# Geometry of the viewer: [scroller client width, first page box width,
# page count, zoom label, scroller scrollHeight, scroller clientHeight].
_PDF_GEOMETRY_JS = """() => {
  const scroller = document.querySelector('.content-tab-view .pdf-scroller');
  const pages = document.querySelectorAll('.content-tab-view .pdf-page');
  const box = pages[0].getBoundingClientRect();
  const styles = getComputedStyle(scroller.querySelector('.pdf-pages'));
  const padding = parseFloat(styles.paddingLeft) + parseFloat(styles.paddingRight);
  return {
    avail: scroller.clientWidth - padding,
    pageWidth: box.width,
    pageCount: pages.length,
    zoom: document.querySelector('.content-tab-view .pdf-zoom-level').textContent,
    scrollHeight: scroller.scrollHeight,
    clientHeight: scroller.clientHeight,
    touchAction: getComputedStyle(scroller).touchAction,
  };
}"""

# The toolbar's page indicator as the user reads it: "Page N of M" with N
# taken from the page field ("Loading…" before the document is open).
_PDF_STATUS_JS = """() => {
  const status = document.querySelector('.content-tab-view .pdf-status');
  const input = status.querySelector('.pdf-page-input');
  return [...status.childNodes]
    .map(node => (node === input ? input.value : node.textContent))
    .join('');
}"""

# The 1-based page under the vertical middle of the view (the next page
# when the middle falls in a gap; the last page past the end).
_PDF_MIDDLE_PAGE_JS = """() => {
  const s = document.querySelector('.content-tab-view .pdf-scroller');
  const middle = s.getBoundingClientRect().top + s.clientHeight / 2;
  const pages = [...document.querySelectorAll('.content-tab-view .pdf-page')];
  const i = pages.findIndex(p => p.getBoundingClientRect().bottom > middle);
  return (i < 0 ? pages.length : i + 1);
}"""


def _pdf_status(page) -> str:
    return str(page.evaluate(_PDF_STATUS_JS))


def _wait_pdf_status(page, text: str) -> None:
    page.wait_for_function(
        "text => (" + _PDF_STATUS_JS + ")() === text", arg=text, timeout=10000
    )


_PDFJS_MODULE = (
    "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.3.289/legacy/build/pdf.min.mjs"
)


def _wait_pdf_rendered(page) -> None:
    """Wait for the pdf.js viewer to draw the first page.  A viewer error
    fails the test unless the pdf.js CDN really is unreachable from the
    browser (then there is no viewer to test and the test skips)."""
    page.wait_for_function(
        """() => document.querySelector('.content-tab-view .pdf-page canvas')
             || document.querySelector('.content-tab-view .pdf-scroller .content-binary-note')""",
        timeout=60000,
    )
    if page.locator(_PDF_PAGE + " canvas").count() > 0:
        return
    note = page.locator(".content-tab-view .content-binary-note").first.inner_text()
    cdn_ok = page.evaluate(
        "url => fetch(url, {method: 'HEAD'}).then(r => r.ok).catch(() => false)",
        _PDFJS_MODULE,
    )
    if cdn_ok:
        pytest.fail("pdf.js viewer failed with the CDN reachable: " + note)
    pytest.skip("pdf.js CDN unreachable: " + note)


def _inject_file_link(page, path: str, link_id: str) -> None:
    """Append a ``span.kiss-filelink[data-path]`` for *path* to the chat
    output, the link a linkified tool output would carry."""
    page.evaluate(
        """([path, linkId]) => {
          const span = document.createElement('span');
          span.className = 'kiss-filelink';
          span.id = linkId;
          span.dataset.path = path;
          span.textContent = path;
          document.getElementById('output').appendChild(span);
        }""",
        [path, link_id],
    )


def _pdf_bytes(pages: int) -> bytes:
    """A minimal *pages*-page PDF of 200x100pt pages (no xref table;
    pdf.js rebuilds it, like it does for damaged files)."""
    kids = " ".join(f"{3 + i} 0 R" for i in range(pages))
    body = (
        b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        + f"2 0 obj << /Type /Pages /Kids [{kids}] /Count {pages} >> endobj\n".encode()
    )
    for i in range(pages):
        body += (
            f"{3 + i} 0 obj << /Type /Page /Parent 2 0 R"
            " /MediaBox [0 0 200 100] >> endobj\n"
        ).encode()
    return body + b"trailer << /Root 1 0 R >>\n%%EOF\n"


def _pinch(page, factor: float) -> None:
    """Dispatch a two-finger pinch on the viewer's scroller that spreads
    the fingers by *factor* (synthetic TouchEvents: Playwright drives one
    pointer at a time)."""
    page.evaluate(
        """factor => {
          const scroller = document.querySelector('.content-tab-view .pdf-scroller');
          const rect = scroller.getBoundingClientRect();
          const cx = rect.left + 100, cy = rect.top + 100;
          const mk = (id, x, y) =>
            new Touch({identifier: id, target: scroller, clientX: x, clientY: y});
          const fire = (type, touches) => scroller.dispatchEvent(new TouchEvent(type, {
            touches, targetTouches: touches, changedTouches: touches,
            bubbles: true, cancelable: true,
          }));
          fire('touchstart', [mk(1, cx - 50, cy), mk(2, cx + 50, cy)]);
          fire('touchmove', [mk(1, cx - 50 * factor, cy), mk(2, cx + 50 * factor, cy)]);
          fire('touchend', []);
        }""",
        factor,
    )


def test_pdf_click_opens_a_viewer_tab(browser, harness, worktree):
    """A PDF opens in the in-app pdf.js viewer: its single page is drawn
    on a canvas at fit-width scale, the toolbar zooms in and out and
    back to fit width, and the viewer goes with its tab."""
    context, page, frames = _open_page(browser, harness)
    try:
        _open_explorer(page)
        tabs_before = page.locator(".chat-tab").count()
        _explorer_row(page, "report.pdf").click()
        _wait_tab_count(page, tabs_before + 1)
        page.locator(_PDF_VIEWER).wait_for(timeout=15000)
        assert page.locator(".content-tab-view iframe").count() == 0
        _wait_pdf_rendered(page)
        # No error toast: the reply carried the bytes, not an error.
        assert "Cannot display" not in page.locator("body").inner_text()
        geo = page.evaluate(_PDF_GEOMETRY_JS)
        assert geo["pageCount"] == 1
        assert _pdf_status(page) == "Page 1 of 1"
        # The toolbar's Download link saves the file's own bytes under its name.
        download_link = page.locator(_PDF_VIEWER + " .pdf-download")
        assert download_link.inner_text() == "Download"
        assert (download_link.get_attribute("href") or "").startswith("blob:")
        with page.expect_download() as download_info:
            download_link.click()
        download = download_info.value
        assert download.suggested_filename == "report.pdf"
        saved = harness.work_dir / "downloaded-copy.pdf"
        download.save_as(str(saved))
        assert saved.read_bytes() == (harness.work_dir / "report.pdf").read_bytes()
        # A host without a URL (the VS Code panel) passes onDownload
        # instead: the link then calls it and does not navigate.
        callback_calls = page.evaluate(
            """async () => {
              const href = document.querySelector('.content-tab-view .pdf-download').href;
              const buf = await fetch(href).then(r => r.arrayBuffer());
              const holder = document.createElement('div');
              document.body.appendChild(holder);
              let calls = 0;
              const viewer = window.mountPdfViewer(holder, new Uint8Array(buf), {
                name: 'callback.pdf',
                onDownload: () => { calls++; },
              });
              const link = holder.querySelector('.pdf-download');
              const before = location.href;
              link.click();
              const result = {calls, navigated: location.href !== before,
                              title: link.title};
              viewer.dispose();
              holder.remove();
              return result;
            }""",
        )
        assert callback_calls == {"calls": 1, "navigated": False, "title": "Download callback.pdf"}
        # Fit width: the 200pt-wide page fills the box (within a pixel of
        # rounding), and the label shows that scale.
        assert abs(geo["pageWidth"] - geo["avail"]) <= 1.5
        fit_width = geo["pageWidth"]
        assert geo["zoom"] == f"{round(fit_width / 200 * 100)}%"
        # Zoom in: a quarter larger, re-drawn at the new size.
        page.click(_PDF_VIEWER + " .pdf-zoom-in")
        page.wait_for_function(
            f"() => Math.abs(document.querySelector('{_PDF_PAGE}').getBoundingClientRect().width"
            f" - {fit_width * 1.25}) <= 1.5",
            timeout=10000,
        )
        page.wait_for_function(
            f"() => document.querySelector('{_PDF_PAGE} canvas').width >= {fit_width * 1.25 - 2}",
            timeout=15000,
        )
        # Zoom out twice: a quarter smaller than fit width.
        page.click(_PDF_VIEWER + " .pdf-zoom-out")
        page.click(_PDF_VIEWER + " .pdf-zoom-out")
        page.wait_for_function(
            f"() => Math.abs(document.querySelector('{_PDF_PAGE}').getBoundingClientRect().width"
            f" - {fit_width / 1.25}) <= 1.5",
            timeout=10000,
        )
        # The percentage button returns to fit width.
        page.click(_PDF_VIEWER + " .pdf-zoom-level")
        page.wait_for_function(
            f"() => Math.abs(document.querySelector('{_PDF_PAGE}').getBoundingClientRect().width"
            f" - {fit_width}) <= 1.5",
            timeout=10000,
        )
        # Ctrl + wheel zooms too (a trackpad pinch on a desktop).
        page.hover(_PDF_PAGE)
        page.keyboard.down("Control")
        page.mouse.wheel(0, -100)
        page.keyboard.up("Control")
        page.wait_for_function(
            f"() => document.querySelector('{_PDF_PAGE}').getBoundingClientRect().width"
            f" > {fit_width * 1.5}",
            timeout=10000,
        )
        # Clicking the open PDF again reloads it in the same tab: the old
        # viewer (and its worker) go, a fresh one draws the page.
        _explorer_row(page, "report.pdf").click()
        page.wait_for_function(
            f"""() => {{
              const p = document.querySelector('{_PDF_PAGE}');
              return document.querySelectorAll('.pdf-viewer').length === 1 && !!p
                && Math.abs(p.getBoundingClientRect().width - {fit_width}) <= 1.5
                && !!p.querySelector('canvas');
            }}""",
            timeout=15000,
        )
        _wait_tab_count(page, tabs_before + 1)
        # An image opens as a picture.
        _explorer_row(page, "dot.png").click()
        _wait_tab_count(page, tabs_before + 2)
        img = page.locator(".content-tab-view img.content-image")
        img.wait_for(timeout=15000)
        assert (img.get_attribute("src") or "").startswith("blob:")
        # A second PDF has its own pdf.js worker: closing the first tab
        # (which frees that document) leaves the second one drawing.
        second = harness.work_dir / "report2.pdf"
        second.write_bytes((harness.work_dir / "report.pdf").read_bytes())
        # Back to the chat through its GROUP-STRIP entry: the main-row
        # entry stands for the whole group and would return to the tab
        # last viewed there (the picture), not to the chat.
        page.evaluate(
            "document.querySelector('#tab-list .chat-tab:not(.content-tab)').click()"
        )
        page.wait_for_selector("#output", state="visible", timeout=15000)
        _inject_file_link(page, str(second), "lnk-pdf2")
        page.click("#lnk-pdf2")
        _wait_tab_count(page, tabs_before + 3)
        _wait_pdf_rendered(page)
        # The chat's main-row entry is highlighted too, so the active
        # content tab (and its close button) is the strip's.
        page.locator("#tab-list .chat-tab", has_text="report.pdf").first.click()
        page.locator("#tab-list .chat-tab.active .chat-tab-close").click()
        _wait_tab_count(page, tabs_before + 2)
        page.locator("#tab-list .chat-tab", has_text="report2.pdf").first.click()
        page.locator(_PDF_VIEWER).wait_for(timeout=15000)
        page.click(_PDF_VIEWER + " .pdf-zoom-in")
        page.wait_for_function(
            f"() => document.querySelector('{_PDF_PAGE} canvas').width >= {fit_width * 1.25 - 2}",
            timeout=15000,
        )
        # Closing a PDF tab removes its viewer.
        page.locator("#tab-list .chat-tab.active .chat-tab-close").click()
        _wait_tab_count(page, tabs_before + 1)
        assert page.locator(".pdf-viewer").count() == 0
    finally:
        context.close()


_PDFJS_BUILD_GLOB = "https://cdn.jsdelivr.net/npm/pdfjs-dist@*/legacy/build/*"


def _route_pdfjs(context, decide) -> list[str]:
    """Intercept the pdf.js module and worker downloads; ``decide(route,
    n)`` handles the n-th (1-based) request for a file.  Returns the
    requested file names, one entry per request (a retried module import
    carries a query string, which is dropped here)."""
    requests: list[str] = []

    def handler(route):
        name = route.request.url.rsplit("/", 1)[1].split("?")[0]
        requests.append(name)
        decide(route, requests.count(name))

    context.route(_PDFJS_BUILD_GLOB, handler)
    return requests


def _open_report_pdf(page) -> None:
    _open_explorer(page)
    tabs_before = page.locator(".chat-tab").count()
    _explorer_row(page, "report.pdf").click()
    _wait_tab_count(page, tabs_before + 1)
    page.locator(_PDF_VIEWER).wait_for(timeout=15000)


def _pdf_note(page) -> str:
    """The viewer's error note (the Download link shares its class)."""
    note = page.locator(_PDF_VIEWER + " div.content-binary-note")
    note.wait_for(timeout=15000)
    return str(note.inner_text())


def test_pdf_viewer_retries_a_dropped_cdn_fetch(browser, harness, worktree):
    """A connection that drops under the first download of pdf.js (the
    module or its worker: a network change, not an HTTP error) is retried
    once, so the page is drawn instead of the tab saying "Cannot display"."""
    context, page, frames = _open_page(browser, harness)
    requests = _route_pdfjs(
        context,
        lambda route, n: route.abort("connectionfailed") if n == 1 else route.continue_(),
    )
    try:
        _open_report_pdf(page)
        _wait_pdf_rendered(page)
        assert page.locator(_PDF_VIEWER + " div.content-binary-note").count() == 0
        assert Counter(requests) == {"pdf.min.mjs": 2, "pdf.worker.min.mjs": 2}
    finally:
        context.close()


def test_pdf_viewer_gives_up_after_one_retry(browser, harness, worktree):
    """With the connection dropping every time, the viewer tries twice
    and then reports the failure; the next viewer starts afresh (and
    draws the page once the CDN is back)."""
    context, page, frames = _open_page(browser, harness)
    requests = _route_pdfjs(context, lambda route, n: route.abort("connectionfailed"))
    try:
        _open_report_pdf(page)
        # Whichever of the module import and the worker fetch fails first
        # names the error (the import's message goes on to name the URL).
        assert _pdf_note(page).startswith("Cannot display report.pdf: TypeError: Failed to fetch")
        assert Counter(requests) == {"pdf.min.mjs": 2, "pdf.worker.min.mjs": 2}
        context.unroute(_PDFJS_BUILD_GLOB)
        _explorer_row(page, "report.pdf").click()
        # The reopened tab replaces the failed viewer (and its note).
        page.locator(_PDF_VIEWER + " div.content-binary-note").wait_for(
            state="detached", timeout=15000
        )
        _wait_pdf_rendered(page)
        assert page.locator(_PDF_VIEWER + " div.content-binary-note").count() == 0
    finally:
        context.close()


def test_pdf_viewer_does_not_retry_an_http_error(browser, harness, worktree):
    """An HTTP error for the worker script is final: one request, and
    the note names the status."""
    context, page, frames = _open_page(browser, harness)

    def decide(route, n):
        if route.request.url.endswith("/pdf.worker.min.mjs"):
            route.fulfill(status=503, body="")
        else:
            route.continue_()

    requests = _route_pdfjs(context, decide)
    try:
        _open_report_pdf(page)
        assert _pdf_note(page) == "Cannot display report.pdf: Error: pdf.js worker HTTP 503"
        assert requests.count("pdf.worker.min.mjs") == 1
    finally:
        context.close()


def test_pdf_scrolls_and_pinch_zooms_on_a_phone(browser, harness, worktree):
    """In mobile mode the PDF fits the screen width, the page list is a
    real scroller (one finger pans it, the browser is told so through
    touch-action) and a two-finger pinch zooms it."""
    context = browser.new_context(
        ignore_https_errors=True,
        viewport={"width": 390, "height": 740},
        is_mobile=True,
        has_touch=True,
    )
    page = context.new_page()
    try:
        goto_retrying_network_change(page, harness.base_url + "/")
        page.wait_for_selector("#task-input", state="visible", timeout=30000)
        page.wait_for_selector(".chat-tab", timeout=30000)
        assert page.locator("body.remote-desktop").count() == 0
        _inject_file_link(page, str(harness.work_dir / "report.pdf"), "lnk-pdf")
        page.click("#lnk-pdf")
        page.locator(_PDF_VIEWER).wait_for(timeout=15000)
        _wait_pdf_rendered(page)
        geo = page.evaluate(_PDF_GEOMETRY_JS)
        # Fit to the phone's width: no horizontal overflow.
        assert abs(geo["pageWidth"] - geo["avail"]) <= 1.5
        assert geo["pageWidth"] <= 390
        assert geo["touchAction"] == "pan-x pan-y"
        fit_width = geo["pageWidth"]
        # Pinch out: the page quadruples (the 2:1 test page is then
        # taller than the phone too), overflows the phone in both
        # directions and the scroller (not the document) carries it.
        _pinch(page, 4.0)
        page.wait_for_function(
            f"() => Math.abs(document.querySelector('{_PDF_PAGE}').getBoundingClientRect().width"
            f" - {fit_width * 4}) <= 3",
            timeout=10000,
        )
        scrolled = page.evaluate(
            """() => {
              const s = document.querySelector('.content-tab-view .pdf-scroller');
              s.scrollLeft = 10000; s.scrollTop = 10000;
              return {left: s.scrollLeft, top: s.scrollTop,
                      overflowX: s.scrollWidth - s.clientWidth,
                      overflowY: s.scrollHeight - s.clientHeight,
                      pageScrolled: document.scrollingElement.scrollTop};
            }""",
        )
        assert scrolled["overflowX"] > fit_width * 0.9
        assert scrolled["overflowY"] > 0
        assert scrolled["left"] == scrolled["overflowX"]
        assert scrolled["top"] == scrolled["overflowY"]
        assert scrolled["pageScrolled"] == 0
        # Pinch in from 4x: a fifth of that is below fit width.
        _pinch(page, 0.2)
        page.wait_for_function(
            f"() => document.querySelector('{_PDF_PAGE}').getBoundingClientRect().width"
            f" < {fit_width}",
            timeout=10000,
        )
    finally:
        context.close()


def test_pdf_zoom_keeps_the_point_under_the_gesture(browser, harness, worktree):
    """Zooming a multi-page PDF anchors the document point at the centre
    of the view (the gaps between pages do not scale, so page 5 must be
    anchored on page 5, not on a scaled scroll offset), pages that
    scroll a screen away give their canvas back, and the toolbar's page
    indicator follows the page under the middle of the view."""
    pdf = harness.work_dir / "pages8.pdf"
    pdf.write_bytes(_pdf_bytes(8))
    context, page, frames = _open_page(browser, harness)
    try:
        _inject_file_link(page, str(pdf), "lnk-pdf8")
        page.click("#lnk-pdf8")
        page.locator(_PDF_VIEWER).wait_for(timeout=15000)
        _wait_pdf_rendered(page)
        assert page.locator(_PDF_PAGE).count() == 8
        assert _pdf_status(page) == "Page 1 of 8"
        # Scrolling to the end puts the last page under the middle.
        page.evaluate(
            "document.querySelector('.content-tab-view .pdf-scroller').scrollTop = 1e6"
        )
        _wait_pdf_status(page, "Page 8 of 8")
        # Scroll so that page 5 sits 40px below the top of the view.
        page.evaluate(
            """() => {
              const s = document.querySelector('.content-tab-view .pdf-scroller');
              const p = document.querySelectorAll('.content-tab-view .pdf-page')[4];
              s.scrollTop = p.getBoundingClientRect().top - s.getBoundingClientRect().top
                            + s.scrollTop - 40;
            }""",
        )
        # The indicator names the page under the middle of the view (page
        # 5 when it is taller than half the view, otherwise a later one).
        expected_page = page.evaluate(_PDF_MIDDLE_PAGE_JS)
        assert expected_page >= 5
        _wait_pdf_status(page, f"Page {expected_page} of 8")
        # Pages 1-2 are more than a screen above: no canvas any more,
        # while page 5 is drawn.
        page.wait_for_function(
            """() => {
              const pages = document.querySelectorAll('.content-tab-view .pdf-page');
              return !pages[0].querySelector('canvas') && !!pages[4].querySelector('canvas');
            }""",
            timeout=15000,
        )
        before = page.evaluate(
            """() => {
              const s = document.querySelector('.content-tab-view .pdf-scroller');
              const p = document.querySelectorAll('.content-tab-view .pdf-page')[4];
              const r = p.getBoundingClientRect(), b = s.getBoundingClientRect();
              return {cy: s.clientHeight / 2, top: r.top - b.top, width: r.width};
            }""",
        )
        page.click(_PDF_VIEWER + " .pdf-zoom-in")
        after = page.evaluate(
            """() => {
              const s = document.querySelector('.content-tab-view .pdf-scroller');
              const p = document.querySelectorAll('.content-tab-view .pdf-page')[4];
              const r = p.getBoundingClientRect(), b = s.getBoundingClientRect();
              return {cy: s.clientHeight / 2, top: r.top - b.top, width: r.width};
            }""",
        )
        ratio = after["width"] / before["width"]
        assert abs(ratio - 1.25) < 0.02
        # The point of page 5 at the view's centre is still at the centre.
        assert abs((after["cy"] - after["top"]) - (before["cy"] - before["top"]) * ratio) <= 2
        # The zoom kept the same page under the middle.
        assert _pdf_status(page) == f"Page {expected_page} of 8"
        # Growing the view (the zoom is manual now, so the pages stay put)
        # moves its middle down without a scroll event: put the middle
        # 40px above page 4's bottom, then add 200px of height and the
        # middle lands on page 5.
        page.evaluate(
            """() => {
              const s = document.querySelector('.content-tab-view .pdf-scroller');
              const p = document.querySelectorAll('.content-tab-view .pdf-page')[3];
              s.scrollTop = p.getBoundingClientRect().bottom - s.getBoundingClientRect().top
                            + s.scrollTop - 40 - s.clientHeight / 2;
            }""",
        )
        _wait_pdf_status(page, "Page 4 of 8")
        size = page.viewport_size
        scroll_top = page.evaluate(
            "document.querySelector('.content-tab-view .pdf-scroller').scrollTop"
        )
        page.set_viewport_size({"width": size["width"], "height": size["height"] + 200})
        _wait_pdf_status(page, "Page 5 of 8")
        assert (
            page.evaluate(
                "document.querySelector('.content-tab-view .pdf-scroller').scrollTop"
            )
            == scroll_top
        )
    finally:
        context.close()


# Where page *n* (1-based) starts relative to the top of the view, the
# view's scroll offset, and the margin above the first page at offset 0.
_PDF_PAGE_TOP_JS = """n => {
  const s = document.querySelector('.content-tab-view .pdf-scroller');
  const p = document.querySelectorAll('.content-tab-view .pdf-page')[n - 1];
  return {
    top: p.getBoundingClientRect().top - s.getBoundingClientRect().top,
    scrollTop: s.scrollTop,
    padding: parseFloat(getComputedStyle(s.querySelector('.pdf-pages')).paddingTop),
    focused: document.activeElement === document.querySelector('.pdf-page-input'),
  };
}"""


def test_pdf_page_field_jumps_to_the_typed_page(browser, harness, worktree):
    """The page number in the toolbar is a field: a number typed into it
    and Enter scrolls that page to the top of the view (out-of-range
    numbers go to the first or last page, a non-number goes nowhere),
    scrolling leaves a number being typed alone, and Escape abandons
    the edit."""
    pdf = harness.work_dir / "pages8.pdf"
    pdf.write_bytes(_pdf_bytes(8))
    context, page, frames = _open_page(browser, harness)
    try:
        _inject_file_link(page, str(pdf), "lnk-pdf8")
        page.click("#lnk-pdf8")
        page.locator(_PDF_VIEWER).wait_for(timeout=15000)
        _wait_pdf_rendered(page)
        field = page.locator(_PDF_VIEWER + " .pdf-page-input")
        assert _pdf_status(page) == "Page 1 of 8"
        assert field.get_attribute("inputmode") == "numeric"
        assert field.get_attribute("enterkeyhint") == "go"
        # Focusing the field selects the number, so typing replaces it.
        field.click()
        assert page.evaluate(
            """() => {
              const f = document.activeElement;
              return f.className === 'pdf-page-input'
                && f.selectionStart === 0 && f.selectionEnd === f.value.length;
            }"""
        )
        page.keyboard.type("5")
        page.keyboard.press("Enter")
        # Page 5 starts at the top of the view, under the same margin as
        # the first page at offset 0; the field is left, showing the
        # page under the middle again (page 5 or, if the pages are
        # shorter than half the view, a later one).
        geo = page.evaluate(_PDF_PAGE_TOP_JS, 5)
        assert abs(geo["top"] - geo["padding"]) <= 1.5
        assert geo["focused"] is False
        middle_page = page.evaluate(_PDF_MIDDLE_PAGE_JS)
        assert middle_page >= 5
        _wait_pdf_status(page, f"Page {middle_page} of 8")
        # A number past the end goes to the last page (the view cannot
        # scroll that far, so it ends at the bottom).
        field.fill("99")
        page.keyboard.press("Enter")
        _wait_pdf_status(page, "Page 8 of 8")
        assert page.evaluate(
            """() => {
              const s = document.querySelector('.content-tab-view .pdf-scroller');
              return s.scrollTop + s.clientHeight === s.scrollHeight;
            }"""
        )
        # Zero goes to the first page: back at the very top.
        field.fill("0")
        page.keyboard.press("Enter")
        _wait_pdf_status(page, "Page 1 of 8")
        assert page.evaluate(_PDF_PAGE_TOP_JS, 1)["scrollTop"] == 0
        # A non-number goes nowhere; the field shows the real page again.
        field.fill("abc")
        page.keyboard.press("Enter")
        geo = page.evaluate(_PDF_PAGE_TOP_JS, 1)
        assert geo["scrollTop"] == 0
        assert geo["focused"] is False
        assert _pdf_status(page) == "Page 1 of 8"
        # A number being typed survives a scroll (the field only catches
        # up once it is left), and Escape abandons it without scrolling.
        field.click()
        page.keyboard.type("3")
        page.evaluate(
            "document.querySelector('.content-tab-view .pdf-scroller').scrollTop = 1e6"
        )
        # Two frames: the indicator refresh runs on the next one.
        page.evaluate(
            "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
        )
        assert field.input_value() == "3"
        assert page.evaluate(_PDF_PAGE_TOP_JS, 8)["focused"] is True
        scroll_top = page.evaluate(_PDF_PAGE_TOP_JS, 8)["scrollTop"]
        page.keyboard.press("Escape")
        geo = page.evaluate(_PDF_PAGE_TOP_JS, 8)
        assert geo["focused"] is False
        assert geo["scrollTop"] == scroll_top
        assert _pdf_status(page) == "Page 8 of 8"
    finally:
        context.close()


def _pdf_scroll_top(page) -> float:
    return float(page.evaluate(_PDF_PAGE_TOP_JS, 1)["scrollTop"])


def _assert_pdf_page_at_top(page, n: int) -> None:
    """Page *n* starts at the top of the view, under the margin the
    first page has at scroll offset 0."""
    geo = page.evaluate(_PDF_PAGE_TOP_JS, n)
    assert abs(geo["top"] - geo["padding"]) <= 1.5, geo


def _scroll_pdf_into_page(page, n: int, offset: int) -> None:
    """Scroll so that page *n* starts *offset* pixels above the view's top."""
    page.evaluate(
        """([n, offset]) => {
          const s = document.querySelector('.content-tab-view .pdf-scroller');
          const p = document.querySelectorAll('.content-tab-view .pdf-page')[n - 1];
          s.scrollTop = p.getBoundingClientRect().top - s.getBoundingClientRect().top
                        + s.scrollTop + offset;
        }""",
        [n, offset],
    )


def _wait_pdf_page_width(page, width: float) -> None:
    page.wait_for_function(
        f"() => Math.abs(document.querySelector('{_PDF_PAGE}').getBoundingClientRect().width"
        f" - {width}) <= 1.5",
        timeout=10000,
    )


def test_pdf_keyboard_shortcuts_move_pages_and_zoom(browser, harness, worktree):
    """With the viewer on screen and nothing focused, PageDown / ArrowRight
    and PageUp / ArrowLeft move one page forward and back, Home / End go
    to the first / last page, Ctrl/Cmd+0 fits the width and Ctrl/Cmd
    with + / - zooms.  The keys are left alone while typed into the page
    field, with Shift or Alt held, while the viewer's tab is hidden, and
    the arrows pan instead once the pages are wider than the view."""
    pdf = harness.work_dir / "pages8.pdf"
    pdf.write_bytes(_pdf_bytes(8))
    context, page, frames = _open_page(browser, harness)
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    try:
        _inject_file_link(page, str(pdf), "lnk-pdf8")
        page.click("#lnk-pdf8")
        page.locator(_PDF_VIEWER).wait_for(timeout=15000)
        _wait_pdf_rendered(page)
        assert _pdf_status(page) == "Page 1 of 8"
        # No click into the viewer first: the keys work as soon as the
        # viewer is shown.
        page.evaluate("document.activeElement.blur()")
        page.keyboard.press("PageDown")
        _assert_pdf_page_at_top(page, 2)
        page.keyboard.press("ArrowRight")
        _assert_pdf_page_at_top(page, 3)
        page.keyboard.press("PageUp")
        _assert_pdf_page_at_top(page, 2)
        page.keyboard.press("ArrowLeft")
        assert _pdf_scroll_top(page) == 0
        # Before the first page there is nothing to go to.
        page.keyboard.press("ArrowLeft")
        assert _pdf_scroll_top(page) == 0
        page.keyboard.press("End")
        _wait_pdf_status(page, "Page 8 of 8")
        at_bottom = page.evaluate(
            """() => {
              const s = document.querySelector('.content-tab-view .pdf-scroller');
              return s.scrollTop + s.clientHeight === s.scrollHeight;
            }"""
        )
        assert at_bottom
        bottom = _pdf_scroll_top(page)
        page.keyboard.press("PageDown")
        assert _pdf_scroll_top(page) == bottom
        page.keyboard.press("Home")
        assert _pdf_scroll_top(page) == 0
        _wait_pdf_status(page, "Page 1 of 8")
        # From inside page 3 (its top 40px above the view), forward goes
        # to the top of page 4 and back to the top of page 2.
        _scroll_pdf_into_page(page, 3, 40)
        page.keyboard.press("PageDown")
        _assert_pdf_page_at_top(page, 4)
        _scroll_pdf_into_page(page, 3, 40)
        page.keyboard.press("PageUp")
        _assert_pdf_page_at_top(page, 2)
        # Zoom: Ctrl+= a quarter larger, Ctrl+- back, Ctrl+Shift+= (the
        # + key) in again, Cmd+0 back to fit width.
        geo = page.evaluate(_PDF_GEOMETRY_JS)
        fit_width = geo["pageWidth"]
        assert abs(fit_width - geo["avail"]) <= 1.5
        page.keyboard.press("Control+Equal")
        _wait_pdf_page_width(page, fit_width * 1.25)
        page.keyboard.press("Control+Minus")
        _wait_pdf_page_width(page, fit_width)
        page.keyboard.press("Control+Shift+Equal")
        _wait_pdf_page_width(page, fit_width * 1.25)
        page.keyboard.press("Meta+0")
        _wait_pdf_page_width(page, fit_width)
        # Zoomed in past the view's width the arrows are the browser's
        # (they pan); PageDown still turns the page.
        page.keyboard.press("Home")
        page.keyboard.press("Control+Equal")
        _wait_pdf_page_width(page, fit_width * 1.25)
        page.keyboard.press("Home")
        assert _pdf_scroll_top(page) == 0
        page.keyboard.press("ArrowRight")
        assert _pdf_scroll_top(page) == 0
        page.keyboard.press("PageDown")
        _assert_pdf_page_at_top(page, 2)
        # Ctrl+0 restores fit-width mode, so the pages follow a resize.
        page.keyboard.press("Control+0")
        _wait_pdf_page_width(page, fit_width)
        size = page.viewport_size
        page.set_viewport_size({"width": size["width"] - 200, "height": size["height"]})
        page.wait_for_function(
            "() => { const g = (" + _PDF_GEOMETRY_JS + ")();"
            " return Math.abs(g.pageWidth - g.avail) <= 1.5 && g.pageWidth < "
            + str(fit_width - 100)
            + "; }",
            timeout=10000,
        )
        page.set_viewport_size(size)
        _wait_pdf_page_width(page, fit_width)
        page.keyboard.press("Home")
        assert _pdf_scroll_top(page) == 0
        # Keys typed into the page field are the field's.
        page.locator(_PDF_VIEWER + " .pdf-page-input").click()
        page.keyboard.press("PageDown")
        assert _pdf_scroll_top(page) == 0
        page.keyboard.press("Escape")
        # Shift or Alt combinations are not the viewer's.
        page.keyboard.press("Shift+PageDown")
        page.keyboard.press("Alt+ArrowRight")
        assert _pdf_scroll_top(page) == 0
        # A hidden viewer (another tab is shown) leaves the keys alone,
        # and comes back at the same place (its zero-size box while
        # hidden must not re-fit the width).
        page.keyboard.press("PageDown")
        _assert_pdf_page_at_top(page, 2)
        page_two = _pdf_scroll_top(page)
        # The chat's GROUP-STRIP entry shows the chat; its main-row entry
        # would return to the group's last viewed tab, the viewer itself.
        page.evaluate(
            "document.querySelector('#tab-list .chat-tab:not(.content-tab)').click()"
        )
        page.wait_for_selector("#output", state="visible", timeout=15000)
        page.evaluate("document.activeElement.blur()")
        page.keyboard.press("PageDown")
        page.locator("#tab-list .chat-tab", has_text="pages8.pdf").first.click()
        page.locator(_PDF_VIEWER).wait_for(timeout=15000)
        assert _pdf_scroll_top(page) == page_two
        _assert_pdf_page_at_top(page, 2)
        page.evaluate("document.activeElement.blur()")
        page.keyboard.press("PageDown")
        _assert_pdf_page_at_top(page, 3)
        # Closing the tab takes the listener with it: the key is nobody's.
        # (The chat's main-row entry is highlighted as well, so the
        # active tab's close button is the strip's.)
        page.locator("#tab-list .chat-tab.active .chat-tab-close").click()
        page.wait_for_function(
            "() => document.querySelectorAll('.pdf-viewer').length === 0", timeout=15000
        )
        page.keyboard.press("PageDown")
        page.keyboard.press("Control+0")
        assert errors == []
    finally:
        context.close()


def _root_paths(page) -> list[str]:
    return [
        str(p)
        for p in page.eval_on_selector_all(
            "#explorer-tree > .explorer-row.is-root",
            "els => els.map(e => e.dataset.explorerPath)",
        )
    ]


def _wait_first_root(page, path: str) -> None:
    """Wait until the first top-level Explorer folder is *path*."""
    page.wait_for_function(
        """p => {
             const row = document.querySelector('#explorer-tree > .explorer-row.is-root');
             return !!row && row.dataset.explorerPath === p;
           }""",
        arg=path,
        timeout=15000,
    )


def test_add_folder_set_work_dir_and_remove(browser, harness, worktree):
    """Add Folder to Explorer / Set as Working Directory / Remove Folder.

    The added folder is listed from the REAL file system, actions on it
    are confined to it (a New File... lands on disk inside it), the
    switch of the working directory reaches the daemon and the removed
    folder leaves nothing behind but the files on disk.
    """
    context, page, frames = _open_page(browser, harness)
    plain = str(harness.plain_dir)
    repo = str(harness.work_dir)
    try:
        _open_explorer(page)
        assert _root_paths(page) == [repo]
        # The working directory carries a check mark, no buttons.
        assert page.locator(".explorer-row.is-workdir .explorer-root-mark").count() == 1
        assert page.locator(".explorer-row.is-root .explorer-root-btn").count() == 0
        # Add the plain folder through the picker in "add" mode.
        page.click("#explorer-add-folder")
        page.wait_for_selector("#folder-picker:not([hidden])", timeout=5000)
        assert page.locator("#folder-picker-title").inner_text() == "Add Folder to Explorer"
        assert page.locator("#folder-picker .folder-picker-select").inner_text() == "Add Folder"
        inp = page.locator("#folder-picker .folder-picker-input")
        inp.fill(plain)
        inp.press("Enter")
        page.wait_for_selector(
            "#folder-picker .explorer-note:text-is('(no subfolders)')", timeout=15000,
        )
        picks_before = len(_sent(frames, "setWorkDir"))
        page.click("#folder-picker .folder-picker-select")
        page.wait_for_selector("#folder-picker", state="hidden")
        page.wait_for_function(
            "document.querySelectorAll('#explorer-tree > .explorer-row.is-root').length === 2",
            timeout=15000,
        )
        assert _root_paths(page) == [repo, plain]
        # Listed from disk, confined to itself; the work dir is untouched.
        only = page.locator(_row_at(os.path.join(plain, "only.txt")))
        only.wait_for(timeout=15000)
        assert only.get_attribute("data-explorer-root") == plain
        listed = [f for f in _sent(frames, "listDir") if f.get("path") == plain]
        assert listed and listed[-1]["workDir"] == plain
        assert len(_sent(frames, "setWorkDir")) == picks_before
        assert page.evaluate("JSON.parse(localStorage.getItem('kiss-explorer-roots'))") == [plain]
        # A file of the added folder opens as a content tab.
        tabs_before = page.locator(".chat-tab").count()
        only.click()
        _wait_tab_count(page, tabs_before + 1)
        assert "only" in page.locator(".content-tab-view").last.inner_text()
        # New File... on the added folder lands on disk inside it (the
        # daemon accepted the folder as the action's workDir).
        plain_root = page.locator(_row_at(plain, ".is-root"))
        plain_root.click(button="right")
        labels = _menu_labels(page)
        assert "Add Folder to Explorer..." in labels
        assert "Set as Working Directory" in labels
        assert "Remove Folder from Explorer" in labels
        assert "Rename..." not in labels and "Delete" not in labels
        _menu_item(page, "New File...").click()
        box = page.locator("#explorer-tree .explorer-input")
        box.wait_for(timeout=5000)
        box.fill("added.txt")
        box.press("Enter")
        page.wait_for_selector(
            _row_at(os.path.join(plain, "added.txt")), timeout=15000,
        )
        assert (harness.plain_dir / "added.txt").is_file()
        (harness.plain_dir / "added.txt").unlink()
        # Set as Working Directory (the buttons show on hover) is the
        # "..." > Working directory flow for that folder: the daemon
        # lists it first (the 'workdir:' listDir check), then setWorkDir
        # makes it the global working directory (the daemon persists it;
        # no saveConfig), and the old working directory stays listed as
        # an added folder.
        _click_root_button(page, plain, "set")
        _wait_first_root(page, plain)
        assert _root_paths(page) == [plain, repo]
        checks = [
            f for f in _sent(frames, "listDir")
            if str(f.get("token", "")).startswith("workdir:") and f.get("path") == plain
        ]
        assert checks and checks[-1]["workDir"] == plain
        assert _sent(frames, "setWorkDir")[-1]["workDir"] == plain
        assert not _sent(frames, "saveConfig")
        assert page.locator(".explorer-row.is-workdir").get_attribute("data-explorer-path") == plain
        page.wait_for_selector(
            _row_at(os.path.join(plain, "only.txt")), timeout=15000,
        )
        # Switch back through the repo row's context menu.
        page.locator(_row_at(repo, ".is-root")).click(button="right")
        _menu_item(page, "Set as Working Directory").click()
        _wait_first_root(page, repo)
        assert _root_paths(page) == [repo, plain]
        assert _sent(frames, "setWorkDir")[-1]["workDir"] == repo
        assert not _sent(frames, "saveConfig")
        # Remove the plain folder with its button: gone from the tree and
        # from storage, still on disk.
        fs_before = len(_sent(frames, "fsAction"))
        _click_root_button(page, plain, "remove")
        page.wait_for_function(
            "document.querySelectorAll('#explorer-tree > .explorer-row.is-root').length === 1",
            timeout=15000,
        )
        assert _root_paths(page) == [repo]
        # The repo stayed remembered from the switch (it is shown once,
        # as the working directory); the plain folder is forgotten.
        assert page.evaluate("JSON.parse(localStorage.getItem('kiss-explorer-roots'))") == [repo]
        assert len(_sent(frames, "fsAction")) == fs_before
        assert (harness.plain_dir / "only.txt").is_file()
        # The Explorer keeps its working directory's tree through all this.
        page.wait_for_selector(".explorer-row.is-file", timeout=15000)
        _wait_explorer_root(page, "repo")
    finally:
        context.close()


def test_history_groups_tasks_by_chat_with_day_separators(browser, harness, worktree):
    """The Tasks view groups the daemon's history by chat, newest chat
    first, under "Today" / "Yesterday" / date separators."""
    import datetime as _dt

    import kiss.agents.sorcar.persistence as th

    def _add(task: str, chat_id: str, ts: float) -> str:
        task_id, chat = th._add_task(task, chat_id)
        th._save_task_result("done", task_id=task_id)
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute("UPDATE task_history SET timestamp = ? WHERE id = ?", (ts, task_id))
        return chat

    noon = _dt.datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    today = noon.timestamp()
    day = 86400.0
    chat_a = _add("alpha one", "", today - 600)
    _add("alpha two", chat_a, today)
    chat_b = _add("beta one", "", today - 300)
    chat_c = _add("gamma one", "", today - day)
    _add("beta zero", chat_b, today - day - 60)
    chat_d = _add("delta one", "", today - 3 * day)
    # Chat B has been summarised (as `upsert_chat_summary` does when a
    # task finishes); its panel header shows the summary, not "beta zero".
    with th._rw_lock.write_lock():
        th._get_db().execute(
            "INSERT INTO chat_summaries (chat_id, summary, last_launched) VALUES (?, ?, ?)",
            (chat_b, "Beta rollout and follow-up", int((today - 300) * 1000)),
        )
    context, page, frames = _open_page(browser, harness)
    try:
        page.click("#activity-tasks")
        page.wait_for_selector("#history-list .history-chat-group", timeout=15000)
        assert (
            page.locator(f".history-chat-group[data-chat-id='{chat_b}'] .history-chat-title")
            .inner_text()
            == "Beta rollout and follow-up"
        )
        shape = page.evaluate(
            """() => Array.from(document.getElementById('history-list').children)
                 .filter(el => el.style.display !== 'none')
                 .map(el => el.classList.contains('history-day-sep')
                   ? 'sep:' + el.textContent
                   : el.classList.contains('history-chat-group')
                     ? 'chat:' + el.dataset.chatId + '[' +
                       Array.from(el.querySelectorAll('.sidebar-item-text'))
                         .map(t => t.textContent).join(',') + ']'
                     : el.className)"""
        )
        three_days = noon - _dt.timedelta(days=3)
        label = page.evaluate(
            "ts => new Date(ts * 1000).toLocaleDateString(undefined, "
            "{weekday: 'short', month: 'short', day: 'numeric'})",
            three_days.timestamp(),
        )
        assert shape == [
            "sep:Today",
            f"chat:{chat_a}[alpha two,alpha one]",
            f"chat:{chat_b}[beta one,beta zero]",
            "sep:Yesterday",
            f"chat:{chat_c}[gamma one]",
            f"sep:{label}",
            f"chat:{chat_d}[delta one]",
        ]
        # Clicking a task inside a chat block still opens it in a tab
        # (no events were persisted, so the task shows read-only).
        # Capture inbound WebSocket frames over CDP first, so the test
        # can prove the DAEMON registered the fresh tab (a canonical
        # `tabs_state` snapshot naming it), not just that the page
        # created one locally.
        received: list[str] = []
        cdp = context.new_cdp_session(page)
        cdp.on(
            "Network.webSocketFrameReceived",
            lambda ev: received.append(
                ev.get("response", {}).get("payloadData", ""),
            ),
        )
        cdp.send("Network.enable")
        pre_tab_ids = page.evaluate(
            "Array.from(document.querySelectorAll('.chat-tab'))"
            ".map(t => t.dataset.tabId)"
        )
        # Chat panels are collapsed by default (nothing is running):
        # the header of a chat not yet summarised shows the chat's FIRST
        # task and opens the panel.
        group_a = page.locator(f".history-chat-group[data-chat-id='{chat_a}']")
        assert "collapsed" in (group_a.get_attribute("class") or "")
        header_a = group_a.locator(".history-chat-header")
        assert header_a.locator(".history-chat-title").inner_text() == "alpha one"
        header_a.click()
        assert "collapsed" not in (group_a.get_attribute("class") or "")
        page.locator(".sidebar-item", has_text="alpha one").click()
        # Opening a history task creates a fresh REGISTERED tab; the
        # daemon's canonical `tabs_state` snapshot prunes the blank
        # never-registered placeholder while REGISTERED tabs (e.g. one
        # left by an earlier test in this workspace) rightly survive, so
        # neither the tab count nor "every old id vanished" is a stable
        # outcome.  Assert the designed outcome instead: the surviving
        # ACTIVE tab is fresh (an id the page did not have before the
        # click), it shows the clicked task in its read-only task panel,
        # and a canonical `tabs_state` snapshot names that fresh id.
        page.wait_for_function(
            "document.getElementById('task-panel-text')"
            " && document.getElementById('task-panel-text').textContent"
            "      === 'alpha one'",
            timeout=15000,
        )
        page.wait_for_function(
            "pre => { const act = document.querySelector('.chat-tab.active');"
            " return !!act && !pre.includes(act.dataset.tabId); }",
            arg=pre_tab_ids,
            timeout=15000,
        )
        active_id = page.evaluate(
            "document.querySelector('.chat-tab.active').dataset.tabId"
        )
        for _ in range(100):
            if any(
                '"tabs_state"' in frame and active_id in frame
                for frame in received
            ):
                break
            page.wait_for_timeout(50)
        else:
            raise AssertionError(
                "no canonical tabs_state snapshot named the fresh tab "
                f"{active_id!r}: registration never reached the daemon"
            )
    finally:
        context.close()
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "DELETE FROM task_history WHERE chat_id IN (?, ?, ?, ?)",
                (chat_a, chat_b, chat_c, chat_d),
            )


def test_history_click_survives_mid_press_refresh(browser, harness, worktree):
    """A real mouse press held on a history row while a changed-data
    refresh arrives over the WebSocket still opens the task: the
    destructive rebuild is deferred past the browser-synthesized click."""
    import datetime as _dt

    import kiss.agents.sorcar.persistence as th

    def _add(task: str, ts: float) -> str:
        task_id, chat = th._add_task(task, "")
        th._save_task_result("done", task_id=task_id)
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "UPDATE task_history SET timestamp = ? WHERE id = ?",
                (ts, task_id),
            )
        return chat

    noon = _dt.datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    today = noon.timestamp()
    chat_a = _add("hold target", today - 60)
    chat_b = _add("other row", today - 120)
    context, page, frames = _open_page(browser, harness)
    try:
        # Capture INBOUND WebSocket frames over CDP so the test can
        # prove the changed-data reply arrived while the mouse was still
        # held (a fixed sleep could silently miss the race).
        received: list[str] = []
        cdp = context.new_cdp_session(page)
        cdp.on(
            "Network.webSocketFrameReceived",
            lambda ev: received.append(
                ev.get("response", {}).get("payloadData", ""),
            ),
        )
        cdp.send("Network.enable")
        page.click("#activity-tasks")
        page.wait_for_selector("#history-list .history-chat-group", timeout=15000)
        # Open the collapsed chat panel so its row can be pressed; the
        # explicit expand survives the parked rebuild below.
        page.locator(
            f".history-chat-group[data-chat-id='{chat_a}'] .history-chat-header"
        ).click()
        row = page.locator(".sidebar-item", has_text="hold target")
        row.scroll_into_view_if_needed()
        box = row.bounding_box()
        assert box is not None
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.down()
        # A real refetch with DIFFERENT data (the search narrows the
        # page to one row) arrives while the button is held: the
        # destructive rebuild must be parked, keeping both rows and the
        # pressed node alive.
        received.clear()
        page.evaluate(
            """() => {
                 const s = document.getElementById('history-search');
                 s.value = 'hold';
                 s.dispatchEvent(new Event('input', {bubbles: true}));
               }"""
        )
        for _ in range(100):
            if any(
                '"history"' in frame and "other row" not in frame
                for frame in received
                if '"sessions"' in frame
            ):
                break
            page.wait_for_timeout(50)
        else:
            raise AssertionError("filtered history reply never arrived")
        # The CDP event proves network receipt; the page's message task
        # is queued behind it.  Drain the renderer's task queue (each
        # evaluate is a full round-trip through the page's event loop)
        # plus an idle beat, so renderHistory has DEMONSTRABLY processed
        # the reply while the mouse is still held ...
        page.evaluate("0")
        page.evaluate("0")
        page.wait_for_timeout(300)
        # ... and the changed page was parked, keeping both rows AND the
        # pressed row itself alive.
        assert page.locator("#history-list .sidebar-item").count() == 2
        assert row.count() == 1
        page.mouse.up()
        # The browser-synthesized click still opens the pressed task.
        page.wait_for_function(
            "document.getElementById('task-panel-text')"
            " && document.getElementById('task-panel-text').textContent"
            "      === 'hold target'",
            timeout=15000,
        )
        # The parked filtered page lands right after the click, and the
        # surviving row is the filtered one.
        page.wait_for_function(
            "() => { const rows = document.querySelectorAll("
            "'#history-list .sidebar-item');"
            " return rows.length === 1"
            " && rows[0].textContent.includes('hold target'); }",
            timeout=15000,
        )
    finally:
        context.close()
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "DELETE FROM task_history WHERE chat_id IN (?, ?)",
                (chat_a, chat_b),
            )
