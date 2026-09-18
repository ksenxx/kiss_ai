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
import re

import pytest
from playwright.sync_api import sync_playwright

from kiss.tests.agents.vscode.test_activity_bar import _explorer_row, _sent
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


def _open_page(browser, harness):
    """Open the remote page in desktop mode with clipboard access and
    dialogs auto-answered; record sent WS frames and dialog messages."""
    context = browser.new_context(
        ignore_https_errors=True,
        viewport={"width": 1400, "height": 900},
        permissions=["clipboard-read", "clipboard-write"],
    )
    page = context.new_page()
    sent_frames: list[dict] = []
    dialogs: list[dict] = []
    answers: dict[str, object] = {"prompt": None, "confirm": True}

    def _on_ws(ws) -> None:
        def _on_sent(payload) -> None:
            try:
                sent_frames.append(json.loads(payload))
            except Exception:
                pass

        ws.on("framesent", _on_sent)

    def _on_dialog(dialog) -> None:
        dialogs.append({"type": dialog.type, "message": dialog.message})
        if dialog.type == "prompt":
            answer = answers["prompt"]
            if answer is None:
                dialog.dismiss()
            else:
                dialog.accept(str(answer))
        elif dialog.type == "confirm":
            if answers["confirm"]:
                dialog.accept()
            else:
                dialog.dismiss()
        else:
            dialog.accept()

    page.on("websocket", _on_ws)
    page.on("dialog", _on_dialog)
    page.goto(harness.base_url + "/")
    page.wait_for_selector("#task-input", state="visible", timeout=30000)
    page.wait_for_selector("body.remote-desktop", state="attached")
    page.wait_for_function(
        "document.getElementById('meta-workdir').textContent.length > 1",
        timeout=30000,
    )
    return context, page, sent_frames, dialogs, answers


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


def _wait_dialog(page, dialogs: list[dict], before: int) -> None:
    """Wait until a browser dialog beyond the first *before* was seen."""
    for _ in range(100):
        if len(dialogs) > before:
            return
        page.wait_for_timeout(50)
    raise AssertionError("no dialog appeared")


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
    context, page, frames, dialogs, answers = _open_page(browser, harness)
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
            f"window.__clip === {json.dumps(str(harness.work_dir) + '/feature.txt')}",
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
    context, page, frames, dialogs, answers = _open_page(browser, harness)
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
    context, page, frames, dialogs, answers = _open_page(browser, harness)
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
            ".explorer-row[data-explorer-path$='/dir/fresh.py']", timeout=15000,
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
            ".explorer-row.is-dir[data-explorer-path$='/dir/made']", timeout=15000,
        )
        assert (harness.work_dir / "dir" / "made").is_dir()

        # Rename... pre-selects the stem and renames on Enter; the
        # editor tab opened for the file follows the new name.
        _explorer_row(page, "fresh.py").click(button="right")
        _menu_item(page, "Rename...").click()
        inp = page.locator(".explorer-row.is-editing .explorer-input")
        inp.wait_for(timeout=5000)
        assert inp.input_value() == "fresh.py"
        sel = inp.evaluate("el => [el.selectionStart, el.selectionEnd]")
        assert sel == [0, 5]
        inp.fill("renamed.py")
        inp.press("Enter")
        page.wait_for_selector(
            ".explorer-row[data-explorer-path$='/dir/renamed.py']", timeout=15000,
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

        # Delete asks first; a dismissed dialog keeps the file.
        answers["confirm"] = False
        _explorer_row(page, "renamed.py").click(button="right")
        n_dialogs = len(dialogs)
        _menu_item(page, "Delete").click()
        _wait_dialog(page, dialogs, n_dialogs)
        assert dialogs[-1]["type"] == "confirm"
        assert "renamed.py" in dialogs[-1]["message"]
        assert (harness.work_dir / "dir" / "renamed.py").is_file()
        answers["confirm"] = True
        _explorer_row(page, "renamed.py").click(button="right")
        _menu_item(page, "Delete").click()
        page.wait_for_selector(
            ".explorer-row[data-explorer-path$='/dir/renamed.py']",
            state="detached",
            timeout=15000,
        )
        assert not (harness.work_dir / "dir" / "renamed.py").exists()
        # The editor tab of the deleted file closed with it.
        _wait_tab_count(page, tabs_before)
        _explorer_row(page, "made").click(button="right")
        _menu_item(page, "Delete").click()
        page.wait_for_selector(
            ".explorer-row[data-explorer-path$='/dir/made']",
            state="detached",
            timeout=15000,
        )
    finally:
        context.close()


def test_copy_paste_cut_and_conflict_prompt(browser, harness, worktree):
    context, page, frames, dialogs, answers = _open_page(browser, harness)
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
            ".explorer-row[data-explorer-path$='/main-only copy.txt']", timeout=15000,
        )
        assert (harness.work_dir / "main-only copy.txt").read_text() == "m\n"
        # Paste into dir/, then again: the second paste collides and the
        # replace confirmation is dismissed -> nothing changes.
        _explorer_row(page, "dir").click(button="right")
        _menu_item(page, "Paste").click()
        page.wait_for_selector(
            ".explorer-row[data-explorer-path$='/dir/main-only.txt']", timeout=15000,
        )
        (harness.work_dir / "dir" / "main-only.txt").write_text("keep\n")
        answers["confirm"] = False
        n_dialogs = len(dialogs)
        _explorer_row(page, "dir").click(button="right")
        _menu_item(page, "Paste").click()
        _wait_dialog(page, dialogs, n_dialogs)
        assert dialogs[-1]["type"] == "confirm"
        assert "already exists" in dialogs[-1]["message"]
        assert (harness.work_dir / "dir" / "main-only.txt").read_text() == "keep\n"
        # Accepting replaces it.
        answers["confirm"] = True
        _explorer_row(page, "dir").click(button="right")
        _menu_item(page, "Paste").click()
        for _ in range(50):
            if (harness.work_dir / "dir" / "main-only.txt").read_text() == "m\n":
                break
            page.wait_for_timeout(100)
        assert (harness.work_dir / "dir" / "main-only.txt").read_text() == "m\n"
        # Cut + Paste moves.
        page.locator(".explorer-row[data-explorer-path$='/main-only copy.txt']").click(
            button="right",
        )
        _menu_item(page, "Cut").click()
        _explorer_row(page, "dir").click(button="right")
        _menu_item(page, "Paste").click()
        page.wait_for_selector(
            ".explorer-row[data-explorer-path$='/dir/main-only copy.txt']", timeout=15000,
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
    context, page, frames, dialogs, answers = _open_page(browser, harness)
    try:
        _open_explorer(page)
        tabs_before = page.locator(".chat-tab").count()
        answers["prompt"] = "sentinel"
        page.locator(".explorer-row[aria-level='1']").click(button="right")
        _menu_item(page, "Find in Folder...").click()
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
    context, page, frames, dialogs, answers = _open_page(browser, harness)
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
        assert opened and opened[0]["path"].endswith("/feature.txt")
    finally:
        context.close()


def test_source_control_lists_every_worktree(browser, harness, worktree):
    context, page, frames, dialogs, answers = _open_page(browser, harness)
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
            f"#scm-changes .scm-row[data-scm-worktree='{worktree}']",
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
    context, page, frames, dialogs, answers = _open_page(browser, harness)
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
    context, page, frames, dialogs, answers = _open_page(browser, harness)
    try:
        _open_scm(page)
        first = page.locator("#scm-graph .scm-commit", has_text="first: add a.txt")
        # Create Tag... (name, then optional message) -> the tag shows
        # up on the commit after the refresh.
        answers["prompt"] = "ui-tag"
        first.click(button="right")
        _menu_item(page, "Create Tag...").click()
        page.wait_for_selector(
            "#scm-graph .scm-commit .scm-ref.is-tag:text-is('ui-tag')", timeout=15000,
        )
        actions = _sent(frames, "gitAction")
        assert actions[-1]["action"] == "createTag"
        assert actions[-1]["name"] == "ui-tag" and actions[-1]["message"] == "ui-tag"
        # A dismissed prompt sends nothing.
        answers["prompt"] = None
        first.click(button="right")
        _menu_item(page, "Create Branch...").click()
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
        answers["prompt"] = "v1"
        first.click(button="right")
        _menu_item(page, "Compare with...").click()
        _wait_tab_count(page, tabs_before + 1)
        _wait_content(page, "feature-file-sentinel-7c1e")
        titles = page.eval_on_selector_all(
            ".chat-tab", "els => els.map(e => e.textContent)",
        )
        assert any("v1 \u2194 " + harness.shas["first"][:7] in t for t in titles)
    finally:
        context.close()


def test_folder_picker_changes_the_workspace(browser, harness, worktree):
    context, page, frames, dialogs, answers = _open_page(browser, harness)
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
        assert inp.input_value().endswith("/dir")
        dir_item.dblclick()
        page.wait_for_function(
            "document.querySelector('.folder-picker-input').value.endsWith('/dir')",
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
            "document.querySelector('.folder-picker-input').value.endsWith('/repo')",
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
            ".explorer-row[data-explorer-path$='/plain/only.txt']", timeout=15000,
        )
        # The daemon was told: setWorkDir + saved config; the settings
        # box follows; the Source Control view reports no repository.
        assert _sent(frames, "setWorkDir")[-1]["workDir"] == str(harness.plain_dir)
        saved = _sent(frames, "saveConfig")
        assert saved and saved[-1]["config"]["work_dir"] == str(harness.plain_dir)
        assert page.locator("#cfg-work-dir").input_value() == str(harness.plain_dir)
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
        assert page.locator("#cfg-work-dir").input_value() == str(harness.plain_dir)
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
        assert _sent(frames, "saveConfig")[-1]["config"]["work_dir"] == str(harness.work_dir)
    finally:
        context.close()


def test_pdf_click_opens_a_viewer_tab(browser, harness, worktree):
    context, page, frames, dialogs, answers = _open_page(browser, harness)
    try:
        _open_explorer(page)
        tabs_before = page.locator(".chat-tab").count()
        _explorer_row(page, "report.pdf").click()
        _wait_tab_count(page, tabs_before + 1)
        frame = page.locator(".content-tab-view iframe.content-pdf-frame")
        frame.wait_for(timeout=15000)
        src = frame.get_attribute("src") or ""
        assert src.startswith("blob:")
        assert frame.get_attribute("sandbox") is None
        assert frame.get_attribute("title") == "report.pdf"
        # No error toast: the reply carried the bytes, not an error.
        assert "Cannot display binary file" not in page.locator("body").inner_text()
        # The bytes the frame shows are the file's.
        size = page.evaluate(
            "src => fetch(src).then(r => r.blob()).then(b => [b.size, b.type])", src,
        )
        assert size == [len((harness.work_dir / "report.pdf").read_bytes()), "application/pdf"]
        # An image opens as a picture.
        _explorer_row(page, "dot.png").click()
        _wait_tab_count(page, tabs_before + 2)
        img = page.locator(".content-tab-view img.content-image")
        img.wait_for(timeout=15000)
        assert (img.get_attribute("src") or "").startswith("blob:")
        # Closing the PDF tab releases its blob URL.
        page.locator(".chat-tab", has_text="report.pdf").first.click()
        page.locator(".chat-tab.active .chat-tab-close").click()
        _wait_tab_count(page, tabs_before + 1)
        revoked = page.evaluate(
            "src => fetch(src).then(() => false).catch(() => true)", src,
        )
        assert revoked is True
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
    context, page, frames, dialogs, answers = _open_page(browser, harness)
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
        saves_before = len(_sent(frames, "saveConfig"))
        page.click("#folder-picker .folder-picker-select")
        page.wait_for_selector("#folder-picker", state="hidden")
        page.wait_for_function(
            "document.querySelectorAll('#explorer-tree > .explorer-row.is-root').length === 2",
            timeout=15000,
        )
        assert _root_paths(page) == [repo, plain]
        # Listed from disk, confined to itself; the work dir is untouched.
        only = page.locator(f".explorer-row[data-explorer-path='{plain}/only.txt']")
        only.wait_for(timeout=15000)
        assert only.get_attribute("data-explorer-root") == plain
        listed = [f for f in _sent(frames, "listDir") if f.get("path") == plain]
        assert listed and listed[-1]["workDir"] == plain
        assert len(_sent(frames, "saveConfig")) == saves_before
        assert page.evaluate("JSON.parse(localStorage.getItem('kiss-explorer-roots'))") == [plain]
        # A file of the added folder opens as a content tab.
        tabs_before = page.locator(".chat-tab").count()
        only.click()
        _wait_tab_count(page, tabs_before + 1)
        assert "only" in page.locator(".content-tab-view").last.inner_text()
        # New File... on the added folder lands on disk inside it (the
        # daemon accepted the folder as the action's workDir).
        plain_root = page.locator(f".explorer-row.is-root[data-explorer-path='{plain}']")
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
            f".explorer-row[data-explorer-path='{plain}/added.txt']", timeout=15000,
        )
        assert (harness.plain_dir / "added.txt").is_file()
        (harness.plain_dir / "added.txt").unlink()
        # Set as Working Directory (the buttons show on hover): the daemon
        # and the saved config follow, the old working directory stays
        # listed as an added folder.
        plain_root.hover()
        page.locator(".explorer-root-set").first.click()
        _wait_first_root(page, plain)
        assert _root_paths(page) == [plain, repo]
        assert _sent(frames, "setWorkDir")[-1]["workDir"] == plain
        assert _sent(frames, "saveConfig")[-1]["config"]["work_dir"] == plain
        assert page.locator("#cfg-work-dir").input_value() == plain
        assert page.locator(".explorer-row.is-workdir").get_attribute("data-explorer-path") == plain
        page.wait_for_selector(
            f".explorer-row[data-explorer-path='{plain}/only.txt']", timeout=15000,
        )
        # Switch back through the repo row's context menu.
        page.locator(f".explorer-row.is-root[data-explorer-path='{repo}']").click(button="right")
        _menu_item(page, "Set as Working Directory").click()
        _wait_first_root(page, repo)
        assert _root_paths(page) == [repo, plain]
        assert _sent(frames, "saveConfig")[-1]["config"]["work_dir"] == repo
        # Remove the plain folder with its button: gone from the tree and
        # from storage, still on disk.
        fs_before = len(_sent(frames, "fsAction"))
        page.locator(f".explorer-row.is-root[data-explorer-path='{plain}']").hover()
        page.locator(".explorer-root-remove").first.click()
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
    context, page, frames, dialogs, answers = _open_page(browser, harness)
    try:
        page.click("#activity-tasks")
        page.wait_for_selector("#history-list .history-chat-group", timeout=15000)
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
        # the header shows the chat's FIRST task and opens the panel.
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
    context, page, frames, dialogs, answers = _open_page(browser, harness)
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
