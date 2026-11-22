# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests of the Projects page in a real browser (``media/projects.js``).

* A new project that is saved, or cancelled, closes the Projects tab and
  takes the content window off the screen; the chat header then offers the
  saved project after ``Default``, the choice of a chat in no project.
* The pin of a project's panel looks like the pin of a project's page in
  Claude Code Desktop: a 28px ghost icon button with an 8px corner, no
  border and no fill until the pointer is over it, holding a 20px pushpin
  in the text's colour, an outline while the project is not pinned and
  solid while it is, named "Pin project" / "Unpin project".  On a phone
  the button answers a finger 4px outside its drawing (a 36px target).
* The form of a new project takes no edit while its create is on its way,
  and a second Save creates nothing.  When the connection drops before the
  daemon's answer, the form says so and takes edits again; the create is
  not sent a second time.
* Save stores what the user set: a setting left alone that another
  connection changed meanwhile keeps following the product's settings.

Real ``RemoteAccessServer`` + headless Chromium; the projects are stored
by the daemon.  ``test/projectsPage.test.js`` covers the page's logic.
"""

from __future__ import annotations

import functools
import json
import sys
import time

from kiss.core.config import DEFAULT_CONFIG
from kiss.server.commands import _PROJECTS_PUBLISH_LOCK, _CommandsMixin
from kiss.tests.conftest import goto_retrying_network_change, mark_version_seen
from kiss.tests.server.test_explorer_scm_commands import (
    _request,
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.webapp.test_phone_machine_dropdown import _IPHONE_UA
from kiss.tests.webapp.test_workspace_sections import (
    browser,  # noqa: F401  (module fixture used by param name)
)

_NEW = ".projects-new-slot .project-panel"
_NO_CONTENT_TAB_JS = "() => !window._testApi.openTabs().some(t => t.isContentTab)"
_PANE_OPEN_JS = "() => document.body.classList.contains('content-pane-open')"
_HEADER_PROJECTS_JS = """() => ({
  hidden: document.getElementById('chat-project-wrap').hidden,
  options: Array.from(document.getElementById('chat-project').options, o => o.textContent),
  label: document.querySelector('#chat-project-wrap .chat-header-select-label').textContent,
})"""
_PANEL_ID_JS = """(name) => Array.from(
  document.querySelectorAll('.projects-list .project-panel')
).find(p => p.querySelector('.project-name').value === name).dataset.projectId"""
# Whether the page lists a project named *name*.
_LISTED_JS = """(name) => Array.from(
  document.querySelectorAll('.projects-list .project-name')
).some(input => input.value === name)"""
# What a panel's form holds: its fields (the keys, which are secrets, left
# out), its pin and how many of its key groups are unfolded.
_FORM_JS = """(panel) => ({
  fields: Array.from(
    panel.querySelectorAll('input:not([data-key]), textarea, select'),
    field => (field.type === 'checkbox' ? field.checked : field.value),
  ),
  pinned: panel.querySelector('.project-pin-btn').getAttribute('aria-pressed'),
  unfolded: panel.querySelectorAll('.config-subpanel-body:not([hidden])').length,
})"""
# The class of every control of a panel the browser lets the user work.
_ENABLED_JS = """(panel) => Array.from(
  panel.querySelectorAll('input, textarea, select, button')
).filter(control => !control.disabled).map(control => control.className)"""
# What the page says in a panel whose save was out when the connection dropped.
_UNCONFIRMED = (
    "The server connection dropped before this save was confirmed. "
    "Once it is back, save again if the project is not listed as you saved it."
)
# The colour a token paints, read off an element that wears it.
_TOKEN_JS = """(token) => {
  const probe = document.body.appendChild(document.createElement('i'));
  probe.style.color = 'var(' + token + ')';
  const colour = getComputedStyle(probe).color;
  probe.remove();
  return colour;
}"""
_PIN_LOOK_JS = """(pin) => {
  const style = getComputedStyle(pin);
  const box = pin.getBoundingClientRect();
  const icon = pin.querySelector('svg').getBoundingClientRect();
  const name = pin.parentElement.querySelector('.project-name').getBoundingClientRect();
  return {
    size: [box.width, box.height],
    icon: [icon.width, icon.height],
    radius: style.borderRadius,
    border: style.borderTopWidth,
    fill: style.backgroundColor,
    held: style.backgroundImage !== 'none',
    ink: style.color,
    shown: Array.from(pin.querySelectorAll('path'))
      .filter(p => getComputedStyle(p).display !== 'none')
      .map(p => p.getAttribute('class')),
    names: [pin.getAttribute('aria-label'), pin.title],
    clearOfName: box.left >= name.right,
  };
}"""
# Whether a touch 3px outside the pin's top left corner lands on the pin.
_NEAR_MISS_JS = """(pin) => {
  const box = pin.getBoundingClientRect();
  return document.elementFromPoint(box.left - 3, box.top - 3) === pin;
}"""


def _note_projects_answer(answers, payload):
    """Keep in *answers* the frame *payload* of the daemon that answers a page's ``getProjects``.

    That answer alone names the chats' projects; the ``projects`` event of a change does not.
    """
    try:
        frame = json.loads(payload)
    except ValueError:
        return
    if isinstance(frame, dict) and frame.get("type") == "projects" and "chatProjects" in frame:
        answers.append(frame)


def _watch_socket(answers, socket):
    """Note in *answers* the daemon's answers to ``getProjects`` over *socket*, a page's."""
    socket.on("framereceived", functools.partial(_note_projects_answer, answers))


def _open_page(browser, harness, **context_options):  # noqa: F811
    """A page of the daemon that has the answers to what it asks on its own; ``(context, page)``.

    A page that connects sends ``ready`` and then ``getProjects``, and is on
    screen before the daemon, which handles the commands of a connection one
    after another, has got to them.  A test that went on from there would be
    ahead of the page's own traffic: ``getProjects`` takes the lock a project
    is stored under, so with a test holding that lock to keep a create on its
    way it would wait for the lock, and the page's ``saveProject`` behind it
    would not reach the daemon.
    """
    context = browser.new_context(ignore_https_errors=True, **context_options)
    mark_version_seen(context)
    # A phone's one-time swipe hint, a toast over the sidebar, was read before.
    context.add_init_script("localStorage.setItem('kiss-swipe-hint-dismissed', '1')")
    page = context.new_page()
    answers: list[dict] = []
    page.on("websocket", functools.partial(_watch_socket, answers))
    goto_retrying_network_change(page, harness.base_url + "/")
    page.wait_for_selector("#task-input", state="visible", timeout=30000)
    page.wait_for_function("() => !!(window._testApi && window._testApi.getActiveTabId())")
    deadline = time.monotonic() + 30
    while not answers:
        assert time.monotonic() < deadline, "the daemon did not answer the page's getProjects"
        page.wait_for_timeout(25)
    return context, page


def _wait_for_connect_focus(page):
    """Block until the composer has the keyboard this connection gives it, retries included.

    The daemon answers ``ready`` with ``focusInput``: the page puts the
    keyboard in the composer at once and again 100 and 300 ms later, unless a
    content view has it.  A locked form has not (the browser takes the focus
    off a button that is disabled), so a key typed at it right after any of
    the three would land in the composer.  A 350 ms timer the page sets once
    the composer has the focus ends after the two retries.
    """
    page.wait_for_function("() => document.activeElement === document.getElementById('task-input')")
    page.evaluate("() => new Promise(resolve => setTimeout(resolve, 350))")


def _open_projects(page):
    """Press the sidebar's Projects row; a phone's sidebar, a swipe away, is slid in first."""
    row = page.locator("#projects-row")
    if not row.is_visible():
        page.evaluate("() => document.getElementById('sidebar').classList.add('open')")
    row.click()
    page.wait_for_selector(".projects-tab-view .projects-new-btn", state="visible")


def _save_new_project(page, name):
    """Create project *name* through the New project panel; the page closes on the answer."""
    _open_projects(page)
    page.click(".projects-new-btn")
    page.fill(f"{_NEW} .project-name", name)
    page.click(f"{_NEW} .project-save")
    page.wait_for_function(_NO_CONTENT_TAB_JS)


def _stored_projects(harness):  # noqa: F811
    """Every project the daemon stores, read over a connection of its own."""
    return _request(harness, {"type": "getProjects"}, "projects")["projects"]


def _save_settings(harness, **config):  # noqa: F811
    """Change the product's settings over a connection of its own, as another page does."""
    _request(harness, {"type": "saveConfig", "config": config}, "configData")


def _wait_for_create_at_the_daemon():
    """Block until a daemon thread is in ``saveProject``, waiting for the lock the test holds."""
    code = _CommandsMixin._cmd_save_project.__code__
    deadline = time.monotonic() + 30
    while not any(frame.f_code is code for frame in sys._current_frames().values()):
        assert time.monotonic() < deadline, "the create never reached the daemon"
        time.sleep(0.01)


async def _drop_page_sockets(harness):  # noqa: F811
    """Close the daemon's end of every page's socket, as a lost link does; the page reconnects."""
    for socket in list(harness.server._printer._remote_clients):
        await socket.close()


def test_new_project_saved_or_cancelled_closes_tab_and_content_window(
    browser, harness,  # noqa: F811
):
    """Cancel and Save of a new project both leave the chat alone on screen."""
    context, page = _open_page(browser, harness, viewport={"width": 1400, "height": 900})
    try:
        assert page.evaluate("() => document.body.classList.contains('remote-desktop')")
        before = page.evaluate(_HEADER_PROJECTS_JS)["options"]

        _open_projects(page)
        assert page.evaluate(_PANE_OPEN_JS), "the page opens in the content window"
        page.click(".projects-new-btn")
        page.fill(f"{_NEW} .project-name", "Never saved")
        page.click(f"{_NEW} .project-cancel")
        page.wait_for_function(_NO_CONTENT_TAB_JS)
        assert not page.evaluate(_PANE_OPEN_JS), "Cancel takes the content window away"
        assert page.evaluate(_HEADER_PROJECTS_JS)["options"] == before, "nothing was stored"

        _save_new_project(page, "Closes on save")
        assert not page.evaluate(_PANE_OPEN_JS), "Save takes the content window away"
        page.wait_for_function(
            "() => Array.from(document.getElementById('chat-project').options,"
            " o => o.textContent).includes('Closes on save')"
        )
        header = page.evaluate(_HEADER_PROJECTS_JS)
        assert not header["hidden"]
        assert header["options"][0] == "Default", "a chat in no project is in Default"
        assert header["label"] == "Default"
        assert "No project" not in header["options"]
    finally:
        context.close()


def test_project_pin_looks_like_claude_code_desktops(browser, harness):  # noqa: F811
    """The pin's rest, hover, held and pinned looks, and its names."""
    context, page = _open_page(browser, harness, viewport={"width": 1400, "height": 900})
    try:
        _save_new_project(page, "Pinned look")
        _open_projects(page)
        page.wait_for_selector(".projects-list .project-panel")
        panel = f'.project-panel[data-project-id="{page.evaluate(_PANEL_ID_JS, "Pinned look")}"]'
        pin = page.locator(f"{panel} .project-pin-btn")
        pin.scroll_into_view_if_needed()
        text_colour = page.evaluate(_TOKEN_JS, "--fg")
        hover_fill = page.evaluate(_TOKEN_JS, "--hover")

        assert pin.evaluate(_PIN_LOOK_JS) == {
            "size": [28, 28], "icon": [20, 20], "radius": "8px", "border": "0px",
            "fill": "rgba(0, 0, 0, 0)", "held": False, "ink": text_colour,
            "shown": ["project-pin-outline"], "names": ["Pin project", "Pin project"],
            "clearOfName": True,
        }
        pin.hover()
        page.wait_for_function(
            "([pin, fill]) => getComputedStyle(pin).backgroundColor === fill",
            arg=[pin.element_handle(), hover_fill],
        )
        page.mouse.down()
        assert pin.evaluate(_PIN_LOOK_JS)["held"], "the fill is laid on twice while held"
        page.mouse.up()
        # The press is stored by the daemon, whose answer paints the pin.
        page.wait_for_selector(f'{panel} .project-pin-btn[aria-pressed="true"]')
        pinned = pin.evaluate(_PIN_LOOK_JS)
        assert pinned["shown"] == ["project-pin-solid"]
        assert pinned["names"] == ["Unpin project", "Unpin project"]
        assert pinned["size"] == [28, 28] and pinned["ink"] == text_colour
        assert not pin.evaluate(_NEAR_MISS_JS), "a pointer must be on the button"
    finally:
        context.close()


def test_project_pin_has_a_finger_sized_target_on_a_phone(browser, harness):  # noqa: F811
    """On a phone a touch just outside the 28px drawing still presses the pin."""
    context, page = _open_page(
        browser, harness, viewport={"width": 393, "height": 852}, device_scale_factor=3,
        is_mobile=True, has_touch=True, user_agent=_IPHONE_UA,
    )
    try:
        _open_projects(page)
        page.click(".projects-new-btn")
        pin = page.locator(f"{_NEW} .project-pin-btn")
        pin.scroll_into_view_if_needed()
        assert pin.evaluate(_PIN_LOOK_JS)["size"] == [28, 28], "the drawing keeps its size"
        assert pin.evaluate(_NEAR_MISS_JS)
    finally:
        context.close()


def test_new_project_form_takes_no_edit_while_its_create_is_on_its_way(
    browser, harness,  # noqa: F811
):
    """A mouse and a keyboard change nothing on the form of a pending create; one project is made.

    The create is held at the daemon by the lock it stores a project
    under, so the form is pending for as long as the test needs.
    """
    context, page = _open_page(browser, harness, viewport={"width": 1400, "height": 900})
    try:
        _wait_for_connect_focus(page)
        _open_projects(page)
        page.click(".projects-new-btn")
        page.fill(f"{_NEW} .project-name", "Held create")
        page.fill(f"{_NEW} .project-description", "As sent")
        panel = page.locator(_NEW)
        # The daemon stores a project under this lock: while the test holds it,
        # the create stays on its way.
        with _PROJECTS_PUBLISH_LOCK:
            page.click(f"{_NEW} .project-save")
            page.wait_for_selector(f"{_NEW} .project-save:disabled")
            sent = panel.evaluate(_FORM_JS)
            assert panel.evaluate(_ENABLED_JS) == ["config-update-btn project-cancel"]
            # Every kind of control is clicked, typed into or picked from, Save too.
            for selector in (".project-name", ".project-description", ".project-max-budget"):
                page.click(f"{_NEW} {selector}", force=True)
                page.keyboard.type("9")
            page.click(f"{_NEW} .project-workdir", force=True)
            page.keyboard.press("ArrowDown")
            page.keyboard.press("Enter")
            for selector in (
                'input[data-setting="use_memory"]', ".project-pin-btn",
                ".config-subpanel-toggle >> nth=0", ".project-new-folder", ".project-save",
            ):
                page.click(f"{_NEW} {selector}", force=True)
            assert panel.evaluate(_FORM_JS) == sent, "the form is as it was sent"
            assert page.locator("#folder-picker:not([hidden])").count() == 0
            assert page.input_value("#task-input") == "", "what was typed went nowhere else"
        # The daemon's answer closes the page: one project, as the form was sent.
        page.wait_for_function(_NO_CONTENT_TAB_JS)
        made = [p for p in _stored_projects(harness) if p["name"].startswith("Held create")]
        assert [(p["name"], p["description"], p["settings"], p["isPinned"]) for p in made] == [
            ("Held create", "As sent", {}, False)
        ]
    finally:
        context.close()


def test_new_project_form_is_released_when_the_connection_drops_before_the_answer(
    browser, harness,  # noqa: F811
):
    """A create whose answer is lost with the connection gives the form back and is not re-sent."""
    context, page = _open_page(browser, harness, viewport={"width": 1400, "height": 900})
    try:
        _open_projects(page)
        page.click(".projects-new-btn")
        page.fill(f"{_NEW} .project-name", "Answer lost")
        with _PROJECTS_PUBLISH_LOCK:
            page.click(f"{_NEW} .project-save")
            page.wait_for_selector(f"{_NEW} .project-save:disabled")
            _wait_for_create_at_the_daemon()
            harness.run(_drop_page_sockets(harness))
            # No answer can come any more: the form says so and takes edits again.
            page.wait_for_selector(f"{_NEW} .project-save:enabled")
            assert page.locator(f"{_NEW} .project-error").inner_text() == _UNCONFIRMED
            assert page.input_value(f"{_NEW} .project-name") == "Answer lost"
        # The daemon made the project all the same.  The page, connected again,
        # lists it under the form, which the user still has.
        page.wait_for_function(_LISTED_JS, arg="Answer lost")
        assert page.locator(f"{_NEW} .project-error").is_visible()
        page.fill(f"{_NEW} .project-name", "Second try")
        page.click(f"{_NEW} .project-save")
        page.wait_for_function(_NO_CONTENT_TAB_JS)
        names = [p["name"] for p in _stored_projects(harness)]
        assert (names.count("Answer lost"), names.count("Second try")) == (1, 1)
    finally:
        context.close()


def test_a_setting_left_alone_is_not_stored_when_another_connection_changed_it(
    browser, harness, monkeypatch,  # noqa: F811
):
    """Save stores what the user set, not this page's older copy of the product's settings."""
    # The daemon applies saved settings to this process too: undone with the test.
    for name in (
        "max_budget", "claude_code_use_subscription", "codex_use_subscription",
        "tool_output_compaction",
    ):
        monkeypatch.setattr(DEFAULT_CONFIG, name, getattr(DEFAULT_CONFIG, name))
    before = _request(harness, {"type": "getConfig"}, "configData")["config"]
    _save_settings(harness, max_budget=25, compaction=True)
    context, page = _open_page(browser, harness, viewport={"width": 1400, "height": 900})
    try:
        _open_projects(page)
        page.click(".projects-new-btn")
        page.wait_for_function(
            "() => document.querySelector('.projects-new-slot .project-max-budget').value === '25'"
        )
        page.fill(f"{_NEW} .project-name", "Follows the product")
        page.uncheck(f'{_NEW} input[data-setting="use_memory"]')
        # Another connection changes the product's settings; this page is not told.
        _save_settings(harness, max_budget=99, compaction=False)
        assert page.input_value(f"{_NEW} .project-max-budget") == "25"
        page.click(f"{_NEW} .project-save")
        page.wait_for_function(_NO_CONTENT_TAB_JS)
        stored = [
            p["settings"] for p in _stored_projects(harness) if p["name"] == "Follows the product"
        ]
        assert stored == [{"use_memory": False}], "the budget and Compaction follow the product"
    finally:
        _save_settings(harness, max_budget=before["max_budget"], compaction=before["compaction"])
        context.close()
