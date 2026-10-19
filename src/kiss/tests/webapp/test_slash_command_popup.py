# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Browser test for the composer's ``/`` command popup.

Against the real :class:`RemoteAccessServer` page in headless
Chromium: slash runs seeded in the task history make the popup list
the three most recently run commands first (``Recent``), the ones run
often next (``Frequent``), then the rest; the popup opens for a slash
typed after other text; accepting an item puts ``/name `` at the
slash and keeps the text before it.
"""

from __future__ import annotations

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar import sea_commands
from kiss.tests.server.test_content_tab_file_links import (
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.webapp.test_content_tab_file_links import _open_page, browser  # noqa: F401

_ITEMS_JS = (
    "() => Array.from(document.querySelectorAll('#autocomplete .ac-item'))"
    ".map(el => el.dataset.text)"
)
_SECTIONS_JS = (
    "() => Array.from(document.querySelectorAll('#autocomplete .ac-section'))"
    ".map(el => el.textContent)"
)


def _run(task: str, ts: float) -> None:
    """Record a finished run of *task* at time *ts* in the task history."""
    task_id, _chat = th._add_task(task, "")
    db = th._get_db()
    with th._rw_lock.write_lock(), th._immediate_txn(db):
        db.execute("UPDATE task_history SET timestamp = ? WHERE id = ?", (ts, task_id))


@pytest.fixture
def seeded_history() -> list[str]:
    """Seed slash runs; return the registered command names."""
    names = sea_commands.list_commands()
    assert {"slack", "gmail", "gcal", "github"} <= set(names)
    _run("/gcal tomorrow", 9e9)
    _run("/slack hello", 8e9)
    _run("/slack again", 1e9)
    _run("/github prs", 7e9)
    _run("/gmail inbox", 6e9)
    _run("/gmail archive", 5e8)
    return names


class TestSlashCommandPopup:
    def test_popup_ranks_recent_then_frequent_and_opens_mid_text(
        self, browser, harness, seeded_history,  # noqa: F811
    ) -> None:
        context, page, sent = _open_page(browser, harness)
        try:
            page.click("#task-input")
            page.keyboard.type("please fix this /")
            page.wait_for_selector("#autocomplete .ac-item", state="visible", timeout=30000)
            # The reply to the popup's own getSeaCommands carries the
            # ranking; wait until the Recent group is painted.
            page.wait_for_function(
                "() => document.querySelector('#autocomplete .ac-section')"
                "?.textContent === 'Recent'",
                timeout=30000,
            )
            assert page.evaluate(_SECTIONS_JS) == ["Recent", "Frequent", "Commands"]
            items = page.evaluate(_ITEMS_JS)
            assert items[:4] == ["gcal", "slack", "github", "gmail"]
            rest = [n for n in seeded_history if n not in items[:4]]
            assert items[4:] == rest
            assert any(f.get("type") == "getSeaCommands" for f in sent)

            page.keyboard.type("gm")
            page.wait_for_function(
                "() => document.querySelectorAll('#autocomplete .ac-item').length === 1",
                timeout=30000,
            )
            page.keyboard.press("Tab")
            assert page.input_value("#task-input") == "please fix this /gmail "
            display_js = "() => document.querySelector('#autocomplete').style.display"
            assert page.evaluate(display_js) == "none"
        finally:
            context.close()
