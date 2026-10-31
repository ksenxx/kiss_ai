# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Review 2, Q2 finding 3: a piece of one write is not ownership of a shared progress log.

The ``/ask`` context showed the whole current file of the task's newest
progress-log write whenever that write's text was still somewhere in it:
text another task appended after it was shown as this task's, and an
``Edit`` whose new text is a common word (``done``) claimed an unrelated
log.  The log is now rebuilt from the task's own successful ``Write`` /
``Edit`` calls, so only text those calls wrote is shown.  Real history
rows and events, real files; nothing is mocked.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from kiss.agents.sorcar import task_digest
from kiss.tests.agents.sorcar.test_task_digest_progress_log_wave3 import (
    _edit_call,
    _log_of,
    _task,
    _write,
    _write_call,
)


def test_text_another_task_appended_after_the_tasks_write_is_not_shown(tmp_path: Path) -> None:
    log = tmp_path / "tmp" / "PROGRESS.md"
    a_text = "# Progress\nA: investigating parser\n"
    task_a = _task(tmp_path, _write_call(log, a_text), time.time() - 600)
    _write(log, a_text + "B: discovered unrelated private deployment details\n")

    assert _log_of(task_a) == a_text.rstrip()
    assert "deployment" not in task_digest.context(task_a)


def test_a_common_word_an_edit_wrote_does_not_claim_another_tasks_log(tmp_path: Path) -> None:
    log = tmp_path / "tmp" / "PROGRESS.md"
    edit = _edit_call(log, {"old_string": "parser", "new_string": "done"}, "edited")
    task_a = _task(tmp_path, edit, time.time() - 600)
    _write(log, "# Task B\nB: analysis done\n")

    assert _log_of(task_a) == "done"
    assert "Task B" not in task_digest.context(task_a)


def test_the_merged_copy_another_task_appended_to_shows_only_the_tasks_text(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    worktree_log = repo / ".kiss-worktrees" / "kiss_wt-abc" / "tmp" / "PROGRESS.md"
    main_log = repo / "tmp" / "PROGRESS.md"
    text = "# worktree task's log\n- step 1\n"
    _write(worktree_log, text)
    task_id = _task(repo, _write_call(worktree_log, text), time.time() - 600)
    shutil.rmtree(repo / ".kiss-worktrees")
    _write(main_log, text + "- another task's private step\n")

    assert _log_of(task_id) == text.rstrip()
    assert "private step" not in task_digest.context(task_id)


def test_the_tasks_own_edits_are_replayed_on_the_text_it_wrote(tmp_path: Path) -> None:
    log = tmp_path / "tmp" / "PROGRESS.md"
    multi = json.dumps([
        {"old_string": "* b\n", "new_string": "* b\n* c\n"},
        {"old_string": "* ", "new_string": "+ ", "replace_all": True},
    ])
    events = [
        *_write_call(log, "# log\n- a\n"),
        *_edit_call(log, {"old_string": "- a\n", "new_string": "- a\n- b\n"}, "ok"),
        *_edit_call(
            log, {"old_string": "- ", "new_string": "* ", "extras": {"replace_all": "True"}}, "ok",
        ),
        *_edit_call(log, {"extras": {"edits": multi}}, "ok"),
        # Old text the task never wrote (another task's): only the new text counts.
        *_edit_call(log, {"old_string": "B's line", "new_string": "- d"}, "ok"),
        *_edit_call(log, {"old_string": "B's other line", "new_string": "- e"}, "ok"),
        # Deleting its own line removes it; deleting another task's text changes nothing.
        *_edit_call(log, {"old_string": "+ c\n", "new_string": ""}, "ok"),
        *_edit_call(log, {"old_string": "B's third line", "new_string": ""}, "ok"),
    ]
    task_id = _task(tmp_path, events, time.time() - 600)

    assert _log_of(task_id) == "# log\n+ a\n+ b\n- d\n- e"


def test_a_task_that_only_appended_to_an_older_log_shows_its_own_edits(tmp_path: Path) -> None:
    log = tmp_path / "tmp" / "PROGRESS.md"
    _write(log, "# an earlier task's log\nolder private notes\n- last step\n")
    events = [
        *_edit_call(log, {"old_string": "- last step\n",
                          "new_string": "- last step\n- A: step 1\n"}, "ok"),
        *_edit_call(log, {"old_string": "- A: step 1\n",
                          "new_string": "- A: step 1\n- A: step 2\n"}, "ok"),
    ]
    task_id = _task(tmp_path, events, time.time() - 600)

    assert _log_of(task_id) == "- last step\n- A: step 1\n- A: step 2"
    assert "older private notes" not in task_digest.context(task_id)


def test_a_log_the_task_blanked_is_not_shown(tmp_path: Path) -> None:
    log = tmp_path / "tmp" / "PROGRESS.md"
    task_id = _task(tmp_path, [*_write_call(log, "# log\n"), *_write_call(log, " \n")],
                    time.time() - 600)

    assert "Progress log" not in task_digest.context(task_id)
