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
import sys
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar import task_digest
from kiss.agents.sorcar.docker_tools import DockerTools
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.tests.agents.sorcar.test_docker_script_generation import host_bash
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


def test_edits_of_a_crlf_log_are_replayed_as_the_tool_applied_them(tmp_path: Path) -> None:
    # A14-5 (astra): Edit matches LF-only old text against a CRLF file by
    # converting the edit to CRLF; the replay must do the same, else the
    # digest keeps the old status and appends the new one.
    log = tmp_path / "tmp" / "PROGRESS.md"
    tools = UsefulTools(work_dir=str(tmp_path))
    tools.Write(str(log), "# log\r\n- old status\r\n")
    single = tools.Edit(str(log), old_string="- old status\n", new_string="- new status\n")
    multi = json.dumps([
        {"old_string": "- new status\n", "new_string": "- new status\n- step 2\n"},
        {"old_string": "- ", "new_string": "* ", "replace_all": True},
    ])
    batch = tools.Edit(str(log), edits=multi)
    # S11 re-fix round 2: old text of one line is found as it is, and the
    # lines its new text adds take the file's CRLF all the same; those of
    # an ``exact`` edit (a ``sed -i`` applied as an Edit) stay as given.
    # The replay converted neither and showed a bare LF the file never held.
    one_line = {"old_string": "step 2", "new_string": "step 2\n* step 3"}
    as_sed = {"old_string": "step 3", "new_string": "step 3\n* step 4"}
    results = [
        single,
        batch,
        tools.Edit(str(log), one_line["old_string"], one_line["new_string"]),
        tools.Edit(str(log), as_sed["old_string"], as_sed["new_string"], exact=True),
    ]
    assert not any(result.startswith("Error") for result in results)
    assert log.read_bytes() == b"# log\r\n* new status\r\n* step 2\r\n* step 3\n* step 4\r\n"
    events = [
        *_write_call(log, "# log\r\n- old status\r\n"),
        *_edit_call(log, {"old_string": "- old status\n", "new_string": "- new status\n"}, single),
        *_edit_call(log, {"extras": {"edits": multi}}, batch),
        *_edit_call(log, one_line, results[2]),
        *_edit_call(log, {**as_sed, "extras": {"exact": "True"}}, results[3]),
    ]
    task_id = _task(tmp_path, events, time.time() - 600)

    assert _log_of(task_id) == "# log\r\n* new status\r\n* step 2\r\n* step 3\n* step 4"


def test_an_exact_edit_is_not_replayed_on_a_whitespace_similar_block(tmp_path: Path) -> None:
    # An ``exact`` edit matches verbatim only.  Its old text was another
    # task's block, indented unlike the task's own: the replay matched the
    # task's block ignoring whitespace and showed a status the task never
    # wrote there.
    log = tmp_path / "tmp" / "PROGRESS.md"
    tools = UsefulTools(work_dir=str(tmp_path))
    own = "# log\n  - old\n  - next\n"
    tools.Write(str(log), own)
    with log.open("a", encoding="utf-8", newline="") as other_task:
        other_task.write(" - old\n - next\n")
    edit = {"old_string": " - old\n - next\n", "new_string": " - done\n - next\n"}
    result = tools.Edit(str(log), edit["old_string"], edit["new_string"], exact=True)
    assert log.read_text(encoding="utf-8") == own + edit["new_string"], result
    events = [
        *_write_call(log, own),
        *_edit_call(log, {**edit, "extras": {"exact": "True"}}, result),
    ]
    task_id = _task(tmp_path, events, time.time() - 600)

    assert _log_of(task_id) == (own + edit["new_string"]).rstrip()


@pytest.mark.skipif(sys.platform != "linux", reason="the container scripts need GNU coreutils")
def test_a_container_batch_edit_that_stopped_part_way_replays_what_it_applied(
    tmp_path: Path,
) -> None:
    # S11 re-fix 5: the container Edit applies a batch one edit at a time,
    # stops at the first that fails and names the edits already in the file
    # (the host one writes all or nothing); the replay took that answer for
    # a call that wrote nothing and showed the entries the edits replaced.
    log = tmp_path / "tmp" / "PROGRESS.md"
    original = "# log\n- first old\n- second old\n- third old\n"
    _write(log, original)
    tools = DockerTools(host_bash)
    stopped = json.dumps([
        {"old_string": "first old", "new_string": "first done"},
        {"old_string": "second old", "new_string": "second done"},
        {"old_string": "missing", "new_string": "never"},
        {"old_string": "third old", "new_string": "third done"},
    ])
    refused = json.dumps([
        {"old_string": "missing", "new_string": "never"},
        {"old_string": "third old", "new_string": "third done"},
    ])
    stopped_result = tools.Edit(str(log), edits=stopped)
    refused_result = tools.Edit(str(log), edits=refused)
    assert stopped_result.startswith("Error: edit 3: edits 1-2 applied; String not found")
    assert refused_result.startswith("Error: edit 1: String not found")
    applied = "# log\n- first done\n- second done\n- third old\n"
    assert log.read_text(encoding="utf-8") == applied
    # A Write the container could not make (the parent is a file) answered
    # with the failed command's output as it was, which the replay, like
    # every reader of the tools' "Error" convention, took for a success.
    blocker = tmp_path / "blocker"
    _write(blocker, "x")
    unmade = blocker / "PROGRESS.md"
    unmade_result = tools.Write(str(unmade), "never written")
    assert "exit code:" in unmade_result and not unmade.exists()
    events = [
        *_write_call(log, original),
        *_edit_call(log, {"extras": {"edits": stopped}}, stopped_result),
        *_edit_call(log, {"extras": {"edits": refused}}, refused_result),
        {"type": "tool_call", "name": "Write", "path": str(unmade), "content": "never written"},
        {"type": "tool_result", "content": unmade_result},
    ]

    assert _log_of(_task(tmp_path, events, time.time() - 600)) == applied.rstrip()


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
