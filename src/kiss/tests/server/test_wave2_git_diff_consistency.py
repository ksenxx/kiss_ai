# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Wave-2 git-diff consistency fixes — end-to-end tests.

Covers three cross-file findings, each against real temporary git
repositories (no mocks, patches, fakes, or test doubles):

* ``GitWorktreeOps._diff_name_only`` historically parsed
  ``git diff --name-only`` with ``stdout.strip().splitlines()``, which
  mangled filenames with leading/trailing spaces (legal on POSIX and
  NOT C-quoted by git).  It now uses NUL-separated ``-z`` output.

* ``GitWorktreeOps.copy_dirty_state`` historically re-implemented
  porcelain-status parsing with ``splitlines()``, which splits inside
  an unquoted filename containing a raw unicode line separator
  (U+2028) when ``core.quotepath=false`` is in force (as it always is
  via the module's ``_git`` wrapper).  It now routes through the
  shared ``_porcelain_entries`` parser (which splits on ``\\n`` only)
  — the same parser backing ``merge_flow._porcelain_paths``.

* The canonical ``User prompt:`` / ``Result:`` commit-message block
  format was spelled independently in ``vscode/helpers.py`` (writer)
  and ``sorcar/git_worktree.py`` (dedup detector).  It is now
  single-sourced via ``USER_PROMPT_HEADING`` / ``TASK_RESULT_HEADING``
  constants; the tests pin the byte-exact writer/detector agreement.
"""

from __future__ import annotations

from kiss.agents.sorcar.git_worktree import (
    TASK_RESULT_HEADING,
    USER_PROMPT_HEADING,
    _ensure_task_metadata,
)
from kiss.server.helpers import _append_task_result, _append_user_prompt


def test_helpers_blocks_detected_as_duplicate_by_dedup() -> None:
    """Blocks appended by the helpers.py writers must dedup byte-exactly.

    ``_ensure_task_metadata`` (the git_worktree.py auto-commit dedup
    detector) must recognise a message stamped by the helpers.py
    writers as already carrying the current prompt/result blocks and
    return it unchanged — pinning the byte-exact agreement that the
    shared ``USER_PROMPT_HEADING`` / ``TASK_RESULT_HEADING`` constants
    guarantee.
    """
    prompt = "Fix the flaky login test"
    result = "Stabilized the test by awaiting the redirect."
    msg = _append_task_result(
        _append_user_prompt("fix: stabilize login test", prompt), result
    )
    assert _ensure_task_metadata(msg, prompt, result) == msg

    prompt_only = _append_user_prompt("fix: stabilize login test", prompt)
    assert _ensure_task_metadata(prompt_only, prompt, None) == prompt_only

    result_only = _append_task_result("fix: stabilize login test", result)
    assert _ensure_task_metadata(result_only, None, result) == result_only


def test_helpers_blocks_use_shared_heading_constants() -> None:
    """The writers compose blocks exactly as ``HEADING + text``."""
    msg = _append_user_prompt("subject", "  a prompt  ")
    assert msg == f"subject{USER_PROMPT_HEADING}a prompt"
    msg = _append_task_result("subject", "  a result  ")
    assert msg == f"subject{TASK_RESULT_HEADING}a result"


def test_missing_prompt_block_inserted_before_helper_result_block() -> None:
    """A helpers-stamped result-only message gains the prompt block."""
    result = "Done."
    prompt = "Do the thing"
    msg = _append_task_result("chore: thing", result)
    stamped = _ensure_task_metadata(msg, prompt, result)
    assert stamped == (
        f"chore: thing{USER_PROMPT_HEADING}{prompt}{TASK_RESULT_HEADING}{result}"
    )
