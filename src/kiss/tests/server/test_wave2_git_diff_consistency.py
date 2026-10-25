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

* The canonical ``User prompt:`` / ``Result:`` commit-message blocks
  are written and recognised by one function,
  ``git_worktree._ensure_task_metadata`` (the auto-commit path in
  ``commit_message.py`` calls it too); the tests pin that stamping is
  idempotent and composes blocks exactly as ``HEADING + text``.
"""

from __future__ import annotations

from kiss.agents.sorcar.commit_message import fallback_commit_message
from kiss.agents.sorcar.git_worktree import (
    TASK_RESULT_HEADING,
    USER_PROMPT_HEADING,
    _ensure_task_metadata,
)


def test_stamped_blocks_detected_as_duplicate_by_dedup() -> None:
    """A message the auto-commit path stamped must not be stamped twice.

    ``_ensure_task_metadata`` must recognise a message carrying the
    current prompt/result blocks (as written by
    ``fallback_commit_message`` / ``generate_commit_message_from_diff``)
    and return it unchanged.
    """
    prompt = "Fix the flaky login test"
    result = "Stabilized the test by awaiting the redirect."
    msg = fallback_commit_message(prompt, result)
    assert _ensure_task_metadata(msg, prompt, result) == msg

    prompt_only = fallback_commit_message(prompt, None)
    assert _ensure_task_metadata(prompt_only, prompt, None) == prompt_only

    result_only = fallback_commit_message(None, result)
    assert _ensure_task_metadata(result_only, None, result) == result_only


def test_stamped_blocks_use_shared_heading_constants() -> None:
    """The blocks are composed exactly as ``HEADING + text``."""
    msg = _ensure_task_metadata("subject", "  a prompt  ", None)
    assert msg == f"subject{USER_PROMPT_HEADING}a prompt"
    msg = _ensure_task_metadata("subject", None, "  a result  ")
    assert msg == f"subject{TASK_RESULT_HEADING}a result"


def test_missing_prompt_block_inserted_before_result_block() -> None:
    """A result-only stamped message gains the prompt block."""
    result = "Done."
    prompt = "Do the thing"
    msg = _ensure_task_metadata("chore: thing", None, result)
    stamped = _ensure_task_metadata(msg, prompt, result)
    assert stamped == (
        f"chore: thing{USER_PROMPT_HEADING}{prompt}{TASK_RESULT_HEADING}{result}"
    )
