# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: the task result is stamped into commit messages as Markdown.

``finish()`` returns the result as HTML.  Stamping that HTML verbatim
under ``Result:`` made the block invisible in VS Code's Git hovers
(blame, Timeline, Source Control Graph), which render commit messages
as Markdown with raw HTML disabled and drop every HTML block.  The
result is now converted to Markdown by
:func:`kiss.agents.sorcar.git_worktree.result_to_commit_text`.

A second bug fixed alongside: ``git commit -m`` normalises the stored
message (``--cleanup=whitespace``), so a prompt with a trailing space on
a line never matched its stored copy in
:func:`~kiss.agents.sorcar.git_worktree._ensure_task_metadata` and the
``User prompt:`` block was appended twice at merge time.
"""

import tempfile
from pathlib import Path

from kiss.agents.sorcar.commit_message import (
    _append_task_result,
    _append_user_prompt,
    generate_commit_message_from_diff,
)
from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    MergeResult,
    _ensure_task_metadata,
    git_cleanup_whitespace,
    result_to_commit_text,
)
from kiss.tests.agents.sorcar.test_merge_message_task_and_result import (
    _agent_manual_commit,
    _create_worktree,
    _git,
    _head_message,
    _make_repo,
)

HTML_RESULT = (
    "<h3>Summary</h3><p>Fixed <code>foo_bar()</code> in <b>x.py</b>.</p>"
    "<ul><li>one</li><li>two</li></ul>"
)
MARKDOWN_RESULT = "### Summary\n\nFixed `foo_bar()` in **x.py**.\n\n- one\n- two"
PROMPT_WITH_TRAILING_SPACE = "fix the bug in foo_bar \n\nUse the fast model."


class TestResultToCommitText:
    def test_html_result_becomes_markdown(self) -> None:
        assert result_to_commit_text(HTML_RESULT) == MARKDOWN_RESULT

    def test_plain_text_is_only_stripped(self) -> None:
        assert result_to_commit_text("  done: a < b\n") == "done: a < b"

    def test_empty_and_none_give_empty_string(self) -> None:
        assert result_to_commit_text(None) == ""
        assert result_to_commit_text("  \n ") == ""


class TestGitCleanupWhitespace:
    def test_matches_git_commit_normalisation(self) -> None:
        raw = (
            "\n\nsubject  \n\n\n\n  body line \t\n   \nnbsp\u00a0\nline\u2028sep\n"
            "a\u000bb\u000c\nlast\n\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "f.txt").write_text("x\n")
            _git("add", "f.txt", cwd=repo)
            assert _git("commit", "-m", raw, cwd=repo).returncode == 0
            assert _head_message(repo) == git_cleanup_whitespace(raw)
        assert git_cleanup_whitespace(raw) == (
            "subject\n\n  body line\n\nnbsp\u00a0\nline\u2028sep\na\u000bb\u000c\nlast"
        )


class TestAppendTaskResult:
    def test_html_result_is_appended_as_markdown(self) -> None:
        assert _append_task_result("subject", HTML_RESULT) == (
            f"subject\n\nResult:\n{MARKDOWN_RESULT}"
        )

    def test_fallback_message_without_diff_uses_markdown(self) -> None:
        msg = generate_commit_message_from_diff(
            "", user_prompt="do it", task_result=HTML_RESULT,
        )
        assert msg == (
            "kiss: auto-commit agent work\n\nUser prompt:\ndo it"
            f"\n\nResult:\n{MARKDOWN_RESULT}"
        )
        assert "<" not in msg


class TestEnsureTaskMetadata:
    def test_stamps_markdown_result(self) -> None:
        msg = _ensure_task_metadata("subject", "prompt", HTML_RESULT)
        assert msg == f"subject\n\nUser prompt:\nprompt\n\nResult:\n{MARKDOWN_RESULT}"

    def test_html_result_deduplicates_against_its_markdown_copy(self) -> None:
        stamped = _ensure_task_metadata("subject", "prompt", HTML_RESULT)
        assert _ensure_task_metadata(stamped, "prompt", HTML_RESULT) == stamped

    def test_prompt_with_trailing_whitespace_is_not_stamped_twice(self) -> None:
        stamped = _ensure_task_metadata("subject", PROMPT_WITH_TRAILING_SPACE, "done")
        stored = git_cleanup_whitespace(stamped)
        again = _ensure_task_metadata(stored, PROMPT_WITH_TRAILING_SPACE, "done")
        assert again == stored
        assert again.count("User prompt:") == 1

    def test_prompt_with_leading_indent_is_not_stamped_twice(self) -> None:
        prompt = "  indented prompt\nsecond line"
        stored = git_cleanup_whitespace(_append_task_result(
            _append_user_prompt("subject", prompt), HTML_RESULT,
        ))
        assert _ensure_task_metadata(stored, prompt, HTML_RESULT) == stored

    def test_legacy_html_result_block_is_replaced_by_markdown(self) -> None:
        legacy = f"subject\n\nUser prompt:\ndo it\n\nResult:\n{HTML_RESULT}"
        msg = _ensure_task_metadata(legacy, "do it", HTML_RESULT)
        assert msg == f"subject\n\nUser prompt:\ndo it\n\nResult:\n{MARKDOWN_RESULT}"
        legacy_no_prompt = f"subject\n\nResult:\n{HTML_RESULT}"
        assert _ensure_task_metadata(legacy_no_prompt, None, HTML_RESULT) == (
            f"subject\n\nResult:\n{MARKDOWN_RESULT}"
        )

    def test_prompt_ending_like_a_legacy_result_block_is_kept_intact(self) -> None:
        prompt = "inspect this output:\n\nResult:\n<p>done</p>"
        stored = f"subject\n\nUser prompt:\n{prompt}"
        msg = _ensure_task_metadata(stored, prompt, "<p>done</p>")
        assert msg == f"{stored}\n\nResult:\ndone"
        assert msg.count("User prompt:") == 1

    def test_stored_prompt_only_message_gets_result_appended_once(self) -> None:
        stored = git_cleanup_whitespace(
            _ensure_task_metadata("subject", PROMPT_WITH_TRAILING_SPACE, None)
        )
        msg = _ensure_task_metadata(stored, PROMPT_WITH_TRAILING_SPACE, "done ")
        assert msg == f"{stored}\n\nResult:\ndone"

    def test_result_only_message_gets_prompt_inserted_before_it(self) -> None:
        stored = git_cleanup_whitespace(_ensure_task_metadata("subject", None, HTML_RESULT))
        msg = _ensure_task_metadata(stored, "the prompt", HTML_RESULT)
        assert msg == f"subject\n\nUser prompt:\nthe prompt\n\nResult:\n{MARKDOWN_RESULT}"

    def test_prompt_only_and_result_only_dedup(self) -> None:
        assert _ensure_task_metadata("s\n\nUser prompt:\np", "p ", None) == (
            "s\n\nUser prompt:\np"
        )
        assert _ensure_task_metadata("s", "p", None) == "s\n\nUser prompt:\np"
        assert _ensure_task_metadata("s\n\nResult:\nr", None, "r\n") == "s\n\nResult:\nr"
        assert _ensure_task_metadata("s", None, "r") == "s\n\nResult:\nr"
        assert _ensure_task_metadata("s\n", None, None) == "s"


class TestSquashMergeStampsMarkdownResultOnce:
    """The real branch-HEAD → squash-merge round trip through git."""

    def test_auto_commit_then_merge_has_one_prompt_and_markdown_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            branch = "kiss/wt-markdown-result"
            wt_dir = _create_worktree(repo, branch)
            baseline = GitWorktreeOps.head_sha(wt_dir)
            assert baseline is not None
            _agent_manual_commit(wt_dir, "first.txt")
            (wt_dir / "second.txt").write_text("more\n")
            _git("add", "second.txt", cwd=wt_dir)
            auto_msg = _append_task_result(
                f"feat: second\n\nUser prompt:\n{PROMPT_WITH_TRAILING_SPACE}", HTML_RESULT,
            )
            assert GitWorktreeOps.commit_staged(wt_dir, auto_msg)
            assert "<h3>" not in _head_message(wt_dir)

            result = GitWorktreeOps.squash_merge_from_baseline(
                repo,
                branch,
                baseline,
                user_prompt=PROMPT_WITH_TRAILING_SPACE,
                task_result=HTML_RESULT,
            )
            assert result == MergeResult.SUCCESS
            msg = _head_message(repo)
            assert msg.count("User prompt:") == 1
            assert msg.count("Result:") == 1
            assert msg.endswith(
                f"User prompt:\n{git_cleanup_whitespace(PROMPT_WITH_TRAILING_SPACE)}"
                f"\n\nResult:\n{MARKDOWN_RESULT}"
            )
            assert "<" not in msg
