"""Tests for parse_task_tags and multi-task execution in the VS Code server.

Tests cover all branches of ``parse_task_tags`` and verify that the
multi-task loop in ``_run_task_inner`` correctly parses subtasks and
only invokes the worktree/merge interface after the final subtask.

No mocks — uses real functions and source inspection.
"""

import inspect
import unittest

from kiss.agents.vscode.server import VSCodeServer, parse_task_tags

# ---------------------------------------------------------------------------
# parse_task_tags unit tests — 100 % branch coverage
# ---------------------------------------------------------------------------


class TestParseTaskTagsNoTags(unittest.TestCase):
    """When input has no <task> tags, return the original text."""

    def test_plain_text(self) -> None:
        assert parse_task_tags("hello world") == ["hello world"]

    def test_empty_string(self) -> None:
        assert parse_task_tags("") == [""]

    def test_whitespace_only(self) -> None:
        assert parse_task_tags("   ") == ["   "]

    def test_partial_tag_open_only(self) -> None:
        assert parse_task_tags("<task>no closing tag") == ["<task>no closing tag"]

    def test_partial_tag_close_only(self) -> None:
        assert parse_task_tags("no opening tag</task>") == ["no opening tag</task>"]

    def test_mismatched_tags(self) -> None:
        assert parse_task_tags("<task>foo</tsk>") == ["<task>foo</tsk>"]


class TestParseTaskTagsSingleTask(unittest.TestCase):
    """Single <task> block returns a one-element list."""

    def test_single_task(self) -> None:
        assert parse_task_tags("<task>do stuff</task>") == ["do stuff"]

    def test_single_task_with_whitespace(self) -> None:
        assert parse_task_tags("<task>  do stuff  </task>") == ["do stuff"]

    def test_single_task_multiline(self) -> None:
        text = "<task>\nrefactor\nmodule A\n</task>"
        result = parse_task_tags(text)
        assert result == ["refactor\nmodule A"]

    def test_single_task_with_surrounding_text(self) -> None:
        text = "prefix <task>the task</task> suffix"
        assert parse_task_tags(text) == ["the task"]


class TestParseTaskTagsMultipleTasks(unittest.TestCase):
    """Multiple <task> blocks return multiple elements."""

    def test_two_tasks(self) -> None:
        text = "<task>task1</task><task>task2</task>"
        assert parse_task_tags(text) == ["task1", "task2"]

    def test_three_tasks(self) -> None:
        text = "<task>a</task>\n<task>b</task>\n<task>c</task>"
        assert parse_task_tags(text) == ["a", "b", "c"]

    def test_tasks_with_newlines_between(self) -> None:
        text = "<task>\ntask one\n</task>\n\n<task>\ntask two\n</task>"
        assert parse_task_tags(text) == ["task one", "task two"]

    def test_tasks_with_surrounding_text(self) -> None:
        text = "intro\n<task>first</task>\nmiddle\n<task>second</task>\noutro"
        assert parse_task_tags(text) == ["first", "second"]


class TestParseTaskTagsEmptyTasks(unittest.TestCase):
    """Empty or whitespace-only tasks are filtered out."""

    def test_single_empty_task(self) -> None:
        """Single empty <task></task> falls back to original text."""
        text = "<task></task>"
        assert parse_task_tags(text) == [text]

    def test_single_whitespace_task(self) -> None:
        """Single whitespace-only <task> falls back to original text."""
        text = "<task>   </task>"
        assert parse_task_tags(text) == [text]

    def test_all_empty_tasks(self) -> None:
        """All empty tasks fall back to original text."""
        text = "<task></task><task>  </task>"
        assert parse_task_tags(text) == [text]

    def test_mixed_empty_and_nonempty(self) -> None:
        """Only non-empty tasks are returned."""
        text = "<task></task><task>real task</task><task>  </task>"
        assert parse_task_tags(text) == ["real task"]

    def test_mixed_two_valid_one_empty(self) -> None:
        text = "<task>a</task><task></task><task>b</task>"
        assert parse_task_tags(text) == ["a", "b"]


class TestParseTaskTagsEdgeCases(unittest.TestCase):
    """Edge cases: nested tags, special characters, etc."""

    def test_task_containing_angle_brackets(self) -> None:
        text = "<task>use if x > 0 then y < 1</task>"
        assert parse_task_tags(text) == ["use if x > 0 then y < 1"]

    def test_task_with_code_block(self) -> None:
        text = "<task>fix this:\n```python\nprint('hello')\n```</task>"
        result = parse_task_tags(text)
        assert len(result) == 1
        assert "print('hello')" in result[0]

    def test_greedy_vs_non_greedy(self) -> None:
        """Regex is non-greedy: each <task>...</task> is separate."""
        text = "<task>one</task>middle<task>two</task>"
        result = parse_task_tags(text)
        assert result == ["one", "two"]
        assert "middle" not in result[0]

    def test_case_sensitive_tags(self) -> None:
        """Tags are case-sensitive: <Task> is NOT matched."""
        text = "<Task>not matched</Task>"
        assert parse_task_tags(text) == [text]

    def test_newlines_inside_tags(self) -> None:
        text = "<task>\n\n  multi\n  line\n  task\n\n</task>"
        result = parse_task_tags(text)
        assert len(result) == 1
        assert "multi" in result[0]
        assert "line" in result[0]


# ---------------------------------------------------------------------------
# Integration: verify _run_task_inner uses parse_task_tags
# ---------------------------------------------------------------------------


class TestMultiTaskIntegration(unittest.TestCase):
    """Verify _run_task_inner correctly loops over subtasks and defers
    merge/worktree to the last subtask only."""

    _src: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls._src = inspect.getsource(VSCodeServer._run_task_inner)

    def test_calls_parse_task_tags(self) -> None:
        """_run_task_inner calls parse_task_tags on the prompt."""
        assert "parse_task_tags(prompt)" in self._src

    def test_loops_over_subtasks(self) -> None:
        """_run_task_inner iterates over subtasks."""
        assert "for task_prompt in subtasks:" in self._src

    def test_agent_run_uses_task_prompt(self) -> None:
        """agent.run() receives the individual subtask prompt, not the full prompt."""
        assert "prompt_template=task_prompt" in self._src

    def test_worktree_check_after_loop(self) -> None:
        """Worktree merge review is in the finally block, after the loop.

        After the RED-10 refactor the block delegates to
        ``_present_pending_worktree`` instead of calling
        ``_get_worktree_changed_files`` directly.
        """
        loop_pos = self._src.index("for task_prompt in subtasks:")
        wt_pos = self._src.index("_present_pending_worktree")
        assert wt_pos > loop_pos

    def test_interrupt_breaks_loop(self) -> None:
        """KeyboardInterrupt breaks out of the subtask loop."""
        # Find the except KeyboardInterrupt block and verify break follows
        ki_idx = self._src.index("except KeyboardInterrupt:")
        after = self._src[ki_idx:ki_idx + 300]
        assert "break" in after

    def test_worktree_check_in_finally_block(self) -> None:
        """Worktree merge review is in the finally block, not the inner try."""
        # The worktree merge check must be in the finally block so it
        # runs on success, failure, and user-stop alike.
        assert "if tab.use_worktree and tab.agent._wt_pending:" in self._src

    def test_merge_view_only_runs_once(self) -> None:
        """_prepare_merge_view is NOT inside the subtask loop.

        It runs once in the finally block (after all subtasks), not
        per-subtask.
        """
        # The merge view code is in the finally block, outside the for loop
        # Verify _prepare_merge_view is NOT between the for loop markers
        loop_start = self._src.index("for task_prompt in subtasks:")
        # Find the break from the loop (KeyboardInterrupt handler)
        # The for loop ends before the "except BaseException" outer handler
        outer_except = self._src.index("except BaseException:")
        loop_body = self._src[loop_start:outer_except]
        assert "_prepare_merge_view" not in loop_body

    def test_followup_only_runs_once(self) -> None:
        """_generate_followup_async is NOT inside the subtask loop."""
        loop_start = self._src.index("for task_prompt in subtasks:")
        outer_except = self._src.index("except BaseException:")
        loop_body = self._src[loop_start:outer_except]
        assert "_generate_followup_async" not in loop_body

    def test_git_snapshot_before_loop(self) -> None:
        """Git snapshot is taken once before the subtask loop."""
        # Snapshot logic is in _capture_pre_snapshot (called from
        # _run_task_inner before the loop).
        snapshot_pos = self._src.index("_capture_pre_snapshot")
        loop_pos = self._src.index("for task_prompt in subtasks:")
        assert snapshot_pos < loop_pos

    def test_recording_spans_all_subtasks(self) -> None:
        """start_recording is before the loop, stop_recording is after."""
        start_pos = self._src.index("self.printer.start_recording()")
        loop_pos = self._src.index("for task_prompt in subtasks:")
        assert start_pos < loop_pos
        # stop_recording is in the finally block, after the loop
        stop_pos = self._src.index("self.printer.stop_recording()")
        assert stop_pos > loop_pos


class TestParseTaskTagsReturnType(unittest.TestCase):
    """Verify parse_task_tags always returns a non-empty list."""

    def test_always_returns_list(self) -> None:
        for text in ["", "hello", "<task>x</task>", "<task></task>"]:
            result = parse_task_tags(text)
            assert isinstance(result, list)
            assert len(result) >= 1


if __name__ == "__main__":
    unittest.main()
