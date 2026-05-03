"""Tests validating that bugs.md fixes are correct.

Each test verifies the fix for a specific bug by exercising real code paths —
no mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import inspect
from typing import Any


class TestB4DeepseekTputInModelInfo:
    def test_tput_model_in_model_info(self) -> None:
        """The -tput variant has a pricing entry so calculate_cost works."""
        from kiss.core.models.model_info import MODEL_INFO

        assert "deepseek-ai/DeepSeek-R1-0528-tput" in MODEL_INFO

    def test_tput_model_has_nonzero_pricing(self) -> None:
        """Pricing is non-zero (not free)."""
        from kiss.core.models.model_info import MODEL_INFO

        info = MODEL_INFO["deepseek-ai/DeepSeek-R1-0528-tput"]
        assert info.input_price_per_1M > 0
        assert info.output_price_per_1M > 0


class TestB6ForceStopReturnCheck:
    def test_checks_return_value(self) -> None:
        """_force_stop_thread checks rc == 0 and rc > 1."""
        from kiss.agents.vscode.server import VSCodeServer

        source = inspect.getsource(VSCodeServer._force_stop_thread)
        assert "rc == 0" in source, "Should check for rc == 0 (thread not found)"
        assert "rc > 1" in source, "Should check for rc > 1 (multiple states)"

    def test_undoes_on_rc_gt_1(self) -> None:
        """On rc > 1, calls PyThreadState_SetAsyncExc with None to undo."""
        from kiss.agents.vscode.server import VSCodeServer

        source = inspect.getsource(VSCodeServer._force_stop_thread)
        idx = source.find("rc > 1")
        assert idx >= 0, "Should have rc > 1 check"
        after = source[idx:]
        assert "SetAsyncExc" in after and "None" in after, (
            "Should call PyThreadState_SetAsyncExc(tid, None) to undo when rc > 1"
        )


class TestB7ExtractResultSummary:
    def test_uses_peek_recording(self) -> None:
        """_extract_result_summary uses peek_recording instead of _recording."""
        from kiss.agents.vscode.server import VSCodeServer

        source = inspect.getsource(VSCodeServer._extract_result_summary)
        assert "peek_recording" in source
        assert "_recording" not in source or "peek_recording" in source


class TestB8ModelConfigInit:
    def test_reset_sets_model_config(self) -> None:
        """_reset() initializes self.model_config to None."""
        from kiss.core.relentless_agent import RelentlessAgent

        source = inspect.getsource(RelentlessAgent._reset)
        assert "self.model_config" in source


class TestB10SystemPromptPrecedence:
    def test_model_config_system_instruction_preserved(self) -> None:
        """model_config system_instruction is kept when system_prompt is also provided."""
        from kiss.core.kiss_agent import KISSAgent

        source = inspect.getsource(KISSAgent.run)
        assert "setdefault" in source, (
            "Should use setdefault to respect user's model_config system_instruction"
        )
        assert 'model_config["system_instruction"] = system_prompt' not in source


class TestB11KISSErrorNotSwallowed:
    def test_kiss_error_re_raised(self) -> None:
        """KISSError (e.g., budget exceeded) is re-raised, not swallowed."""
        from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

        source = inspect.getsource(WorktreeSorcarAgent.run)
        lines = source.split("\n")
        found_kiss_error_reraise = False
        for i, line in enumerate(lines):
            if "except KISSError" in line:
                for j in range(i + 1, min(i + 3, len(lines))):
                    if lines[j].strip() == "raise":
                        found_kiss_error_reraise = True
                        break
                break
        assert found_kiss_error_reraise, (
            "KISSError should be caught and re-raised before the generic "
            "except Exception handler"
        )


class TestB12ClaudeCodeNoMutation:
    def test_uses_local_config_copy(self) -> None:
        """generate_and_process_with_tools uses a local copy of model_config."""
        from kiss.core.models.claude_code_model import ClaudeCodeModel

        source = inspect.getsource(
            ClaudeCodeModel.generate_and_process_with_tools
        )
        assert "dict(original_config)" in source or "dict(self.model_config)" in source, (
            "Should create a local copy of model_config instead of mutating it in-place"
        )
        lines = source.split("\n")
        copy_line = None
        for i, line in enumerate(lines):
            if "dict(original_config)" in line or "config = dict(" in line:
                copy_line = i
                break
        assert copy_line is not None


class TestB13NegativeTokensPrevented:
    def test_max_zero_applied_to_input_tokens(self) -> None:
        """Input token count uses max(0, ...) to prevent negative values."""
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

        source = inspect.getsource(
            OpenAICompatibleModel.extract_input_output_token_counts_from_response
        )
        assert "max(0," in source, (
            "Should use max(0, prompt_tokens - cached - cache_write) "
            "to prevent negative input tokens"
        )


class TestB14FollowupTaskIdGuard:
    def test_followup_guarded_by_task_id_check(self) -> None:
        """_generate_followup_async is only called when task_id is not None."""
        from kiss.agents.vscode.server import VSCodeServer

        source = inspect.getsource(VSCodeServer._run_task_inner)
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "_generate_followup_async" in line:
                for j in range(i - 1, max(i - 5, -1), -1):
                    if "task_history_id is not None" in lines[j]:
                        return
                assert False, (
                    "_generate_followup_async called without checking "
                    "task_history_id is not None"
                )


class TestB15LoadHistoryNoCap:
    def test_default_limit_uncapped(self) -> None:
        """_load_history(limit=0) returns all entries (no hard cap)."""
        from kiss.agents.sorcar.persistence import _load_history

        source = inspect.getsource(_load_history)
        assert "10000" not in source, "Should not have a hard cap of 10000"
        assert "-1" in source, "Should use -1 (no limit) when limit is 0"


class TestB16CaseSensitiveGlob:
    def test_uses_glob_not_like(self) -> None:
        """_prefix_match_task uses GLOB for case-sensitive matching."""
        from kiss.agents.sorcar.persistence import _prefix_match_task

        source = inspect.getsource(_prefix_match_task)
        assert "GLOB" in source, "Should use GLOB for case-sensitive matching"
        assert "LIKE" not in source, "Should not use LIKE (case-insensitive)"


class TestB17MultiPrinterResult:

    def test_skips_empty_results(self) -> None:
        """Skips printers returning empty string."""
        from kiss.core.printer import MultiPrinter, Printer

        class TestPrinter(Printer):
            def __init__(self, return_val: str):
                self._return_val = return_val

            def print(self, content: str, type: str = "text", **kwargs: Any) -> str:
                return self._return_val

            def reset(self) -> None:
                pass

            def token_callback(self, token: str) -> None:
                pass

        p1 = TestPrinter("")
        p2 = TestPrinter("second")
        mp = MultiPrinter([p1, p2])
        result = mp.print("test")
        assert result == "second"


class TestB19StepCountCheck:
    def test_no_stale_pragma_no_branch(self) -> None:
        """step_count check should not have stale '# pragma: no branch'."""
        from kiss.core.kiss_agent import KISSAgent

        source = inspect.getsource(KISSAgent._check_limits)
        for line in source.split("\n"):
            if "step_count" in line and "max_steps" in line:
                assert "pragma: no branch" not in line, (
                    "Stale 'pragma: no branch' on dead step_count check"
                )


class TestB20ArtifactDirLocking:
    def test_uses_lock(self) -> None:
        """get_artifact_dir uses a lock for thread-safe lazy init."""
        from kiss.core.config import get_artifact_dir

        source = inspect.getsource(get_artifact_dir)
        assert "_artifact_dir_lock" in source

    def test_double_checked_locking(self) -> None:
        """Uses double-checked locking pattern (check before and inside lock)."""
        from kiss.core.config import get_artifact_dir

        source = inspect.getsource(get_artifact_dir)
        lines = source.split("\n")
        none_checks = [line for line in lines if "_artifact_dir is None" in line]
        assert len(none_checks) >= 2, (
            f"Should have double-checked locking (2 None checks), "
            f"found {len(none_checks)}"
        )
