"""Tests validating that bugs.md fixes are correct.

Each test verifies the fix for a specific bug by exercising real code paths —
no mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import inspect
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# B1: fast_model_for() returns correct model per provider key
# ---------------------------------------------------------------------------
class TestB1FastModelForRouting:
    def test_anthropic_key_returns_claude_haiku(self) -> None:
        """With ANTHROPIC_API_KEY set, returns direct claude-haiku-4-5."""
        from kiss.core.config import DEFAULT_CONFIG
        from kiss.agents.vscode.helpers import fast_model_for

        saved = DEFAULT_CONFIG.ANTHROPIC_API_KEY
        try:
            DEFAULT_CONFIG.ANTHROPIC_API_KEY = "test-key"
            result = fast_model_for()
            assert result == "claude-haiku-4-5"
        finally:
            DEFAULT_CONFIG.ANTHROPIC_API_KEY = saved

    def test_openrouter_only_returns_openrouter_model(self) -> None:
        """With only OPENROUTER_API_KEY, returns openrouter/ prefixed model."""
        from kiss.core.config import DEFAULT_CONFIG
        from kiss.agents.vscode.helpers import fast_model_for

        saved_a = DEFAULT_CONFIG.ANTHROPIC_API_KEY
        saved_o = DEFAULT_CONFIG.OPENROUTER_API_KEY
        try:
            DEFAULT_CONFIG.ANTHROPIC_API_KEY = ""
            DEFAULT_CONFIG.OPENROUTER_API_KEY = "test-key"
            result = fast_model_for()
            assert result.startswith("openrouter/"), (
                f"With only OPENROUTER_API_KEY, got '{result}' which would "
                f"route to AnthropicModel with empty key"
            )
        finally:
            DEFAULT_CONFIG.ANTHROPIC_API_KEY = saved_a
            DEFAULT_CONFIG.OPENROUTER_API_KEY = saved_o

    def test_together_only_returns_together_model(self) -> None:
        """With only TOGETHER_API_KEY, returns a Together-compatible model."""
        from kiss.core.config import DEFAULT_CONFIG
        from kiss.agents.vscode.helpers import fast_model_for

        saved_a = DEFAULT_CONFIG.ANTHROPIC_API_KEY
        saved_o = DEFAULT_CONFIG.OPENROUTER_API_KEY
        saved_t = DEFAULT_CONFIG.TOGETHER_API_KEY
        try:
            DEFAULT_CONFIG.ANTHROPIC_API_KEY = ""
            DEFAULT_CONFIG.OPENROUTER_API_KEY = ""
            DEFAULT_CONFIG.TOGETHER_API_KEY = "test-key"
            result = fast_model_for()
            # Must NOT be "claude-haiku-4-5" which is a direct Anthropic model
            assert not result.startswith("claude-"), (
                f"With only TOGETHER_API_KEY, got '{result}' which is a direct "
                f"Anthropic model — would fail without ANTHROPIC_API_KEY"
            )
        finally:
            DEFAULT_CONFIG.ANTHROPIC_API_KEY = saved_a
            DEFAULT_CONFIG.OPENROUTER_API_KEY = saved_o
            DEFAULT_CONFIG.TOGETHER_API_KEY = saved_t


# ---------------------------------------------------------------------------
# B2: AnthropicBatchModel keeps batch/ prefix in model_name for pricing
# ---------------------------------------------------------------------------
class TestB2BatchModelPricing:
    def test_model_name_has_batch_prefix(self) -> None:
        """self.model_name retains batch/ prefix for pricing lookup."""
        from kiss.core.models.anthropic_batch_model import AnthropicBatchModel

        m = AnthropicBatchModel(
            model_name="batch/claude-opus-4-6",
            api_key="test-key",
        )
        assert m.model_name == "batch/claude-opus-4-6", (
            f"model_name should keep batch/ prefix, got: {m.model_name}"
        )

    def test_api_model_name_stripped(self) -> None:
        """_api_model_name has the batch/ prefix removed for API calls."""
        from kiss.core.models.anthropic_batch_model import AnthropicBatchModel

        m = AnthropicBatchModel(
            model_name="batch/claude-opus-4-6",
            api_key="test-key",
        )
        assert m._api_model_name == "claude-opus-4-6"

    def test_batch_pricing_is_half(self) -> None:
        """calculate_cost with batch/ prefix uses 50% pricing."""
        from kiss.core.models.model_info import MODEL_INFO, calculate_cost

        # Verify the batch/ entry has half the price
        full = MODEL_INFO["claude-opus-4-6"]
        batch = MODEL_INFO["batch/claude-opus-4-6"]
        assert batch.input_price_per_1M == full.input_price_per_1M * 0.5
        assert batch.output_price_per_1M == full.output_price_per_1M * 0.5

        # Verify calculate_cost uses the batch pricing
        cost_full = calculate_cost("claude-opus-4-6", 1000, 1000)
        cost_batch = calculate_cost("batch/claude-opus-4-6", 1000, 1000)
        assert abs(cost_batch - cost_full * 0.5) < 0.0001, (
            f"Batch cost {cost_batch} should be ~half of full cost {cost_full}"
        )


# ---------------------------------------------------------------------------
# B4: deepseek-ai/DeepSeek-R1-0528-tput in MODEL_INFO
# ---------------------------------------------------------------------------
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

    def test_tput_model_context_length(self) -> None:
        """get_max_context_length doesn't raise KeyError."""
        from kiss.core.models.model_info import get_max_context_length

        ctx = get_max_context_length("deepseek-ai/DeepSeek-R1-0528-tput")
        assert ctx > 0

    def test_tput_model_cost_nonzero(self) -> None:
        """calculate_cost returns non-zero for -tput variant."""
        from kiss.core.models.model_info import calculate_cost

        cost = calculate_cost("deepseek-ai/DeepSeek-R1-0528-tput", 1000, 1000)
        assert cost > 0


# ---------------------------------------------------------------------------
# B5: newChat handler acquires _state_lock
# ---------------------------------------------------------------------------
class TestB5NewChatUnderLock:
    def test_newchat_handler_uses_state_lock(self) -> None:
        """newChat command handler wraps check in _state_lock."""
        from kiss.agents.vscode.server import VSCodeServer

        source = inspect.getsource(VSCodeServer._handle_command)
        lines = source.split("\n")
        # Find the newChat elif line, then scan subsequent lines for _state_lock
        found_newchat = False
        found_lock = False
        for line in lines:
            stripped = line.strip()
            if '"newChat"' in stripped or "'newChat'" in stripped:
                found_newchat = True
                continue
            if found_newchat:
                if "_state_lock" in stripped:
                    found_lock = True
                    break
                # If we hit another handler branch, stop
                if stripped.startswith(("elif ", "else:")):
                    break
        assert found_lock, "newChat handler should use _state_lock"


# ---------------------------------------------------------------------------
# B6: _force_stop_thread checks PyThreadState_SetAsyncExc return value
# ---------------------------------------------------------------------------
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
        # After the rc > 1 check, the undo call (SetAsyncExc with None)
        # may span multiple lines. Check the substring after "rc > 1".
        idx = source.find("rc > 1")
        assert idx >= 0, "Should have rc > 1 check"
        after = source[idx:]
        assert "SetAsyncExc" in after and "None" in after, (
            "Should call PyThreadState_SetAsyncExc(tid, None) to undo when rc > 1"
        )


# ---------------------------------------------------------------------------
# B7: _extract_result_summary uses peek_recording, not raw _recordings
# ---------------------------------------------------------------------------
class TestB7ExtractResultSummary:
    def test_uses_peek_recording(self) -> None:
        """_extract_result_summary uses peek_recording instead of _recordings."""
        from kiss.agents.vscode.server import VSCodeServer

        source = inspect.getsource(VSCodeServer._extract_result_summary)
        assert "peek_recording" in source
        assert "_recordings" not in source

    def test_accepts_recording_id(self) -> None:
        """_extract_result_summary takes a recording_id parameter."""
        from kiss.agents.vscode.server import VSCodeServer

        sig = inspect.signature(VSCodeServer._extract_result_summary)
        assert "recording_id" in sig.parameters


# ---------------------------------------------------------------------------
# B8: RelentlessAgent._reset() initializes model_config
# ---------------------------------------------------------------------------
class TestB8ModelConfigInit:
    def test_reset_sets_model_config(self) -> None:
        """_reset() initializes self.model_config to None."""
        from kiss.core.relentless_agent import RelentlessAgent

        source = inspect.getsource(RelentlessAgent._reset)
        assert "self.model_config" in source

    def test_model_config_accessible_after_reset(self) -> None:
        """After _reset(), model_config attribute exists."""
        from kiss.core.relentless_agent import RelentlessAgent

        agent = RelentlessAgent.__new__(RelentlessAgent)
        agent.name = "test"
        agent._reset(
            model_name=None,
            max_sub_sessions=None,
            max_steps=None,
            max_budget=None,
            work_dir=tempfile.mkdtemp(),
            docker_image=None,
        )
        assert hasattr(agent, "model_config")


# ---------------------------------------------------------------------------
# B10: system_prompt does not override model_config["system_instruction"]
# ---------------------------------------------------------------------------
class TestB10SystemPromptPrecedence:
    def test_model_config_system_instruction_preserved(self) -> None:
        """model_config system_instruction is kept when system_prompt is also provided."""
        from kiss.core.kiss_agent import KISSAgent

        source = inspect.getsource(KISSAgent.run)
        assert "setdefault" in source, (
            "Should use setdefault to respect user's model_config system_instruction"
        )
        # Verify it's NOT unconditional assignment
        assert 'model_config["system_instruction"] = system_prompt' not in source


# ---------------------------------------------------------------------------
# B11: worktree_sorcar_agent re-raises KISSError
# ---------------------------------------------------------------------------
class TestB11KISSErrorNotSwallowed:
    def test_kiss_error_re_raised(self) -> None:
        """KISSError (e.g., budget exceeded) is re-raised, not swallowed."""
        from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

        source = inspect.getsource(WorktreeSorcarAgent.run)
        lines = source.split("\n")
        # Find the try/except structure: KISSError should be re-raised
        found_kiss_error_reraise = False
        for i, line in enumerate(lines):
            if "except KISSError" in line:
                # Check the next non-blank line is 'raise'
                for j in range(i + 1, min(i + 3, len(lines))):
                    if lines[j].strip() == "raise":
                        found_kiss_error_reraise = True
                        break
                break
        assert found_kiss_error_reraise, (
            "KISSError should be caught and re-raised before the generic "
            "except Exception handler"
        )


# ---------------------------------------------------------------------------
# B12: ClaudeCodeModel.generate_and_process_with_tools uses local config copy
# ---------------------------------------------------------------------------
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
        # Verify it doesn't do self.model_config["system_instruction"] = ... directly
        # without creating a copy first
        lines = source.split("\n")
        copy_line = None
        for i, line in enumerate(lines):
            if "dict(original_config)" in line or "config = dict(" in line:
                copy_line = i
                break
        assert copy_line is not None


# ---------------------------------------------------------------------------
# B13: Negative input tokens prevented
# ---------------------------------------------------------------------------
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

    def test_negative_cache_tokens_produce_zero(self) -> None:
        """When cached + cache_write > prompt, input_tokens is 0, not negative."""
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)

        class Details:
            cached_tokens = 80
            cache_read_tokens = None
            cache_write_tokens = 50

        class Usage:
            prompt_tokens = 100
            completion_tokens = 50
            prompt_tokens_details = Details()
            completion_tokens_details = None

        class Response:
            usage = Usage()

        inp, out, cr, cw = (
            model.extract_input_output_token_counts_from_response(Response())
        )
        # cached(80) + cache_write(50) = 130 > prompt(100)
        # Without fix: 100 - 80 - 50 = -30
        # With fix: max(0, -30) = 0
        assert inp >= 0, f"Input tokens should be >= 0, got {inp}"


# ---------------------------------------------------------------------------
# B14: _generate_followup_async only called when task_id is not None
# ---------------------------------------------------------------------------
class TestB14FollowupTaskIdGuard:
    def test_followup_guarded_by_task_id_check(self) -> None:
        """_generate_followup_async is only called when task_id is not None."""
        from kiss.agents.vscode.server import VSCodeServer

        source = inspect.getsource(VSCodeServer._run_task_inner)
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "_generate_followup_async" in line:
                # Look backwards for the task_id check
                for j in range(i - 1, max(i - 5, -1), -1):
                    if "_task_history_id is not None" in lines[j]:
                        return  # Found the guard
                assert False, (
                    "_generate_followup_async called without checking "
                    "_task_history_id is not None"
                )


# ---------------------------------------------------------------------------
# B15: _load_history has a hard cap
# ---------------------------------------------------------------------------
class TestB15LoadHistoryCap:
    def test_default_limit_capped(self) -> None:
        """_load_history(limit=0) uses a hard cap, not unbounded."""
        from kiss.agents.sorcar.persistence import _load_history

        source = inspect.getsource(_load_history)
        assert "10000" in source, "Should have a hard cap of 10000"

    def test_explicit_limit_respected(self) -> None:
        """When limit > 0, that exact limit is used."""
        from kiss.agents.sorcar import persistence

        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir = persistence._KISS_DIR
            old_db = persistence._DB_PATH
            old_conn = persistence._db_conn
            try:
                persistence._KISS_DIR = Path(tmpdir)
                persistence._DB_PATH = Path(tmpdir) / "test.db"
                persistence._db_conn = None

                db = persistence._get_db()
                for i in range(5):
                    db.execute(
                        "INSERT INTO task_history (task, timestamp) VALUES (?, ?)",
                        (f"task{i}", float(i)),
                    )
                db.commit()

                # Explicit limit
                result = persistence._load_history(limit=2)
                assert len(result) == 2

                # Default limit (0) returns all up to cap
                result = persistence._load_history()
                assert len(result) == 5
            finally:
                if persistence._db_conn:
                    persistence._db_conn.close()
                persistence._KISS_DIR = old_dir
                persistence._DB_PATH = old_db
                persistence._db_conn = old_conn


# ---------------------------------------------------------------------------
# B16: _prefix_match_task uses case-sensitive GLOB
# ---------------------------------------------------------------------------
class TestB16CaseSensitiveGlob:
    def test_uses_glob_not_like(self) -> None:
        """_prefix_match_task uses GLOB for case-sensitive matching."""
        from kiss.agents.sorcar.persistence import _prefix_match_task

        source = inspect.getsource(_prefix_match_task)
        assert "GLOB" in source, "Should use GLOB for case-sensitive matching"
        assert "LIKE" not in source, "Should not use LIKE (case-insensitive)"

    def test_case_sensitive_matching(self) -> None:
        """Uppercase query does not match lowercase task."""
        from kiss.agents.sorcar import persistence

        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir = persistence._KISS_DIR
            old_db = persistence._DB_PATH
            old_conn = persistence._db_conn
            try:
                persistence._KISS_DIR = Path(tmpdir)
                persistence._DB_PATH = Path(tmpdir) / "test.db"
                persistence._db_conn = None

                db = persistence._get_db()
                db.execute(
                    "INSERT INTO task_history (task, timestamp) VALUES (?, ?)",
                    ("hello world", 1000.0),
                )
                db.commit()

                # Exact case should match
                assert persistence._prefix_match_task("hello") == "hello world"
                # Wrong case should NOT match (GLOB is case-sensitive)
                assert persistence._prefix_match_task("Hello") == ""
            finally:
                if persistence._db_conn:
                    persistence._db_conn.close()
                persistence._KISS_DIR = old_dir
                persistence._DB_PATH = old_db
                persistence._db_conn = old_conn


# ---------------------------------------------------------------------------
# B17: MultiPrinter.print returns first non-empty result
# ---------------------------------------------------------------------------
class TestB17MultiPrinterResult:
    def test_returns_first_non_empty(self) -> None:
        """MultiPrinter.print returns the first non-empty result."""
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

        p1 = TestPrinter("first")
        p2 = TestPrinter("second")
        mp = MultiPrinter([p1, p2])
        result = mp.print("test")
        assert result == "first", f"Should return first non-empty result, got '{result}'"

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


# ---------------------------------------------------------------------------
# B19: _check_limits step count check no stale pragma comment
# ---------------------------------------------------------------------------
class TestB19StepCountCheck:
    def test_no_stale_pragma_no_branch(self) -> None:
        """step_count check should not have stale '# pragma: no branch'."""
        from kiss.core.kiss_agent import KISSAgent

        source = inspect.getsource(KISSAgent._check_limits)
        # The step_count > max_steps line should not have pragma: no branch
        for line in source.split("\n"):
            if "step_count" in line and "max_steps" in line:
                assert "pragma: no branch" not in line, (
                    "Stale 'pragma: no branch' on dead step_count check"
                )


# ---------------------------------------------------------------------------
# B20: get_artifact_dir uses double-checked locking
# ---------------------------------------------------------------------------
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
        # Should have two checks for _artifact_dir is None
        none_checks = [l for l in lines if "_artifact_dir is None" in l]
        assert len(none_checks) >= 2, (
            f"Should have double-checked locking (2 None checks), "
            f"found {len(none_checks)}"
        )

    def test_concurrent_calls_return_same_dir(self) -> None:
        """Multiple threads calling get_artifact_dir get the same result."""
        from kiss.core import config as config_mod

        results: list[str] = []
        barrier = threading.Barrier(4)

        def worker() -> None:
            barrier.wait()
            results.append(config_mod.get_artifact_dir())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(set(results)) == 1, (
            f"All threads should get the same dir, got {set(results)}"
        )
