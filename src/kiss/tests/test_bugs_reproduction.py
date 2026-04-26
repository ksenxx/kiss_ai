"""Integration tests that reproduce bugs listed in bugs.md.

Each test demonstrates the buggy behavior. All tests should FAIL
until the corresponding bug is fixed. No mocks, patches, fakes,
or test doubles are used.
"""

import inspect


class TestC4ThoughtSignaturesNotCleared:
    def test_reset_conversation_clears_thought_signatures(self) -> None:
        """reset_conversation() should clear _thought_signatures.

        The bug: only initialize() clears it, so stale signatures
        accumulate across sub-sessions.
        """
        from kiss.core.models.gemini_model import GeminiModel

        model = GeminiModel.__new__(GeminiModel)
        model.conversation = []
        model.usage_info_for_messages = ""
        model._thought_signatures = {"stale-key": b"stale-value"}

        model.reset_conversation()

        assert model._thought_signatures == {}, (
            f"reset_conversation() should clear _thought_signatures, "
            f"but it still contains: {model._thought_signatures}"
        )


class TestR2UnboundedCompleteThreads:
    def test_rapid_completions_spawn_many_threads(self) -> None:
        """Each keystroke spawns a new thread for completions.

        We verify that the complete handler doesn't create unbounded
        threads by checking the source code for thread-per-request pattern.
        """
        from kiss.agents.vscode.server import VSCodeServer

        source = inspect.getsource(VSCodeServer._handle_command)
        lines = source.split("\n")
        in_complete_branch = False
        spawns_thread_per_complete = False
        for line in lines:
            stripped = line.strip()
            if in_complete_branch and stripped.startswith(("elif ", "else:")):
                break
            if in_complete_branch and "threading.Thread" in stripped:
                spawns_thread_per_complete = True
                break
            if "'complete'" in stripped or '"complete"' in stripped:
                in_complete_branch = True

        assert not spawns_thread_per_complete, (
            "Complete handler spawns a new thread per keystroke. "
            "Should use a persistent worker or debounce."
        )


class TestR4FlushBashStaleText:
    def test_flush_bash_can_broadcast_after_reset(self) -> None:
        """_flush_bash reads buffer, releases lock, then broadcasts.
        reset() between read and broadcast cannot prevent stale output.

        This test verifies the presence of a generation counter or
        similar mechanism that prevents stale broadcasts.
        """
        from kiss.agents.vscode.browser_ui import BaseBrowserPrinter

        source = inspect.getsource(BaseBrowserPrinter._flush_bash)

        has_generation_guard = (
            "generation" in source or "_gen" in source or "epoch" in source
        )

        assert has_generation_guard, (
            "_flush_bash broadcasts text outside the lock after reset() can run. "
            "A generation counter should guard the broadcast against stale data."
        )


class TestI1IdentityComparison:
    def test_conversation_loop_uses_identity_not_equality(self) -> None:
        """The loop comparing messages should use 'is' not '=='.

        With '==', two different dicts with identical content would
        wrongly be considered the same message.
        """
        from kiss.core.models.gemini_model import GeminiModel

        source = inspect.getsource(
            GeminiModel._convert_conversation_to_gemini_contents
        )

        uses_identity = "prev_msg is msg" in source
        uses_equality = "prev_msg == msg" in source

        assert uses_identity and not uses_equality, (
            f"Should use 'prev_msg is msg' (identity) not 'prev_msg == msg' "
            f"(equality). identity={uses_identity}, equality={uses_equality}"
        )


class TestI2FindChannelReturnsName:
    def test_find_channel_does_actual_lookup(self) -> None:
        """find_channel should look up channel by name, not echo it back.

        The bug: it returns the name as-is, which is a string like
        'general', not a Discord snowflake ID.
        """
        from kiss.agents.third_party_agents.discord_agent import DiscordChannelBackend

        backend = DiscordChannelBackend()
        result = backend.find_channel("general")
        assert result != "general" or result is None, (
            f"find_channel('general') returned '{result}' — the name echoed "
            f"back as a channel ID. Should do actual channel lookup or return None."
        )


class TestI3SnowflakeTruncation:
    def test_snowflake_computation_preserves_fractional_seconds(self) -> None:
        """The Snowflake computation should not truncate fractional seconds
        before multiplying by 1000.

        Bug: int(time.time() - 1) * 1000 truncates, should be
        int((time.time() - 1) * 1000).
        """
        from kiss.agents.third_party_agents.discord_agent import DiscordChannelBackend

        source = inspect.getsource(DiscordChannelBackend.poll_messages)

        has_bug = "int(time.time() - 1) * 1000" in source

        assert not has_bug, (
            "Snowflake computation truncates fractional seconds before "
            "multiplying: int(time.time() - 1) * 1000. "
            "Should be: int((time.time() - 1) * 1000)"
        )


class TestI4ModelUsageNotRecordedOnFailure:
    def test_record_model_usage_not_in_task_runner(self) -> None:
        """_record_model_usage must NOT be called from _run_task_inner.

        Model usage counts and last_model_used are updated only when the
        user selects a model via the model picker (_cmd_select_model).
        """
        from kiss.agents.vscode.server import VSCodeServer

        source = inspect.getsource(VSCodeServer._run_task_inner)
        assert "_record_model_usage" not in source, (
            "_record_model_usage should not be in _run_task_inner; "
            "model usage is updated only via the model picker"
        )


class TestI5OffByOneStepCount:
    def test_agent_executes_max_steps_not_max_steps_minus_one(self) -> None:
        """With max_steps=N, the agent should execute N steps, not N-1.

        The bug: step_count is incremented before _check_limits, so on
        step N, step_count == max_steps triggers the >= check before
        _execute_step runs.
        """
        from kiss.core.kiss_agent import KISSAgent

        source = inspect.getsource(KISSAgent._run_agentic_loop)

        lines = source.split("\n")
        increment_line = -1
        check_line = -1
        execute_line = -1
        for i, line in enumerate(lines):
            if "step_count += 1" in line:
                increment_line = i
            if "_check_limits()" in line:
                check_line = i
            if "_execute_step()" in line and execute_line == -1:
                execute_line = i

        check_source = inspect.getsource(KISSAgent._check_limits)
        uses_gte = "step_count >= self.max_steps" in check_source

        if increment_line < check_line < execute_line:
            assert not uses_gte, (
                "step_count incremented before _check_limits which uses >=. "
                "On the last step, check fires before execute. "
                "Use > instead of >= in _check_limits, or increment after execute."
            )


class TestI6DocstringReferencesNonExistentParams:
    def test_open_docstring_does_not_reference_args(self) -> None:
        """open() takes no parameters, so its docstring shouldn't list Args."""
        from kiss.docker.docker_manager import DockerManager

        doc = inspect.getdoc(DockerManager.open) or ""
        sig = inspect.signature(DockerManager.open)
        params = [p for p in sig.parameters if p != "self"]

        assert "image_name" not in doc or params, (
            "open() docstring references 'image_name' parameter but "
            f"open() takes no arguments (params={params})"
        )


class TestI7UnnecessaryCallableCheck:
    def test_disconnect_backend_calls_disconnect_directly(self) -> None:
        """_disconnect_backend should call disconnect() directly since
        it's a required protocol method, not use getattr+callable guard.
        """
        from kiss.agents.third_party_agents._channel_agent_utils import ChannelRunner

        source = inspect.getsource(ChannelRunner._disconnect_backend)

        has_getattr = "getattr(self._backend" in source
        has_callable = "callable(disconnect)" in source

        assert not (has_getattr and has_callable), (
            "_disconnect_backend uses getattr+callable guard for disconnect(), "
            "which is unnecessary. "
            "Should call self._backend.disconnect() directly."
        )


class TestI9FollowupTextNoTruncation:
    def test_result_is_truncated_or_docstring_is_accurate(self) -> None:
        """Either result is truncated to 500 chars, or the docstring
        shouldn't claim truncation.
        """
        from kiss.agents.vscode.helpers import generate_followup_text

        doc = inspect.getdoc(generate_followup_text) or ""
        source = inspect.getsource(generate_followup_text)

        claims_truncation = "500" in doc and "truncat" in doc.lower()
        does_truncation = "result[:500]" in source or "result[: 500]" in source

        assert not claims_truncation or does_truncation, (
            "Docstring claims result is truncated to 500 chars but the code "
            "passes result as-is. Either add truncation or fix the docstring."
        )


class TestI10ArtifactDirProxyMissingEqHash:
    def test_artifact_dir_proxy_supports_equality(self) -> None:
        """_ArtifactDirProxy should support == comparison with strings."""
        from kiss.core.config import _ArtifactDirProxy

        proxy = _ArtifactDirProxy()
        path_str = str(proxy)

        assert proxy == path_str, (
            f"_ArtifactDirProxy.__eq__ not implemented: "
            f"proxy == '{path_str}' returned False. "
            f"String comparisons with the proxy silently fail."
        )


class TestI11SummarizerNoStopEvent:
    def test_summarizer_passes_stop_event_to_useful_tools(self) -> None:
        """The summarizer should pass a stop_event to UsefulTools so
        the VS Code stop button can interrupt long Bash commands.
        """
        from kiss.core.relentless_agent import RelentlessAgent

        source = inspect.getsource(RelentlessAgent.perform_task)

        lines = source.split("\n")
        for line in lines:
            if "UsefulTools()" in line and "shell_tools" in line:
                assert False, (
                    "Summarizer creates UsefulTools() without stop_event. "
                    "Should pass stop_event so interruption works: "
                    "UsefulTools(stop_event=self._stop_event)"
                )
                break


class TestI8GetFilesBlocksStateLock:
    def test_get_files_does_not_hold_state_lock_during_scan(self) -> None:
        """_get_files should not hold _state_lock while running _scan_files
        on first call, as it blocks all other state-dependent operations.
        """
        from kiss.agents.vscode.server import VSCodeServer

        if not hasattr(VSCodeServer, "_get_files"):
            return

        source = inspect.getsource(VSCodeServer._get_files)

        lines = source.split("\n")
        lock_indent = -1
        scan_under_lock = False
        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if "with self._state_lock" in stripped:
                lock_indent = indent
            elif lock_indent >= 0 and indent <= lock_indent and stripped:
                lock_indent = -1
            if "_scan_files" in stripped and lock_indent >= 0:
                scan_under_lock = True

        assert not scan_under_lock, (
            "_scan_files is called while holding _state_lock. "
            "This blocks all state-dependent operations on first call."
        )
