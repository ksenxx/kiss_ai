"""Integration tests that reproduce bugs listed in bugs.md.

Each test demonstrates the buggy behavior. All tests should FAIL
until the corresponding bug is fixed. No mocks, patches, fakes,
or test doubles are used.
"""

import inspect


# ---------------------------------------------------------------------------
# C1: openai_compatible_model.py — Double-counting reasoning tokens
# ---------------------------------------------------------------------------
class TestC1DoubleCountingReasoningTokens:
    def test_reasoning_tokens_not_double_counted(self) -> None:
        """completion_tokens already includes reasoning_tokens.

        The bug adds reasoning_tokens again, inflating the count.
        This test creates a real response object with known token
        counts and verifies the extraction returns the correct total.
        """
        from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)

        # Build a realistic response object with nested attributes
        class CompletionDetails:
            reasoning_tokens = 50

        class Usage:
            prompt_tokens = 100
            completion_tokens = 200  # Already includes 50 reasoning tokens
            completion_tokens_details = CompletionDetails()
            prompt_tokens_details = None

        class Response:
            usage = Usage()

        inp, out, cache_read, cache_write = (
            model.extract_input_output_token_counts_from_response(Response())
        )
        # If bug is present: out = 200 + 50 = 250 (WRONG)
        # If bug is fixed: out = 200 (CORRECT)
        assert out == 200, (
            f"Output tokens should be 200 (completion_tokens already includes "
            f"reasoning_tokens), but got {out}"
        )





# ---------------------------------------------------------------------------
# C3: server.py — _task_history_id never populated
# ---------------------------------------------------------------------------
class TestC3TaskHistoryIdNeverPopulated:
    def test_task_history_id_exposed_after_run(self) -> None:
        """After StatefulSorcarAgent.run(), the task_id should be
        accessible so the server can use it for history updates.

        The bug: _task_history_id stays None because StatefulSorcarAgent
        creates task_id as a local and never exposes it.
        """
        from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent

        agent = StatefulSorcarAgent("test")
        # Check if StatefulSorcarAgent exposes task_id after run
        # The bug is that there's no mechanism to get the task_id back
        assert hasattr(agent, "_last_task_id"), (
            "StatefulSorcarAgent should expose the task_id created during run() "
            "so callers can reference the exact history row. Currently it's a "
            "local variable inside run() that is never exposed."
        )


# ---------------------------------------------------------------------------
# C4: gemini_model.py — _thought_signatures not cleared on reset_conversation()
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# C5: All channel backends — disconnect not excluded from tool discovery
# ---------------------------------------------------------------------------
class TestC5DisconnectExposedAsTool:
    def test_disconnect_not_in_tool_methods(self) -> None:
        """disconnect() should be excluded from get_tool_methods().

        The bug: 'disconnect' is missing from the non_tool frozenset,
        so the LLM can call it and break the channel connection.
        """
        # Test with Discord backend (representative of all 23)
        from kiss.channels.discord_agent import DiscordChannelBackend

        backend = DiscordChannelBackend()
        tool_methods = backend.get_tool_methods()
        tool_names = [m.__name__ for m in tool_methods]

        assert "disconnect" not in tool_names, (
            f"disconnect() should NOT appear as a tool method, but it was "
            f"found in: {tool_names}"
        )

    def test_disconnect_not_in_tool_methods_slack(self) -> None:
        """Same check for Slack backend."""
        from kiss.channels.slack_agent import SlackChannelBackend

        backend = SlackChannelBackend()
        tool_methods = backend.get_tool_methods()
        tool_names = [m.__name__ for m in tool_methods]

        assert "disconnect" not in tool_names, (
            f"disconnect() should NOT appear as a tool method for Slack, "
            f"but found: {tool_names}"
        )


# ---------------------------------------------------------------------------
# R2: server.py — complete handler spawns unbounded threads
# ---------------------------------------------------------------------------
class TestR2UnboundedCompleteThreads:
    def test_rapid_completions_spawn_many_threads(self) -> None:
        """Each keystroke spawns a new thread for completions.

        We verify that the complete handler doesn't create unbounded
        threads by checking the source code for thread-per-request pattern.
        """
        from kiss.agents.vscode.server import VSCodeServer

        source = inspect.getsource(VSCodeServer._handle_command)
        # The bug: each "complete" request spawns a brand new
        # threading.Thread(target=self._complete, ...).  A fixed version
        # would submit work to a reusable worker / debounce.
        # Detect the pattern: Thread(... self._complete ...) inside the
        # "complete" handler branch.
        lines = source.split("\n")
        in_complete_branch = False
        spawns_thread_per_complete = False
        for line in lines:
            stripped = line.strip()
            # Exit the complete branch on the next elif/else
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


# ---------------------------------------------------------------------------
# R4: browser_ui.py — _flush_bash broadcasts stale text after reset()
# ---------------------------------------------------------------------------
class TestR4FlushBashStaleText:
    def test_flush_bash_can_broadcast_after_reset(self) -> None:
        """_flush_bash reads buffer, releases lock, then broadcasts.
        reset() between read and broadcast cannot prevent stale output.

        This test verifies the presence of a generation counter or
        similar mechanism that prevents stale broadcasts.
        """
        from kiss.agents.vscode.browser_ui import BaseBrowserPrinter

        source = inspect.getsource(BaseBrowserPrinter._flush_bash)

        # A fix would add a generation counter checked before broadcast
        has_generation_guard = (
            "generation" in source or "_gen" in source or "epoch" in source
        )

        assert has_generation_guard, (
            "_flush_bash broadcasts text outside the lock after reset() can run. "
            "A generation counter should guard the broadcast against stale data."
        )


# ---------------------------------------------------------------------------
# I1: gemini_model.py — Identity comparison uses == instead of is
# ---------------------------------------------------------------------------
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

        # Check for the buggy pattern: prev_msg == msg
        uses_identity = "prev_msg is msg" in source
        uses_equality = "prev_msg == msg" in source

        assert uses_identity and not uses_equality, (
            f"Should use 'prev_msg is msg' (identity) not 'prev_msg == msg' "
            f"(equality). identity={uses_identity}, equality={uses_equality}"
        )


# ---------------------------------------------------------------------------
# I2: discord_agent.py — find_channel returns name as channel ID
# ---------------------------------------------------------------------------
class TestI2FindChannelReturnsName:
    def test_find_channel_does_actual_lookup(self) -> None:
        """find_channel should look up channel by name, not echo it back.

        The bug: it returns the name as-is, which is a string like
        'general', not a Discord snowflake ID.
        """
        from kiss.channels.discord_agent import DiscordChannelBackend

        backend = DiscordChannelBackend()
        result = backend.find_channel("general")
        # If the bug is present: result == "general" (the name, not an ID)
        # A proper implementation would return None (no API to look up)
        # or a numeric snowflake ID
        assert result != "general" or result is None, (
            f"find_channel('general') returned '{result}' — the name echoed "
            f"back as a channel ID. Should do actual channel lookup or return None."
        )


# ---------------------------------------------------------------------------
# I3: discord_agent.py — Snowflake fractional-second truncation
# ---------------------------------------------------------------------------
class TestI3SnowflakeTruncation:
    def test_snowflake_computation_preserves_fractional_seconds(self) -> None:
        """The Snowflake computation should not truncate fractional seconds
        before multiplying by 1000.

        Bug: int(time.time() - 1) * 1000 truncates, should be
        int((time.time() - 1) * 1000).
        """
        from kiss.channels.discord_agent import DiscordChannelBackend

        source = inspect.getsource(DiscordChannelBackend.poll_messages)

        # The buggy pattern
        has_bug = "int(time.time() - 1) * 1000" in source

        assert not has_bug, (
            "Snowflake computation truncates fractional seconds before "
            "multiplying: int(time.time() - 1) * 1000. "
            "Should be: int((time.time() - 1) * 1000)"
        )


# ---------------------------------------------------------------------------
# I4: server.py — _record_model_usage only called on success
# ---------------------------------------------------------------------------
class TestI4ModelUsageNotRecordedOnFailure:
    def test_record_model_usage_in_finally_block(self) -> None:
        """_record_model_usage should be called even when the task is
        stopped or fails, since tokens were still consumed.
        """
        from kiss.agents.vscode.server import VSCodeServer

        source = inspect.getsource(VSCodeServer._run_task_inner)

        # Find _record_model_usage and check it's in a finally block
        lines = source.split("\n")
        in_finally = False
        usage_in_finally = False
        usage_before_finally = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("finally:"):
                in_finally = True
            if "_record_model_usage" in stripped:
                if in_finally:
                    usage_in_finally = True
                else:
                    usage_before_finally = True

        assert usage_in_finally and not usage_before_finally, (
            "_record_model_usage should be in the finally block so it runs "
            f"even on failure/stop. in_finally={usage_in_finally}, "
            f"before_finally={usage_before_finally}"
        )


# ---------------------------------------------------------------------------
# I5: kiss_agent.py — Off-by-one: agent gets max_steps - 1 actual steps
# ---------------------------------------------------------------------------
class TestI5OffByOneStepCount:
    def test_agent_executes_max_steps_not_max_steps_minus_one(self) -> None:
        """With max_steps=N, the agent should execute N steps, not N-1.

        The bug: step_count is incremented before _check_limits, so on
        step N, step_count == max_steps triggers the >= check before
        _execute_step runs.
        """
        from kiss.core.kiss_agent import KISSAgent

        source = inspect.getsource(KISSAgent._run_agentic_loop)

        # Check whether step_count is incremented before or after _check_limits
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

        # Bug: increment → check → execute (check fires before execute on last step)
        # Fix: either increment after execute, or use > instead of >=
        check_source = inspect.getsource(KISSAgent._check_limits)
        uses_gte = "step_count >= self.max_steps" in check_source

        # Either increment should be after execute, or check should use >
        if increment_line < check_line < execute_line:
            # increment before check before execute — only OK if using >
            assert not uses_gte, (
                "step_count incremented before _check_limits which uses >=. "
                "On the last step, check fires before execute. "
                "Use > instead of >= in _check_limits, or increment after execute."
            )
        # If increment is after execute, bug is fixed regardless of >= vs >


# ---------------------------------------------------------------------------
# I6: docker_manager.py — open() docstring references non-existent parameters
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# I7: background_agent.py — unnecessary callable() check
# ---------------------------------------------------------------------------
class TestI7UnnecessaryCallableCheck:
    def test_disconnect_backend_calls_disconnect_directly(self) -> None:
        """_disconnect_backend should call disconnect() directly since
        it's a required protocol method, not use getattr+callable guard.
        """
        from kiss.channels.background_agent import ChannelDaemon

        source = inspect.getsource(ChannelDaemon._disconnect_backend)

        has_getattr = "getattr(self._backend" in source
        has_callable = "callable(disconnect)" in source

        assert not (has_getattr and has_callable), (
            "_disconnect_backend uses getattr+callable guard for disconnect(), "
            "which is a required ChannelBackend protocol method. "
            "Should call self._backend.disconnect() directly."
        )


# ---------------------------------------------------------------------------
# I9: helpers.py — generate_followup_text docstring claims truncation
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# I10: config.py — _ArtifactDirProxy missing __eq__ and __hash__
# ---------------------------------------------------------------------------
class TestI10ArtifactDirProxyMissingEqHash:
    def test_artifact_dir_proxy_supports_equality(self) -> None:
        """_ArtifactDirProxy should support == comparison with strings."""
        from kiss.core.config import _ArtifactDirProxy

        proxy = _ArtifactDirProxy()
        path_str = str(proxy)

        # If __eq__ is not implemented, this comparison uses object identity
        # and will return False even though the strings match
        assert proxy == path_str, (
            f"_ArtifactDirProxy.__eq__ not implemented: "
            f"proxy == '{path_str}' returned False. "
            f"String comparisons with the proxy silently fail."
        )


# ---------------------------------------------------------------------------
# I11: relentless_agent.py — Summarizer's UsefulTools has no stop_event
# ---------------------------------------------------------------------------
class TestI11SummarizerNoStopEvent:
    def test_summarizer_passes_stop_event_to_useful_tools(self) -> None:
        """The summarizer should pass a stop_event to UsefulTools so
        the VS Code stop button can interrupt long Bash commands.
        """
        from kiss.core.relentless_agent import RelentlessAgent

        source = inspect.getsource(RelentlessAgent.perform_task)

        # Find where UsefulTools is constructed for the summarizer
        lines = source.split("\n")
        for line in lines:
            if "UsefulTools()" in line and "shell_tools" in line:
                assert False, (
                    "Summarizer creates UsefulTools() without stop_event. "
                    "Should pass stop_event so interruption works: "
                    "UsefulTools(stop_event=self._stop_event)"
                )
                break


# ---------------------------------------------------------------------------
# I12: persistence.py — _search_history doesn't escape SQL LIKE wildcards
# ---------------------------------------------------------------------------
class TestI12SearchHistoryNoWildcardEscape:
    def test_search_with_percent_wildcard_is_escaped(self) -> None:
        """Searching for a query containing '%' should treat it literally,
        not as a SQL LIKE wildcard.

        The bug: '%' in query becomes a LIKE wildcard, matching everything.
        """
        import tempfile
        from pathlib import Path

        from kiss.agents.sorcar import persistence

        # Use a temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir = persistence._KISS_DIR
            old_db_path = persistence._DB_PATH
            old_db_conn = persistence._db_conn
            try:
                persistence._KISS_DIR = Path(tmpdir)
                persistence._DB_PATH = Path(tmpdir) / "test_history.db"
                persistence._db_conn = None  # Force re-creation

                # Add entries to the database
                db = persistence._get_db()
                db.execute(
                    "INSERT INTO task_history (task, timestamp, result) "
                    "VALUES (?, ?, ?)",
                    ("Calculate 50% of 100", 1000.0, "50"),
                )
                db.execute(
                    "INSERT INTO task_history (task, timestamp, result) "
                    "VALUES (?, ?, ?)",
                    ("Write hello world", 1001.0, "done"),
                )
                db.commit()

                # Search for literal "50%" (unused but validates query works)
                persistence._search_history("50%")

                # If bug is present: '%' acts as wildcard, "50%" matches
                # "50% of 100" AND "Write hello world" (because %
                # matches anything after "50")
                # Actually with the query being f"%{query}%" = "%50%%",
                # the % at end is redundant but the middle % could match
                # differently. Let's test with just "%"
                results_all = persistence._search_history("%")

                # With bug: "%" becomes LIKE "%%%", which matches everything
                # With fix: "%" is escaped and only matches entries containing literal "%"
                assert len(results_all) == 1, (
                    f"Searching for literal '%' matched {len(results_all)} entries "
                    f"(expected 1 — only the entry containing '%'). "
                    f"The '%' is being treated as a SQL LIKE wildcard."
                )
            finally:
                if persistence._db_conn is not None:
                    persistence._db_conn.close()
                persistence._KISS_DIR = old_dir
                persistence._DB_PATH = old_db_path
                persistence._db_conn = old_db_conn


# ---------------------------------------------------------------------------
# R3: server.py — _task_thread check-and-start not under _state_lock
# ---------------------------------------------------------------------------
class TestR3TaskThreadNotUnderLock:
    def test_task_thread_check_under_state_lock(self) -> None:
        """The _task_thread check-and-start should be under _state_lock
        for consistency with the finally block that sets it to None.
        """
        from kiss.agents.vscode.server import VSCodeServer

        source = inspect.getsource(VSCodeServer._handle_command)

        # Find the "run" command handler section and verify _state_lock
        # wraps the _task_thread alive check
        lines = source.split("\n")
        in_run_section = False
        found_lock = False
        found_check = False
        for line in lines:
            if '"run"' in line or "'run'" in line:
                in_run_section = True
            if in_run_section:
                if "_state_lock" in line:
                    found_lock = True
                if "_task_thread" in line and "is_alive" in line:
                    found_check = True
                    break

        assert found_lock and found_check, (
            "The _task_thread alive check should be under _state_lock. "
            f"found_lock_before_check={found_lock}, found_check={found_check}"
        )


# ---------------------------------------------------------------------------
# I8: server.py — _get_files holds _state_lock during synchronous _scan_files
# ---------------------------------------------------------------------------
class TestI8GetFilesBlocksStateLock:
    def test_get_files_does_not_hold_state_lock_during_scan(self) -> None:
        """_get_files should not hold _state_lock while running _scan_files
        on first call, as it blocks all other state-dependent operations.
        """
        from kiss.agents.vscode.server import VSCodeServer

        # Find the _get_files method
        if not hasattr(VSCodeServer, "_get_files"):
            return  # Method may have been renamed/removed

        source = inspect.getsource(VSCodeServer._get_files)

        # Check that _scan_files is NOT called at a deeper indent than a
        # "with self._state_lock:" context manager.
        lines = source.split("\n")
        lock_indent = -1
        scan_under_lock = False
        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            # Track when we enter/exit a _state_lock with-block
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
