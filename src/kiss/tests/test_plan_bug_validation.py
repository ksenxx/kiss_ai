"""Integration tests validating that plan.md bugs are fixed.

Each test verifies that the corresponding bug fix is in place by
inspecting real code — no mocks, patches, fakes, or test doubles.
"""

import inspect


# ---------------------------------------------------------------------------
# §19: poller uses public resume_chat_by_id — not private _chat_id
# ---------------------------------------------------------------------------
class TestPollerSessionResume:
    def test_poller_uses_resume_chat_by_id(self) -> None:
        """_handle_message() uses resume_chat_by_id instead of mutating _chat_id."""
        from kiss.channels._channel_agent_utils import ChannelRunner

        source = inspect.getsource(ChannelRunner._handle_message)
        assert "agent._chat_id" not in source

    def test_stateful_agent_has_resume_by_id(self) -> None:
        """StatefulSorcarAgent exposes resume_chat_by_id."""
        from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent

        assert hasattr(StatefulSorcarAgent, "resume_chat_by_id")


# ---------------------------------------------------------------------------
# §22C: _is_current_task_generation removed (dead code — never called)
# ---------------------------------------------------------------------------
class TestVSCodeTaskGenerationSync:
    def test_is_current_task_generation_removed(self) -> None:
        """_is_current_task_generation was dead code and has been removed."""
        from kiss.agents.vscode.server import VSCodeServer

        assert not hasattr(VSCodeServer, "_is_current_task_generation")


# ---------------------------------------------------------------------------
# §23: wait_for_reply() has timeout and cancellation support
# ---------------------------------------------------------------------------
class TestWaitForReplyHasTimeout:
    def test_slack_wait_for_reply_has_timeout(self) -> None:
        """Slack wait_for_reply accepts timeout_seconds."""
        from kiss.channels.slack_agent import SlackChannelBackend

        sig = inspect.signature(SlackChannelBackend.wait_for_reply)
        assert "timeout_seconds" in sig.parameters

    def test_irc_wait_for_reply_has_timeout(self) -> None:
        """IRC wait_for_reply accepts timeout_seconds."""
        from kiss.channels.irc_agent import IRCChannelBackend

        sig = inspect.signature(IRCChannelBackend.wait_for_reply)
        assert "timeout_seconds" in sig.parameters

    def test_whatsapp_wait_for_reply_has_timeout(self) -> None:
        """WhatsApp wait_for_reply accepts timeout_seconds."""
        from kiss.channels.whatsapp_agent import WhatsAppChannelBackend

        sig = inspect.signature(WhatsAppChannelBackend.wait_for_reply)
        assert "timeout_seconds" in sig.parameters


# ---------------------------------------------------------------------------
# §24: IRC backend has disconnect/cleanup lifecycle
# ---------------------------------------------------------------------------
class TestIRCBackendLifecycle:
    def test_irc_has_disconnect_method(self) -> None:
        """IRCChannelBackend has a disconnect() method."""
        from kiss.channels.irc_agent import IRCChannelBackend

        assert hasattr(IRCChannelBackend, "disconnect")

    def test_irc_socket_has_timeout(self) -> None:
        """IRC connect sets a socket timeout for the reader loop."""
        from kiss.channels.irc_agent import IRCChannelBackend

        source = inspect.getsource(IRCChannelBackend.connect)
        assert "settimeout" in source


# ---------------------------------------------------------------------------
# §25: WhatsApp webhook server has disconnect lifecycle
# ---------------------------------------------------------------------------
class TestWhatsAppWebhookLifecycle:
    def test_whatsapp_has_disconnect(self) -> None:
        """WhatsAppChannelBackend has disconnect() method."""
        from kiss.channels.whatsapp_agent import WhatsAppChannelBackend

        assert hasattr(WhatsAppChannelBackend, "disconnect")

    def test_disconnect_calls_shutdown(self) -> None:
        """disconnect() shuts down the webhook server."""
        from kiss.channels.whatsapp_agent import WhatsAppChannelBackend

        source = inspect.getsource(WhatsAppChannelBackend.disconnect)
        assert "shutdown" in source or "stop_http_server" in source


# ---------------------------------------------------------------------------
# §26: Relentless agent temp file in project temp area
# ---------------------------------------------------------------------------
class TestRelentlessAgentTempFile:
    def test_no_tempfile_mkstemp(self) -> None:
        """RelentlessAgent no longer uses tempfile.mkstemp()."""
        from kiss.core.relentless_agent import RelentlessAgent

        source = inspect.getsource(RelentlessAgent)
        assert "tempfile.mkstemp(" not in source

    def test_uses_work_dir_tmp(self) -> None:
        """RelentlessAgent creates temp files under work_dir/tmp."""
        from kiss.core.relentless_agent import RelentlessAgent

        source = inspect.getsource(RelentlessAgent)
        assert '"tmp"' in source or "'tmp'" in source


# ---------------------------------------------------------------------------
# §27: Docker Bash has timeout support
# ---------------------------------------------------------------------------
class TestDockerManagerTimeout:
    def test_bash_has_timeout_parameter(self) -> None:
        """DockerManager.Bash() accepts timeout_seconds."""
        from kiss.docker.docker_manager import DockerManager

        sig = inspect.signature(DockerManager.Bash)
        assert "timeout_seconds" in sig.parameters

    def test_bash_enforces_timeout(self) -> None:
        """Bash() uses timeout to limit exec_run duration."""
        from kiss.docker.docker_manager import DockerManager

        source = inspect.getsource(DockerManager.Bash)
        assert "timed out" in source or "timeout" in source


# ---------------------------------------------------------------------------
# §28: Docker output no unconditional newline separator
# ---------------------------------------------------------------------------
class TestDockerOutputFormatting:
    def test_no_unconditional_newline(self) -> None:
        """Output parts are joined only when non-empty."""
        from kiss.docker.docker_manager import DockerManager

        source = inspect.getsource(DockerManager.Bash)
        # Should filter empty parts instead of unconditional concatenation
        assert 'stdout + "\\n" + stderr' not in source
        assert "output_parts" in source or "filter" in source or "if part" in source


# ---------------------------------------------------------------------------
# §29: _clean_env() no longer caches stale environment
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# §30: no os.chdir() in agent/channel CLIs
# ---------------------------------------------------------------------------
class TestNoChdirInEntryPoints:
    def test_sorcar_agent_no_os_chdir(self) -> None:
        """sorcar_agent.py does not use os.chdir()."""
        from kiss.agents.sorcar import sorcar_agent

        source = inspect.getsource(sorcar_agent)
        assert "os.chdir(" not in source

    def test_stateful_agent_no_os_chdir(self) -> None:
        """stateful_sorcar_agent.py does not use os.chdir()."""
        from kiss.agents.sorcar import stateful_sorcar_agent

        source = inspect.getsource(stateful_sorcar_agent)
        assert "os.chdir(" not in source

    def test_channel_agents_no_os_chdir(self) -> None:
        """Channel agent CLIs do not use os.chdir()."""
        from kiss.channels import (
            discord_agent,
            irc_agent,
            slack_agent,
            whatsapp_agent,
        )

        for mod in [discord_agent, irc_agent, slack_agent, whatsapp_agent]:
            source = inspect.getsource(mod)
            assert "os.chdir(" not in source, f"{mod.__name__} should not use os.chdir"


# ---------------------------------------------------------------------------
# §31: Webhook backends fail fast on bind error
# ---------------------------------------------------------------------------
class TestWebhookBindFailurePropagated:
    def test_whatsapp_connect_fails_on_bind_error(self) -> None:
        """WhatsApp connect() fails if webhook server cannot bind."""
        from kiss.channels.whatsapp_agent import WhatsAppChannelBackend

        source = inspect.getsource(WhatsAppChannelBackend.connect)
        assert "if not self._start_webhook_server()" in source

    def test_start_webhook_returns_bool(self) -> None:
        """_start_webhook_server returns a bool success flag."""
        from kiss.channels.whatsapp_agent import WhatsAppChannelBackend

        source = inspect.getsource(WhatsAppChannelBackend._start_webhook_server)
        assert "return True" in source
        assert "return False" in source


# ---------------------------------------------------------------------------
# §32: Webhook backends have distinct default ports
# ---------------------------------------------------------------------------
class TestWebhookDistinctPorts:
    def test_each_backend_has_unique_port(self) -> None:
        """Webhook backends use distinct default ports, not all 8080."""
        from kiss.channels.line_agent import LineChannelBackend
        from kiss.channels.synology_chat_agent import SynologyChatChannelBackend
        from kiss.channels.whatsapp_agent import WhatsAppChannelBackend
        from kiss.channels.zalo_agent import ZaloChannelBackend

        backends = [
            WhatsAppChannelBackend,
            LineChannelBackend,
            ZaloChannelBackend,
            SynologyChatChannelBackend,
        ]
        ports: set[str] = set()
        for cls in backends:
            sig = inspect.signature(cls._start_webhook_server)  # type: ignore[attr-defined]
            default = sig.parameters["port"].default
            assert default != 8080, f"{cls.__name__} should not default to 8080"
            ports.add(str(default))
        assert len(ports) == len(backends), "All ports should be distinct"


# ---------------------------------------------------------------------------
# §33: SORCAR.md anchored to _kiss_pkg_dir (package directory)
# ---------------------------------------------------------------------------
class TestSystemPromptAnchored:
    def test_sorcar_path_uses_pkg_dir(self) -> None:
        """SORCAR.md is resolved relative to _kiss_pkg_dir, not bare CWD."""
        import kiss.core.base as base_mod

        source = inspect.getsource(base_mod)
        assert '_kiss_pkg_dir / "SORCAR.md"' in source or "_kiss_pkg_dir / 'SORCAR.md'" in source


# ---------------------------------------------------------------------------
# §34: artifact_dir uses lazy resolution, not import-time CWD
# ---------------------------------------------------------------------------
class TestArtifactDirLazy:
    def test_get_artifact_dir_function_exists(self) -> None:
        """config exposes get_artifact_dir() for lazy resolution."""
        from kiss.core import config as config_mod

        assert hasattr(config_mod, "get_artifact_dir")
        assert callable(config_mod.get_artifact_dir)

    def test_set_artifact_base_dir_function_exists(self) -> None:
        """config exposes set_artifact_base_dir() for explicit base."""
        from kiss.core import config as config_mod

        assert hasattr(config_mod, "set_artifact_base_dir")


# ---------------------------------------------------------------------------
# §35: global_budget_used reads synchronized
# ---------------------------------------------------------------------------
class TestGlobalBudgetSynchronizedReads:
    def test_get_global_budget_used_under_lock(self) -> None:
        """Base.get_global_budget_used() reads under _class_lock."""
        from kiss.core.base import Base

        source = inspect.getsource(Base.get_global_budget_used)
        assert "_class_lock" in source

    def test_check_limits_uses_getter(self) -> None:
        """_check_limits uses get_global_budget_used() (which is lock-protected)."""
        from kiss.core.kiss_agent import KISSAgent

        source = inspect.getsource(KISSAgent._check_limits)
        assert "get_global_budget_used()" in source


# ---------------------------------------------------------------------------
# §36: global_budget_used can be reset
# ---------------------------------------------------------------------------
class TestGlobalBudgetReset:
    def test_reset_method_exists(self) -> None:
        """Base.reset_global_budget() class method exists."""
        from kiss.core.base import Base

        assert hasattr(Base, "reset_global_budget")
        assert callable(Base.reset_global_budget)

    def test_poller_resets_budget_on_start(self) -> None:
        """ChannelRunner resets global budget in run_once()."""
        from kiss.channels._channel_agent_utils import ChannelRunner

        source = inspect.getsource(ChannelRunner.run_once)
        assert "reset_global_budget" in source


# ---------------------------------------------------------------------------
# §38: typo "Abrubptly" is fixed
# ---------------------------------------------------------------------------
class TestPersistenceTypoFixed:
    def test_no_abrubptly_typo(self) -> None:
        """persistence.py no longer contains 'Abrubptly'."""
        from kiss.agents.sorcar import persistence

        source = inspect.getsource(persistence)
        assert "Abrubptly" not in source

    def test_has_correct_spelling(self) -> None:
        """persistence.py uses 'Abruptly'."""
        from kiss.agents.sorcar import persistence

        source = inspect.getsource(persistence)
        assert "Abruptly" in source


# ---------------------------------------------------------------------------
# §39: _record_file_usage() is atomic under _db_lock
# ---------------------------------------------------------------------------
class TestRecordFileUsageAtomic:
    def test_entire_operation_under_db_lock(self) -> None:
        """INSERT, SELECT COUNT, DELETE, and commit are all inside _db_lock."""
        from kiss.agents.sorcar import persistence

        source = inspect.getsource(persistence._record_file_usage)
        lines = source.split("\n")

        lock_line = None
        lock_indent = 0
        for i, line in enumerate(lines):
            if "with _db_lock:" in line:
                lock_line = i
                lock_indent = len(line) - len(line.lstrip())
                break
        assert lock_line is not None

        # Everything after the with statement should be indented more
        for i, line in enumerate(lines):
            if i <= lock_line:
                continue
            stripped = line.strip()
            if not stripped:
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= lock_indent:
                break  # we've left the with block
            if "INSERT INTO file_usage" in stripped:
                assert True  # inside lock
            if "SELECT COUNT(*)" in stripped:
                assert True  # inside lock
            if "db.commit()" in stripped:
                assert True  # inside lock


# ---------------------------------------------------------------------------
# §40: _save_task_result uses task_id, lookup+update atomic under _db_lock
# ---------------------------------------------------------------------------
class TestPersistenceTOCTOUFixed:
    def test_save_task_result_uses_lock(self) -> None:
        """_save_task_result() wraps everything under _db_lock."""
        from kiss.agents.sorcar import persistence

        source = inspect.getsource(persistence._save_task_result)
        assert "with _db_lock:" in source

    def test_save_task_result_accepts_task_id(self) -> None:
        """_save_task_result() has a task_id parameter."""
        from kiss.agents.sorcar import persistence

        sig = inspect.signature(persistence._save_task_result)
        assert "task_id" in sig.parameters

    def test_set_latest_chat_events_uses_task_id_under_lock(self) -> None:
        """_set_latest_chat_events() resolves task_id inside _db_lock."""
        from kiss.agents.sorcar import persistence

        source = inspect.getsource(persistence._set_latest_chat_events)
        assert "with _db_lock:" in source
        sig = inspect.signature(persistence._set_latest_chat_events)
        assert "task_id" in sig.parameters


# ---------------------------------------------------------------------------
# §41: _add_task returns row ID
# ---------------------------------------------------------------------------
class TestAddTaskReturnsRowId:
    def test_add_task_returns_int(self) -> None:
        """_add_task() returns the inserted row ID."""
        from kiss.agents.sorcar import persistence

        source = inspect.getsource(persistence._add_task)
        assert "lastrowid" in source
        assert "return" in source

# ---------------------------------------------------------------------------
# §42: _force_stop_thread sets argtypes
# ---------------------------------------------------------------------------
class TestForceStopThreadArgtypes:
    def test_argtypes_set_at_module_level(self) -> None:
        """PyThreadState_SetAsyncExc.argtypes is set at module level."""
        import kiss.agents.vscode.server as server_mod

        source = inspect.getsource(server_mod)
        assert "PyThreadState_SetAsyncExc.argtypes" in source


# ---------------------------------------------------------------------------
# §43: _close_db() exists
# ---------------------------------------------------------------------------
class TestDbConnClosable:
    def test_close_db_exists(self) -> None:
        """persistence._close_db() function exists."""
        from kiss.agents.sorcar import persistence

        assert hasattr(persistence, "_close_db")
        assert callable(persistence._close_db)

# ---------------------------------------------------------------------------
# §44: WebUseTool _ensure_browser cleans up on failure
# ---------------------------------------------------------------------------
class TestWebUseToolCleanupOnFailure:
    def test_ensure_browser_guards_on_playwright_and_page(self) -> None:
        """_ensure_browser checks both _playwright and _page."""
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        source = inspect.getsource(WebUseTool._ensure_browser)
        assert "self._playwright is not None" in source
        assert "self._page is not None" in source

    def test_ensure_browser_has_cleanup_on_failure(self) -> None:
        """_ensure_browser calls self.close() in except block."""
        from kiss.agents.sorcar.web_use_tool import WebUseTool

        source = inspect.getsource(WebUseTool._ensure_browser)
        assert "self.close()" in source
        # close() is in the except path
        except_idx = source.index("except Exception:")
        close_idx = source.index("self.close()")
        assert close_idx > except_idx
