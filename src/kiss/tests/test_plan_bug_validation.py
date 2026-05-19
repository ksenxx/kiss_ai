"""Integration tests validating that plan.md bugs are fixed.

Each test verifies that the corresponding bug fix is in place by
inspecting real code — no mocks, patches, fakes, or test doubles.
"""

import inspect


class TestPollerSessionResume:

    def test_stateful_agent_has_resume_by_id(self) -> None:
        """ChatSorcarAgent exposes resume_chat_by_id."""
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

        assert hasattr(ChatSorcarAgent, "resume_chat_by_id")


class TestVSCodeTaskGenerationSync:
    def test_is_current_task_generation_removed(self) -> None:
        """_is_current_task_generation was dead code and has been removed."""
        from kiss.agents.vscode.server import VSCodeServer

        assert not hasattr(VSCodeServer, "_is_current_task_generation")


class TestWaitForReplyHasTimeout:
    def test_slack_wait_for_reply_has_timeout(self) -> None:
        """Slack wait_for_reply accepts timeout_seconds."""
        from kiss.agents.third_party_agents.slack_agent import SlackChannelBackend

        sig = inspect.signature(SlackChannelBackend.wait_for_reply)
        assert "timeout_seconds" in sig.parameters

    def test_irc_wait_for_reply_has_timeout(self) -> None:
        """IRC wait_for_reply accepts timeout_seconds."""
        from kiss.agents.third_party_agents.irc_agent import IRCChannelBackend

        sig = inspect.signature(IRCChannelBackend.wait_for_reply)
        assert "timeout_seconds" in sig.parameters

    def test_whatsapp_wait_for_reply_has_timeout(self) -> None:
        """WhatsApp wait_for_reply accepts timeout_seconds."""
        from kiss.agents.third_party_agents.whatsapp_agent import WhatsAppChannelBackend

        sig = inspect.signature(WhatsAppChannelBackend.wait_for_reply)
        assert "timeout_seconds" in sig.parameters


class TestIRCBackendLifecycle:
    def test_irc_has_disconnect_method(self) -> None:
        """IRCChannelBackend has a disconnect() method."""
        from kiss.agents.third_party_agents.irc_agent import IRCChannelBackend

        assert hasattr(IRCChannelBackend, "disconnect")



class TestWhatsAppWebhookLifecycle:
    def test_whatsapp_has_disconnect(self) -> None:
        """WhatsAppChannelBackend has disconnect() method."""
        from kiss.agents.third_party_agents.whatsapp_agent import WhatsAppChannelBackend

        assert hasattr(WhatsAppChannelBackend, "disconnect")



class TestRelentlessAgentTempFile:
    def test_no_tempfile_mkstemp(self) -> None:
        """RelentlessAgent no longer uses tempfile.mkstemp()."""
        from kiss.core.relentless_agent import RelentlessAgent

        source = inspect.getsource(RelentlessAgent)
        assert "tempfile.mkstemp(" not in source



class TestDockerManagerTimeout:
    def test_bash_has_timeout_parameter(self) -> None:
        """DockerManager.Bash() accepts timeout_seconds."""
        from kiss.docker.docker_manager import DockerManager

        sig = inspect.signature(DockerManager.Bash)
        assert "timeout_seconds" in sig.parameters









class TestWebhookDistinctPorts:
    def test_each_backend_has_unique_port(self) -> None:
        """Webhook backends use distinct default ports, not all 8080."""
        from kiss.agents.third_party_agents.line_agent import LineChannelBackend
        from kiss.agents.third_party_agents.synology_chat_agent import SynologyChatChannelBackend
        from kiss.agents.third_party_agents.whatsapp_agent import WhatsAppChannelBackend
        from kiss.agents.third_party_agents.zalo_agent import ZaloChannelBackend

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




class TestGlobalBudgetReset:
    def test_reset_method_exists(self) -> None:
        """Base.reset_global_budget() class method exists."""
        from kiss.core.base import Base

        assert hasattr(Base, "reset_global_budget")
        assert callable(Base.reset_global_budget)







class TestPersistenceTOCTOUFixed:

    def test_save_task_result_accepts_task_id(self) -> None:
        """_save_task_result() has a task_id parameter."""
        from kiss.agents.sorcar import persistence

        sig = inspect.signature(persistence._save_task_result)
        assert "task_id" in sig.parameters





class TestDbConnClosable:
    def test_close_db_exists(self) -> None:
        """persistence._close_db() function exists."""
        from kiss.agents.sorcar import persistence

        assert hasattr(persistence, "_close_db")
        assert callable(persistence._close_db)
