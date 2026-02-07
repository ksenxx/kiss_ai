"""Test suite for Claude Coding Agent.

These tests verify the Claude Coding Agent functionality using real API calls.
NO MOCKS are used - all tests exercise actual behavior.
"""

import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path

from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent
from kiss.core import DEFAULT_CONFIG
from kiss.tests.conftest import requires_anthropic_api_key


@requires_anthropic_api_key
class TestClaudeCodingAgentPermissions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.readable_dir = self.temp_dir / "readable"
        self.writable_dir = self.temp_dir / "writable"
        self.readable_dir.mkdir()
        self.writable_dir.mkdir()

        self.agent = ClaudeCodingAgent("test-agent")
        self.agent._reset(
            model_name="claude-sonnet-4-5",
            readable_paths=[str(self.readable_dir)],
            writable_paths=[str(self.writable_dir)],
            base_dir=str(self.temp_dir),
            max_steps=DEFAULT_CONFIG.agent.max_steps,
            max_budget=DEFAULT_CONFIG.agent.max_agent_budget,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _run_permission(self, tool_name, tool_args):
        from claude_agent_sdk import ToolPermissionContext

        context = ToolPermissionContext()
        return asyncio.run(self.agent.permission_handler(tool_name, tool_args, context))

    def test_permission_handler_read_allowed(self):
        from claude_agent_sdk import PermissionResultAllow

        result = self._run_permission("Read", {"path": str(self.readable_dir / "test.txt")})
        self.assertIsInstance(result, PermissionResultAllow)

    def test_permission_handler_read_denied(self):
        from claude_agent_sdk import PermissionResultDeny

        result = self._run_permission("Read", {"path": "/tmp/outside/test.txt"})
        self.assertIsInstance(result, PermissionResultDeny)

    def test_permission_handler_write_allowed(self):
        from claude_agent_sdk import PermissionResultAllow

        result = self._run_permission("Write", {"path": str(self.writable_dir / "output.txt")})
        self.assertIsInstance(result, PermissionResultAllow)

    def test_permission_handler_write_denied(self):
        from claude_agent_sdk import PermissionResultDeny

        result = self._run_permission("Write", {"path": str(self.readable_dir / "readonly.txt")})
        self.assertIsInstance(result, PermissionResultDeny)

    def test_permission_handler_tools_without_path(self):
        from claude_agent_sdk import PermissionResultAllow

        result = self._run_permission("SomeOtherTool", {})
        self.assertIsInstance(result, PermissionResultAllow)

    def test_permission_handler_file_path_key(self):
        from claude_agent_sdk import PermissionResultAllow

        result = self._run_permission(
            "Read", {"file_path": str(self.readable_dir / "test.txt")}
        )
        self.assertIsInstance(result, PermissionResultAllow)

    def test_permission_handler_grep_glob(self):
        from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

        for tool in ["Grep", "Glob"]:
            result = self._run_permission(tool, {"path": str(self.readable_dir / "test.txt")})
            self.assertIsInstance(result, PermissionResultAllow)
            result = self._run_permission(tool, {"path": "/tmp/outside/test.txt"})
            self.assertIsInstance(result, PermissionResultDeny)

    def test_permission_handler_edit_multiedit(self):
        from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

        for tool in ["Edit", "MultiEdit"]:
            result = self._run_permission(tool, {"path": str(self.writable_dir / "file.txt")})
            self.assertIsInstance(result, PermissionResultAllow)
            result = self._run_permission(
                tool, {"path": str(self.readable_dir / "readonly.txt")}
            )
            self.assertIsInstance(result, PermissionResultDeny)


@requires_anthropic_api_key
class TestClaudeCodingAgentRun(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()
        self.project_root = Path(DEFAULT_CONFIG.agent.artifact_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_agent_run_simple_task(self):
        agent = ClaudeCodingAgent("test-agent")
        result = agent.run(
            model_name="claude-sonnet-4-5",
            prompt_template="Write a simple Python function that adds two numbers.",
            readable_paths=[str(self.project_root / "src")],
            writable_paths=[str(self.output_dir)],
            base_dir=str(self.temp_dir),
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
