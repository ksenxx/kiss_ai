# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

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
from kiss.core.utils import is_subpath


class TestClaudeCodingAgentPermissions(unittest.TestCase):
    """Tests for ClaudeCodingAgent permission handling."""

    def setUp(self):
        """Set up test fixtures."""
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
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_is_subpath_for_exact_match(self):
        """Test is_subpath returns True for exact path match."""
        target = Path(self.readable_dir).resolve()
        whitelist = [Path(self.readable_dir).resolve()]
        self.assertTrue(is_subpath(target, whitelist))

    def test_is_subpath_for_child_path(self):
        """Test is_subpath returns True for child paths."""
        child_path = Path(self.readable_dir, "subdir", "file.txt").resolve()
        whitelist = [Path(self.readable_dir).resolve()]
        self.assertTrue(is_subpath(child_path, whitelist))

    def test_is_subpath_for_unrelated_path(self):
        """Test is_subpath returns False for unrelated paths."""
        unrelated = Path("/tmp/unrelated/path").resolve()
        whitelist = [Path(self.readable_dir).resolve()]
        self.assertFalse(is_subpath(unrelated, whitelist))

    def test_permission_handler_allows_read_in_readable_path(self):
        """Test permission_handler allows Read for readable paths."""
        from claude_agent_sdk import PermissionResultAllow, ToolPermissionContext

        file_path = str(self.readable_dir / "test.txt")
        context = ToolPermissionContext()
        result = asyncio.run(self.agent.permission_handler("Read", {"path": file_path}, context))
        self.assertIsInstance(result, PermissionResultAllow)

    def test_permission_handler_denies_read_outside_readable_path(self):
        """Test permission_handler denies Read outside readable paths."""
        from claude_agent_sdk import PermissionResultDeny, ToolPermissionContext

        file_path = "/tmp/outside/test.txt"
        context = ToolPermissionContext()
        result = asyncio.run(self.agent.permission_handler("Read", {"path": file_path}, context))
        self.assertIsInstance(result, PermissionResultDeny)

    def test_permission_handler_allows_write_in_writable_path(self):
        """Test permission_handler allows Write for writable paths."""
        from claude_agent_sdk import PermissionResultAllow, ToolPermissionContext

        file_path = str(self.writable_dir / "output.txt")
        context = ToolPermissionContext()
        result = asyncio.run(self.agent.permission_handler("Write", {"path": file_path}, context))
        self.assertIsInstance(result, PermissionResultAllow)

    def test_permission_handler_denies_write_outside_writable_path(self):
        """Test permission_handler denies Write outside writable paths."""
        from claude_agent_sdk import PermissionResultDeny, ToolPermissionContext

        file_path = str(self.readable_dir / "readonly.txt")
        context = ToolPermissionContext()
        result = asyncio.run(self.agent.permission_handler("Write", {"path": file_path}, context))
        self.assertIsInstance(result, PermissionResultDeny)

    def test_permission_handler_allows_tools_without_path(self):
        """Test permission_handler allows tools without path parameter."""
        from claude_agent_sdk import PermissionResultAllow, ToolPermissionContext

        context = ToolPermissionContext()
        result = asyncio.run(self.agent.permission_handler("SomeOtherTool", {}, context))
        self.assertIsInstance(result, PermissionResultAllow)

    def test_permission_handler_handles_file_path_key(self):
        """Test permission_handler handles 'file_path' key."""
        from claude_agent_sdk import PermissionResultAllow, ToolPermissionContext

        file_path = str(self.readable_dir / "test.txt")
        context = ToolPermissionContext()
        result = asyncio.run(
            self.agent.permission_handler("Read", {"file_path": file_path}, context)
        )
        self.assertIsInstance(result, PermissionResultAllow)

    def test_permission_handler_for_grep(self):
        """Test permission_handler handles Grep tool."""
        from claude_agent_sdk import (
            PermissionResultAllow,
            PermissionResultDeny,
            ToolPermissionContext,
        )

        context = ToolPermissionContext()
        # In readable path
        file_path = str(self.readable_dir / "test.txt")
        result = asyncio.run(self.agent.permission_handler("Grep", {"path": file_path}, context))
        self.assertIsInstance(result, PermissionResultAllow)

        # Outside readable path
        file_path = "/tmp/outside/test.txt"
        result = asyncio.run(self.agent.permission_handler("Grep", {"path": file_path}, context))
        self.assertIsInstance(result, PermissionResultDeny)

    def test_permission_handler_for_glob(self):
        """Test permission_handler handles Glob tool."""
        from claude_agent_sdk import (
            PermissionResultAllow,
            PermissionResultDeny,
            ToolPermissionContext,
        )

        context = ToolPermissionContext()
        # In readable path
        file_path = str(self.readable_dir / "*.py")
        result = asyncio.run(self.agent.permission_handler("Glob", {"path": file_path}, context))
        self.assertIsInstance(result, PermissionResultAllow)

        # Outside readable path
        file_path = "/tmp/outside/*.py"
        result = asyncio.run(self.agent.permission_handler("Glob", {"path": file_path}, context))
        self.assertIsInstance(result, PermissionResultDeny)

    def test_permission_handler_for_edit(self):
        """Test permission_handler handles Edit tool."""
        from claude_agent_sdk import (
            PermissionResultAllow,
            PermissionResultDeny,
            ToolPermissionContext,
        )

        context = ToolPermissionContext()
        # In writable path
        file_path = str(self.writable_dir / "edit.txt")
        result = asyncio.run(self.agent.permission_handler("Edit", {"path": file_path}, context))
        self.assertIsInstance(result, PermissionResultAllow)

        # Outside writable path
        file_path = str(self.readable_dir / "readonly.txt")
        result = asyncio.run(self.agent.permission_handler("Edit", {"path": file_path}, context))
        self.assertIsInstance(result, PermissionResultDeny)

    def test_permission_handler_for_multiedit(self):
        """Test permission_handler handles MultiEdit tool."""
        from claude_agent_sdk import (
            PermissionResultAllow,
            PermissionResultDeny,
            ToolPermissionContext,
        )

        context = ToolPermissionContext()
        # In writable path
        file_path = str(self.writable_dir / "multi.txt")
        result = asyncio.run(
            self.agent.permission_handler("MultiEdit", {"path": file_path}, context)
        )
        self.assertIsInstance(result, PermissionResultAllow)

        # Outside writable path
        file_path = str(self.readable_dir / "readonly.txt")
        result = asyncio.run(
            self.agent.permission_handler("MultiEdit", {"path": file_path}, context)
        )
        self.assertIsInstance(result, PermissionResultDeny)


class TestClaudeCodingAgentRun(unittest.TestCase):
    """Integration tests for ClaudeCodingAgent.run() method.

    These tests make real API calls to Claude.
    """

    def setUp(self):
        """Set up test fixtures with a temp directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()

        self.project_root = Path(DEFAULT_CONFIG.agent.artifact_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_agent_run_simple_task(self):
        """Test running a simple code generation task."""
        agent = ClaudeCodingAgent("test-agent")

        task = """Write a simple Python function that adds two numbers."""

        result = agent.run(
            model_name="claude-sonnet-4-5",
            prompt_template=task,
            readable_paths=[str(self.project_root / "src")],
            writable_paths=[str(self.output_dir)],
            base_dir=str(self.temp_dir),
        )

        # Result should be a string summary
        self.assertIsNotNone(result)
        if result:
            print(result)
            self.assertIsInstance(result, str)

    def test_agent_run_returns_string_summary(self):
        """Test that agent run returns a string summary."""
        agent = ClaudeCodingAgent("test-agent")

        task = "Write a simple factorial function, test it, and mke it efficient."

        result = agent.run(
            model_name="claude-sonnet-4-5",
            prompt_template=task,
            readable_paths=[str(self.project_root / "src")],
            writable_paths=[str(self.output_dir)],
            base_dir=str(self.temp_dir),
        )

        self.assertIsNotNone(result)
        if result:
            print(result)
            self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
