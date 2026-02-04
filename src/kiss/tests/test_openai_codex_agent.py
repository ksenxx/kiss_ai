# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Test suite for OpenAI Codex Coding Agent.

These tests verify the OpenAI Codex Agent functionality using real API calls.
NO MOCKS are used - all tests exercise actual behavior.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from kiss.agents.coding_agents.openai_codex_agent import OpenAICodexAgent
from kiss.core import DEFAULT_CONFIG
from kiss.core.utils import is_subpath, resolve_path


class TestOpenAICodexAgentPermissions(unittest.TestCase):
    """Tests for OpenAICodexAgent permission handling."""

    def setUp(self):
        """Set up test fixtures.

        Creates a temporary directory structure with separate readable and
        writable subdirectories, and initializes an OpenAICodexAgent instance.
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        self.readable_dir = self.temp_dir / "readable"
        self.writable_dir = self.temp_dir / "writable"
        self.readable_dir.mkdir()
        self.writable_dir.mkdir()

        self.agent = OpenAICodexAgent("test-agent")
        self.agent._reset(
            model_name="gpt-5.2-codex",
            readable_paths=[str(self.readable_dir)],
            writable_paths=[str(self.writable_dir)],
            base_dir=str(self.temp_dir),
            max_steps=DEFAULT_CONFIG.agent.max_steps,
            max_budget=DEFAULT_CONFIG.agent.max_agent_budget,
            formatter=None,
        )

    def tearDown(self):
        """Clean up test fixtures.

        Removes the temporary directory and all its contents.
        """
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

    def test_resolve_path_relative(self):
        """Test resolve_path handles relative paths.

        Verifies that relative paths are correctly resolved against
        the base directory.
        """
        resolved = resolve_path("test.txt", str(self.temp_dir))
        expected = (Path(self.temp_dir) / "test.txt").resolve()
        self.assertEqual(resolved, expected)

    def test_resolve_path_absolute(self):
        """Test resolve_path handles absolute paths.

        Verifies that absolute paths are returned as-is, ignoring
        the base directory.
        """
        abs_path = "/tmp/absolute.txt"
        resolved = resolve_path(abs_path, str(self.temp_dir))
        self.assertEqual(resolved, Path(abs_path).resolve())

    def test_tools_are_created(self):
        """Test that tools are created with correct names.

        Verifies that _create_tools returns tools with the expected names
        including read_file, write_file, list_dir, and run_shell.
        """
        tools = self.agent._create_tools()
        names = {t.name for t in tools}
        self.assertIn("read_file", names)
        self.assertIn("write_file", names)
        self.assertIn("list_dir", names)
        self.assertIn("run_shell", names)

    def test_reset_adds_base_dir_to_paths(self):
        """Test that _reset adds base_dir to readable and writable paths.

        Verifies that the base_dir is automatically included in both
        readable_paths and writable_paths after agent reset.
        """
        self.assertIn(self.temp_dir.resolve(), self.agent.readable_paths)
        self.assertIn(self.temp_dir.resolve(), self.agent.writable_paths)

    def test_reset_creates_base_dir(self):
        """Test that _reset creates base_dir if it doesn't exist.

        Verifies that calling _reset with a non-existent base_dir path
        causes the directory to be created automatically.
        """
        new_dir = self.temp_dir / "new_workdir"
        self.assertFalse(new_dir.exists())

        agent = OpenAICodexAgent("test-agent-2")
        agent._reset("gpt-5.2-codex", None, None, str(new_dir), 10, 0.5, None)
        self.assertTrue(new_dir.exists())


class TestOpenAICodexAgentRun(unittest.TestCase):
    """Integration tests for OpenAICodexAgent.run() method.

    These tests make real API calls to OpenAI.
    """

    def setUp(self):
        """Set up test fixtures with a temp directory.

        Creates a temporary directory structure with an output subdirectory
        and stores the project root path for readable paths configuration.
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()

        self.project_root = Path(DEFAULT_CONFIG.agent.artifact_dir)

    def tearDown(self):
        """Clean up test fixtures.

        Removes the temporary directory and all its contents.
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_agent_run_simple_task(self):
        """Test running a simple code generation task.

        Verifies that the OpenAICodexAgent can execute a basic code generation
        task and return a non-null string result.
        """
        agent = OpenAICodexAgent("test-agent")

        task = """Write a simple Python function that adds two numbers."""

        result = agent.run(
            model_name="gpt-5.2-codex",
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
        """Test that agent run returns a string summary.

        Verifies that the OpenAICodexAgent returns a string summary after
        executing a more complex task involving code generation and testing.
        """
        agent = OpenAICodexAgent("test-agent")

        task = "Write a simple factorial function, test it, and make it efficient."

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
