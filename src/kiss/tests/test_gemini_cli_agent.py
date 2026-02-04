# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Test suite for Gemini CLI Coding Agent.

These tests verify the Gemini CLI Agent functionality using real API calls.
NO MOCKS are used - all tests exercise actual behavior.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from kiss.agents.coding_agents.gemini_cli_agent import DEFAULT_GEMINI_MODEL, GeminiCliAgent
from kiss.core import DEFAULT_CONFIG
from kiss.core.utils import is_subpath, resolve_path
from kiss.tests.conftest import requires_gemini_api_key


@requires_gemini_api_key
class TestGeminiCliAgentPermissions(unittest.TestCase):
    """Tests for GeminiCliAgent permission handling."""

    def setUp(self):
        """Set up test fixtures.

        Creates a temporary directory structure with separate readable and
        writable subdirectories, and initializes a GeminiCliAgent instance.
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        self.readable_dir = self.temp_dir / "readable"
        self.writable_dir = self.temp_dir / "writable"
        self.readable_dir.mkdir()
        self.writable_dir.mkdir()

        self.agent = GeminiCliAgent("test-agent")
        self.agent._reset(
            model_name=DEFAULT_GEMINI_MODEL,
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
        """Test that tools are created as callable functions.

        Verifies that _create_tools returns the expected set of ADK tools
        including read_file, write_file, list_dir, run_shell, and web_search.
        """
        tools = self.agent._create_tools()
        # ADK tools are Python functions directly
        self.assertEqual(len(tools), 5)
        names = {t.__name__ for t in tools}
        self.assertIn("read_file", names)
        self.assertIn("write_file", names)
        self.assertIn("list_dir", names)
        self.assertIn("run_shell", names)
        self.assertIn("web_search", names)

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

        agent = GeminiCliAgent("test-agent-2")
        agent._reset(DEFAULT_GEMINI_MODEL, None, None, str(new_dir), 10, 0.5, None)
        self.assertTrue(new_dir.exists())


@requires_gemini_api_key
class TestGeminiCliAgentTools(unittest.TestCase):
    """Tests for GeminiCliAgent tool functions."""

    def setUp(self):
        """Set up test fixtures.

        Creates a temporary directory with a test file and initializes
        a GeminiCliAgent with tools for testing tool functionality.
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_file = self.temp_dir / "test.txt"
        self.test_file.write_text("Hello, World!")

        self.agent = GeminiCliAgent("test-agent")
        self.agent._reset(
            model_name=DEFAULT_GEMINI_MODEL,
            readable_paths=[str(self.temp_dir)],
            writable_paths=[str(self.temp_dir)],
            base_dir=str(self.temp_dir),
            max_steps=10,
            max_budget=1.0,
            formatter=None,
        )
        self.tools = self.agent._create_tools()
        self.tools_by_name = {t.__name__: t for t in self.tools}

    def tearDown(self):
        """Clean up test fixtures.

        Removes the temporary directory and all its contents.
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_file_success(self):
        """Test read_file tool reads a file successfully.

        Verifies that read_file returns status 'success' and the correct
        content for an existing file.
        """
        read_file = self.tools_by_name["read_file"]
        result = read_file("test.txt")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], "Hello, World!")

    def test_read_file_not_found(self):
        """Test read_file tool handles missing file.

        Verifies that read_file returns status 'error' when attempting
        to read a file that does not exist.
        """
        read_file = self.tools_by_name["read_file"]
        result = read_file("nonexistent.txt")
        self.assertEqual(result["status"], "error")

    def test_write_file_success(self):
        """Test write_file tool writes content successfully.

        Verifies that write_file returns status 'success' and creates
        a file with the specified content.
        """
        write_file = self.tools_by_name["write_file"]
        result = write_file("output.txt", "Test content")
        self.assertEqual(result["status"], "success")
        # Verify the file was created
        output_path = self.temp_dir / "output.txt"
        self.assertTrue(output_path.exists())
        self.assertEqual(output_path.read_text(), "Test content")

    def test_list_dir_success(self):
        """Test list_dir tool lists directory contents.

        Verifies that list_dir returns status 'success' and includes
        the test file in its entries output.
        """
        list_dir = self.tools_by_name["list_dir"]
        result = list_dir(".")
        self.assertEqual(result["status"], "success")
        self.assertIn("[file] test.txt", result["entries"])

    def test_run_shell_success(self):
        """Test run_shell tool executes commands.

        Verifies that run_shell returns status 'success' and captures
        the command's stdout output.
        """
        run_shell = self.tools_by_name["run_shell"]
        result = run_shell("echo 'hello'")
        self.assertEqual(result["status"], "success")
        self.assertIn("hello", result["stdout"])

    def test_run_shell_failure(self):
        """Test run_shell tool handles command failure.

        Verifies that run_shell returns status 'error' and the correct
        exit code when a command fails.
        """
        run_shell = self.tools_by_name["run_shell"]
        result = run_shell("exit 1")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["exit_code"], 1)


@requires_gemini_api_key
class TestGeminiCliAgentRun(unittest.TestCase):
    """Integration tests for GeminiCliAgent.run() method.

    These tests make real API calls to Google Gemini.
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

        Verifies that the GeminiCliAgent can execute a basic code generation
        task and return a non-null string result.
        """
        agent = GeminiCliAgent("test_agent")

        task = """Write a simple Python function that adds two numbers."""

        result = agent.run(
            model_name=DEFAULT_GEMINI_MODEL,
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

        Verifies that the GeminiCliAgent returns a string summary after
        executing a more complex task involving code generation and testing.
        """
        agent = GeminiCliAgent("test_agent")

        task = "Write a simple factorial function, test it, and make it efficient."

        result = agent.run(
            model_name=DEFAULT_GEMINI_MODEL,
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
