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
from kiss.tests.conftest import requires_gemini_api_key


@requires_gemini_api_key
class TestGeminiCliAgentPermissions(unittest.TestCase):
    def setUp(self):
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
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tools_are_created(self):
        tools = self.agent._create_tools()
        self.assertEqual(len(tools), 5)
        names = {t.__name__ for t in tools}
        self.assertIn("read_file", names)
        self.assertIn("write_file", names)
        self.assertIn("list_dir", names)
        self.assertIn("run_shell", names)
        self.assertIn("web_search", names)

    def test_reset_adds_base_dir_to_paths(self):
        self.assertIn(self.temp_dir.resolve(), self.agent.readable_paths)
        self.assertIn(self.temp_dir.resolve(), self.agent.writable_paths)

    def test_reset_creates_base_dir(self):
        new_dir = self.temp_dir / "new_workdir"
        self.assertFalse(new_dir.exists())
        agent = GeminiCliAgent("test-agent-2")
        agent._reset(DEFAULT_GEMINI_MODEL, None, None, str(new_dir), 10, 0.5, None)
        self.assertTrue(new_dir.exists())


@requires_gemini_api_key
class TestGeminiCliAgentTools(unittest.TestCase):
    def setUp(self):
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
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_file_success(self):
        result = self.tools_by_name["read_file"]("test.txt")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], "Hello, World!")

    def test_read_file_not_found(self):
        result = self.tools_by_name["read_file"]("nonexistent.txt")
        self.assertEqual(result["status"], "error")

    def test_write_file_success(self):
        result = self.tools_by_name["write_file"]("output.txt", "Test content")
        self.assertEqual(result["status"], "success")
        output_path = self.temp_dir / "output.txt"
        self.assertTrue(output_path.exists())
        self.assertEqual(output_path.read_text(), "Test content")

    def test_list_dir_success(self):
        result = self.tools_by_name["list_dir"](".")
        self.assertEqual(result["status"], "success")
        self.assertIn("[file] test.txt", result["entries"])

    def test_run_shell_success(self):
        result = self.tools_by_name["run_shell"]("echo 'hello'")
        self.assertEqual(result["status"], "success")
        self.assertIn("hello", result["stdout"])

    def test_run_shell_failure(self):
        result = self.tools_by_name["run_shell"]("exit 1")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["exit_code"], 1)


@requires_gemini_api_key
class TestGeminiCliAgentRun(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()
        self.project_root = Path(DEFAULT_CONFIG.agent.artifact_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_agent_run_simple_task(self):
        agent = GeminiCliAgent("test_agent")
        result = agent.run(
            model_name=DEFAULT_GEMINI_MODEL,
            prompt_template="Write a simple Python function that adds two numbers.",
            readable_paths=[str(self.project_root / "src")],
            writable_paths=[str(self.output_dir)],
            base_dir=str(self.temp_dir),
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
