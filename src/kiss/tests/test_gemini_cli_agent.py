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
