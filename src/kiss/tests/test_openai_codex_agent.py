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
from kiss.tests.conftest import requires_openai_api_key


@requires_openai_api_key
class TestOpenAICodexAgentPermissions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.readable_dir = self.temp_dir / "readable"
        self.writable_dir = self.temp_dir / "writable"
        self.readable_dir.mkdir()
        self.writable_dir.mkdir()

        self.agent = OpenAICodexAgent("test-agent")
        self.agent._reset(
            model_name="gpt-5.3-codex",
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
        names = {t.name for t in tools}
        self.assertIn("read_file", names)
        self.assertIn("write_file", names)
        self.assertIn("list_dir", names)
        self.assertIn("run_shell", names)

    def test_reset_adds_base_dir_to_paths(self):
        self.assertIn(self.temp_dir.resolve(), self.agent.readable_paths)
        self.assertIn(self.temp_dir.resolve(), self.agent.writable_paths)

    def test_reset_creates_base_dir(self):
        new_dir = self.temp_dir / "new_workdir"
        self.assertFalse(new_dir.exists())
        agent = OpenAICodexAgent("test-agent-2")
        agent._reset("gpt-5.3-codex", None, None, str(new_dir), 10, 0.5, None)
        self.assertTrue(new_dir.exists())


@requires_openai_api_key
class TestOpenAICodexAgentRun(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()
        self.project_root = Path(DEFAULT_CONFIG.agent.artifact_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_agent_run_simple_task(self):
        agent = OpenAICodexAgent("test-agent")
        result = agent.run(
            model_name="gpt-5.3-codex",
            prompt_template="Write a simple Python function that adds two numbers.",
            readable_paths=[str(self.project_root / "src")],
            writable_paths=[str(self.output_dir)],
            base_dir=str(self.temp_dir),
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
