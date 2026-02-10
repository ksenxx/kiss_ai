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
