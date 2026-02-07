"""Test suite for OpenAICompatibleModel with configurable model.

Usage:
    pytest src/kiss/tests/test_a_model.py --model=gpt-5.2
"""

import json
import re
import unittest

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.rag.simple_rag import SimpleRAG
from kiss.tests.conftest import DEFAULT_MODEL, simple_calculator, skip_if_no_api_key_for_model

TEST_TIMEOUT = 60


@pytest.fixture
def model_name(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--model"))


class TestAModel(unittest.TestCase):
    model_name = DEFAULT_MODEL

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_non_agentic(self) -> None:
        skip_if_no_api_key_for_model(self.model_name)
        agent = KISSAgent(f"Test Agent for {self.model_name}")
        result = agent.run(
            model_name=self.model_name,
            prompt_template="What is 2 + 2? Answer with just the number.",
            is_agentic=False,
            max_budget=1.0,
        )
        self.assertIsNotNone(result)
        self.assertIn("4", re.sub(r"[,\\s]", "", result))
        self.assertGreater(len(json.loads(agent.get_trajectory())), 0)

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_agentic(self) -> None:
        skip_if_no_api_key_for_model(self.model_name)
        agent = KISSAgent(f"Test Agent for {self.model_name}")
        result = agent.run(
            model_name=self.model_name,
            prompt_template=(
                "Use the simple_calculator tool with expression='8934 * 2894' to calculate. "
                "Then call finish with the result of the simple_calculator tool."
            ),
            tools=[simple_calculator],
            max_steps=10,
            max_budget=1.0,
        )
        self.assertIsNotNone(result)
        self.assertIn("25854996", re.sub(r"[,\\s]", "", result))
        self.assertGreaterEqual(len(json.loads(agent.get_trajectory())), 5)

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_embedding(self) -> None:
        skip_if_no_api_key_for_model(self.model_name)
        from kiss.core.models.model_info import MODEL_INFO

        is_embedding = (
            self.model_name in MODEL_INFO and MODEL_INFO[self.model_name].is_embedding_supported
        )
        if not is_embedding:
            self.skipTest(f"{self.model_name} does not support embedding")

        rag = SimpleRAG(model_name=self.model_name)
        docs = [
            {"id": "1", "text": "The Eiffel Tower is in Paris."},
            {"id": "2", "text": "Mount Everest is the tallest mountain in the world."},
            {"id": "3", "text": "Python is a popular programming language."},
        ]
        rag.add_documents(docs)
        results = rag.query("What city is the Eiffel Tower located in?", top_k=1)
        self.assertTrue(results)
        self.assertIn("Eiffel Tower", results[0]["text"])
        self.assertIn("Paris", results[0]["text"])


def pytest_configure(config: pytest.Config) -> None:
    model = config.getoption("--model", default=DEFAULT_MODEL)
    TestAModel.model_name = model


if __name__ == "__main__":
    import sys

    model = DEFAULT_MODEL
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]
            sys.argv.pop(i)
            break
        elif arg == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
            sys.argv.pop(i + 1)
            sys.argv.pop(i)
            break

    TestAModel.model_name = model
    print(f"Testing model: {model}")
    unittest.main(verbosity=2)
