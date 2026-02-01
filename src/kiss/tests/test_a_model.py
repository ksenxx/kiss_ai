# Author: Koushik Sen (ksen@berkeley.edu)
# Test suite for OpenAICompatibleModel

"""Test suite for OpenAICompatibleModel with configurable model.

Usage:
    pytest src/kiss/tests/test_openai_compatible.py --model=gpt-5.2
    pytest src/kiss/tests/test_openai_compatible.py --model=openrouter/openai/gpt-4o
"""

import json
import re
import unittest

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.rag.simple_rag import SimpleRAG
from kiss.tests.conftest import DEFAULT_MODEL, simple_calculator

TEST_TIMEOUT = 60


@pytest.fixture
def model_name(request):
    """Get the model name from command line or use default."""
    return request.config.getoption("--model")


class TestAModel(unittest.TestCase):
    """Test a model with configurable model."""

    model_name = DEFAULT_MODEL

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_non_agentic(self):
        """Test non-agentic mode."""
        agent = KISSAgent(f"Test Agent for {self.model_name}")
        result = agent.run(
            model_name=self.model_name,
            prompt_template="What is 2 + 2? Answer with just the number.",
            is_agentic=False,
            max_budget=1.0,
        )
        self.assertIsNotNone(result)
        result_clean = re.sub(r"[,\\s]", "", result)
        self.assertIn("4", result_clean)
        trajectory = json.loads(agent.get_trajectory())
        self.assertGreater(len(trajectory), 0)

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_agentic(self):
        """Test agentic mode with tool calling."""
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
        result_clean = re.sub(r"[,\\s]", "", result)
        self.assertIn("25854996", result_clean)
        trajectory = json.loads(agent.get_trajectory())
        self.assertGreaterEqual(len(trajectory), 5)

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_embedding(self):
        """Test embedding mode."""

        # Test that embedding works with SimpleRAG using this model
        from kiss.core.models.model_info import MODEL_INFO

        is_embedding = (
            self.model_name in MODEL_INFO and MODEL_INFO[self.model_name].is_embedding_supported
        )
        if not is_embedding:
            self.skipTest(f"{self.model_name} does not support embedding")
        rag = SimpleRAG(model_name=self.model_name)
        # Add documents to the RAG index
        docs = [
            {"id": "1", "text": "The Eiffel Tower is in Paris."},
            {"id": "2", "text": "Mount Everest is the tallest mountain in the world."},
            {"id": "3", "text": "Python is a popular programming language."},
        ]
        rag.add_documents(docs)
        # Query something related to one of the docs
        query = "What city is the Eiffel Tower located in?"
        results = rag.query(query, top_k=1)
        self.assertTrue(results)
        # best match should be the Eiffel Tower fact
        self.assertIn("Eiffel Tower", results[0]["text"])
        self.assertIn("Paris", results[0]["text"])


def pytest_configure(config):
    """Configure the test class with the model name from command line."""
    model = config.getoption("--model", default=DEFAULT_MODEL)
    TestAModel.model_name = model


if __name__ == "__main__":
    import sys

    # Check for --model argument when running directly
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
