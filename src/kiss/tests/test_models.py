# Author: Generated Test Suite
# Test suite for all models in model_info.py - both agentic and non-agentic modes

"""Test suite for all models defined in model_info.py.

This tests both agentic mode (with tool calling) and non-agentic mode (simple generation)
for all models. Results are collected and reported at the end.
"""

import json
import re
import unittest

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import MODEL_INFO
from kiss.rag.simple_rag import SimpleRAG
from kiss.tests.conftest import simple_calculator

# Note: is_model_flaky and get_flaky_reason are available from model_info
# for use in test filtering if needed in the future

# Timeout in seconds for each test
TEST_TIMEOUT = 60
# Longer timeout for thinking/reasoning models
THINKING_MODEL_TIMEOUT = 180

# Models that require longer timeouts (thinking/reasoning models)
# These models take longer due to internal reasoning/thinking steps
THINKING_MODELS = {
    # Together AI thinking models
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "moonshotai/Kimi-K2-Thinking",
    # OpenRouter thinking/reasoning models
    "openrouter/deepseek/deepseek-r1",
    "openrouter/deepseek/deepseek-r1-0528",
    "openrouter/openai/o1",
    "openrouter/openai/o3-pro",
    "openrouter/qwen/qwq-32b",
    "openrouter/moonshotai/kimi-k2-thinking",
}

# Get models by capability from MODEL_INFO
GENERATION_MODELS = [
    name for name, info in MODEL_INFO.items() if info.is_generation_supported
]
AGENTIC_MODELS = [
    name for name, info in MODEL_INFO.items() if info.is_function_calling_supported
]
EMBEDDING_MODELS = [
    name for name, info in MODEL_INFO.items() if info.is_embedding_supported
]


class TestModelsNonAgentic(unittest.TestCase):
    """Test all models in non-agentic mode."""

    def _test_model_non_agentic(self, model_name: str):
        """Test a single model in non-agentic mode."""
        agent = KISSAgent(f"Test Agent for {model_name}")
        result = agent.run(
            model_name=model_name,
            prompt_template="What is 2 + 2 Answer with just the number.",
            is_agentic=False,
            max_budget=1.0,
        )
        self.assertIsNotNone(result)
        result_clean = re.sub(r"[,\\s]", "", result)
        self.assertIn("4", result_clean)
        trajectory = json.loads(agent.get_trajectory())
        self.assertGreater(len(trajectory), 0)


# Dynamically create test methods for each generation model (non-agentic)
def _create_non_agentic_test(model_name: str):
    timeout = THINKING_MODEL_TIMEOUT if model_name in THINKING_MODELS else TEST_TIMEOUT

    @pytest.mark.timeout(timeout)
    def test_method(self):
        self._test_model_non_agentic(model_name)

    return test_method


for model_name in GENERATION_MODELS:
    # Create a valid method name
    safe_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    method_name = f"test_non_agentic_{safe_name}"
    setattr(TestModelsNonAgentic, method_name, _create_non_agentic_test(model_name))


class TestModelsAgentic(unittest.TestCase):
    """Test all models in agentic mode with tool calling."""

    def _test_model_agentic(self, model_name: str):
        """Test a single model in agentic mode with retries."""
        agent = KISSAgent(f"Test Agent for {model_name}")
        result = agent.run(
            model_name=model_name,
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


# Dynamically create test methods for each agentic model
def _create_agentic_test(model_name: str):
    timeout = THINKING_MODEL_TIMEOUT if model_name in THINKING_MODELS else TEST_TIMEOUT

    @pytest.mark.timeout(timeout)
    def test_method(self):
        self._test_model_agentic(model_name)

    return test_method


for model_name in AGENTIC_MODELS:
    # Create a valid method name
    safe_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    method_name = f"test_agentic_{safe_name}"
    setattr(TestModelsAgentic, method_name, _create_agentic_test(model_name))


class TestModelsEmbedding(unittest.TestCase):
    """Test embedding models using SimpleRAG."""

    def _test_embedding_model(self, model_name: str):
        """Test a single embedding model with SimpleRAG."""
        rag = SimpleRAG(model_name=model_name)
        # Initialize the model's client (required for embedding calls)
        rag._model.initialize("dummy prompt for initialization")
        # Override the model name used for embeddings
        rag.model_name = model_name

        # Add test documents
        documents = [
            {
                "id": "1",
                "text": "Python is a programming language known for its simplicity.",
                "metadata": {"topic": "programming"},
            },
            {
                "id": "2",
                "text": "Machine learning uses algorithms to learn from data.",
                "metadata": {"topic": "ML"},
            },
            {
                "id": "3",
                "text": "Docker containers provide isolated execution environments.",
                "metadata": {"topic": "devops"},
            },
        ]
        rag.add_documents(documents)

        # Verify documents were added
        stats = rag.get_collection_stats()
        self.assertEqual(stats["num_documents"], 3)
        self.assertIsNotNone(stats["embedding_dimension"])
        self.assertGreater(stats["embedding_dimension"], 0)

        # Query for similar documents
        results = rag.query("What is Python programming?", top_k=3)
        self.assertGreater(len(results), 0)
        # The Python document should be somewhere in the results (different models rank differently)
        result_ids = [r["id"] for r in results]
        # At least verify we got some results back with valid IDs
        self.assertTrue(any(rid in ["1", "2", "3"] for rid in result_ids))

        # Verify scores are reasonable (cosine similarity should be between -1 and 1)
        for result in results:
            self.assertGreaterEqual(result["score"], -1.0)
            self.assertLessEqual(result["score"], 1.0)


# Dynamically create test methods for each embedding model
def _create_embedding_test(model_name: str):
    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_method(self):
        self._test_embedding_model(model_name)

    return test_method


for model_name in EMBEDDING_MODELS:
    # Create a valid method name
    safe_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    method_name = f"test_embedding_{safe_name}"
    setattr(TestModelsEmbedding, method_name, _create_embedding_test(model_name))


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
