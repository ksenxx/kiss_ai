# Author: Generated Test Suite
# Quick test suite for selected reliable models from each LLM provider

"""Quick test suite for selected reliable models from each provider.

This tests 2 reliable and fast models from each LLM provider in both
agentic mode (with tool calling) and non-agentic mode (simple generation).
"""

import json
import re
import unittest

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import MODEL_INFO
from kiss.rag.simple_rag import SimpleRAG
from kiss.tests.conftest import simple_calculator

# Timeout in seconds for each test
TEST_TIMEOUT = 60

# =============================================================================
# Selected reliable and fast models - 1 per provider for quick testing
# Selection criteria:
# - Fastest model from each provider
# - Reliable function calling (fc=True)
# - Diverse provider coverage
# =============================================================================

# =============================================================================
# Selected reliable and fast models - 1 per provider for quick testing
# Note: Only OpenAI-compatible models are supported (OpenAI, Together AI, OpenRouter)
# =============================================================================
ALL_GENERATION_MODELS = [
    "gpt-4.1-mini",  # OpenAI - fast, reliable
    "Qwen/Qwen3-Next-80B-A3B-Instruct",  # Together AI Qwen - fast MoE
    "zai-org/GLM-4.6",  # Together AI GLM - fast, reliable
    "claude-haiku-4-5",  # Anthropic - fast
    "openrouter/google/gemini-2.5-flash",  # OpenRouter Gemini - fast
    "openrouter/amazon/nova-lite-v1",  # OpenRouter Nova - fast, affordable
    "openrouter/minimax/minimax-m2.1",  # OpenRouter MiniMax - latest, fast
    "openrouter/bytedance-seed/seed-1.6",  # OpenRouter ByteDance - reliable
]

# Embedding models - fastest from each provider
EMBEDDING_MODELS = [
    "text-embedding-3-small",  # OpenAI (fastest)
]

# Filter to only models that support function calling for agentic tests
AGENTIC_MODELS = [m for m in ALL_GENERATION_MODELS if MODEL_INFO[m].is_function_calling_supported]


class TestModelsQuickNonAgentic(unittest.TestCase):
    """Test selected models in non-agentic mode."""

    def _test_model_non_agentic(self, model_name: str):
        """Test a single model in non-agentic mode."""
        agent = KISSAgent(f"Test Agent for {model_name}")
        result = agent.run(
            model_name=model_name,
            prompt_template="What is 2 + 2? Answer with just the number.",
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
    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_method(self):
        self._test_model_non_agentic(model_name)

    return test_method


for model_name in ALL_GENERATION_MODELS:
    # Create a valid method name
    safe_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    method_name = f"test_non_agentic_{safe_name}"
    setattr(TestModelsQuickNonAgentic, method_name, _create_non_agentic_test(model_name))


class TestModelsQuickAgentic(unittest.TestCase):
    """Test selected models in agentic mode with tool calling."""

    def _test_model_agentic(self, model_name: str):
        """Test a single model in agentic mode."""
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
    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_method(self):
        self._test_model_agentic(model_name)

    return test_method


for model_name in AGENTIC_MODELS:
    # Create a valid method name
    safe_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    method_name = f"test_agentic_{safe_name}"
    setattr(TestModelsQuickAgentic, method_name, _create_agentic_test(model_name))


class TestModelsQuickEmbedding(unittest.TestCase):
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
        # At least verify we got some results back with valid IDs
        result_ids = [r["id"] for r in results]
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
    setattr(TestModelsQuickEmbedding, method_name, _create_embedding_test(model_name))


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
