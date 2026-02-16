# Author: Koushik Sen (ksen@berkeley.edu)
# Script to test all models from model_info.py

"""Run tests on all models from model_info.py and report failures.

This script runs test_a_model.py on each model in model_info.py,
testing non-agentic, agentic, and embedding modes as appropriate.

Usage:
    # Test all models (WARNING: This takes a very long time!)
    python src/kiss/tests/run_all_models_test.py

    # Test only Together AI models
    python src/kiss/tests/run_all_models_test.py --provider together

    # Test only OpenRouter models
    python src/kiss/tests/run_all_models_test.py --provider openrouter

    # Test only OpenAI models
    python src/kiss/tests/run_all_models_test.py --provider openai

    # Test only Anthropic models
    python src/kiss/tests/run_all_models_test.py --provider anthropic

    # Test only Gemini models
    python src/kiss/tests/run_all_models_test.py --provider gemini

    # Test a specific model
    python src/kiss/tests/run_all_models_test.py --model "openrouter/z-ai/glm-4.7-flash"

    # Skip slow reasoning models
    python src/kiss/tests/run_all_models_test.py --skip-slow
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass

from kiss.core.models.model_info import MODEL_INFO

# Models known to be slow (reasoning models, thinking models)
SLOW_MODELS = {
    # OpenAI reasoning models
    "o1",
    "o1-mini",
    "o1-pro",
    "o3-pro",
    # Together AI slow models
    "Qwen/QwQ-32B",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "moonshotai/Kimi-K2-Thinking",
    "deepseek-ai/DeepSeek-R1",
    # OpenRouter slow models - DeepSeek
    "openrouter/deepseek/deepseek-r1",
    "openrouter/deepseek/deepseek-r1-0528",
    # OpenRouter slow models - OpenAI
    "openrouter/openai/o1",
    "openrouter/openai/o1-pro",
    "openrouter/openai/o3-pro",
    # OpenRouter slow models - Qwen thinking models
    "openrouter/qwen/qwq-32b",
    "openrouter/qwen/qwen-plus-2025-07-28:thinking",
    "openrouter/qwen/qwen3-235b-a22b-thinking-2507",
    "openrouter/qwen/qwen3-max-thinking",
    "openrouter/qwen/qwen3-30b-a3b-thinking-2507",
    "openrouter/qwen/qwen3-next-80b-a3b-thinking",
    "openrouter/qwen/qwen3-vl-235b-a22b-thinking",
    "openrouter/qwen/qwen3-vl-30b-a3b-thinking",
    "openrouter/qwen/qwen3-vl-8b-thinking",
    # OpenRouter slow models - MoonshotAI
    "openrouter/moonshotai/kimi-k2-thinking",
    # OpenRouter slow models - ByteDance
    "openrouter/bytedance-seed/seed-2.0-thinking",
    # OpenRouter slow models - AllenAI
    "openrouter/allenai/olmo-3-32b-think",
    "openrouter/allenai/olmo-3-7b-think",
    "openrouter/allenai/olmo-3.1-32b-think",
    # OpenRouter slow models - Anthropic thinking
    "openrouter/anthropic/claude-3.7-sonnet:thinking",
    # OpenRouter slow models - Baidu thinking
    "openrouter/baidu/ernie-4.5-21b-a3b-thinking",
}

# Models known to have issues (503 errors, empty responses, etc.)
SKIP_MODELS = {
    # Models that return empty or have gen=False
    "openrouter/cohere/command-r-plus-08-2024",  # gen=False
    "openrouter/deepseek/deepseek-chat-v3.1",  # gen=False
    "openrouter/deepseek/deepseek-v3.1-terminus",  # gen=False
    "openrouter/inception/mercury",  # Returns empty
    "openrouter/inception/mercury-coder",  # Returns empty
    "openrouter/mistralai/devstral-medium",  # gen=False
    "openrouter/mistralai/devstral-small",  # gen=False
    "openrouter/nousresearch/deephermes-3-mistral-24b-preview",  # gen=False
    "openrouter/meta-llama/llama-3.1-405b-instruct",  # gen=False
    "openrouter/meta-llama/llama-3.3-70b-instruct",  # gen=False
    "openrouter/qwen/qwen3-235b-a22b",  # Returns empty
    # Image generation models (gen=False - can't do text generation)
    "openrouter/google/gemini-2.5-flash-image",  # Image generation
    "openrouter/google/gemini-3-pro-image-preview",  # Image generation
    "openrouter/openai/gpt-5-image",  # Image generation
    "openrouter/openai/gpt-5-image-mini",  # Image generation
    # Audio models (not suitable for text-only testing)
    "openrouter/openai/gpt-audio",  # Audio model
    "openrouter/openai/gpt-audio-mini",  # Audio model
    "openrouter/openai/gpt-4o-audio-preview",  # Audio model
    # Router/meta models
    "openrouter/switchpoint/router",  # Router model
}


@dataclass
class TestResult:
    """Result of a test run."""

    model_name: str
    test_type: str  # "non_agentic", "agentic", "embedding"
    passed: bool
    error_message: str = ""


@dataclass
class ModelTestResults:
    """Results for all tests on a model."""

    model_name: str
    non_agentic: TestResult | None = None
    agentic: TestResult | None = None
    embedding: TestResult | None = None


def run_test(model_name: str, test_name: str, timeout: int = 120) -> TestResult:
    """Run a specific test for a model."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "src/kiss/tests/test_a_model.py",
        f"--model={model_name}",
        f"-k={test_name}",
        "-q",  # Quiet mode for faster output
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = result.returncode == 0
        error_message = ""
        if not passed:
            # Extract relevant error info
            output = result.stdout + result.stderr
            # Look for FAILED or ERROR lines
            lines = output.split("\n")
            error_lines = [
                line
                for line in lines
                if "FAILED" in line or "ERROR" in line or "AssertionError" in line
            ]
            if error_lines:
                error_message = "\n".join(error_lines[:5])  # First 5 error lines
            else:
                error_message = output[-500:] if len(output) > 500 else output

        return TestResult(
            model_name=model_name,
            test_type=test_name,
            passed=passed,
            error_message=error_message,
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            model_name=model_name,
            test_type=test_name,
            passed=False,
            error_message="TIMEOUT",
        )
    except Exception as e:
        return TestResult(
            model_name=model_name,
            test_type=test_name,
            passed=False,
            error_message=str(e),
        )


def test_model(model_name: str) -> ModelTestResults:
    """Run all applicable tests for a model."""
    info = MODEL_INFO.get(model_name)
    if info is None:
        print(f"  WARNING: {model_name} not found in MODEL_INFO")
        return ModelTestResults(model_name=model_name)

    results = ModelTestResults(model_name=model_name)

    # Test non-agentic mode (if generation is supported)
    if info.is_generation_supported:
        print("  Running non-agentic test...")
        results.non_agentic = run_test(model_name, "test_non_agentic")
        status = "✓" if results.non_agentic.passed else "✗"
        print(f"    Non-agentic: {status}")

    # Test agentic mode (if function calling is supported)
    if info.is_function_calling_supported:
        print("  Running agentic test...")
        results.agentic = run_test(model_name, "test_agentic")
        status = "✓" if results.agentic.passed else "✗"
        print(f"    Agentic: {status}")

    # Test embedding mode (if embedding is supported)
    if info.is_embedding_supported:
        print("  Running embedding test...")
        results.embedding = run_test(model_name, "test_embedding")
        status = "✓" if results.embedding.passed else "✗"
        print(f"    Embedding: {status}")

    return results


def get_provider_prefix(provider: str) -> list[str]:
    """Get the model name prefixes for a provider."""
    provider_prefixes = {
        "openai": ["gpt-", "o1", "o3", "text-embedding-"],
        "anthropic": ["claude-"],
        "gemini": ["gemini-", "models/"],
        "together": [
            "meta-llama/",
            "Qwen/",
            "mistralai/",
            "deepseek-ai/",
            "google/",
            "moonshotai/",
            "nvidia/",
            "arcee-ai/",
            "refuel-ai/",
            "marin-community/",
            "essentialai/",
            "zai-org/",
            "deepcogito/",
            "Alibaba-NLP/",
            "BAAI/",
            "togethercomputer/",
            "intfloat/",
        ],
        "openrouter": ["openrouter/"],
    }
    return provider_prefixes.get(provider.lower(), [])


def filter_models_by_provider(models: list[str], provider: str) -> list[str]:
    """Filter models to only include those from a specific provider."""
    prefixes = get_provider_prefix(provider)
    if not prefixes:
        print(f"Unknown provider: {provider}")
        print("Valid providers: openai, anthropic, gemini, together, openrouter")
        return []

    return [m for m in models if any(m.startswith(p) for p in prefixes)]


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test all models from model_info.py")
    parser.add_argument(
        "--provider",
        type=str,
        help="Filter by provider (openai, anthropic, gemini, together, openrouter)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Test a specific model by name",
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow reasoning/thinking models",
    )
    parser.add_argument(
        "--skip-known-issues",
        action="store_true",
        help="Skip models with known issues (503 errors, empty responses)",
    )
    args = parser.parse_args()

    all_models = list(MODEL_INFO.keys())

    # Filter by specific model
    if args.model:
        if args.model in MODEL_INFO:
            all_models = [args.model]
        else:
            print(f"Model '{args.model}' not found in MODEL_INFO")
            return 1

    # Filter by provider
    if args.provider:
        all_models = filter_models_by_provider(all_models, args.provider)
        if not all_models:
            return 1

    # Skip slow models
    if args.skip_slow:
        original_count = len(all_models)
        all_models = [m for m in all_models if m not in SLOW_MODELS]
        skipped = original_count - len(all_models)
        if skipped > 0:
            print(f"Skipping {skipped} slow models")

    # Skip models with known issues
    if args.skip_known_issues:
        original_count = len(all_models)
        all_models = [m for m in all_models if m not in SKIP_MODELS]
        skipped = original_count - len(all_models)
        if skipped > 0:
            print(f"Skipping {skipped} models with known issues")

    print(f"Testing {len(all_models)} models from model_info.py\n")
    print("=" * 80)

    all_results: list[ModelTestResults] = []
    failed_models: dict[str, list[str]] = {}  # model_name -> list of failed test types

    for i, model_name in enumerate(all_models, 1):
        print(f"\n[{i}/{len(all_models)}] Testing: {model_name}")
        results = test_model(model_name)
        all_results.append(results)

        # Track failures
        failures = []
        if results.non_agentic and not results.non_agentic.passed:
            failures.append("non_agentic")
        if results.agentic and not results.agentic.passed:
            failures.append("agentic")
        if results.embedding and not results.embedding.passed:
            failures.append("embedding")

        if failures:
            failed_models[model_name] = failures

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_tests = 0
    passed_tests = 0

    for results in all_results:
        for test_result in [results.non_agentic, results.agentic, results.embedding]:
            if test_result is not None:
                total_tests += 1
                if test_result.passed:
                    passed_tests += 1

    print(f"\nTotal tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")

    if failed_models:
        print(f"\n{'=' * 80}")
        print("FAILED MODELS")
        print("=" * 80)
        for model_name, failures in sorted(failed_models.items()):
            print(f"\n{model_name}:")
            for failure_type in failures:
                # Find the result
                for results in all_results:
                    if results.model_name == model_name:
                        test_result = getattr(results, failure_type)
                        if test_result:
                            print(f"  - {failure_type}: {test_result.error_message[:200]}")
                        break
    else:
        print("\nAll tests passed!")

    # Return exit code
    return 1 if failed_models else 0


if __name__ == "__main__":
    sys.exit(main())
