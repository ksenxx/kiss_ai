# Author: Koushik Sen (ksen@berkeley.edu)
# Script to test all models from model_info.py

"""Run tests on all models from model_info.py and report failures.

This script runs test_a_model.py on each model in model_info.py,
testing non-agentic, agentic, and embedding modes as appropriate.
"""

import subprocess
import sys
from dataclasses import dataclass

from kiss.core.models.model_info import MODEL_INFO


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
        sys.executable, "-m", "pytest",
        "src/kiss/tests/test_a_model.py",
        f"--model={model_name}",
        f"-k={test_name}",
        "-v",
        f"--timeout={timeout}",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,  # Extra buffer for pytest overhead
        )
        passed = result.returncode == 0
        error_message = ""
        if not passed:
            # Extract relevant error info
            output = result.stdout + result.stderr
            # Look for FAILED or ERROR lines
            lines = output.split('\n')
            error_lines = [
                line for line in lines
                if 'FAILED' in line or 'ERROR' in line or 'AssertionError' in line
            ]
            if error_lines:
                error_message = '\n'.join(error_lines[:5])  # First 5 error lines
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


def main():
    """Main entry point."""
    all_models = list(MODEL_INFO.keys())
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
        print("\nAll tests passed! 🎉")

    # Return exit code
    return 1 if failed_models else 0


if __name__ == "__main__":
    sys.exit(main())
