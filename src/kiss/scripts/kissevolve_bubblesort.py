#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""OpenEvolve test: Evolve bubble sort to discover O(n log n) algorithms.

This script uses OpenEvolve to evolve bubble sort (O(n²)) and attempts to
discover faster sorting algorithms like quicksort or mergesort (O(n log n)).
"""

import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from kiss.agents.kiss import get_run_simple_coding_agent

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kiss.agents.kiss_evolve.kiss_evolve import KISSEvolve

# Initial bubble sort algorithm (O(n²))
INITIAL_CODE = """
def sort_array(arr):
    \"""Sort an array using bubble sort algorithm.\"""
    n = len(arr)
    arr = arr.copy()  # Don't modify original
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""


def evaluate_correctness_of_code(code: str) -> dict:
    """Evaluate the correctness of a sorting code variant by running test cases.

    Args:
        code: The code string to evaluate

    Returns:
        Dict with 'correctness' and 'error'
    """
    # Create a temporary file to execute the code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        # Compile and execute the code
        namespace: dict[str, Any] = {}
        exec(compile(code, temp_file, "exec"), namespace)

        # Check if the function exists
        if "sort_array" not in namespace:
            return {
                "correctness": False,
                "error": "Function 'sort_array' not found",
            }

        func = namespace["sort_array"]

        # Test correctness with multiple test cases
        test_cases = [
            ([3, 1, 2], [1, 2, 3]),
            ([5, 3, 4, 1, 2], [1, 2, 3, 4, 5]),
            ([1], [1]),
            ([], []),
            ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),  # Already sorted
            ([3, 3, 3, 3], [3, 3, 3, 3]),  # All same
            ([-1, -3, -2, 0, 2], [-3, -2, -1, 0, 2]),  # Negative numbers
        ]

        # Verify correctness
        for input_arr, expected in test_cases:
            try:
                result = func(input_arr)
                if result != expected:
                    return {
                        "correctness": False,
                        "error": (
                            f"Incorrect result for {input_arr}: got {result}, expected {expected}"
                        ),
                    }
            except Exception as e:
                return {
                    "correctness": False,
                    "error": f"Exception during execution: {e}",
                }

        return {
            "correctness": True,
            "error": "None",
        }

    except SyntaxError as e:
        return {
            "correctness": False,
            "error": f"Syntax error: {e}",
        }
    except Exception as e:
        return {
            "correctness": False,
            "error": f"Execution error: {e}",
        }
    finally:
        # Clean up temp file
        try:
            Path(temp_file).unlink()
        except Exception:
            pass


def evaluate_performance_of_code(code: str) -> dict:
    """Evaluate the performance of a sorting code variant by measuring execution time.

    Args:
        code: The code string to evaluate

    Returns:
        Dict with 'fitness', 'metrics', 'artifacts', and optionally 'error'
    """
    # Create a temporary file to execute the code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        # Compile and execute the code
        namespace: dict[str, Any] = {}
        exec(compile(code, temp_file, "exec"), namespace)

        # Check if the function exists
        if "sort_array" not in namespace:
            return {
                "fitness": 0.0,
                "metrics": {},
                "artifacts": {},
                "error": "Function 'sort_array' not found",
            }

        func = namespace["sort_array"]

        # Measure performance on different input sizes
        # This helps identify O(n log n) vs O(n²) algorithms
        sizes = [100, 500, 1000, 2000]
        times_by_size = {}
        total_time = 0.0
        num_runs = 3

        for size in sizes:
            # Generate random test data
            test_data = [random.randint(-1000, 1000) for _ in range(size)]

            # Run multiple times and take average
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                func(test_data.copy())
                end = time.perf_counter()
                times.append(end - start)

            avg_time = sum(times) / len(times)
            times_by_size[size] = avg_time
            total_time += avg_time

        # Calculate fitness
        # Higher fitness for faster algorithms
        # Bonus for algorithms that scale better (O(n log n) vs O(n²))
        baseline_time = 1.0  # Normalize
        fitness = baseline_time / total_time if total_time > 0 else 0.0

        # Analyze complexity by looking at time growth
        # O(n log n) should grow slower than O(n²)
        growth_ratio = 0.0
        if len(sizes) >= 2:
            # Compare growth from size 500 to 2000
            if times_by_size[500] > 0:
                growth_ratio = times_by_size[2000] / times_by_size[500]
            else:
                growth_ratio = float("inf")

            # For O(n²), time should grow ~16x (4²)
            # For O(n log n), time should grow ~4.6x (4 * log(4) / log(2) ≈ 4.6)
            # Give bonus if growth is closer to O(n log n)
            if growth_ratio < 10:  # Reasonable growth (not O(n²))
                fitness *= 1.5  # Bonus for better scaling

        # Additional metrics
        metrics = {
            "total_time_seconds": total_time,
            "times_by_size": times_by_size,
            "growth_ratio": growth_ratio,
        }

        artifacts = {
            "test_sizes": sizes,
            "performance_data": times_by_size,
        }

        return {
            "fitness": fitness,
            "metrics": metrics,
            "artifacts": artifacts,
        }

    except SyntaxError as e:
        return {"fitness": 0.0, "metrics": {}, "artifacts": {}, "error": f"Syntax error: {e}"}
    except Exception as e:
        return {"fitness": 0.0, "metrics": {}, "artifacts": {}, "error": f"Execution error: {e}"}
    finally:
        # Clean up temp file
        try:
            Path(temp_file).unlink()
        except Exception:
            pass


def analyze_complexity(metrics: dict) -> str:
    """Analyze the complexity of the algorithm based on performance metrics."""
    if not metrics or "times_by_size" not in metrics:
        return "Unknown"

    times = metrics["times_by_size"]
    if not isinstance(times, dict) or len(times) < 2:
        return "Unknown"

    # Compare growth rates
    sizes = sorted(times.keys())
    if len(sizes) >= 2:
        # Compare last two sizes
        size1, size2 = sizes[-2], sizes[-1]
        time1, time2 = times[size1], times[size2]

        size_ratio = size2 / size1
        time_ratio = time2 / time1 if time1 > 0 else float("inf")

        # O(n²) would have time_ratio ≈ size_ratio²
        # O(n log n) would have time_ratio ≈ size_ratio * log(size_ratio) / log(2)
        import math

        expected_n2 = size_ratio**2
        expected_nlogn = size_ratio * math.log2(size_ratio) if size_ratio > 0 else 0

        if abs(time_ratio - expected_nlogn) < abs(time_ratio - expected_n2):
            return "O(n log n)"
        elif abs(time_ratio - expected_n2) < abs(time_ratio - expected_nlogn):
            return "O(n²)"
        else:
            return f"O(n^?) - ratio: {time_ratio:.2f}"

    return "Unknown"


def main() -> None:
    """Run OpenEvolve optimization on bubble sort."""
    print("=" * 80)
    print("OpenEvolve Test: Evolving Bubble Sort to O(n log n)")
    print("=" * 80)
    print("\nInitial code (Bubble Sort - O(n²)):")
    print(INITIAL_CODE)

    # Evaluate initial code
    print("\nEvaluating initial bubble sort...")
    initial_result = evaluate_performance_of_code(INITIAL_CODE)
    print(f"Initial fitness: {initial_result['fitness']:.4f}")
    if initial_result["metrics"]:
        print("Initial metrics:")
        for key, value in initial_result["metrics"].items():
            if key == "times_by_size":
                print("  Performance by size:")
                for size, t in value.items():
                    print(f"    Size {size}: {t * 1000:.4f} ms")
            else:
                print(f"  {key}: {value}")

    initial_complexity = analyze_complexity(initial_result.get("metrics", {}))
    print(f"Initial complexity: {initial_complexity}")

    # Create OpenEvolve optimizer
    print("\n" + "=" * 80)
    print("Starting OpenEvolve optimization...")
    print("=" * 80)
    print("Goal: Discover O(n log n) sorting algorithm (quicksort, mergesort, etc.)")

    # Wrapper function to convert evaluate_correctness_of_code return value to bool
    def test_correctness(code: str) -> bool:
        """Test if code is correct, returning a boolean."""
        result = evaluate_correctness_of_code(code)
        correctness = result.get("correctness", False)
        return bool(correctness)

    optimizer = KISSEvolve(
        code_agent_wrapper=get_run_simple_coding_agent(test_correctness),
        initial_code=INITIAL_CODE,
        evaluation_fn=evaluate_performance_of_code,
        model_names=[("gpt-4o", 1.0)],
        extra_coding_instructions="""
   - You **MUST NOT** use any builtin or standard library functions to sort.
   - You **MUST** implement your best algorithm to sort the array.
""",
        population_size=6,
        max_generations=8,  # More generations to allow discovery
        mutation_rate=0.7,
        elite_size=1,
    )

    # Run evolution
    best_variant = optimizer.evolve()

    # Report results
    print("\n" + "=" * 80)
    print("Evolution Complete!")
    print("=" * 80)
    print(f"\nBest fitness: {best_variant.fitness:.4f}")
    print(f"Generation: {best_variant.generation}")
    if best_variant.metrics:
        print("\nBest variant metrics:")
        for key, value in best_variant.metrics.items():
            if key == "times_by_size" and isinstance(value, dict):
                print("  Performance by size:")
                for size, t in value.items():
                    print(f"    Size {size}: {t * 1000:.4f} ms")
            else:
                print(f"  {key}: {value}")

    best_complexity = analyze_complexity(best_variant.metrics)
    print(f"\nEvolved complexity: {best_complexity}")

    print("\nBest evolved code:")
    print("-" * 80)
    print(best_variant.code)
    print("-" * 80)

    # Compare with initial
    improvement = (
        ((best_variant.fitness - initial_result["fitness"]) / initial_result["fitness"] * 100)
        if initial_result["fitness"] > 0
        else 0
    )

    print(f"\nPerformance improvement: {improvement:.2f}%")

    if best_variant.metrics and "total_time_seconds" in best_variant.metrics:
        initial_time = initial_result["metrics"].get("total_time_seconds", 0)
        evolved_time = best_variant.metrics.get("total_time_seconds", 0)
        if initial_time > 0:
            speedup = initial_time / evolved_time
            print(f"Speedup: {speedup:.2f}x")
            print(f"Initial total time: {initial_time * 1000:.4f} ms")
            print(f"Evolved total time: {evolved_time * 1000:.4f} ms")

    # Check if we discovered O(n log n)
    if "log" in best_complexity.lower() or "nlogn" in best_variant.code.lower():
        print("\n" + "=" * 80)
        print("🎉 SUCCESS: O(n log n) algorithm discovered!")
        print("=" * 80)
    elif best_variant.fitness > initial_result["fitness"] * 2:
        print("\n" + "=" * 80)
        print("✅ Significant improvement achieved!")
        print("=" * 80)

    # Population statistics
    stats = optimizer.get_population_stats()
    print("\nFinal population stats:")
    print(f"  Size: {stats['size']}")
    print(f"  Average fitness: {stats['avg_fitness']:.4f}")
    print(f"  Best fitness: {stats['best_fitness']:.4f}")
    print(f"  Worst fitness: {stats['worst_fitness']:.4f}")


if __name__ == "__main__":
    main()
