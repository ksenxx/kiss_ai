#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""AlgoTune runner using KISSEvolve to optimize algorithm implementations."""

import inspect
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import kiss.agents.kiss_evolve.config  # noqa: F401
import kiss.evals.algotune.config  # noqa: F401
from kiss.agents.kiss import get_run_simple_coding_agent
from kiss.agents.kiss_evolve.kiss_evolve import KISSEvolve
from kiss.core import config as config_module

# Required dependencies for AlgoTune tasks
ALGOTUNE_DEPENDENCIES = ["orjson", "scipy", "scikit-learn"]

# Network-related exceptions that warrant retrying
NETWORK_ERROR_PATTERNS = [
    "Connection reset by peer",
    "ConnectionError",
    "ReadError",
    "TimeoutError",
    "SSLError",
    "RemoteDisconnected",
    "httpx.ReadError",
    "httpcore.ReadError",
]


def _check_and_install_dependencies() -> None:
    """Check for required dependencies and install them if missing.

    Attempts to import required packages (orjson, scipy, scikit-learn) and
    installs any missing ones using uv pip (preferred) or pip fallback.

    Raises:
        RuntimeError: If package installation fails.
    """
    missing = []

    # Map package names to their import names
    import_map = {
        "orjson": "orjson",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
    }

    for package in ALGOTUNE_DEPENDENCIES:
        import_name = import_map.get(package, package)
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}...")
        try:
            # Try using uv first (faster), fall back to pip
            try:
                subprocess.run(
                    ["uv", "pip", "install"] + missing,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            print(f"Successfully installed: {', '.join(missing)}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to install dependencies {missing}: {e.stderr}") from e


def _is_network_error(error: Exception) -> bool:
    """Check if an exception is a network-related error that should be retried.

    Args:
        error: The exception to check.

    Returns:
        True if the error matches known network error patterns, False otherwise.
    """
    error_str = str(error) + str(type(error).__name__)
    return any(pattern in error_str for pattern in NETWORK_ERROR_PATTERNS)


def _ensure_algotune_installed(path: Path, repo_url: str) -> None:
    """Clone AlgoTune repository if it doesn't exist, then add to sys.path.

    Args:
        path: Local path where AlgoTune should be installed.
        repo_url: Git URL for cloning the AlgoTune repository.

    Raises:
        RuntimeError: If git clone fails or git is not installed.
    """
    if not path.exists():
        print(f"AlgoTune not found at {path}, cloning from {repo_url}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"Successfully cloned AlgoTune to {path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone AlgoTune: {e.stderr}") from e
        except FileNotFoundError:
            raise RuntimeError("Git not found. Please install git.")

    # Add to sys.path if not already present
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def get_all_task_names(algotune_path: Path) -> list[str]:
    """Get all available task names from the AlgoTune repository.

    Args:
        algotune_path: Path to the AlgoTune repository root.

    Returns:
        Sorted list of valid task names (directories with matching .py files).
    """
    tasks_dir = algotune_path / "AlgoTuneTasks"
    if not tasks_dir.exists():
        return []

    task_names = []
    for item in sorted(tasks_dir.iterdir()):
        # Skip non-directories and special files
        if not item.is_dir() or item.name.startswith("_"):
            continue
        # Check if it has a matching .py file (valid task)
        if (item / f"{item.name}.py").exists():
            task_names.append(item.name)

    return task_names


def get_task_class(task_name: str):
    """Dynamically import and return a task class from AlgoTune.

    Args:
        task_name: Name of the AlgoTune task to load.

    Returns:
        The task class from the AlgoTune registry.

    Raises:
        ValueError: If task_name is not found in the AlgoTune registry.
    """
    module_name = f"AlgoTuneTasks.{task_name}.{task_name}"
    __import__(module_name, fromlist=[task_name])

    from AlgoTuneTasks.base import TASK_REGISTRY  # type: ignore[import-not-found]

    if task_name in TASK_REGISTRY:
        return TASK_REGISTRY[task_name]
    raise ValueError(f"Task '{task_name}' not found in AlgoTune registry")


def _extract_solve_body(solve_source: str) -> str:
    """Extract the body of a solve method, skipping docstrings.

    Args:
        solve_source: Source code string containing the solve method.

    Returns:
        The body of the solve method with proper indentation for embedding
        in _solve_internal, with docstrings removed.
    """
    lines = solve_source.split("\n")
    body_lines = []
    in_body = False
    in_docstring = False
    docstring_char = None
    base_indent = None

    for line in lines:
        stripped = line.strip()

        # Skip until we find 'def solve'
        if not in_body:
            if stripped.startswith("def solve"):
                in_body = True
            continue

        # Handle docstrings
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                if stripped.count(docstring_char) >= 2:
                    continue  # Single-line docstring
                in_docstring = True
                continue
        else:
            if docstring_char and docstring_char in stripped:
                in_docstring = False
            continue

        # Skip leading empty lines
        if base_indent is None:
            if not stripped:
                continue
            base_indent = len(line) - len(line.lstrip())

        # Re-indent for embedding in _solve_internal
        if stripped:
            original_indent = len(line) - len(line.lstrip())
            relative_indent = max(0, original_indent - base_indent)
            body_lines.append("        " + " " * relative_indent + stripped)
        else:
            body_lines.append("")

    return "\n".join(body_lines)


def _extract_imports(task_file: Path) -> str:
    """Extract import statements from a task file.

    Args:
        task_file: Path to the task's Python file.

    Returns:
        String containing all import statements from the file, excluding
        AlgoTune internal imports. Always includes 'import numpy as np'.
    """
    if not task_file.exists():
        return "import numpy as np"

    imports = []
    content = task_file.read_text()

    for line in content.split("\n"):
        stripped = line.strip()
        # Capture import and from...import statements
        if stripped.startswith("import ") or stripped.startswith("from "):
            # Skip AlgoTune internal imports
            if "AlgoTuneTasks" in stripped or "AlgoTuner" in stripped:
                continue
            imports.append(stripped)

    # Ensure numpy is always available (commonly needed)
    if not any("numpy" in imp for imp in imports):
        imports.insert(0, "import numpy as np")

    return "\n".join(imports)


def _create_initial_code(
    task_name: str, description: str, solve_source: str, task_file: Path
) -> str:
    """Create initial solver code from reference implementation.

    Args:
        task_name: Name of the AlgoTune task.
        description: Task description text.
        solve_source: Source code of the reference solve method.
        task_file: Path to the task's Python file for extracting imports.

    Returns:
        Complete solver code string with imports, Solver class, and solve method.
    """
    body = _extract_solve_body(solve_source)
    imports = _extract_imports(task_file)

    return f'''"""Solver for {task_name} AlgoTune task."""
{imports}

class Solver:
    """Solver for {task_name}."""

    def solve(self, problem: dict):
        """Solve the problem.

        Task: {description[:500]}...
        """
        result = self._solve_internal(problem)
        if isinstance(result, np.ndarray):
            return result.tolist()
        return result

    def _solve_internal(self, problem):
        """Internal solve implementation."""
{body}
'''


def _execute_code(code: str) -> dict[str, Any] | None:
    """Execute code and return namespace, or None on failure.

    Args:
        code: Python code string to execute.

    Returns:
        The execution namespace dictionary if successful, None on any exception.
    """
    try:
        namespace: dict[str, Any] = {}
        exec(compile(code, "<solver>", "exec"), namespace)
        return namespace
    except Exception:
        return None


def create_evaluation_fn(task_instance, test_problems: list, num_timing_runs: int = 5):
    """Create an evaluation function for KISSEvolve.

    Args:
        task_instance: AlgoTune task instance with is_solution method.
        test_problems: List of test problems to evaluate against.
        num_timing_runs: Number of timing runs for performance measurement.

    Returns:
        Evaluation function that takes code string and returns dict with
        fitness, metrics, artifacts, and optional error.
    """

    def evaluate(code: str) -> dict[str, Any]:
        """Evaluate solver code for correctness and performance.

        Args:
            code: Python code string containing Solver class.

        Returns:
            Dictionary with fitness (1/total_time), metrics (total_time_seconds),
            artifacts, and error (if any).
        """
        namespace = _execute_code(code)
        if namespace is None or "Solver" not in namespace:
            return {"fitness": 0.0, "metrics": {}, "artifacts": {}, "error": "Invalid code"}

        try:
            solver = namespace["Solver"]()
        except Exception as e:
            return {"fitness": 0.0, "metrics": {}, "artifacts": {}, "error": str(e)}

        # Test correctness
        for i, problem in enumerate(test_problems):
            try:
                solution = solver.solve(problem)
                if not task_instance.is_solution(problem, solution):
                    return {
                        "fitness": 0.0,
                        "metrics": {},
                        "artifacts": {},
                        "error": f"Incorrect on problem {i}",
                    }
            except Exception as e:
                return {
                    "fitness": 0.0,
                    "metrics": {},
                    "artifacts": {},
                    "error": f"Error on problem {i}: {e}",
                }

        # Measure performance
        total_time = 0.0
        for problem in test_problems:
            times = []
            for _ in range(num_timing_runs):
                start = time.perf_counter()
                solver.solve(problem)
                times.append(time.perf_counter() - start)
            total_time += sum(times) / len(times)

        return {
            "fitness": 1.0 / total_time if total_time > 0 else 0.0,
            "metrics": {"total_time_seconds": total_time},
            "artifacts": {},
        }

    return evaluate


def create_correctness_test_fn(task_instance, test_problems: list):
    """Create a correctness test function for the coding agent.

    Args:
        task_instance: AlgoTune task instance with is_solution method.
        test_problems: List of test problems to validate against.

    Returns:
        Test function that takes code string and returns True if all
        problems are solved correctly, False otherwise.
    """

    def test(code: str) -> bool:
        """Test solver code for correctness on all problems.

        Args:
            code: Python code string containing Solver class.

        Returns:
            True if solver produces correct solutions for all test problems,
            False on any error or incorrect solution.
        """
        namespace = _execute_code(code)
        if namespace is None or "Solver" not in namespace:
            return False

        try:
            solver = namespace["Solver"]()
            for problem in test_problems:
                solution = solver.solve(problem)
                if not task_instance.is_solution(problem, solution):
                    return False
            return True
        except Exception:
            return False

    return test


def run_algotune(
    max_retries: int = 3,
    **kwargs,
) -> dict[str, Any]:
    """Run KISSEvolve on an AlgoTune task.

    Args:
        max_retries: Maximum number of retries for network errors.
        **kwargs: Override config parameters.

    Returns:
        Dictionary with optimization results or error information.
    """

    _check_and_install_dependencies()

    algotune_cfg = config_module.DEFAULT_CONFIG.algotune  # type: ignore[attr-defined]
    algotune_path = Path(algotune_cfg.algotune_path)
    _ensure_algotune_installed(algotune_path, algotune_cfg.algotune_repo_url)

    print("=" * 80)
    print(f"AlgoTune Optimization: {algotune_cfg.task}")
    print("=" * 80)

    # Load task
    print(f"\nLoading task: {algotune_cfg.task}...")
    task_class = get_task_class(algotune_cfg.task)
    task_instance = task_class()

    # Load description
    desc_path = algotune_path / "AlgoTuneTasks" / algotune_cfg.task / "description.txt"
    description = desc_path.read_text() if desc_path.exists() else algotune_cfg.task
    description = description.replace("{", "{{").replace("}", "}}")

    # Generate test problems
    num_problems = algotune_cfg.num_test_problems
    problem_size = algotune_cfg.problem_size
    random_seed = algotune_cfg.random_seed
    print(f"Generating {num_problems} test problems (size={problem_size})...")
    test_problems = [
        task_instance.generate_problem(problem_size, random_seed=random_seed + i)
        for i in range(num_problems)
    ]

    # Create initial code
    print("Creating initial solver from reference implementation...")
    solve_source = inspect.getsource(task_instance.solve)
    task_name = algotune_cfg.task
    task_file = algotune_path / "AlgoTuneTasks" / task_name / f"{task_name}.py"
    initial_code = _create_initial_code(task_name, description, solve_source, task_file)

    print("\nInitial code (truncated):")
    print("-" * 40)
    print(initial_code[:800] + "..." if len(initial_code) > 800 else initial_code)
    print("-" * 40)

    # Evaluate reference
    print("\nEvaluating reference implementation...")
    eval_fn = create_evaluation_fn(task_instance, test_problems, algotune_cfg.num_timing_runs)
    initial_result = eval_fn(initial_code)

    if initial_result.get("error"):
        print(f"Error in reference: {initial_result['error']}")
        return {"error": initial_result["error"]}

    initial_time = initial_result["metrics"]["total_time_seconds"]
    print(f"Reference fitness: {initial_result['fitness']:.4f}")
    print(f"Reference time: {initial_time * 1000:.2f} ms")

    # Create optimizer
    print(f"\nInitializing model: {algotune_cfg.model}...")
    test_fn = create_correctness_test_fn(task_instance, test_problems)

    extra_instructions = f"""
## Task-Specific Instructions ##
- Optimizing solver for '{task_name}' AlgoTune benchmark.
- Output format must match reference exactly.
- Focus on ALGORITHMIC optimizations (better data structures, faster algorithms).
- Use numpy, scipy, and standard Python libraries.
- Goal: achieve SPEEDUP over reference implementation.

## Task Description ##
{description[:1500]}
"""

    evolve_config = config_module.DEFAULT_CONFIG.kiss_evolve  # type: ignore[attr-defined]
    pop_size = evolve_config.population_size
    max_gen = evolve_config.max_generations
    print(f"\nStarting KISSEvolve (pop={pop_size}, gen={max_gen})...")

    optimizer = KISSEvolve(
        code_agent_wrapper=get_run_simple_coding_agent(test_fn),
        initial_code=initial_code,
        evaluation_fn=eval_fn,
        model_names=[(algotune_cfg.model, 1.0)],
        extra_coding_instructions=extra_instructions,
    )

    # Run evolution with retry logic for network errors
    last_error = None
    for attempt in range(max_retries):
        try:
            best = optimizer.evolve()
            break  # Success, exit retry loop
        except Exception as e:
            last_error = e
            if _is_network_error(e) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                print(f"\n⚠️  Network error encountered: {type(e).__name__}")
                print(f"   Retrying in {wait_time}s... (attempt {attempt + 2}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise
    else:
        # All retries exhausted
        raise RuntimeError(
            f"Evolution failed after {max_retries} attempts due to network errors"
        ) from last_error

    # Results
    if best.metrics:
        evolved_time = best.metrics.get("total_time_seconds", initial_time)
    else:
        evolved_time = initial_time
    # Ensure evolved_time is a float for calculations
    evolved_time = float(evolved_time or initial_time)
    speedup = initial_time / evolved_time if evolved_time > 0 else 1.0

    print("\n" + "=" * 80)
    print("Evolution Complete!")
    print("=" * 80)
    print(f"\nBest fitness: {best.fitness:.4f} (generation {best.generation})")
    print(f"\n{'=' * 40}")
    print(f"SPEEDUP: {speedup:.2f}x")
    print(f"Reference: {initial_time * 1000:.2f} ms → Optimized: {evolved_time * 1000:.2f} ms")
    print(f"{'=' * 40}")
    print("\nBest evolved code:")
    print("-" * 80)
    print(best.code)
    print("-" * 80)

    return {
        "task_name": config_module.DEFAULT_CONFIG.algotune.task,  # type: ignore[attr-defined]
        "initial_fitness": initial_result["fitness"],
        "best_fitness": best.fitness,
        "initial_time": initial_time,
        "evolved_time": evolved_time,
        "speedup": speedup,
        "best_code": best.code,
        "generation": best.generation,
    }


def run_all_tasks(max_retries: int = 3) -> list[dict[str, Any]]:
    """Run KISSEvolve on all AlgoTune tasks.

    Args:
        max_retries: Maximum number of retries for network errors per task.

    Returns:
        List of results for each task.
    """
    # Check and install required dependencies before starting
    _check_and_install_dependencies()

    algotune_cfg = config_module.DEFAULT_CONFIG.algotune  # type: ignore[attr-defined]
    algotune_path = Path(algotune_cfg.algotune_path)
    _ensure_algotune_installed(algotune_path, algotune_cfg.algotune_repo_url)

    task_names = get_all_task_names(algotune_path)
    if not task_names:
        print(f"No tasks found in {algotune_path}/AlgoTuneTasks")
        return []

    print("=" * 80)
    print(f"Running AlgoTune optimization on {len(task_names)} tasks")
    print("=" * 80)
    print(f"Tasks: {', '.join(task_names[:10])}{'...' if len(task_names) > 10 else ''}")
    print()

    results = []
    successful = 0
    failed = 0

    for i, task_name in enumerate(task_names, 1):
        print(f"\n[{i}/{len(task_names)}] Processing: {task_name}")
        print("-" * 40)

        try:
            result = run_algotune(max_retries=max_retries, task=task_name)
            results.append(result)

            if "error" in result:
                failed += 1
                print(f"  ❌ Failed: {result['error']}")
            else:
                successful += 1
                print(f"  ✅ Speedup: {result['speedup']:.2f}x")
        except Exception as e:
            failed += 1
            error_type = type(e).__name__
            results.append({"task_name": task_name, "error": f"{error_type}: {e}"})
            print(f"  ❌ Exception ({error_type}): {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tasks: {len(task_names)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful > 0:
        speedups = [r["speedup"] for r in results if "speedup" in r]
        print("\nSpeedup statistics:")
        print(f"  Min: {min(speedups):.2f}x")
        print(f"  Max: {max(speedups):.2f}x")
        print(f"  Avg: {sum(speedups) / len(speedups):.2f}x")

        print("\nTop 5 speedups:")
        sorted_results = sorted(
            [r for r in results if "speedup" in r],
            key=lambda x: x["speedup"],
            reverse=True,
        )
        for r in sorted_results[:5]:
            print(f"  {r['task_name']}: {r['speedup']:.2f}x")

    return results


def main():
    """Main entry point.

    Reads configuration from DEFAULT_CONFIG.algotune and runs either
    all tasks (if all_tasks is True) or a single task optimization.
    """
    config = config_module.DEFAULT_CONFIG.algotune  # type: ignore[attr-defined]

    if config.all_tasks:
        results = run_all_tasks()
        successful = sum(1 for r in results if "error" not in r)
        print(f"\n✅ Completed {successful}/{len(results)} tasks successfully")
    else:
        result = run_algotune()
        if "error" not in result:
            print(f"\n✅ Optimization complete! Speedup: {result['speedup']:.2f}x")
        else:
            print(f"\n❌ Optimization failed: {result['error']}")


if __name__ == "__main__":
    main()
