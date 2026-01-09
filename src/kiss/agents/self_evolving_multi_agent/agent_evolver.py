# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Coding Agent Evolver - Evolves the AdvancedCodingAgent for better performance.

This module evolves the AdvancedCodingAgent to optimize for:
1. Fewer LLM calls (efficiency)
2. Lower token/budget usage
3. Accurate completion of long-horizon coding tasks
"""

from __future__ import annotations

import importlib.resources
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kiss.agents.self_evolving_multi_agent.config  # noqa: F401
from kiss.agents.kiss_evolve.kiss_evolve import CodeVariant, KISSEvolve
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.utils import get_config_value
from kiss.docker.docker_manager import DockerManager


def _load_base_agent_code() -> str:
    """Load the base agent code from multi_agent.py.

    This works for both:
    - Development installations (editable install with `pip install -e .`)
    - Wheel installations (pip install kiss-*.whl)

    Returns:
        The source code of multi_agent.py as a string
    """
    try:
        # Python 3.9+ way using importlib.resources.files
        package_files = importlib.resources.files("kiss.agents.self_evolving_multi_agent")
        agent_file = package_files.joinpath("multi_agent.py")
        return agent_file.read_text(encoding="utf-8")
    except (AttributeError, TypeError):
        # Fallback for older Python versions
        import importlib.resources as resources

        with resources.open_text(
            "kiss.agents.self_evolving_multi_agent", "multi_agent.py"
        ) as f:
            return f.read()


def _create_run_task_wrapper(agent_code: str) -> str:
    """Create a wrapper that adds a run_task function to the agent code.

    The evolver expects a run_task(task, model_name, docker) function.
    This wrapper adds that function using the SelfEvolvingMultiAgent class
    from the loaded code.

    Args:
        agent_code: The source code of self_evolving_multi_agent.py

    Returns:
        Modified code with a run_task function added
    """
    wrapper = '''

# === EVOLVER WRAPPER ===
# This wrapper is added by agent_evolver to provide the run_task interface

def run_task(task: str, model_name: str, docker: "DockerManager") -> dict:
    """Run a self evolving multi agent task using SelfEvolvingMultiAgent.

    This is a wrapper function for the evolver that uses the SelfEvolvingMultiAgent
    class defined above.

    Args:
        task: The task description
        model_name: The LLM model to use
        docker: The Docker manager for execution

    Returns:
        Dictionary with result and metrics
    """
    # Create agent with the provided docker instance
    agent = SelfEvolvingMultiAgent(
        model_name=model_name,
        max_steps=30,
        max_budget=1.5,
    )

    # Override the docker manager
    agent.docker = docker

    # Reset state
    agent.state = AgentState()
    agent.trajectory = []

    # Setup workspace is already done by the evolver
    docker.workdir = "/workspace"

    # Create orchestrator agent
    orchestrator = KISSAgent(name="Multi Agent Orchestrator")

    try:
        result = orchestrator.run(
            model_name=agent.model_name,
            prompt_template=ORCHESTRATOR_PROMPT,
            arguments={
                "task": task,
                "todo_list": agent._format_todo_list(),
                "completed_tasks": agent._format_completed_tasks(),
                "last_error": agent.state.last_error or "None",
            },
            tools=agent._create_tools(),
            max_steps=agent.max_steps,
            max_budget=agent.max_budget,
        )

        # Return metrics for the evolver
        return {
            "result": result,
            "metrics": {
                "llm_calls": len(agent.state.completed_tasks) + 1,  # sub-agents + orchestrator
                "steps": len(agent.state.completed_tasks),
            },
            "stats": agent.get_stats(),
        }

    except Exception as e:
        return {
            "result": str(e),
            "metrics": {"llm_calls": 10, "steps": 0},  # Penalize errors
            "error": str(e),
        }
'''
    return agent_code + wrapper


# Load the base agent code from the actual file
# This ensures the evolver always uses the current implementation
_RAW_AGENT_CODE = _load_base_agent_code()
BASE_AGENT_CODE = _create_run_task_wrapper(_RAW_AGENT_CODE)


@dataclass
class EvaluationTask:
    """A task for evaluating self evolving multi agent performance."""

    name: str
    description: str
    test_script: str  # Python script that returns True if task succeeded
    expected_files: list[str]  # Files that should exist after completion
    timeout: int = 300  # Timeout in seconds
    complexity: str = "simple"  # simple, medium, long_horizon


# Long-horizon evaluation tasks that require multi-step planning and execution
EVALUATION_TASKS = [
    # Simple task for baseline
    EvaluationTask(
        name="fibonacci",
        description="""
        Create a Python script that:
        1. Generates the first 15 Fibonacci numbers
        2. Saves them to 'fibonacci.txt', one number per line
        3. The script should be named 'fib.py'
        """,
        test_script="""
import os
if not os.path.exists('fibonacci.txt'):
    print("FAIL: fibonacci.txt not found")
    exit(1)
with open('fibonacci.txt') as f:
    nums = [int(x.strip()) for x in f.readlines() if x.strip()]
expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
if nums == expected:
    print("PASS")
    exit(0)
else:
    print(f"FAIL: Got {nums}")
    exit(1)
""",
        expected_files=["fibonacci.txt", "fib.py"],
        timeout=120,
        complexity="simple",
    ),
    # Medium complexity - multiple files and data processing
    EvaluationTask(
        name="data_pipeline",
        description="""
        Build a data processing pipeline:
        1. Create 'generator.py' that generates 100 random integers (1-1000)
           and saves to 'raw_data.txt'
        2. Create 'processor.py' that reads 'raw_data.txt', filters numbers > 500,
           sorts them, removes duplicates, and saves to 'processed.txt'
        3. Create 'analyzer.py' that reads 'processed.txt' and writes statistics
           (count, min, max, mean) to 'stats.json'
        4. Run all three scripts in order
        """,
        test_script="""
import os
import json

# Check all files exist
required = [
    'generator.py', 'processor.py', 'analyzer.py',
    'raw_data.txt', 'processed.txt', 'stats.json'
]
for f in required:
    if not os.path.exists(f):
        print(f"FAIL: {f} not found")
        exit(1)

# Verify raw_data.txt has 100 numbers
with open('raw_data.txt') as f:
    raw = [int(x.strip()) for x in f.readlines() if x.strip()]
if len(raw) != 100:
    print(f"FAIL: raw_data.txt should have 100 numbers, got {len(raw)}")
    exit(1)

# Verify processed.txt is filtered, sorted, unique
with open('processed.txt') as f:
    processed = [int(x.strip()) for x in f.readlines() if x.strip()]
if processed != sorted(set([x for x in raw if x > 500])):
    print("FAIL: processed.txt not correctly filtered/sorted/deduped")
    exit(1)

# Verify stats.json
with open('stats.json') as f:
    stats = json.load(f)
expected_keys = ['count', 'min', 'max', 'mean']
for key in expected_keys:
    if key not in stats:
        print(f"FAIL: stats.json missing '{key}'")
        exit(1)
if stats['count'] != len(processed):
    print(f"FAIL: count mismatch")
    exit(1)

print("PASS")
exit(0)
""",
        expected_files=[
            "generator.py", "processor.py", "analyzer.py",
            "raw_data.txt", "processed.txt", "stats.json",
        ],
        timeout=300,
        complexity="medium",
    ),
    # Long-horizon task - full project with testing
    EvaluationTask(
        name="calculator_project",
        description="""
        Build a complete calculator project:
        1. Create 'calculator.py' with a Calculator class that has methods:
           add, subtract, multiply, divide, power, sqrt
           - Division by zero should raise ValueError
           - sqrt of negative should raise ValueError
        2. Create 'test_calculator.py' with unit tests for all Calculator
           methods including edge cases
        3. Create 'main.py' that demonstrates all calculator operations
           and saves results to 'demo_output.txt'
        4. Run the tests and main script
        """,
        test_script="""
import os
import sys

# Check all files exist
required = ['calculator.py', 'test_calculator.py', 'main.py', 'demo_output.txt']
for f in required:
    if not os.path.exists(f):
        print(f"FAIL: {f} not found")
        exit(1)

# Import and test calculator
sys.path.insert(0, '.')
try:
    from calculator import Calculator
    calc = Calculator()

    # Test basic operations
    assert calc.add(2, 3) == 5, "add failed"
    assert calc.subtract(5, 3) == 2, "subtract failed"
    assert calc.multiply(4, 3) == 12, "multiply failed"
    assert calc.divide(10, 2) == 5, "divide failed"
    assert calc.power(2, 3) == 8, "power failed"
    assert abs(calc.sqrt(16) - 4) < 0.001, "sqrt failed"

    # Test error cases
    try:
        calc.divide(1, 0)
        print("FAIL: divide by zero should raise ValueError")
        exit(1)
    except ValueError:
        pass

    try:
        calc.sqrt(-1)
        print("FAIL: sqrt of negative should raise ValueError")
        exit(1)
    except ValueError:
        pass

except Exception as e:
    print(f"FAIL: Calculator error: {e}")
    exit(1)

# Verify demo_output.txt has content
with open('demo_output.txt') as f:
    content = f.read()
if len(content) < 50:
    print("FAIL: demo_output.txt too short")
    exit(1)

print("PASS")
exit(0)
""",
        expected_files=[
            "calculator.py", "test_calculator.py", "main.py", "demo_output.txt"
        ],
        timeout=360,
        complexity="long_horizon",
    ),
    # Long-horizon task - web scraper simulation with multiple components
    EvaluationTask(
        name="text_analyzer_suite",
        description="""
        Build a text analysis suite:
        1. Create 'text_utils.py' with functions: word_count, char_count,
           sentence_count, word_frequency, find_longest_word
        2. Create 'file_handler.py' with functions: read_text_file,
           write_json_file, append_to_file
        3. Create 'analyzer.py' that uses both modules to:
           - Generate a sample text file 'sample.txt' with at least 5 paragraphs
             (each paragraph has 3+ sentences)
           - Analyze the text and save complete analysis to 'analysis.json'
             (word_count, char_count, sentence_count, top_10_words, longest_word)
           - Save a summary report to 'report.txt'
        4. Create 'test_utils.py' with tests for text_utils functions
        5. Run tests and analyzer
        """,
        test_script="""
import os
import json
import sys

# Check all files exist
required = [
    'text_utils.py', 'file_handler.py', 'analyzer.py', 'test_utils.py',
    'sample.txt', 'analysis.json', 'report.txt'
]
for f in required:
    if not os.path.exists(f):
        print(f"FAIL: {f} not found")
        exit(1)

# Check sample.txt has enough content
with open('sample.txt') as f:
    text = f.read()
paragraphs = [p.strip() for p in text.split('\\n\\n') if p.strip()]
if len(paragraphs) < 5:
    print(f"FAIL: sample.txt should have at least 5 paragraphs, got {len(paragraphs)}")
    exit(1)

# Check analysis.json
with open('analysis.json') as f:
    analysis = json.load(f)
required_keys = [
    'word_count', 'char_count', 'sentence_count', 'top_10_words', 'longest_word'
]
for key in required_keys:
    if key not in analysis:
        print(f"FAIL: analysis.json missing '{key}'")
        exit(1)

# Verify word_count is reasonable
if analysis['word_count'] < 100:
    print(f"FAIL: word_count too low ({analysis['word_count']})")
    exit(1)

# Test text_utils module
sys.path.insert(0, '.')
try:
    from text_utils import word_count, char_count, sentence_count
    test_text = "Hello world. This is a test."
    assert word_count(test_text) == 6, "word_count failed"
    assert char_count(test_text) >= 20, "char_count failed"
except Exception as e:
    print(f"FAIL: text_utils error: {e}")
    exit(1)

# Check report.txt has content
with open('report.txt') as f:
    report = f.read()
if len(report) < 100:
    print("FAIL: report.txt too short")
    exit(1)

print("PASS")
exit(0)
""",
        expected_files=[
            "text_utils.py", "file_handler.py", "analyzer.py", "test_utils.py",
            "sample.txt", "analysis.json", "report.txt",
        ],
        timeout=420,
        complexity="long_horizon",
    ),
]




def evaluate_agent_code(
    agent_code: str,
    tasks: list[EvaluationTask],
    model_name: str,
) -> dict[str, Any]:
    """Evaluate agent code on a set of tasks.

    Fitness is computed based on:
    1. Task accuracy (primary - 60% weight)
    2. Efficiency: fewer LLM calls (20% weight)
    3. Speed: faster completion (10% weight)
    4. Complexity bonus: harder tasks worth more (10% weight)

    Args:
        agent_code: The Python code for the agent
        tasks: List of evaluation tasks
        model_name: LLM model to use

    Returns:
        Dictionary with fitness and metrics
    """
    results: dict[str, Any] = {
        "fitness": 0.0,
        "metrics": {
            "tasks_passed": 0,
            "tasks_total": len(tasks),
            "total_time": 0.0,
            "avg_time": 0.0,
            "total_llm_calls": 0,
            "avg_llm_calls": 0.0,
            "efficiency_score": 0.0,
        },
        "artifacts": {},
        "error": None,
    }

    # Create a namespace to exec the agent code
    namespace: dict[str, Any] = {}

    try:
        # Execute the agent code to get the run_task function
        exec(agent_code, namespace)
        run_task_fn = namespace.get("run_task")
        if run_task_fn is None:
            results["error"] = "Agent code does not define run_task function"
            return results
    except Exception as e:
        results["error"] = f"Failed to compile agent code: {e}"
        return results

    passed = 0
    total_time = 0.0
    total_llm_calls = 0
    complexity_score = 0.0
    max_complexity_score = 0.0

    # Complexity weights for different task types
    complexity_weights = {
        "simple": 1.0,
        "medium": 2.0,
        "long_horizon": 3.0,
    }

    for task in tasks:
        task_start = time.time()
        task_passed = False
        task_llm_calls = 0
        task_complexity = complexity_weights.get(task.complexity, 1.0)
        max_complexity_score += task_complexity

        try:
            with DockerManager("python:3.12-slim", workdir="/") as docker:
                # Setup workspace - create directory first, then change workdir
                docker.run_bash_command("mkdir -p /workspace", "Creating workspace")
                docker.workdir = "/workspace"

                # Run the agent and capture metrics
                agent_result = run_task_fn(task.description, model_name, docker)

                # Extract metrics if returned as dict
                if isinstance(agent_result, dict):
                    metrics = agent_result.get("metrics", {})
                    task_llm_calls = metrics.get("llm_calls", 1)
                else:
                    # Estimate LLM calls if not provided (conservative estimate)
                    task_llm_calls = 5

                total_llm_calls += task_llm_calls

                # Run the test script
                docker.run_bash_command(
                    f"cat > /tmp/test.py << 'EOF'\n{task.test_script}\nEOF",
                    "Creating test script",
                )
                test_result = docker.run_bash_command(
                    "python /tmp/test.py",
                    "Running test",
                )

                if "PASS" in test_result:
                    task_passed = True
                    passed += 1
                    complexity_score += task_complexity

        except Exception as e:
            results["artifacts"][task.name] = f"Error: {e}"
            task_llm_calls = 10  # Penalize errors with high LLM call estimate

        task_time = time.time() - task_start
        total_time += task_time
        results["artifacts"][task.name] = {
            "passed": task_passed,
            "time": task_time,
            "llm_calls": task_llm_calls,
            "complexity": task.complexity,
        }

    # Calculate metrics
    results["metrics"]["tasks_passed"] = passed
    results["metrics"]["total_time"] = total_time
    results["metrics"]["avg_time"] = total_time / len(tasks) if tasks else 0
    results["metrics"]["total_llm_calls"] = total_llm_calls
    results["metrics"]["avg_llm_calls"] = total_llm_calls / len(tasks) if tasks else 0

    # Calculate fitness components
    # 1. Accuracy score (60% weight) - weighted by task complexity
    accuracy_score = complexity_score / max_complexity_score if max_complexity_score > 0 else 0

    # 2. Efficiency score (20% weight) - fewer LLM calls is better
    # Baseline: assume 10 LLM calls per task is average, fewer is better
    avg_calls = total_llm_calls / len(tasks) if tasks else 10
    # Score: 1.0 if avg_calls <= 3, decreases to 0.0 at avg_calls >= 15
    efficiency_score = max(0, min(1, (15 - avg_calls) / 12))
    results["metrics"]["efficiency_score"] = efficiency_score

    # 3. Speed score (10% weight) - faster is better
    # Baseline: assume 120 seconds per task is average
    avg_time = total_time / len(tasks) if tasks else 120
    # Score: 1.0 if avg_time <= 30s, decreases to 0.0 at avg_time >= 180s
    speed_score = max(0, min(1, (180 - avg_time) / 150))

    # 4. Complexity bonus (10% weight) - reward passing harder tasks
    # Already factored into accuracy_score via complexity weights

    # Combined fitness with weights
    results["fitness"] = (
        accuracy_score * 0.60 +      # Primary: task accuracy
        efficiency_score * 0.25 +    # Secondary: LLM efficiency
        speed_score * 0.15           # Tertiary: speed
    )

    return results


def create_code_agent_wrapper(model_name: str) -> Callable[..., str]:
    """Create a code agent wrapper for KISSEvolve.

    Args:
        model_name: The LLM model to use for code generation

    Returns:
        A function that generates code variations
    """

    def code_agent_wrapper(
        prompt_template: str,
        arguments: dict[str, str],
        model_name: str = model_name,
    ) -> str:
        """Generate code using an LLM agent."""
        agent = KISSAgent(name="CodeEvolver")
        result = agent.run(
            model_name=model_name,
            prompt_template=prompt_template,
            arguments=arguments,
            is_agentic=True,
            max_steps=100,
            max_budget=10.0,
        )
        return result

    return code_agent_wrapper


class AgentEvolver:
    """Evolves the SelfEvolvingMultiAgent for better efficiency and accuracy.

    The evolver optimizes for:
    1. Fewer LLM calls - reduce token usage and API costs
    2. Lower budget consumption - efficient resource usage
    3. Accurate completion of long-horizon tasks - maintain correctness
    """

    def __init__(
        self,
        model_name: str | None = None,
        population_size: int | None = None,
        max_generations: int | None = None,
        mutation_rate: float | None = None,
        elite_size: int | None = None,
        tasks: list[EvaluationTask] | None = None,
        focus_on_efficiency: bool = True,
    ):
        """Initialize the evolver.

        Args:
            model_name: LLM model to use
            population_size: Number of variants per generation
            max_generations: Maximum generations
            mutation_rate: Probability of mutation
            elite_size: Number of elite variants to keep
            tasks: Evaluation tasks (defaults to EVALUATION_TASKS)
            focus_on_efficiency: Whether to prioritize efficiency optimization
        """
        cfg = DEFAULT_CONFIG.self_evolving_multi_agent  # type: ignore[attr-defined]

        self.model_name = get_config_value(model_name, cfg, "evolver_model")
        self.population_size = get_config_value(population_size, cfg, "evolver_population_size")
        self.max_generations = get_config_value(max_generations, cfg, "evolver_max_generations")
        self.mutation_rate = get_config_value(mutation_rate, cfg, "evolver_mutation_rate")
        self.elite_size = get_config_value(elite_size, cfg, "evolver_elite_size")
        self.tasks = tasks if tasks is not None else EVALUATION_TASKS
        self.focus_on_efficiency = focus_on_efficiency

    def evolve(self) -> CodeVariant:
        """Run the evolutionary optimization.

        Returns:
            The best code variant found
        """
        print("=" * 60)
        print("Self Evolving Multi Agent Evolver")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Population: {self.population_size}")
        print(f"Generations: {self.max_generations}")
        print(f"Tasks: {len(self.tasks)}")
        print(f"Focus: {'Efficiency + Accuracy' if self.focus_on_efficiency else 'Accuracy'}")
        print()

        # Print task breakdown
        for task in self.tasks:
            print(f"  - {task.name} ({task.complexity})")
        print()

        def evaluation_fn(code: str) -> dict[str, Any]:
            return evaluate_agent_code(code, self.tasks, self.model_name)

        # Efficiency-focused optimization instructions
        efficiency_instructions = """
## Primary Optimization Goals ##
Your goal is to evolve the coding agent to be MORE EFFICIENT while maintaining accuracy.

### Key Efficiency Improvements ###
1. **Reduce LLM Calls**
   - Combine multiple small tasks into single comprehensive plans
   - Execute multiple bash commands in sequence without intermediate LLM calls
   - Avoid unnecessary verification steps that require LLM reasoning
   - Use direct execution instead of sub-agents when tasks are simple

2. **Optimize Planning**
   - Create comprehensive plans upfront that don't require replanning
   - Group related file operations together
   - Minimize back-and-forth between planning and execution

3. **Streamline Prompts**
   - Keep prompts concise but complete
   - Remove redundant instructions
   - Focus on actionable guidance

4. **Smart Sub-Agent Usage**
   - Only use sub-agents for truly complex sub-tasks
   - For simple file writes or commands, use direct tools
   - Reduce sub-agent max_steps and max_budget

5. **Batch Operations**
   - Write multiple files in a single script when possible
   - Run test commands together
   - Combine related bash commands

### Code Structure Guidelines ###
- The run_task function must return a dict with 'result' and 'metrics' keys
- metrics should include 'llm_calls' to track efficiency
- Maintain the core planning/execution architecture but optimize it
- Keep error handling but make it lightweight

### What NOT to Change ###
- Don't remove essential error handling
- Don't skip verification for complex tasks
- Don't sacrifice accuracy for speed
- Keep the basic tool interface (run_bash, read_file, write_file)
"""

        evolver = KISSEvolve(
            code_agent_wrapper=create_code_agent_wrapper('gemini-3-pro-preview'),
            initial_code=BASE_AGENT_CODE,
            evaluation_fn=evaluation_fn,
            model_names=[(self.model_name, 1.0)],
            population_size=self.population_size,
            max_generations=self.max_generations,
            mutation_rate=self.mutation_rate,
            elite_size=self.elite_size,
            extra_coding_instructions=efficiency_instructions if self.focus_on_efficiency else """
Focus on improving the agent's:
1. Task understanding and planning
2. Error handling and recovery
3. Code verification before finishing
4. Handling of long-horizon multi-step tasks
""",
        )

        best = evolver.evolve()

        print("\n" + "=" * 60)
        print("EVOLUTION COMPLETE")
        print("=" * 60)
        print(f"Best fitness: {best.fitness:.4f}")
        print(f"Metrics: {best.metrics}")
        if best.metrics:
            passed = best.metrics.get('tasks_passed', 'N/A')
            total = best.metrics.get('tasks_total', 'N/A')
            print(f"  - Tasks passed: {passed}/{total}")
            print(f"  - Avg LLM calls: {best.metrics.get('avg_llm_calls', 'N/A'):.1f}")
            eff = best.metrics.get('efficiency_score', 'N/A')
            print(f"  - Efficiency score: {eff:.3f}")

        return best

    def save_best(self, variant: CodeVariant, path: str = "evolved_agent.py") -> None:
        """Save the best variant to a file.

        Args:
            variant: The code variant to save
            path: Output file path
        """
        Path(path).write_text(variant.code)
        print(f"Saved best variant to {path}")

    def run_baseline_evaluation(self) -> dict[str, Any]:
        """Run evaluation on the base agent code to establish baseline.

        Returns:
            Baseline evaluation results
        """
        print("Running baseline evaluation...")
        results = evaluate_agent_code(BASE_AGENT_CODE, self.tasks, self.model_name)
        print(f"Baseline fitness: {results['fitness']:.4f}")
        print(f"Baseline metrics: {results['metrics']}")
        return results


def main() -> None:
    """Main entry point for the Coding Agent Evolver.

    This evolves the SelfEvolvingMultiAgent to optimize for:
    - Fewer LLM calls
    - Lower token/budget usage
    - Accurate completion of long-horizon coding tasks
    """
    config = DEFAULT_CONFIG.self_evolving_multi_agent  # type: ignore[attr-defined]

    print("=" * 70)
    print("Self Evolving Multi Agent Evolver")
    print("Optimizing for: Efficiency + Accuracy on Long-Horizon Tasks")
    print("=" * 70)

    if config.evolver_test_only:
        # Just test the base agent
        print("\nRunning baseline evaluation only...")
        print("-" * 50)

        # Test on a subset of tasks
        test_tasks = [t for t in EVALUATION_TASKS if t.complexity in ["simple", "medium"]][:2]
        result = evaluate_agent_code(
            BASE_AGENT_CODE, test_tasks, config.evolver_model
        )

        print("\nBaseline Results:")
        print(f"  Fitness: {result['fitness']:.4f}")
        metrics = result['metrics']
        print(f"  Tasks Passed: {metrics['tasks_passed']}/{metrics['tasks_total']}")
        print(f"  Avg LLM Calls: {metrics['avg_llm_calls']:.1f}")
        print(f"  Efficiency Score: {metrics['efficiency_score']:.3f}")
        print(f"\nFull results:\n{json.dumps(result, indent=2)}")
        return

    # Create evolver with efficiency focus
    evolver = AgentEvolver(
        model_name=config.evolver_model,
        population_size=config.evolver_population_size,
        max_generations=config.evolver_max_generations,
        focus_on_efficiency=True,
    )

    # Run baseline first
    print("\n--- Baseline Evaluation ---")
    baseline = evolver.run_baseline_evaluation()

    # Run evolution
    print("\n--- Starting Evolution ---")
    best = evolver.evolve()

    # Compare improvement
    print("\n" + "=" * 70)
    print("EVOLUTION SUMMARY")
    print("=" * 70)
    print(f"\nBaseline fitness: {baseline['fitness']:.4f}")
    print(f"Evolved fitness:  {best.fitness:.4f}")
    base_fitness = baseline['fitness']
    if base_fitness > 0:
        improvement = ((best.fitness - base_fitness) / base_fitness) * 100
    else:
        improvement = 0
    print(f"Improvement: {improvement:+.1f}%")

    if best.metrics:
        print("\nEvolved Agent Metrics:")
        passed = best.metrics.get('tasks_passed', 'N/A')
        total = best.metrics.get('tasks_total', 'N/A')
        print(f"  - Tasks passed: {passed}/{total}")
        print(f"  - Avg LLM calls: {best.metrics.get('avg_llm_calls', 'N/A'):.1f}")
        print(f"  - Efficiency score: {best.metrics.get('efficiency_score', 'N/A'):.3f}")
        print(f"  - Avg time per task: {best.metrics.get('avg_time', 'N/A'):.1f}s")

    # Save the best variant
    evolver.save_best(best, config.evolver_output)
    print(f"\nBest evolved agent saved to: {config.evolver_output}")


if __name__ == "__main__":
    main()
