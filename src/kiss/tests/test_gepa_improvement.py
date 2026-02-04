# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Test GEPA optimization and measure improvement factor."""

import kiss.core.utils as utils
from kiss.agents.gepa import GEPA
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError


def run_gepa_improvement_test():
    """Run GEPA optimization and measure improvement before/after.

    This function executes a full GEPA optimization cycle and compares
    the performance of the initial prompt vs the optimized prompt on
    held-out test examples.

    Returns:
        Dict containing initial_scores, optimized_scores, initial_prompt,
        optimized_prompt, and improvement_percent.
    """
    print("=" * 70)
    print("GEPA Prompt Optimization - Improvement Factor Test")
    print("=" * 70)

    # A deliberately weak/vague initial prompt that doesn't guide the model well
    initial_prompt = """Here is something: {problem}
Do something with it."""

    # Training examples - mix of math and reasoning
    train_examples = [
        {"problem": "What is 15 + 27?", "_expected": "42"},
        {"problem": "What is 100 - 37?", "_expected": "63"},
        {"problem": "What is 8 * 7?", "_expected": "56"},
        {"problem": "What is 144 / 12?", "_expected": "12"},
        {"problem": "What is 25 + 38?", "_expected": "63"},
        {"problem": "What is 90 - 45?", "_expected": "45"},
        {"problem": "What is 9 * 9?", "_expected": "81"},
        {"problem": "What is 72 / 8?", "_expected": "9"},
    ]

    # Held-out test examples (not used in training)
    test_examples = [
        {"problem": "What is 33 + 19?", "_expected": "52"},
        {"problem": "What is 85 - 29?", "_expected": "56"},
        {"problem": "What is 6 * 11?", "_expected": "66"},
        {"problem": "What is 96 / 6?", "_expected": "16"},
    ]

    call_counter = [0]

    def agent_wrapper(prompt_template: str, arguments: dict[str, str]) -> tuple[str, list]:
        """Run agent with real LLM call, capturing trajectory.

        Args:
            prompt_template: The prompt template to fill and run.
            arguments: Dict of arguments including '_expected' for expected answer.

        Returns:
            Tuple of (result string with expected/actual, trajectory list).
        """
        import json

        expected = arguments.get("_expected", "")
        agent_args = {k: v for k, v in arguments.items() if not k.startswith("_")}

        call_counter[0] += 1
        agent = KISSAgent(f"Agent {call_counter[0]}")

        try:
            result = agent.run(
                model_name="gpt-4o",
                prompt_template=prompt_template,
                arguments=agent_args,
                tools=[utils.finish],
            )
        except KISSError as e:
            # Agent exceeded max steps or other error - return failure
            result = f"status: error\nresult: Agent failed - {e}"

        # Capture agent trajectory for reflection
        trajectory = json.loads(agent.get_trajectory())

        return f"EXPECTED:{expected}\nRESULT:{result}", trajectory

    def evaluation_fn(result: str) -> dict[str, float]:
        """Evaluate result against expected.

        Args:
            result: Result string in format 'EXPECTED:...\nRESULT:...'

        Returns:
            Dict with 'success' and 'correct' scores (0.0 or 1.0).
        """
        import yaml

        try:
            if result.startswith("EXPECTED:"):
                parts = result.split("\nRESULT:", 1)
                expected = parts[0].replace("EXPECTED:", "").strip().lower()
                actual_result = parts[1] if len(parts) > 1 else ""

                result_dict = yaml.safe_load(actual_result) or {}
                actual = str(result_dict.get("result", "")).strip().lower()

                return {
                    "success": 1.0 if result_dict.get("status") == "success" else 0.0,
                    "correct": 1.0 if expected == actual else 0.0,
                }
        except Exception:
            pass
        return {"success": 0.0, "correct": 0.0}

    def evaluate_prompt_on_examples(
        prompt: str, examples: list[dict], label: str
    ) -> dict[str, float]:
        """Evaluate a prompt on a set of examples.

        Args:
            prompt: The prompt template to evaluate.
            examples: List of example dicts with problem and _expected keys.
            label: Label for logging purposes.

        Returns:
            Dict of average scores across all examples.
        """
        total_scores: dict[str, float] = {}
        count = 0

        print(f"\n  Evaluating {label}...")
        for ex in examples:
            result, _ = agent_wrapper(prompt, ex)
            scores = evaluation_fn(result)

            for k, v in scores.items():
                total_scores[k] = total_scores.get(k, 0.0) + v
            count += 1

        avg_scores = {k: v / count for k, v in total_scores.items()}
        return avg_scores

    # =========================================================================
    # Phase 1: Evaluate INITIAL prompt on test set (baseline)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Baseline - Evaluating INITIAL prompt")
    print("=" * 70)
    print(f"\nInitial prompt:\n{'-' * 40}\n{initial_prompt}\n{'-' * 40}")

    initial_test_scores = evaluate_prompt_on_examples(
        initial_prompt, test_examples, "initial prompt on test set"
    )

    print(f"\n  Initial prompt test scores: {initial_test_scores}")

    # =========================================================================
    # Phase 2: Run GEPA optimization
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Running GEPA Optimization")
    print("=" * 70)

    gepa = GEPA(
        agent_wrapper=agent_wrapper,
        initial_prompt_template=initial_prompt,
        evaluation_fn=evaluation_fn,
        max_generations=3,
        population_size=3,
        pareto_size=3,
        mutation_rate=0.8,
        dev_val_split=0.5,
        use_merge=True,
        merge_val_overlap_floor=1,
    )

    print("\nRunning optimization...")
    best_candidate = gepa.optimize(train_examples, dev_minibatch_size=2)

    optimized_prompt = gepa.get_best_prompt()

    print("\nOptimization complete!")
    print(f"  Candidates created: {gepa._candidate_id}")
    print(f"  Pareto frontier size: {len(gepa.get_pareto_frontier())}")
    print(f"  Best candidate val_scores: {best_candidate.val_scores}")

    print(f"\nOptimized prompt:\n{'-' * 40}\n{optimized_prompt}\n{'-' * 40}")

    # =========================================================================
    # Phase 3: Evaluate OPTIMIZED prompt on test set
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Evaluating OPTIMIZED prompt")
    print("=" * 70)

    optimized_test_scores = evaluate_prompt_on_examples(
        optimized_prompt, test_examples, "optimized prompt on test set"
    )

    print(f"\n  Optimized prompt test scores: {optimized_test_scores}")

    # =========================================================================
    # Phase 4: Calculate and display improvement
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS: Improvement Factor")
    print("=" * 70)

    print(f"\n{'Metric':<15} {'Initial':<12} {'Optimized':<12} {'Improvement':<15}")
    print("-" * 55)

    for metric in initial_test_scores:
        initial = initial_test_scores.get(metric, 0.0)
        optimized = optimized_test_scores.get(metric, 0.0)

        if initial > 0:
            improvement = (optimized - initial) / initial * 100
            improvement_str = f"{improvement:+.1f}%"
        elif optimized > 0:
            improvement_str = "+inf%"
        else:
            improvement_str = "0%"

        print(f"{metric:<15} {initial:<12.2f} {optimized:<12.2f} {improvement_str:<15}")

    # Overall improvement
    initial_avg = sum(initial_test_scores.values()) / len(initial_test_scores)
    optimized_avg = sum(optimized_test_scores.values()) / len(optimized_test_scores)

    if initial_avg > 0:
        overall_improvement = (optimized_avg - initial_avg) / initial_avg * 100
    elif optimized_avg > 0:
        overall_improvement = float("inf")
    else:
        overall_improvement = 0.0

    print("-" * 55)
    print(
        f"{'AVERAGE':<15} {initial_avg:<12.2f} {optimized_avg:<12.2f} {overall_improvement:+.1f}%"
    )

    print("\n" + "=" * 70)
    print(f"Total LLM calls made: {call_counter[0]}")
    print("=" * 70)

    return {
        "initial_scores": initial_test_scores,
        "optimized_scores": optimized_test_scores,
        "initial_prompt": initial_prompt,
        "optimized_prompt": optimized_prompt,
        "improvement_percent": overall_improvement,
    }


if __name__ == "__main__":
    run_gepa_improvement_test()
