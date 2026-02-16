"""IMO Agent: Optimized Verification-and-Refinement Pipeline for Olympiad Mathematics.

Optimized version: fewer LLM calls, compressed prompts, merged steps.
Based on arXiv:2507.15855.
"""

from __future__ import annotations

import kiss.agents.imo_agent.config as _imo_config  # noqa: F401
from kiss.agents.imo_agent.imo_problems import (
    get_problem_statement,
    get_validation_info,
)
from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.kiss_agent import KISSAgent

# ── Compressed Prompts ────────────────────────────────────────────────────────

SOLVER_PROMPT = """\
### Problem ###
{problem}

### Instructions ###
Produce a complete, rigorously justified solution. Every step must be logically sound.
If you cannot find a complete solution, present only significant
partial results you can rigorously prove.
Use TeX for all math (e.g., `$n$`).

### Output Format ###
**1. Summary**
* **Verdict:** "Complete solution" or "Partial solution" with the answer/conclusion.
* **Method Sketch:** High-level outline with key lemma statements.

**2. Detailed Solution**
Full step-by-step proof. Only the rigorous proof—no commentary or failed attempts.

Before finalizing, self-review for correctness and rigor. Fix any errors or gaps.
"""

VERIFIER_PROMPT = """\
You are a meticulous IMO grader. Verify the solution step-by-step.
Find and report all issues. Do NOT fix errors—only identify them.

Classify issues as:
* **Critical Error:** Breaks logical chain (logical fallacy or factual error).
* **Justification Gap:** Conclusion may be correct but argument is incomplete.

### Output Format ###
**Summary:**
* **Final Verdict**: "The solution is correct" or
  "The solution contains a Critical Error" (one sentence).
* **List of Findings**: Bulleted list of issues found (Location + Issue).

**Detailed Verification Log:** Step-by-step check with quotes.

### Problem ###
{problem}

### Solution ###
{solution}
"""

CORRECTION_PROMPT = """\
### Problem ###
{problem}

### Your Previous Solution ###
{solution}

### Bug Report ###
{bug_report}

### Instructions ###
Review the bug report. If you agree with items, fix your solution.
If you disagree, add detailed explanations.
Output the improved solution in the same format (Summary + Detailed Solution).
Ensure every step is rigorously justified.
"""

VALIDATION_PROMPT = """\
Validate whether this IMO solution is correct against the known answer.

### Problem ###
{problem}

### Proposed Solution ###
{solution}

### Known Answer ###
{known_answer}

### Validation Criteria ###
{validation_criteria}

### Instructions ###
Check if the solution arrives at the correct answer and satisfies the criteria.
Be lenient on presentation but strict on mathematical correctness.

Start with exactly one of:
**VERDICT: PASS** — if correct and meets criteria.
**VERDICT: FAIL** — if significant errors or wrong answer.
Then briefly explain.
"""


class IMOAgent(Base):
    """IMO problem solver using optimized verification-and-refinement pipeline."""

    def __init__(self, name: str = "IMOAgent") -> None:
        super().__init__(name)
        cfg = config_module.DEFAULT_CONFIG.imo_agent
        self.solver_model = cfg.solver_model
        self.verifier_model = cfg.verifier_model
        self.validator_model = cfg.validator_model
        self.max_refinement_rounds = cfg.max_refinement_rounds
        self.max_attempts = cfg.max_attempts
        self.max_budget = cfg.max_budget
        self.budget_used: float = 0.0
        self.total_tokens_used: int = 0

    def _call_model(self, agent_name: str, model_name: str, prompt: str,
                    arguments: dict[str, str]) -> str:
        """Run a non-agentic KISSAgent call and track costs."""
        agent = KISSAgent(agent_name)
        result = agent.run(
            model_name=model_name,
            prompt_template=prompt,
            arguments=arguments,
            is_agentic=False,
            max_budget=self.max_budget,
            print_to_console=True,
        )
        self.budget_used += agent.budget_used
        self.total_tokens_used += agent.total_tokens_used
        return result

    def _generate_solution(self, problem: str) -> str:
        """Step 1: Generate initial solution (with built-in self-review)."""
        print(f"\n{'='*60}\n  STEP 1: Generating solution\n{'='*60}")
        return self._call_model(
            "IMO-Solver", self.solver_model, SOLVER_PROMPT,
            {"problem": problem},
        )

    def _verify(self, problem: str, solution: str) -> str:
        """Step 2: Verification - generate bug report."""
        print(f"\n{'='*60}\n  STEP 2: Verification\n{'='*60}")
        return self._call_model(
            "IMO-Verifier", self.verifier_model, VERIFIER_PROMPT,
            {"problem": problem, "solution": solution},
        )

    def _correct(self, problem: str, solution: str, bug_report: str) -> str:
        """Step 3: Correct solution based on bug report."""
        print(f"\n{'='*60}\n  STEP 3: Correcting solution\n{'='*60}")
        return self._call_model(
            "IMO-Corrector", self.solver_model, CORRECTION_PROMPT,
            {"problem": problem, "solution": solution, "bug_report": bug_report},
        )

    @staticmethod
    def _is_solution_accepted(bug_report: str) -> bool:
        """Check if the verifier accepted the solution."""
        report_lower = bug_report.lower()
        if "the solution is correct" in report_lower:
            return True
        if "the solution is **correct**" in report_lower:
            return True
        if "critical error" in report_lower:
            return False
        if (
            "the solution is **invalid**" in report_lower
            or "the solution is invalid" in report_lower
        ):
            return False
        if "justification gap" in report_lower and "critical" not in report_lower:
            return True
        return False

    def solve_problem(self, problem_number: int) -> str:
        """Solve an IMO problem using optimized verification-and-refinement.

        Pipeline per attempt:
        1. Generate solution (with self-review baked in)
        2. Verify -> if accepted, done
        3. Correct -> verify -> repeat up to max_refinement_rounds
        """
        problem = get_problem_statement(problem_number)
        print(f"\n{'#'*70}")
        print(f"  IMO 2025 Problem {problem_number}")
        print(f"{'#'*70}")
        print(f"\n{problem}\n")

        best_solution = ""

        for attempt in range(self.max_attempts):
            print(f"\n{'*'*60}")
            print(f"  Attempt {attempt + 1}/{self.max_attempts}")
            print(f"{'*'*60}")

            # Step 1: Generate solution (includes self-review)
            solution = self._generate_solution(problem)

            # Steps 2-3: Verify and refine loop
            accepted = False
            for round_num in range(self.max_refinement_rounds):
                print(f"\n--- Refinement round {round_num + 1}"
                      f"/{self.max_refinement_rounds} ---")

                # Verify
                bug_report = self._verify(problem, solution)

                # Check acceptance
                if self._is_solution_accepted(bug_report):
                    print(f"\n✓ Solution ACCEPTED after "
                          f"{round_num + 1} refinement rounds")
                    accepted = True
                    break

                # Correct
                solution = self._correct(problem, solution, bug_report)

            best_solution = solution
            if accepted:
                break

        return best_solution

    @staticmethod
    def validate_solution(problem_number: int, solution: str,
                          validator_model: str = "gemini-2.5-pro") -> tuple[bool, str]:
        """Independently validate a solution against known answer/criteria."""
        problem = get_problem_statement(problem_number)
        validation_criteria, known_answer = get_validation_info(problem_number)

        validator = KISSAgent("IMO-Validator")
        result = validator.run(
            model_name=validator_model,
            prompt_template=VALIDATION_PROMPT,
            arguments={
                "problem": problem,
                "solution": solution,
                "known_answer": known_answer,
                "validation_criteria": validation_criteria,
            },
            is_agentic=False,
            print_to_console=True,
        )

        passed = "VERDICT: PASS" in result.upper()
        return passed, result


def main() -> None:
    """Main function: solve all IMO 2025 problems with verification."""
    import time

    agent = IMOAgent("IMO2025-Solver")

    # Start with Problem 4 as requested, then do the rest
    problem_order = [4, 1, 2, 3, 5, 6]
    results: dict[int, dict] = {}

    for prob_num in problem_order:
        print(f"\n{'='*70}")
        print(f"  SOLVING IMO 2025 PROBLEM {prob_num}")
        print(f"{'='*70}")

        start = time.time()
        solution = agent.solve_problem(prob_num)
        solve_time = time.time() - start

        # Independent validation
        print(f"\n{'='*60}")
        print(f"  INDEPENDENT VALIDATION - Problem {prob_num}")
        print(f"{'='*60}")
        passed, explanation = IMOAgent.validate_solution(
            prob_num, solution, agent.validator_model
        )

        results[prob_num] = {
            "solution": solution[:500] + "..." if len(solution) > 500 else solution,
            "passed": passed,
            "explanation": explanation[:300],
            "time": f"{solve_time:.1f}s",
        }

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n  Problem {prob_num}: {status} ({solve_time:.1f}s)")

        if not passed:
            print(f"  Validation: {explanation[:200]}")

    # Final summary
    print(f"\n{'='*70}")
    print("  FINAL RESULTS - IMO 2025")
    print(f"{'='*70}")
    total_passed = 0
    for p in sorted(results):
        status = "✓ PASS" if results[p]["passed"] else "✗ FAIL"
        if results[p]["passed"]:
            total_passed += 1
        print(f"  Problem {p}: {status} ({results[p]['time']})")

    print(f"\n  Total: {total_passed}/6 problems solved")
    print(f"  Budget used: ${agent.budget_used:.4f}")
    print(f"  Total tokens: {agent.total_tokens_used}")


if __name__ == "__main__":
    main()
