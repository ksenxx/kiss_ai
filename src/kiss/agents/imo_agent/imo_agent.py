"""IMO Agent: Verification-and-Refinement Pipeline for Olympiad Mathematics.

Based on the approach described in:
  "Winning Gold at IMO 2025 with a Model-Agnostic Verification-and-Refinement Pipeline"
  Yichen Huang, Lin F. Yang (arXiv:2507.15855)

The pipeline:
  Step 1: Initial solution generation with a rigorous solver prompt
  Step 2: Self-improvement (model reviews and improves its own work)
  Step 3: Verification with a meticulous verifier prompt -> bug report
  Step 4: Review of bug report (optional, reduces false positives)
  Step 5: Correction/improvement based on bug report -> go to Step 3
  Step 6: Accept or Reject
"""

from __future__ import annotations

import kiss.agents.imo_agent.config as _imo_config  # noqa: F401
from kiss.agents.imo_agent.imo_problems import (
    get_all_problem_numbers,
    get_problem_statement,
    get_validation_info,
)
from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.kiss_agent import KISSAgent

# ──────────────────────────────────────────────────────────────────────────────
# Prompts (from the paper, Section 3.1 and 3.2)
# ──────────────────────────────────────────────────────────────────────────────

SOLVER_PROMPT = """\
### Problem ###
{problem}

### Core Instructions ###
* **Rigor is Paramount:** Your primary goal is to produce a complete and rigorously \
justified solution. Every step in your solution must be logically sound and clearly \
explained. A correct final answer derived from flawed or incomplete reasoning is \
considered a failure.
* **Honesty About Completeness:** If you cannot find a complete solution, you must \
**not** guess or create a solution that appears correct but contains hidden flaws or \
justification gaps. Instead, you should present only significant partial results that \
you can rigorously prove. A partial result is considered significant if it represents \
a substantial advancement toward a full solution. Examples include:
  * Proving a key lemma.
  * Fully resolving one or more cases within a logically sound case-based proof.
  * Establishing a critical property of the mathematical objects in the problem.
  * For an optimization problem, proving an upper or lower bound without proving \
that this bound is achievable.
* **Use TeX for All Mathematics:** All mathematical variables, expressions, and \
relations must be enclosed in TeX delimiters (e.g., `Let $n$ be an integer.`).

### Output Format ###
Your response MUST be structured into the following sections, in this exact order.

**1. Summary**
Provide a concise overview of your findings. This section must contain two parts:
* **a. Verdict:** State clearly whether you have found a complete solution or a \
partial solution.
  * **For a complete solution:** State the final answer, e.g., "I have successfully \
solved the problem. The final answer is..."
  * **For a partial solution:** State the main rigorous conclusion(s) you were able \
to prove.
* **b. Method Sketch:** Present a high-level, conceptual outline of your solution. \
This sketch should allow an expert to understand the logical flow of your argument \
without reading the full detail. It should include:
  * A narrative of your overall strategy.
  * The full and precise mathematical statements of any key lemmas.
  * If applicable, describe any key constructions or case splits.

**2. Detailed Solution**
Present the full, step-by-step mathematical proof. Each step must be logically \
justified and clearly explained. This section must contain ONLY the complete, rigorous \
proof, free of any internal commentary, alternative approaches, or failed attempts.

### Self-Correction Instruction ###
Before finalizing your output, carefully review your "Method Sketch" and "Detailed \
Solution" to ensure they are clean, rigorous, and strictly adhere to all instructions. \
Verify that every statement contributes directly to the final, coherent mathematical \
argument.
"""

SELF_IMPROVEMENT_PROMPT = """\
### Problem ###
{problem}

### Your Previous Solution ###
{solution}

### Instructions ###
You have an opportunity to improve your solution. Please review your solution \
carefully. Correct errors and fill justification gaps if any. Your output should \
strictly follow the same format as before (Summary + Detailed Solution). Make sure \
every step is rigorously justified.
"""

VERIFIER_PROMPT = """\
You are an expert mathematician and a meticulous grader for an International \
Mathematical Olympiad (IMO) level exam. Your primary task is to rigorously verify \
the provided mathematical solution. A solution is to be judged correct **only if \
every step is rigorously justified.** A solution that arrives at a correct final \
answer through flawed reasoning, educated guesses, or with gaps in its arguments \
must be flagged as incorrect or incomplete.

### Instructions ###

**1. Core Instructions**
* Your sole task is to find and report all issues in the provided solution. You \
must act as a **verifier**, NOT a solver. **Do NOT attempt to correct the errors \
or fill the gaps you find.**
* You must perform a **step-by-step** check of the entire solution.

**2. How to Handle Issues**
When you identify an issue, classify it as:
* **a. Critical Error:** Any error that breaks the logical chain (logical fallacies \
or factual errors). Do NOT check further steps that rely on this error, but DO scan \
independent parts.
* **b. Justification Gap:** Steps where the conclusion may be correct but the \
argument is incomplete. Assume the conclusion is true and continue checking.

**3. Output Format**
* **a. Summary**: Start with:
  * **Final Verdict**: One sentence declaring validity (e.g., "The solution is \
correct" or "The solution contains a Critical Error").
  * **List of Findings**: Bulleted list with Location and Issue for each finding.
* **b. Detailed Verification Log**: Step-by-step check with quotes from the solution.

======================================================================
### Problem ###
{problem}
======================================================================
### Solution ###
{solution}
======================================================================
### Verification Task Reminder ###
Now generate the summary and step-by-step verification log.
"""

BUG_REPORT_REVIEW_PROMPT = """\
### Problem ###
{problem}

### Solution ###
{solution}

### Bug Report ###
{bug_report}

### Instructions ###
Can you carefully review each item in the bug report's list of findings? Are they \
valid or overly strict? An expert grader must distinguish between a genuine flaw and \
a concise argument that is nonetheless sound, and correct their own assessment when \
necessary. If modifications to any item are necessary, produce a new list. Start \
directly with **Summary** (including Final Verdict and List of Findings).
"""

CORRECTION_PROMPT = """\
### Problem ###
{problem}

### Your Previous Solution ###
{solution}

### Bug Report ###
{bug_report}

### Instructions ###
Above is the bug report from a verification step. If you agree with certain items, \
improve your solution so that it is complete and rigorous. Note that the evaluator \
can misunderstand your solution and make mistakes. If you do not agree with certain \
items, add detailed explanations to avoid such misunderstanding. Your new solution \
should follow the same format (Summary + Detailed Solution).
"""

VALIDATION_PROMPT = """\
You are an independent mathematical expert. Your task is to validate whether a \
proposed solution to an IMO problem is correct by checking it against the known \
answer and validation criteria.

### Problem ###
{problem}

### Proposed Solution ###
{solution}

### Known Answer ###
{known_answer}

### Validation Criteria ###
{validation_criteria}

### Instructions ###
1. Check if the proposed solution arrives at the correct answer.
2. Check if the proof/reasoning satisfies the validation criteria.
3. Be somewhat lenient on minor presentation issues but strict on mathematical \
correctness and completeness.

### Output Format ###
Start with exactly one of these lines:
**VERDICT: PASS** — if the solution is correct and meets the criteria.
**VERDICT: FAIL** — if the solution has significant errors or wrong answer.

Then provide a brief explanation of your assessment.
"""


class IMOAgent(Base):
    """IMO problem solver using verification-and-refinement pipeline.

    Based on arXiv:2507.15855. The agent:
    1. Generates a solution using a powerful reasoning model
    2. Self-improves the solution
    3. Iteratively verifies and refines using a verifier model
    4. Does NOT see validation criteria or known answers
    """

    def __init__(self, name: str = "IMOAgent") -> None:
        super().__init__(name)
        cfg = config_module.DEFAULT_CONFIG.imo_agent
        self.solver_model = cfg.solver_model
        self.verifier_model = cfg.verifier_model
        self.validator_model = cfg.validator_model
        self.max_refinement_rounds = cfg.max_refinement_rounds
        self.num_verify_passes = cfg.num_verify_passes
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
        """Step 1: Generate initial solution."""
        print(f"\n{'='*60}\n  STEP 1: Generating initial solution\n{'='*60}")
        return self._call_model(
            "IMO-Solver", self.solver_model, SOLVER_PROMPT,
            {"problem": problem},
        )

    def _self_improve(self, problem: str, solution: str) -> str:
        """Step 2: Self-improvement."""
        print(f"\n{'='*60}\n  STEP 2: Self-improvement\n{'='*60}")
        return self._call_model(
            "IMO-Improver", self.solver_model, SELF_IMPROVEMENT_PROMPT,
            {"problem": problem, "solution": solution},
        )

    def _verify(self, problem: str, solution: str) -> str:
        """Step 3: Verification - generate bug report."""
        print(f"\n{'='*60}\n  STEP 3: Verification\n{'='*60}")
        return self._call_model(
            "IMO-Verifier", self.verifier_model, VERIFIER_PROMPT,
            {"problem": problem, "solution": solution},
        )

    def _review_bug_report(self, problem: str, solution: str, bug_report: str) -> str:
        """Step 4: Review bug report for false positives."""
        print(f"\n{'='*60}\n  STEP 4: Reviewing bug report\n{'='*60}")
        return self._call_model(
            "IMO-BugReviewer", self.verifier_model, BUG_REPORT_REVIEW_PROMPT,
            {"problem": problem, "solution": solution, "bug_report": bug_report},
        )

    def _correct(self, problem: str, solution: str, bug_report: str) -> str:
        """Step 5: Correct solution based on bug report."""
        print(f"\n{'='*60}\n  STEP 5: Correcting solution\n{'='*60}")
        return self._call_model(
            "IMO-Corrector", self.solver_model, CORRECTION_PROMPT,
            {"problem": problem, "solution": solution, "bug_report": bug_report},
        )

    @staticmethod
    def _is_solution_accepted(bug_report: str) -> bool:
        """Check if the verifier accepted the solution (no critical errors)."""
        report_lower = bug_report.lower()
        # Check for positive verdicts
        if "the solution is correct" in report_lower:
            return True
        if "the solution is **correct**" in report_lower:
            return True
        # Check for negative verdicts
        if "critical error" in report_lower:
            return False
        if "the solution is **invalid**" in report_lower:
            return False
        if "the solution is invalid" in report_lower:
            return False
        # If no critical errors found, tentatively accept
        if "justification gap" in report_lower and "critical" not in report_lower:
            return True
        return False

    def solve_problem(self, problem_number: int) -> str:
        """Solve an IMO problem using the verification-and-refinement pipeline.

        The agent does NOT see validation criteria or known answers.

        Args:
            problem_number: The IMO 2025 problem number (1-6).

        Returns:
            The final solution text.
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

            # Step 1: Generate initial solution
            solution = self._generate_solution(problem)

            # Step 2: Self-improvement
            solution = self._self_improve(problem, solution)

            # Steps 3-5: Verification and refinement loop
            accepted = False
            for round_num in range(self.max_refinement_rounds):
                print(f"\n--- Refinement round {round_num + 1}"
                      f"/{self.max_refinement_rounds} ---")

                # Step 3: Verify
                bug_report = self._verify(problem, solution)

                # Step 4: Review bug report
                reviewed_report = self._review_bug_report(
                    problem, solution, bug_report
                )

                # Check if solution passes
                if self._is_solution_accepted(reviewed_report):
                    # Run additional verification passes for robustness
                    all_pass = True
                    for vpass in range(self.num_verify_passes - 1):
                        print(f"  Additional verification pass "
                              f"{vpass + 2}/{self.num_verify_passes}")
                        extra_report = self._verify(problem, solution)
                        if not self._is_solution_accepted(extra_report):
                            all_pass = False
                            reviewed_report = extra_report
                            break

                    if all_pass:
                        print(f"\n✓ Solution ACCEPTED after "
                              f"{round_num + 1} refinement rounds")
                        accepted = True
                        break

                # Step 5: Correct based on bug report
                solution = self._correct(problem, solution, reviewed_report)

            best_solution = solution
            if accepted:
                break

        return best_solution

    @staticmethod
    def validate_solution(problem_number: int, solution: str,
                          validator_model: str = "gemini-2.5-pro") -> tuple[bool, str]:
        """Independently validate a solution against known answer/criteria.

        This is separate from the solver and uses the validation criteria
        that the solver never sees.

        Args:
            problem_number: The IMO 2025 problem number.
            solution: The proposed solution to validate.
            validator_model: Model to use for validation.

        Returns:
            Tuple of (passed: bool, explanation: str)
        """
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
