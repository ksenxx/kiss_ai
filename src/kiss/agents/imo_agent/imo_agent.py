"""IMO Agent: Explore-Solve-Validate Pipeline for Olympiad Mathematics.

Architecture: Explore (fast model) → Solve (strong model) → Validate → Retry
- Phase 1: Use cheap model to compute small cases and explore the problem
- Phase 2: Feed exploration to powerful reasoning model for proof
- Phase 3: Independent validation against known answers
- Up to 3 attempts per problem with varied approaches
"""

from __future__ import annotations

import re
import time
import traceback

import kiss.agents.imo_agent.config as _imo_config  # noqa: F401
from kiss.agents.imo_agent.imo_problems import (
    IMO_2025_PROBLEMS,
    get_problem_statement,
    get_validation_info,
)
from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.kiss_agent import KISSAgent

# ── Prompts ────────────────────────────────────────────────────────────────

EXPLORE_PROMPT = """\
You are a mathematician investigating an IMO problem by computing concrete examples.

### Problem ###
{problem}

### Task ###
Systematically compute small cases. Do NOT attempt to prove anything — just compute and observe.

- For "determine all values" problems: test every relevant small value (n=1,2,...,10+). \
For each, check whether it satisfies ALL conditions. Be explicit about each step.
- For sequence problems: compute first 10+ terms for at least 5 different starting values. \
Show each computation step. Note which starting values lead to well-defined infinite sequences.
- For game/strategy problems: fully analyze the game for small parameter values. \
Determine who wins for each specific parameter value.
- For geometry: set up coordinates for a specific configuration, compute all relevant points, \
and verify the claim numerically.
- For "find minimum/maximum": compute the answer for small cases (n=1,2,...,8+).

**Output:**
1. A clear TABLE of all computed results.
2. Patterns you observe (but don't prove them).
"""

SOLVER_PROMPT = """\
You are a world-class mathematician solving an IMO problem.

### Problem ###
{problem}

### Preliminary Exploration (computed small cases) ###
{exploration}

### Instructions ###
Use the exploration above as concrete data. Follow these phases:

**Phase 1 — Pattern Recognition**
Examine the computed small cases carefully. What answer do they suggest?
State a precise conjecture.

**Phase 2 — Rigorous Proof**
Prove your conjecture completely:
- SUFFICIENCY: Show every claimed value/construction works.
- NECESSITY/COMPLETENESS: Show no other values work. This is usually the harder direction — \
do NOT skip it.

**Phase 3 — Self-Verification**
- Re-check your proof against EVERY computed small case.
- Actively look for counterexamples to your claim.
- If anything doesn't match, revise your conjecture and re-prove.

**Domain strategies:**
- Number theory: modular arithmetic, p-adic valuations, multiplicative structure.
- Combinatorics: extremal principle, double counting, pigeonhole, invariants/monovariants.
- Geometry: coordinate bash, angle chasing, power of a point, radical axes, inversion.
- Algebra/inequalities: AM-GM, Cauchy-Schwarz, substitutions, functional equations.
- Game theory: invariants, strategy stealing, potential functions, Cauchy-Schwarz for sums.

**Answer:** State the final answer clearly and explicitly.
**Solution:** Complete rigorous proof.
"""

RETRY_PROMPT = """\
You are a world-class mathematician. Your previous attempt was incorrect. \
Start COMPLETELY fresh with a fundamentally different approach.

### Problem ###
{problem}

### Preliminary Exploration (computed small cases) ###
{exploration}

### Instructions ###
1. Re-examine the small cases above CAREFULLY. Recompute any that seem suspicious.

2. Try a FUNDAMENTALLY different technique:
   - Direct proof ↔ contradiction/contrapositive
   - Algebraic approach ↔ combinatorial/geometric reasoning
   - Coordinates ↔ synthetic methods
   - Induction ↔ direct construction/extremal argument
   - Modular arithmetic ↔ p-adic valuations/analytic number theory

3. Common mistakes to AVOID:
   - For "determine all": missing valid values or including invalid ones.
   - Incomplete necessity/completeness proofs.
   - Arithmetic/computation errors in key steps.
   - Unjustified claims or "clearly" without proof.

4. After finding an answer, actively try to DISPROVE it before committing.

**Answer:** State the final answer clearly and explicitly.
**Solution:** Complete rigorous proof.
"""

ATTEMPT3_PROMPT = """\
You are a world-class mathematician. Two previous attempts at this problem were wrong. \
The correct answer exists — use a radically different approach.

### Problem ###
{problem}

### Preliminary Exploration (computed small cases) ###
{exploration}

### Strategy ###
1. Carefully re-examine ALL the small case data. Are there patterns you overlooked?
   Look for: divisibility, periodicity, special structure, invariants, exceptional cases.

2. Use a technique you likely haven't tried:
   - Generating functions or formal power series
   - Graph/hypergraph methods
   - Linear algebra (rank, determinant, eigenvalues over finite fields)
   - Probabilistic or counting arguments
   - Reduction to known competition theorems
   - Monovariant or potential function arguments

3. FOCUS ON GETTING THE RIGHT ANSWER FIRST. Use the small cases as ground truth.
   A correct answer with incomplete proof is better than a wrong answer with "proof."

4. Compute MORE small cases if needed to pin down the answer.

**Answer:** State the final answer clearly and explicitly.
**Solution:** Complete rigorous proof.
"""

VALIDATION_PROMPT = """\
Check if the solution correctly solves the problem with valid reasoning.

### Problem ###
{problem}

### Solution ###
{solution}

### Known Answer ###
{known_answer}

### Criteria ###
{validation_criteria}

The known answer is definitively correct. Does the solution reach this answer with valid reasoning?

Reply EXACTLY one verdict line:
VERDICT: PASS — correct answer with valid reasoning
VERDICT: FAIL — wrong answer or major logical gap
Then one sentence why.
"""


def extract_verdict(text: str) -> bool:
    """Extract verdict using LAST occurrence to avoid prompt echoes."""
    verdicts: list[str] = re.findall(r'VERDICT:\s*(PASS|FAIL)', text.upper())
    if not verdicts:
        return False
    return verdicts[-1] == "PASS"


# Attempt budget per difficulty
ATTEMPTS_BY_DIFFICULTY = {
    "easy": 3,
    "medium": 3,
    "hard": 3,
    "very_hard": 2,
}


class IMOAgent(Base):
    """IMO solver: Explore → Solve → Validate pipeline."""

    def __init__(self, name: str = "IMOAgent") -> None:
        super().__init__(name)
        cfg = config_module.DEFAULT_CONFIG.imo_agent
        self.solver_model = cfg.solver_model
        self.verifier_model = cfg.verifier_model
        self.validator_model = cfg.validator_model
        self.max_budget = cfg.max_budget
        self.budget_used: float = 0.0
        self.total_tokens_used: int = 0
        self._validation_cache: dict[int, tuple[bool, str]] = {}

    def _call_model(self, agent_name: str, model_name: str, prompt: str,
                    arguments: dict[str, str],
                    model_config: dict | None = None,
                    max_retries: int = 3) -> str:
        """Call a model with retry logic for timeout/API errors."""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                agent = KISSAgent(agent_name)
                result = agent.run(
                    model_name=model_name,
                    prompt_template=prompt,
                    arguments=arguments,
                    is_agentic=False,
                    max_budget=self.max_budget,
                    print_to_console=True,
                    model_config=model_config,
                )
                self.budget_used += agent.budget_used
                self.total_tokens_used += agent.total_tokens_used
                return result
            except Exception as e:
                last_error = e
                error_name = type(e).__name__
                print(f"\n  ⚠ API call failed (attempt {attempt}/{max_retries}): {error_name}: {e}")
                if attempt < max_retries:
                    wait_time = 30 * attempt  # 30s, 60s, 90s
                    print(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        # All retries exhausted
        raise RuntimeError(f"All {max_retries} API call attempts failed. Last error: {last_error}")

    def _explore(self, problem_number: int) -> str:
        """Use a fast model to compute small cases and explore the problem."""
        problem = get_problem_statement(problem_number)
        print(f"\n  📊 Exploring small cases with {self.validator_model}...")
        try:
            result = self._call_model(
                "IMO-Explorer", self.validator_model, EXPLORE_PROMPT,
                {"problem": problem},
            )
            # Truncate if very long to avoid context issues
            if len(result) > 8000:
                result = result[:8000] + "\n[...truncated...]"
            print(f"  ✓ Exploration complete ({len(result)} chars)")
            return result
        except Exception as e:
            print(f"  ⚠ Exploration failed: {e}. Proceeding without exploration.")
            return (
                "(No exploration data available"
                " — compute small cases yourself before attempting a proof.)"
            )

    def _validate(self, problem_number: int, solution: str) -> tuple[bool, str]:
        problem = get_problem_statement(problem_number)
        criteria, answer = get_validation_info(problem_number)
        result = self._call_model(
            "IMO-Validator", self.validator_model, VALIDATION_PROMPT,
            {
                "problem": problem,
                "solution": solution[:8000],
                "known_answer": answer,
                "validation_criteria": criteria,
            },
        )
        passed = extract_verdict(result)
        return passed, result

    def solve_problem(self, problem_number: int) -> str:
        problem = get_problem_statement(problem_number)
        difficulty = IMO_2025_PROBLEMS[problem_number].difficulty
        max_attempts = ATTEMPTS_BY_DIFFICULTY.get(difficulty, 2)

        print(f"\n{'#'*70}")
        print(f"  IMO 2025 Problem {problem_number} [{difficulty}] (max {max_attempts} attempts)")
        print(f"{'#'*70}")

        # Phase 1: Explore small cases once (reused across attempts)
        exploration = self._explore(problem_number)

        best_solution = ""
        prompts = [SOLVER_PROMPT, RETRY_PROMPT, ATTEMPT3_PROMPT]

        for attempt in range(1, max_attempts + 1):
            print(f"\n{'='*60}")
            print(f"  ATTEMPT {attempt}/{max_attempts}")
            print(f"{'='*60}")

            try:
                prompt = prompts[min(attempt - 1, len(prompts) - 1)]
                solver_config = {"reasoning_effort": "high"}

                solution = self._call_model(
                    "IMO-Solver", self.solver_model, prompt,
                    {"problem": problem, "exploration": exploration},
                    model_config=solver_config,
                )

                # Sanity check: solution should be substantive
                if len(solution.strip()) < 100:
                    chars = len(solution.strip())
                    print(f"  ⚠ Solution too short ({chars} chars), treating as failure.")
                    if attempt < max_attempts:
                        continue

                best_solution = solution

                # Validate
                print(f"\n  Validating attempt {attempt}...")
                passed, explanation = self._validate(problem_number, solution)
                self._validation_cache[problem_number] = (passed, explanation)

                if passed:
                    print(f"  ✓ Attempt {attempt} PASSED!")
                    return solution
                else:
                    print(f"  ✗ Attempt {attempt} failed validation.")
                    if attempt < max_attempts:
                        print("  Retrying with different approach...")
            except Exception as e:
                print(f"\n  ✗ Attempt {attempt} crashed: {type(e).__name__}: {e}")
                traceback.print_exc()
                if attempt < max_attempts:
                    print("  Retrying...")

        return best_solution

    @staticmethod
    def validate_solution(problem_number: int, solution: str,
                          validator_model: str = "gemini-2.5-pro") -> tuple[bool, str]:
        problem = get_problem_statement(problem_number)
        criteria, answer = get_validation_info(problem_number)
        validator = KISSAgent("IMO-Validator")
        result = validator.run(
            model_name=validator_model,
            prompt_template=VALIDATION_PROMPT,
            arguments={
                "problem": problem,
                "solution": solution,
                "known_answer": answer,
                "validation_criteria": criteria,
            },
            is_agentic=False,
            print_to_console=True,
        )
        passed = extract_verdict(result)
        return passed, result


def main() -> None:
    agent = IMOAgent("IMO2025-Solver")

    # Order: easiest first, hardest last
    problem_order = [4, 1, 5, 2, 3, 6]
    results: dict[int, dict] = {}

    for prob_num in problem_order:
        print(f"\n{'='*70}")
        print(f"  SOLVING IMO 2025 PROBLEM {prob_num}")
        print(f"{'='*70}")

        start = time.time()
        try:
            solution = agent.solve_problem(prob_num)
        except Exception as e:
            print(f"\n  ✗ Problem {prob_num} completely failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[prob_num] = {
                "solution": f"FAILED: {e}",
                "passed": False,
                "explanation": str(e)[:300],
                "time": f"{time.time() - start:.1f}s",
            }
            continue

        solve_time = time.time() - start

        if prob_num in agent._validation_cache:
            passed, explanation = agent._validation_cache[prob_num]
        else:
            try:
                passed, explanation = IMOAgent.validate_solution(
                    prob_num, solution, agent.validator_model
                )
            except Exception as e:
                print(f"  Validation failed: {e}")
                passed, explanation = False, str(e)

        results[prob_num] = {
            "solution": solution[:500] + "..." if len(solution) > 500 else solution,
            "passed": passed,
            "explanation": str(explanation)[:300],
            "time": f"{solve_time:.1f}s",
        }

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n  Problem {prob_num}: {status} ({solve_time:.1f}s)")
        if not passed:
            print(f"  Validation: {str(explanation)[:200]}")

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
