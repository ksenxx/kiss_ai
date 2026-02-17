"""IMO 2025 problem statements, validation criteria, and difficulty ratings.

This module contains all 6 problems from the International Mathematical Olympiad 2025.
Problem statements are separated from validation criteria so that the solver agent
never sees the answers or validation criteria.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IMOProblem:
    """Container for an IMO problem with its metadata."""
    number: int
    statement: str
    domain: str
    difficulty: str  # "easy", "medium", "hard", "very_hard"
    day: int
    problem_type: str  # "determine", "prove", "find"
    # Validation criteria - NEVER shown to the solver agent
    validation_criteria: str
    known_answer: str


# ──────────────────────────────────────────────────────────────────────────────
# Problem Statements (what the solver sees)
# ──────────────────────────────────────────────────────────────────────────────

PROBLEM_1_STATEMENT = (
    "A line in the plane is called *sunny* if it is not parallel to any of the "
    "$x$-axis, the $y$-axis, and the line $x+y=0$. Let $n\\ge3$ be a given integer. "
    "Determine all nonnegative integers $k$ such that there exist $n$ distinct lines "
    "in the plane satisfying both of the following:\n"
    "* for all positive integers $a$ and $b$ with $a+b\\le n+1$, the point $(a,b)$ "
    "is on at least one of the lines; and\n"
    "* exactly $k$ of the $n$ lines are sunny."
)

PROBLEM_2_STATEMENT = (
    "Let $\\Omega$ and $\\Gamma$ be circles with centres $M$ and $N$, respectively, "
    "such that the radius of $\\Omega$ is less than the radius of $\\Gamma$. Suppose "
    "circles $\\Omega$ and $\\Gamma$ intersect at two distinct points $A$ and $B$. "
    "Line $MN$ intersects $\\Omega$ at $C$ and $\\Gamma$ at $D$, such that points $C$, "
    "$M$, $N$, and $D$ lie on the line in that order. Let $P$ be the circumcentre of "
    "triangle $ACD$. Line $AP$ intersects $\\Omega$ again at $E\\neq A$. Line $AP$ "
    "intersects $\\Gamma$ again at $F\\neq A$. Let $H$ be the orthocentre of triangle "
    "$PMN$.\n\n"
    "Prove that the line through $H$ parallel to $AP$ is tangent to the circumcircle "
    "of triangle $BEF$.\n\n"
    "(The orthocentre of a triangle is the point of intersection of its altitudes.)"
)

PROBLEM_3_STATEMENT = (
    "Let $\\mathbb N$ denote the set of positive integers. A function "
    "$f:\\mathbb N\\to\\mathbb N$ is said to be bonza if $f(a)$ divides "
    "$b^a-f(b)^{f(a)}$ for all positive integers $a$ and $b$. Determine the smallest "
    "real constant $c$ such that $f(n)\\le cn$ for all bonza functions $f$ and all "
    "positive integers $n$."
)

PROBLEM_4_STATEMENT = (
    "A proper divisor of a positive integer $N$ is a positive divisor of $N$ other "
    "than $N$ itself. The infinite sequence $a_1,a_2,\\ldots$ consists of positive "
    "integers, each of which has at least three proper divisors. For each $n\\ge1$, "
    "the integer $a_{n+1}$ is the sum of the three largest proper divisors of $a_n$.\n\n"
    "Determine all possible values of $a_1$."
)

PROBLEM_5_STATEMENT = (
    "Alice and Bazza are playing the inekoalaty game, a two-player game whose rules "
    "depend on a positive real number $\\lambda$ which is known to both players. On "
    "the $n$th turn of the game (starting with $n=1$) the following happens:\n"
    "* If $n$ is odd, Alice chooses a nonnegative real number $x_n$ such that "
    "$$x_1+x_2+\\cdots+x_n\\le\\lambda n.$$\n"
    "* If $n$ is even, Bazza chooses a nonnegative real number $x_n$ such that "
    "$$x_1^2+x_2^2+\\cdots+x_n^2\\le n.$$\n\n"
    "If a player cannot choose a suitable number $x_n$, the game ends and the other "
    "player wins. If the game goes on forever, neither player wins. All chosen numbers "
    "are known to both players.\n\n"
    "Determine all values of $\\lambda$ for which Alice has a winning strategy and "
    "all those for which Bazza has a winning strategy."
)

PROBLEM_6_STATEMENT = (
    "Consider a $2025\\times2025$ grid of unit squares. Matilda wishes to place on "
    "the grid some rectangular tiles, possibly of different sizes, such that each "
    "side of every tile lies on a grid line and every unit square is covered by at "
    "most one tile.\n\n"
    "Determine the minimum number of tiles Matilda needs to place so that each row "
    "and each column of the grid has exactly one unit square that is not covered by "
    "any tile."
)


# ──────────────────────────────────────────────────────────────────────────────
# Validation Criteria and Known Answers (NEVER shown to solver)
# ──────────────────────────────────────────────────────────────────────────────

IMO_2025_PROBLEMS: dict[int, IMOProblem] = {
    1: IMOProblem(
        number=1,
        statement=PROBLEM_1_STATEMENT,
        domain="Combinatorics",
        difficulty="easy",
        day=1,
        problem_type="determine",
        validation_criteria=(
            "The solution must correctly determine that the answer is k in {0, 1, 3} "
            "(for all n >= 3). The solution must:\n"
            "1. Show that k = 0, k = 1, and k = 3 are achievable (constructions).\n"
            "2. Prove that no other values of k are possible.\n"
            "3. The proof must handle the boundary/extremal cases correctly.\n"
            "4. All constructions must be explicitly verified to satisfy both conditions."
        ),
        known_answer="k ∈ {0, 1, 3}",
    ),
    2: IMOProblem(
        number=2,
        statement=PROBLEM_2_STATEMENT,
        domain="Geometry",
        difficulty="medium",
        day=1,
        problem_type="prove",
        validation_criteria=(
            "The solution must provide a complete and rigorous proof that the line "
            "through H parallel to AP is tangent to the circumcircle of triangle BEF. "
            "The proof must:\n"
            "1. Correctly identify key geometric relationships (e.g., P as circumcenter "
            "of ACD, H as orthocenter of PMN).\n"
            "2. Use valid angle chasing, power of a point, or coordinate/analytic methods.\n"
            "3. Clearly establish the tangency condition (distance equals radius, or "
            "the line meets the circle at exactly one point).\n"
            "4. Have no logical gaps or unjustified claims."
        ),
        known_answer="Proof required (tangency holds).",
    ),
    3: IMOProblem(
        number=3,
        statement=PROBLEM_3_STATEMENT,
        domain="Number Theory / Algebra",
        difficulty="hard",
        day=1,
        problem_type="determine",
        validation_criteria=(
            "The solution must correctly determine that c = 4. Specifically:\n"
            "1. Show c >= 4 by exhibiting a bonza function with f(4) = 16 = 4*4 "
            "(e.g., f(n) = 1 for odd n, f(4) = 16, f(n) = 2 for even n != 4).\n"
            "2. Show c <= 4 by proving f(n) <= 4n for all bonza f and all n.\n"
            "3. The proof that f(a) | b^a - f(b)^{f(a)} constrains f must be rigorous.\n"
            "4. Key steps include: showing f(n) | n^n (from P(n,n)), that f(p) = 1 for "
            "odd primes p (unless f = id), that f(n) is always a power of 2, and that "
            "f(n) <= 2^{v_2(n)+2} <= 4n where v_2 is the 2-adic valuation."
        ),
        known_answer="c = 4",
    ),
    4: IMOProblem(
        number=4,
        statement=PROBLEM_4_STATEMENT,
        domain="Number Theory",
        difficulty="medium",
        day=2,
        problem_type="determine",
        validation_criteria=(
            "The solution must correctly determine all possible values of a_1. The answer "
            "is: a_1 can be any positive integer of the form a_1 = 6 * 12^k * m, where "
            "k >= 0 is a nonnegative integer and m >= 1 is a positive integer with "
            "gcd(m, 10) = 1 (i.e., m is coprime to both 2 and 5). Equivalently, a_1 must "
            "be divisible by 2 and 3 but when all factors of 12 are removed (writing "
            "a_1 = 12^k * r with 12 not dividing r), the remaining factor r must equal "
            "6*m with gcd(m,10)=1. The solution must:\n"
            "1. Show these values work (the sequence is well-defined and infinite).\n"
            "2. Show no other values work.\n"
            "3. Key observations: elements must be divisible by 2 and 3; the three largest "
            "proper divisors of N are N/p1, N/p2, N/p3 for the three smallest prime-power "
            "divisors; analysis of what happens when 5 divides a term."
        ),
        known_answer="a_1 = 6 * 12^k * m where k >= 0 and gcd(m, 10) = 1",
    ),
    5: IMOProblem(
        number=5,
        statement=PROBLEM_5_STATEMENT,
        domain="Algebra / Game Theory",
        difficulty="hard",
        day=2,
        problem_type="determine",
        validation_criteria=(
            "The solution must correctly determine:\n"
            "- Alice has a winning strategy if and only if lambda > 1/sqrt(2).\n"
            "- Bazza has a winning strategy if and only if lambda < 1/sqrt(2).\n"
            "- For lambda = 1/sqrt(2), neither player wins (the game goes on forever).\n"
            "The proof must:\n"
            "1. For lambda > 1/sqrt(2): construct Alice's winning strategy and show "
            "Bazza is eventually forced to violate his constraint.\n"
            "2. For lambda < 1/sqrt(2): construct Bazza's winning strategy and show "
            "Alice is eventually forced to violate her constraint.\n"
            "3. For lambda = 1/sqrt(2): show neither player can force a win.\n"
            "4. Use rigorous analysis of the sum/sum-of-squares constraints."
        ),
        known_answer=(
            "Alice wins iff lambda > 1/sqrt(2); "
            "Bazza wins iff lambda < 1/sqrt(2); "
            "neither wins for lambda = 1/sqrt(2)."
        ),
    ),
    6: IMOProblem(
        number=6,
        statement=PROBLEM_6_STATEMENT,
        domain="Combinatorics",
        difficulty="very_hard",
        day=2,
        problem_type="determine",
        validation_criteria=(
            "The solution must correctly determine the minimum number of tiles is 2112. "
            "Since 2025 = 45^2, the answer is 45^2 + 2*45 - 3 = 2112. In general for "
            "n = k^2, the answer is k^2 + 2k - 3. The proof must:\n"
            "1. Show that 2112 tiles suffice by providing a valid construction (e.g., "
            "(k-1)^2 = 1936 tiles of size k x k plus 4(k-1) = 176 boundary tiles).\n"
            "2. Show that fewer than 2112 tiles are not sufficient (lower bound), e.g., "
            "using the Erdos-Szekeres theorem on the permutation of uncovered cells.\n"
            "3. The construction must explicitly satisfy: each row and each column has "
            "exactly one uncovered unit square, tiles are rectangles with sides on grid "
            "lines, no overlapping.\n"
            "NOTE: This is the hardest problem. The paper's pipeline (arXiv:2507.15855) "
            "failed to solve this problem with any of the three leading models."
        ),
        known_answer="2112",
    ),
}


def get_problem_statement(problem_number: int) -> str:
    """Get the problem statement only (no validation criteria or answer).

    This is what the solver agent sees.
    """
    if problem_number not in IMO_2025_PROBLEMS:
        raise ValueError(f"Problem {problem_number} not found. Valid: 1-6")
    return IMO_2025_PROBLEMS[problem_number].statement


def get_validation_info(problem_number: int) -> tuple[str, str]:
    """Get validation criteria and known answer (for the verifier only).

    Returns:
        Tuple of (validation_criteria, known_answer)
    """
    if problem_number not in IMO_2025_PROBLEMS:
        raise ValueError(f"Problem {problem_number} not found. Valid: 1-6")
    p = IMO_2025_PROBLEMS[problem_number]
    return p.validation_criteria, p.known_answer


def get_all_problem_numbers() -> list[int]:
    """Return all problem numbers."""
    return list(IMO_2025_PROBLEMS.keys())
