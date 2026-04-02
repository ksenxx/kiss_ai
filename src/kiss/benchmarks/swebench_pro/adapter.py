# Author: Koushik Sen (ksen@berkeley.edu)

"""Convert SWE-bench Pro instances into sorcar task prompts."""


def make_sorcar_task(instance: dict) -> str:
    """Build a sorcar task prompt from a SWE-bench Pro instance.

    Returns a task string that tells sorcar:
    - The repo is already cloned at /app
    - The issue to fix (problem_statement)
    - To produce a git diff as the solution

    Args:
        instance: A SWE-bench Pro dataset row with at least
            'problem_statement' and 'repo' fields.

    Returns:
        A formatted task prompt string for sorcar.
    """
    return (
        f"You are working on the repository at /app.\n\n"
        f"## Issue\n{instance['problem_statement']}\n\n"
        f"## Instructions\n"
        f"Fix the issue described above. When done, produce a unified diff "
        f"(git diff) of your changes.\n"
    )
