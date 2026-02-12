"""Demo showing KISSAgent capabilities on a simple arithmetic task with streaming output."""

from kiss.core.kiss_agent import KISSAgent


def simple_calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression.

    Args:
        expression: The arithmetic expression to evaluate (e.g., '2+2', '10*5', '(3+4)*2')

    Returns:
        The result of the expression as a string.
    """
    compiled = compile(expression, "<string>", "eval")
    return str(eval(compiled, {"__builtins__": {}}, {}))


def main() -> None:
    agent = KISSAgent("Arithmetic Demo Agent")
    prompt = (
        "You are a helpful math assistant. Use the simple_calculator tool to solve the "
        "following problems step by step. You MUST think loud.:\n"
        "1. What is 127 * 843?\n"
        "2. What is (1234 + 5678) / 2?\n"
        "3. What is 2**10 - 1?\n"
        "Report each result clearly, then call finish with a summary of all three answers."
    )

    agent.run(
        model_name="claude-sonnet-4-5",
        prompt_template=prompt,
        tools=[simple_calculator],
        max_steps=20,
        max_budget=1.0,
        print_to_browser=True,
        print_to_console=True,
    )


if __name__ == "__main__":
    main()
