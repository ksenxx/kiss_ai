"""Demo showing KISSAgent capabilities on a simple arithmetic task with streaming output."""

import time

from rich.panel import Panel

from kiss.agents.coding_agents.print_to_browser import BrowserPrinter
from kiss.agents.coding_agents.print_to_console import ConsolePrinter
from kiss.core import config as config_module
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
    console_printer = ConsolePrinter()
    browser_printer = BrowserPrinter()
    browser_printer.start()
    browser_printer.reset()

    suppress_next_tool_result = False

    async def token_callback(token: str) -> None:
        nonlocal suppress_next_tool_result
        if suppress_next_tool_result:
            suppress_next_tool_result = False
            return
        console_printer._stream_delta(token)
        browser_printer._broadcast({"type": "text_delta", "text": token})

    config_module.DEFAULT_CONFIG.agent.verbose = False

    agent = KISSAgent("Arithmetic Demo Agent")
    prompt = (
        "You are a helpful math assistant. Use the simple_calculator tool to solve the "
        "following problems step by step. You MUST think loud.:\n"
        "1. What is 127 * 843?\n"
        "2. What is (1234 + 5678) / 2?\n"
        "3. What is 2**10 - 1?\n"
        "Report each result clearly, then call finish with a summary of all three answers."
    )

    console = console_printer._console
    console.print()
    console.print(
        Panel(prompt, title="[bold]Task[/bold]", border_style="cyan", padding=(1, 2))
    )
    console.print()

    def printing_calculator(expression: str) -> str:
        """Evaluate a simple arithmetic expression.

        Args:
            expression: The arithmetic expression to evaluate (e.g., '2+2', '10*5', '(3+4)*2')

        Returns:
            The result of the expression as a string.
        """
        nonlocal suppress_next_tool_result
        console_printer._flush_newline()
        console_printer._format_tool_call("simple_calculator", {"expression": expression})
        browser_printer._broadcast({"type": "text_end"})
        browser_printer._format_tool_call("simple_calculator", {"expression": expression})
        result = simple_calculator(expression)
        console_printer._print_tool_result(result, is_error=False)
        browser_printer._print_tool_result(result, is_error=False)
        suppress_next_tool_result = True
        return result

    start_time = time.time()
    result = agent.run(
        model_name="claude-sonnet-4-5",
        prompt_template=prompt,
        tools=[printing_calculator],
        is_agentic=True,
        max_steps=20,
        max_budget=1.0,
        token_callback=token_callback,
    )
    elapsed = time.time() - start_time

    console_printer._flush_newline()
    console.print()
    console.print(
        Panel(
            result or "(no result)",
            title="Result",
            subtitle=(
                f"steps={agent.step_count}  tokens={agent.total_tokens_used}  "
                f"cost=${agent.budget_used:.4f}  time={elapsed:.1f}s"
            ),
            border_style="bold green",
            padding=(1, 2),
        )
    )

    browser_printer._broadcast({"type": "text_end"})
    browser_printer._broadcast({
        "type": "result",
        "text": result or "(no result)",
        "step_count": agent.step_count,
        "total_tokens": agent.total_tokens_used,
        "cost": f"${agent.budget_used:.4f}",
    })
    browser_printer.stop()


if __name__ == "__main__":
    main()
