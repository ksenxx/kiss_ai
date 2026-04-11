"""CLI entry point for the extensible calculator."""

import sys

from test_data.calculator.evaluator import evaluate


def main(argv: list[str] | None = None) -> int:
    """Run the calculator CLI.

    Usage: python -m test_data.calculator.cli '<expression>'
    """
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: python -m test_data.calculator.cli '<expression>'", file=sys.stderr)
        return 1
    expression = " ".join(args)
    try:
        result = evaluate(expression)
    except (ValueError, ZeroDivisionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    # Print as int if the result is a whole number
    print(int(result) if result == int(result) else result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
