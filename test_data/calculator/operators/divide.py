"""Division operator."""

symbol = "/"
precedence = 2


def eval(a: float, b: float) -> float:
    """Evaluate a / b."""
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b
