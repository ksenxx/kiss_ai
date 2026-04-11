"""Expression tokenizer and evaluator using registered operators."""

from __future__ import annotations

import types

from test_data.calculator.operators import OPERATORS


def tokenize(expression: str) -> list[str]:
    """Split an expression string into number and operator tokens.

    Handles negative numbers at the start or after an operator/open-paren.
    """
    tokens: list[str] = []
    i = 0
    while i < len(expression):
        ch = expression[i]
        if ch.isspace():
            i += 1
            continue
        if ch in ("(", ")"):
            tokens.append(ch)
            i += 1
            continue
        # Check for a number (including leading minus for negative numbers)
        if ch.isdigit() or ch == "." or (
            ch == "-"
            and (not tokens or tokens[-1] == "(" or tokens[-1] in OPERATORS)
        ):
            start = i
            if ch == "-":
                i += 1
            while i < len(expression) and (expression[i].isdigit() or expression[i] == "."):
                i += 1
            token = expression[start:i]
            if token == "-":
                # Standalone minus sign — it's an operator, not a number
                tokens.append(token)
            else:
                tokens.append(token)
            continue
        # Must be an operator symbol
        # Support multi-char operators by trying longest match first
        matched = False
        for length in range(min(3, len(expression) - i), 0, -1):
            candidate = expression[i : i + length]
            if candidate in OPERATORS:
                tokens.append(candidate)
                i += length
                matched = True
                break
        if not matched:
            raise ValueError(f"unexpected character: {ch!r}")
    return tokens


def evaluate(expression: str) -> float:
    """Parse and evaluate an arithmetic expression string.

    Supports operator precedence and parentheses.
    """
    tokens = tokenize(expression)
    pos = [0]  # mutable index for recursive descent

    def _parse_expr(min_prec: int) -> float:
        left = _parse_atom()
        while pos[0] < len(tokens) and tokens[pos[0]] in OPERATORS:
            op_mod = _get_operator(tokens[pos[0]])
            prec = op_mod.precedence
            if prec < min_prec:
                break
            pos[0] += 1
            right = _parse_expr(prec + 1)
            left = op_mod.eval(left, right)
        return left

    def _parse_atom() -> float:
        if pos[0] >= len(tokens):
            raise ValueError("unexpected end of expression")
        token = tokens[pos[0]]
        if token == "(":
            pos[0] += 1
            result = _parse_expr(0)
            if pos[0] >= len(tokens) or tokens[pos[0]] != ")":
                raise ValueError("missing closing parenthesis")
            pos[0] += 1
            return result
        try:
            value = float(token)
            pos[0] += 1
            return value
        except ValueError:
            raise ValueError(f"unexpected token: {token!r}") from None

    result = _parse_expr(0)
    if pos[0] < len(tokens):
        raise ValueError(f"unexpected token: {tokens[pos[0]]!r}")
    return result


def _get_operator(symbol: str) -> types.ModuleType:
    """Look up an operator module by its symbol."""
    mod = OPERATORS.get(symbol)
    if mod is None:
        raise ValueError(f"unknown operator: {symbol!r}")
    return mod  # type: ignore[return-value]
