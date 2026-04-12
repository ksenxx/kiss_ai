"""Tests for the extensible CLI calculator."""

import subprocess
import sys

import pytest

from test_data.calculator.evaluator import _get_operator, evaluate, tokenize
from test_data.calculator.cli import main
from test_data.calculator.operators import OPERATORS


# --- Operator registry tests ---


class TestOperatorRegistry:
    def test_all_operators_registered(self):
        assert set(OPERATORS.keys()) == {"+", "-", "*", "/"}
    def test_each_operator_has_eval_and_precedence(self):
        for sym, mod in OPERATORS.items():
            assert hasattr(mod, "eval"), f"{sym} missing eval"
            assert hasattr(mod, "precedence"), f"{sym} missing precedence"
            assert hasattr(mod, "symbol"), f"{sym} missing symbol"

    def test_mul_div_higher_precedence_than_add_sub(self):
        from test_data.calculator.operators import add, subtract, multiply, divide
        assert multiply.precedence > add.precedence
        assert divide.precedence > subtract.precedence

    def test_add(self):
        from test_data.calculator.operators import add
        assert add.eval(0.1, 0.2) == pytest.approx(0.3)

    def test_subtract(self):
        from test_data.calculator.operators import subtract
        assert subtract.eval(5, 3) == 2
        assert subtract.eval(0, 5) == -5

    def test_multiply(self):
        from test_data.calculator.operators import multiply
        assert multiply.eval(4, 3) == 12
        assert multiply.eval(-2, 3) == -6
        assert multiply.eval(0, 100) == 0

    def test_divide(self):
        from test_data.calculator.operators import divide
        assert divide.eval(10, 2) == 5
        assert divide.eval(7, 2) == 3.5

    def test_divide_by_zero(self):
        from test_data.calculator.operators import divide
        with pytest.raises(ZeroDivisionError):
            divide.eval(1, 0)


# --- Tokenizer tests ---


class TestTokenizer:
    def test_simple(self):
        assert tokenize("2 + 3") == ["2", "+", "3"]

    def test_no_spaces(self):
        assert tokenize("2+3") == ["2", "+", "3"]

    def test_standalone_minus_as_operator(self):
        # "-" followed by another operator: parsed as operator, not negative number
        assert tokenize("- + 3") == ["-", "+", "3"]

    def test_unknown_char_raises(self):
        with pytest.raises(ValueError, match="unexpected character"):
            tokenize("2 @ 3")


# --- Evaluator tests ---


class TestEvaluator:
    def test_addition(self):
        assert evaluate("2 + 3") == 5

    def test_subtraction(self):
        assert evaluate("10 - 4") == 6

    def test_multiplication(self):
        assert evaluate("3 * 7") == 21

    def test_division(self):
        assert evaluate("15 / 3") == 5

    def test_precedence_mul_before_add(self):
        assert evaluate("2 + 3 * 4") == 14

    def test_precedence_div_before_sub(self):
        assert evaluate("10 - 6 / 3") == 8

    def test_chained_operations(self):
        assert evaluate("1 + 2 + 3 + 4") == 10

    def test_mixed_precedence(self):
        assert evaluate("2 * 3 + 4 * 5") == 26

    def test_decimal_result(self):
        assert evaluate("7 / 2") == 3.5

    def test_divide_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            evaluate("1 / 0")

    def test_empty_expression(self):
        with pytest.raises(ValueError):
            evaluate("")

    def test_missing_closing_paren(self):
        with pytest.raises(ValueError, match="missing closing parenthesis"):
            evaluate("(2 + 3")

    def test_trailing_operator(self):
        with pytest.raises(ValueError):
            evaluate("2 +")

    def test_extra_token(self):
        with pytest.raises(ValueError, match="unexpected token"):
            evaluate("2 3")

    def test_unknown_operator_via_get_operator(self):
        with pytest.raises(ValueError, match="unknown operator"):
            _get_operator("^")


# --- CLI tests ---


class TestCLI:
    def test_basic_expression(self, capsys):
        ret = main(["2 + 3"])
        captured = capsys.readouterr()
        assert ret == 0
        assert captured.out.strip() == "5"

    def test_precedence(self, capsys):
        ret = main(["2 + 3 * 4"])
        captured = capsys.readouterr()
        assert ret == 0
        assert captured.out.strip() == "14"

    def test_decimal_output(self, capsys):
        ret = main(["7 / 2"])
        captured = capsys.readouterr()
        assert ret == 0
        assert captured.out.strip() == "3.5"

    def test_no_args(self, capsys):
        ret = main([])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Usage" in captured.err

    def test_invalid_expression(self, capsys):
        ret = main(["2 @ 3"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_subprocess_integration(self):
        result = subprocess.run(
            [sys.executable, "-m", "test_data.calculator.cli", "4 + 5"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "9"
