"""Operator registry built from static imports.

To add a new operator, create a module with `symbol`, `precedence`, and
`eval(a, b)`, then add a static import here and append it to OPERATORS.
"""

from test_data.calculator.operators import add, divide, multiply, subtract
OPERATORS: dict[str, object] = {
    add.symbol: add,
    subtract.symbol: subtract,
    multiply.symbol: multiply,
    divide.symbol: divide,
}