"""Test suite for multiprocessing functionality with compute-heavy tasks."""

import time
import unittest
from collections.abc import Callable
from typing import Any

from kiss.multiprocessing.multiprocess import (
    get_available_cores,
    run_functions_in_parallel,
)


def compute_factorial(n: int) -> int:
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def find_primes_in_range(start: int, end: int) -> list[int]:
    primes = []
    for num in range(start, end + 1):
        if num > 1:
            is_prime = True
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(num)
    return primes


def matrix_multiply(size: int) -> list[list[int]]:
    matrix_a = [[i * size + j for j in range(size)] for i in range(size)]
    matrix_b = [[i * size + j for j in range(size)] for i in range(size)]
    result = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result


def fibonacci_sequence(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def sum_of_squares(n: int) -> int:
    return sum(i * i for i in range(1, n + 1))


class TestMultiprocess(unittest.TestCase):
    def test_get_available_cores(self) -> None:
        cores = get_available_cores()
        self.assertIsInstance(cores, int)
        self.assertGreater(cores, 0)

    def test_parallel_execution(self) -> None:
        tasks: list[tuple[Callable[..., Any], list[Any]]] = [
            (matrix_multiply, [200]),
            (fibonacci_sequence, [200000]),
            (sum_of_squares, [20000000]),
            (compute_factorial, [6000]),
            (find_primes_in_range, [200000, 250000]),
            (matrix_multiply, [180]),
            (fibonacci_sequence, [180000]),
            (sum_of_squares, [18000000]),
            (compute_factorial, [5500]),
            (find_primes_in_range, [250000, 300000]),
        ]

        start_parallel = time.time()
        parallel_results = run_functions_in_parallel(tasks)
        parallel_time = time.time() - start_parallel

        start_sequential = time.time()
        sequential_results = [func(*args) for func, args in tasks]
        sequential_time = time.time() - start_sequential

        self.assertEqual(parallel_results, sequential_results)
        self.assertGreater(parallel_time, 0)
        self.assertGreater(sequential_time, 0)

    def test_result_order_preservation(self) -> None:
        tasks = [
            (compute_factorial, [100]),
            (sum_of_squares, [1000]),
            (fibonacci_sequence, [20]),
        ]
        results = run_functions_in_parallel(tasks)
        self.assertEqual(results[0], compute_factorial(100))
        self.assertEqual(results[1], sum_of_squares(1000))
        self.assertEqual(results[2], fibonacci_sequence(20))

    def test_edge_cases(self) -> None:
        single_results = run_functions_in_parallel([(compute_factorial, [100])])
        self.assertEqual(len(single_results), 1)
        self.assertEqual(single_results[0], compute_factorial(100))

        empty_tasks: list[tuple[Callable[..., Any], list[Any]]] = []
        self.assertEqual(len(run_functions_in_parallel(empty_tasks)), 0)


if __name__ == "__main__":
    unittest.main()
