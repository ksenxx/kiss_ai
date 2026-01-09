# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Test suite for multiprocessing functionality with compute-heavy tasks."""

import time
import unittest

from kiss.multiprocessing.multiprocess import (
    get_available_cores,
    run_functions_in_parallel,
)


def compute_factorial(n: int) -> int:
    """Compute factorial of n - compute-heavy operation."""
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def find_primes_in_range(start: int, end: int) -> list[int]:
    """Find all prime numbers in the given range - compute-heavy operation."""
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
    """Multiply two matrices of given size - compute-heavy operation."""
    matrix_a = [[i * size + j for j in range(size)] for i in range(size)]
    matrix_b = [[i * size + j for j in range(size)] for i in range(size)]

    result = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def fibonacci_sequence(n: int) -> int:
    """Compute nth Fibonacci number using iterative method."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def sum_of_squares(n: int) -> int:
    """Compute sum of squares from 1 to n."""
    return sum(i * i for i in range(1, n + 1))


class TestMultiprocess(unittest.TestCase):
    """Test suite for multiprocessing with compute-heavy tasks."""

    def test_get_available_cores(self):
        """Test that we can get the number of available cores."""
        cores = get_available_cores()
        self.assertIsInstance(cores, int)
        self.assertGreater(cores, 0)

    def test_parallel_execution(self):
        """Test parallel execution with various compute-heavy tasks."""
        # Create diverse compute-heavy tasks
        tasks = [
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

        # Measure parallel execution time
        start_parallel = time.time()
        parallel_results = run_functions_in_parallel(tasks)
        parallel_time = time.time() - start_parallel

        # Measure sequential execution time
        start_sequential = time.time()
        sequential_results = []
        for func, args in tasks:
            sequential_results.append(func(*args))
        sequential_time = time.time() - start_sequential

        # Verify results match
        self.assertEqual(parallel_results, sequential_results)

        # Both should complete
        self.assertGreater(parallel_time, 0)
        self.assertGreater(sequential_time, 0)

    def test_result_order_preservation(self):
        """Test that results are returned in the same order as input tasks."""
        tasks = [
            (compute_factorial, [100]),
            (sum_of_squares, [1000]),
            (fibonacci_sequence, [20]),
        ]

        results = run_functions_in_parallel(tasks)

        self.assertEqual(results[0], compute_factorial(100))
        self.assertEqual(results[1], sum_of_squares(1000))
        self.assertEqual(results[2], fibonacci_sequence(20))

    def test_edge_cases(self):
        """Test edge cases: single task and empty task list."""
        # Single task
        single_tasks = [(compute_factorial, [100])]
        single_results = run_functions_in_parallel(single_tasks)
        self.assertEqual(len(single_results), 1)
        self.assertEqual(single_results[0], compute_factorial(100))

        # Empty tasks
        empty_tasks = []
        empty_results = run_functions_in_parallel(empty_tasks)
        self.assertEqual(len(empty_results), 0)


if __name__ == "__main__":
    unittest.main()
