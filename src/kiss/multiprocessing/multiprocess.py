# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Parallel execution of Python functions using multiprocessing."""

import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any


def _run_parallel(
    tasks: list[tuple[Callable[..., Any], list[Any], dict[str, Any]]],
) -> list[Any]:
    """Run tasks in parallel, returning results in input order.

    Args:
        tasks: List of (function, args, kwargs) tuples.

    Returns:
        List of results in the same order as the input tasks.
    """
    if not tasks:
        return []

    max_workers = min(multiprocessing.cpu_count(), len(tasks))
    results: list[Any] = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(func, *args, **kwargs): idx
            for idx, (func, args, kwargs) in enumerate(tasks)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                raise Exception(f"Function at index {idx} failed with error: {e}") from e

    return results


def run_functions_in_parallel(
    tasks: list[tuple[Callable[..., Any], list[Any]]],
) -> list[Any]:
    """Run a list of functions in parallel using multiprocessing.

    Args:
        tasks: List of tuples, where each tuple contains (function, arguments).

    Returns:
        List of results from each function, in the same order as the input tasks.

    Example:
        >>> def add(a, b):
        ...     return a + b
        >>> tasks = [(add, [1, 2]), (multiply, [3, 4])]
        >>> results = run_functions_in_parallel(tasks)
    """
    return _run_parallel([(func, args, {}) for func, args in tasks])


def run_functions_in_parallel_with_kwargs(
    functions: list[Callable[..., Any]],
    args_list: list[list[Any]] | None = None,
    kwargs_list: list[dict[str, Any]] | None = None,
) -> list[Any]:
    """Run a list of functions in parallel using multiprocessing with support for kwargs.

    Args:
        functions: List of callable functions to execute.
        args_list: Optional list of argument lists for positional arguments.
        kwargs_list: Optional list of keyword argument dictionaries.

    Returns:
        List of results from each function, in the same order as the input functions.

    Example:
        >>> def greet(name, title="Mr."):
        ...     return f"Hello, {title} {name}!"
        >>> results = run_functions_in_parallel_with_kwargs(
        ...     [greet, greet], [["Alice"], ["Bob"]], [{"title": "Dr."}, {}]
        ... )
    """
    if args_list is None:
        args_list = [[] for _ in range(len(functions))]
    if kwargs_list is None:
        kwargs_list = [{} for _ in range(len(functions))]

    if len(functions) != len(args_list):
        raise ValueError(
            f"Number of functions ({len(functions)}) must match "
            f"number of argument lists ({len(args_list)})"
        )
    if len(functions) != len(kwargs_list):
        raise ValueError(
            f"Number of functions ({len(functions)}) must match "
            f"number of kwargs lists ({len(kwargs_list)})"
        )

    return _run_parallel(list(zip(functions, args_list, kwargs_list)))


def get_available_cores() -> int:
    """Get the number of available CPU cores."""
    return multiprocessing.cpu_count()
