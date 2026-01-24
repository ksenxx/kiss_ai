# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Parallel execution of Python functions using multiprocessing."""

import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any


def run_functions_in_parallel(
    tasks: list[tuple[Callable[..., Any], list[Any]]],
) -> list[Any]:
    """Run a list of functions in parallel using multiprocessing.

    Args:
        tasks: List of tuples, where each tuple contains (function, arguments).
               Each function is a callable, and arguments is a list/tuple that can
               be unpacked with *args.

    Returns:
        List of results from each function, in the same order as the input tasks.

    Raises:
        Exception: Any exception raised by the functions will be propagated.

    Example:
        >>> def add(a, b):
        ...     return a + b
        >>> def multiply(x, y):
        ...     return x * y
        >>> tasks = [(add, [1, 2]), (multiply, [3, 4])]
        >>> results = run_functions_in_parallel(tasks)
        >>> print(results)  # [3, 12]
    """
    # Handle empty tasks list
    if len(tasks) == 0:
        return []

    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()

    # If we have fewer tasks than cores, use the number of tasks
    max_workers = min(num_cores, len(tasks))

    results = [None] * len(tasks)

    # Use ProcessPoolExecutor for better exception handling and result ordering
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(func, *args): idx for idx, (func, args) in enumerate(tasks)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                # Re-raise with context about which function failed
                raise Exception(f"Function at index {idx} failed with error: {e}") from e

    return results


def run_functions_in_parallel_with_kwargs(
    functions: list[Callable[..., Any]],
    args_list: list[list[Any]] | None = None,
    kwargs_list: list[dict[str, Any]] | None = None,
) -> list[Any]:
    """Run a list of functions in parallel using multiprocessing with support for kwargs.

    Args:
        functions: List of callable functions to execute.
        args_list: Optional list of argument lists for positional arguments.
                   If None, an empty list is used for each function.
        kwargs_list: Optional list of keyword argument dictionaries.
                     If None, an empty dict is used for each function.

    Returns:
        List of results from each function, in the same order as the input functions.

    Raises:
        ValueError: If the number of functions doesn't match the number of argument lists.
        Exception: Any exception raised by the functions will be propagated.

    Example:
        >>> def greet(name, title="Mr."):
        ...     return f"Hello, {title} {name}!"
        >>> functions = [greet, greet]
        >>> args_list = [["Alice"], ["Bob"]]
        >>> kwargs_list = [{"title": "Dr."}, {}]
        >>> results = run_functions_in_parallel_with_kwargs(functions, args_list, kwargs_list)
        >>> print(results)  # ["Hello, Dr. Alice!", "Hello, Mr. Bob!"]
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

    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()

    # If we have fewer tasks than cores, use the number of tasks
    max_workers = min(num_cores, len(functions))

    results = [None] * len(functions)

    # Use ProcessPoolExecutor for better exception handling and result ordering
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(func, *args, **kwargs): idx
            for idx, (func, args, kwargs) in enumerate(zip(functions, args_list, kwargs_list))
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                # Re-raise with context about which function failed
                raise Exception(f"Function at index {idx} failed with error: {e}") from e

    return results


def get_available_cores() -> int:
    """Get the number of available CPU cores.

    Returns:
        Number of CPU cores available on the system.
    """
    return multiprocessing.cpu_count()
