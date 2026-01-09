# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Parallel execution utilities using multiprocessing."""

from kiss.multiprocessing.multiprocess import (
    get_available_cores,
    run_functions_in_parallel,
    run_functions_in_parallel_with_kwargs,
)

__all__ = [
    "get_available_cores",
    "run_functions_in_parallel",
    "run_functions_in_parallel_with_kwargs",
]
