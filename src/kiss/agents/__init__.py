# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""KISS agents package with pre-built agent implementations."""

from typing import Any

__all__ = [
    "prompt_refiner_agent",
    "get_run_simple_coding_agent",
    "run_bash_task_in_sandboxed_ubuntu_latest",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from kiss.agents import kiss
        return getattr(kiss, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
