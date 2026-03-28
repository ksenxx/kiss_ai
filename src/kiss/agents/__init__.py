"""KISS agents package with pre-built agent implementations."""

__all__ = [
    "prompt_refiner_agent",
    "get_run_simple_coding_agent",
    "run_bash_task_in_sandboxed_ubuntu_latest",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazily import convenience functions from kiss.agents.kiss."""
    if name in __all__:
        from kiss.agents.kiss import (
            get_run_simple_coding_agent,
            prompt_refiner_agent,
            run_bash_task_in_sandboxed_ubuntu_latest,
        )

        _exports = {
            "prompt_refiner_agent": prompt_refiner_agent,
            "get_run_simple_coding_agent": get_run_simple_coding_agent,
            "run_bash_task_in_sandboxed_ubuntu_latest": run_bash_task_in_sandboxed_ubuntu_latest,
        }
        globals().update(_exports)
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
