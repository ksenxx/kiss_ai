# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Docker wrapper module for the KISS agent framework."""

__all__ = ["DockerManager", "DockerTools"]

_LAZY_IMPORTS = {
    "DockerManager": "kiss.docker.docker_manager",
    "DockerTools": "kiss.docker.docker_tools",
}


def __getattr__(name: str) -> type:
    """Lazily import Docker classes to avoid pulling in docker SDK at import time."""
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        cls: type = getattr(module, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
