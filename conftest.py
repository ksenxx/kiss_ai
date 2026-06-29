# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Root pytest configuration.

Coverage is enabled by default through ``--cov``/``--cov-branch`` in the
``addopts`` setting of ``pyproject.toml``. Those flags are registered by the
``pytest-cov`` plugin, so disabling it with ``pytest -p no:cov`` removes the
options and pytest aborts with::

    error: unrecognized arguments: --cov=src/kiss --cov-branch

To keep ``pytest -p no:cov`` (and any other plugin-less invocation) working
without editing the command line, this module registers inert fallback
``--cov``/``--cov-branch`` options *only when the pytest-cov plugin is not
loaded*. They simply absorb the leftover ``addopts`` flags and do nothing, so
coverage stays off (the plugin is gone) while argument parsing still succeeds.
When pytest-cov is active these fallbacks are skipped so its real options are
used.

This module also raises ``RLIMIT_NOFILE`` (the per-process file-descriptor
soft limit) to at least ``_MIN_NOFILE_SOFT`` at import time. macOS ships with a
very low default soft limit (256) which is exhausted by the test suite's many
concurrent UDS sockets and subprocesses, surfacing as
``OSError: [Errno 24] Too many open files`` deep inside asyncio
``socket.accept()`` calls. Bumping the limit here means every pytest
invocation — CI, local, parallel splits, IDE — gets the higher cap without
relying on anyone remembering to run ``ulimit -n 4096`` in their shell.
"""

import sys

import pytest

# Target soft limit for RLIMIT_NOFILE. 4096 comfortably covers the full test
# suite (~4200 tests with sockets, subprocesses, tempfiles) while staying well
# below typical hard limits on Linux/macOS CI runners.
_MIN_NOFILE_SOFT = 4096


def _raise_nofile_soft_limit(target: int = _MIN_NOFILE_SOFT) -> None:
    """Raise the process's ``RLIMIT_NOFILE`` soft limit to at least *target*.

    Args:
        target: Desired minimum soft file-descriptor limit. The actual new
            soft limit will be ``min(target, hard_limit)`` so we never try to
            exceed the (immutable for unprivileged processes) hard cap.

    Returns:
        None. Best-effort: silently skipped on platforms that lack the
        ``resource`` module (e.g. Windows) and on the rare case where
        ``setrlimit`` rejects the call.
    """
    if sys.platform == "win32":
        return
    import resource  # POSIX-only; imported lazily so Windows imports stay clean.

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= target:
        return
    new_soft = target if hard == resource.RLIM_INFINITY else min(target, hard)
    if new_soft <= soft:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except (ValueError, OSError):
        # Some sandboxed environments cap setrlimit; nothing we can do, and
        # failing to raise the limit must not abort the test session.
        return
    print(f"[conftest] Raised RLIMIT_NOFILE soft limit: {soft} -> {new_soft}")


_raise_nofile_soft_limit()


def pytest_addoption(parser: pytest.Parser, pluginmanager: pytest.PytestPluginManager) -> None:
    """Register inert ``--cov`` fallbacks when pytest-cov is disabled.

    Args:
        parser: The pytest command-line option parser to add options to.
        pluginmanager: The pytest plugin manager, used to detect whether the
            ``pytest-cov`` plugin is loaded.

    Returns:
        None. Fallback options are added to *parser* in place only when the
        ``pytest-cov`` plugin is absent (e.g. under ``-p no:cov``).
    """
    if pluginmanager.hasplugin("pytest_cov"):
        # pytest-cov is active and already owns --cov/--cov-branch.
        return
    group = parser.getgroup("cov", "coverage reporting (disabled)")
    group.addoption(
        "--cov",
        action="append",
        nargs="?",
        const=True,
        default=[],
        dest="_cov_disabled",
        help="No-op fallback used when pytest-cov is disabled via -p no:cov.",
    )
    group.addoption(
        "--cov-branch",
        action="store_true",
        default=False,
        dest="_cov_branch_disabled",
        help="No-op fallback used when pytest-cov is disabled via -p no:cov.",
    )
