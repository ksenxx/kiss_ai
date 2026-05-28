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
"""

import pytest


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
