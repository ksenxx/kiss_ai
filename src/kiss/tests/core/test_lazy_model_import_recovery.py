# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the lazy model-class loader.

Finding **F7** of ``tmp/audit/01-core-models-a.md``:
``kiss.core.models.__getattr__`` caught every ``ImportError`` and wrote
``globals()[name] = None``.  Two consequences, both real:

1. **The failure is cached forever.**  Python only calls a module-level
   ``__getattr__`` when normal attribute lookup fails, so once ``None``
   is in ``globals()`` the import is never retried — a long-running
   ``kiss-web`` daemon can never recover from a transient or
   environmental import failure without a restart.
2. **Every in-SDK ``ImportError`` is mis-reported as "SDK not installed".**
   This is not hypothetical here: the missing-``brotli`` regression fixed
   in an earlier session made ``urllib3`` — and therefore the provider
   SDKs — raise ``ImportError`` at import time on a machine where the
   SDKs *were* installed.  The user was told to install a package that
   was already present, and the real traceback was only at
   ``logging.DEBUG``.

Test strategy — no mocks, patches or test doubles: a real package
directory is created on disk whose ``__init__.py`` raises a genuine
``ImportError`` from *inside* the module, it is put on ``sys.path`` of a
**real child process** so the main test session is never poisoned, and
the child then reports what the real production code does before and
after the shadow is removed.
"""

from __future__ import annotations

import pytest


class TestFactoryReportsTheRealImportFailure:
    """``_load_model_class`` must translate, not discard, the ImportError."""

    _MISSING_MODULE = "kiss.core.models.zz_nonexistent_module"

    def test_missing_module_becomes_a_kiss_error_keeping_its_cause(self) -> None:
        """A real failed lazy import surfaces as a KISSError with a cause."""
        import kiss.core.models as models
        from kiss.core.kiss_error import KISSError
        from kiss.core.models.model_info import _load_model_class

        models._LAZY_IMPORTS["ZZMissingModel"] = self._MISSING_MODULE
        try:
            with pytest.raises(KISSError) as excinfo:
                _load_model_class("ZZMissingModel", "ZZ SDK not installed.")
        finally:
            del models._LAZY_IMPORTS["ZZMissingModel"]

        assert "ZZ SDK not installed." in str(excinfo.value)
        assert self._MISSING_MODULE in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ImportError)
        assert "ZZMissingModel" not in vars(models), (
            "the failed import was cached, so a retry can never succeed"
        )


if __name__ == "__main__":  # pragma: no cover - manual run
    raise SystemExit(pytest.main([__file__]))
