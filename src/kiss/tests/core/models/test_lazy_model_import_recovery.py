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

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_MISSING = "kiss_nonexistent_dep_xyz"


_PROBE = """
import json, sys

shadow = sys.argv[1]
sys.path.insert(0, shadow)

import kiss.core.models as models
from kiss.core.models.model_info import model

out = {}
try:
    models.AnthropicModel
    out["first_access"] = "imported"
except ImportError as exc:
    out["first_access"] = f"ImportError: {exc}"

try:
    model("claude-opus-4-7")
    out["factory"] = "constructed"
except BaseException as exc:
    out["factory"] = type(exc).__name__
    out["factory_message"] = str(exc)
    cause = exc.__cause__
    out["cause"] = None if cause is None else f"{type(cause).__name__}: {cause}"

# The environment heals: the shadow goes away and the real SDK is
# importable again, with no interpreter restart.
sys.path.remove(shadow)
for name in [k for k in sys.modules if k.split(".")[0] == "anthropic"]:
    del sys.modules[name]

out["second_access"] = models.AnthropicModel is not None
print(json.dumps(out))
"""


@pytest.fixture
def shadow_anthropic(tmp_path: Path) -> Path:
    """A real ``anthropic`` package whose import raises from inside."""
    package = tmp_path / "shadow" / "anthropic"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        f"import {_MISSING}\n", encoding="utf-8",
    )
    return tmp_path / "shadow"


def _probe(shadow: Path) -> dict[str, object]:
    """Run the probe in a real child process and return its report."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(shadow)],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    report: dict[str, object] = json.loads(proc.stdout)
    return report


class TestFailedLazyImportIsNotCached:
    """A failed SDK import must not poison the process forever."""

    def test_import_recovers_once_the_environment_heals(
        self, shadow_anthropic: Path,
    ) -> None:
        """The second access must succeed after the broken shadow is gone."""
        report = _probe(shadow_anthropic)

        assert report["second_access"] is True, (
            "kiss.core.models cached the failed import as None, so the "
            "class can never be loaded again without restarting"
        )

    def test_in_sdk_import_error_is_reported_with_its_cause(
        self, shadow_anthropic: Path,
    ) -> None:
        """The real ImportError must reach the user, not just "not installed"."""
        report = _probe(shadow_anthropic)

        assert report["first_access"] == f"ImportError: No module named '{_MISSING}'"
        assert report["factory"] == "KISSError"
        cause = report["cause"]
        assert isinstance(cause, str) and _MISSING in cause, (
            f"the KISSError lost the underlying ImportError: cause={cause!r}"
        )


class TestLazyImportStillWorks:
    """The healthy path and the attribute contract are unchanged."""

    def test_every_advertised_class_imports(self) -> None:
        """Each name in ``__all__`` resolves to a real class."""
        import kiss.core.models as models

        for name in models.__all__:
            assert isinstance(getattr(models, name), type), name

    def test_unknown_attribute_still_raises_attribute_error(self) -> None:
        """A name outside the lazy table must raise ``AttributeError``."""
        import kiss.core.models as models

        with pytest.raises(AttributeError):
            models.NoSuchModel  # noqa: B018


if __name__ == "__main__":  # pragma: no cover - manual run
    raise SystemExit(pytest.main([__file__]))
