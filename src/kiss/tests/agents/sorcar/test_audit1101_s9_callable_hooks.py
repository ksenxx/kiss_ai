# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A callable that is no method where a SEA hook belongs is a SEA diagnostic, not a crash.

``_bound`` binds the hook attribute of a SEA class to the instance and
reports a non-method attribute (a string, a class, a ``staticmethod``)
as a ``SeaError``.  A builtin (``tools = len``) and a
``functools.partial`` passed its ``callable`` guard: the builtin has no
``__get__`` and the partial no ``__name__``, so the raw
``AttributeError`` escaped ``check_sea`` and aborted ``lint_all``
before the remaining scripts were checked.  The guard now admits
Python functions only, so both get the ``must be a method`` diagnostic.

Real script files, the real loader; nothing is mocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar.sea_commands import SeaError, check_sea
from kiss.agents.sorcar.sea_lint import lint_all

HOOKS = {"partial": "partial(list)", "builtin": "len"}


def _write(folder: Path, name: str, hook: str) -> Path:
    path = folder / name / f"{name}_sea.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "from functools import partial\n"
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
        "class BadSea(BaseSea):\n"
        "    def description(self):\n"
        "        return 'A non-method hook is a broken SEA, not a crash.'\n\n"
        f"    tools = {hook}\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("case", sorted(HOOKS))
def test_a_callable_non_method_hook_is_a_sea_error(tmp_path: Path, case: str) -> None:
    path = _write(tmp_path, case, HOOKS[case])
    with pytest.raises(SeaError, match="tools of SEA .* must be a method, got"):
        check_sea(path)


def test_lint_reports_the_scripts_as_broken_and_checks_the_rest(tmp_path: Path) -> None:
    """``lint_all`` reports both scripts and still lints the healthy one after them."""
    bad = [_write(tmp_path, case, hook) for case, hook in sorted(HOOKS.items())]
    healthy = tmp_path / "healthy" / "healthy_sea.py"
    healthy.parent.mkdir(parents=True)
    healthy.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
        "class HealthySea(BaseSea):\n"
        "    def settings(self, settings):\n"
        "        return settings | {'model': 'no-such-model-xyz'}\n",
        encoding="utf-8",
    )
    findings = lint_all([*bad, healthy])
    assert [(f.path, f.code) for f in findings] == [
        (bad[0], "broken"), (bad[1], "broken"), (healthy, "unknown-model"),
    ]
    assert all("must be a method" in f.message for f in findings[:2]), findings
