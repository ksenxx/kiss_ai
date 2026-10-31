# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The SEA contract check rejects every getter the SEA lint reports as removed.

``sea_contract.REMOVED_GETTERS`` used to be a hand-kept copy of
``sea_lint.REMOVED_GETTERS`` that had lost ``is_parallel``: a SEA script
defining ``is_parallel()`` passed ``assert_no_removed_getters`` although
the lint calls that getter removed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from kiss.agents.sorcar import sea_lint
from kiss.tests.agents.seas.sea_contract import assert_no_removed_getters

_SEA = (
    "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
    'class Sea(BaseSea):\n    """A SEA."""\n'
)


def _load(path: Path, source: str) -> ModuleType:
    """Write the SEA script *source* to *path* and import it as a module."""
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", sea_lint.REMOVED_GETTERS)
def test_a_getter_the_lint_calls_removed_fails_the_contract(tmp_path: Path, name: str) -> None:
    """Defined on the module or on the SEA class, the getter fails the contract check."""
    function = f"{_SEA}\n\ndef {name}():\n    return None\n"
    with pytest.raises(AssertionError, match=f"^module_getter.{name} is a removed getter"):
        assert_no_removed_getters(_load(tmp_path / "module_getter.py", function))
    method = f"{_SEA}\n    def {name}(self):\n        return None\n"
    with pytest.raises(AssertionError, match=f"^Sea.{name} is a removed getter"):
        assert_no_removed_getters(_load(tmp_path / "class_getter.py", method))
