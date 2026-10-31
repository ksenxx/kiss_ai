# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``load_layers`` and ``sea_layers`` reuse SEAs already loaded instead of importing them again.

The bare-command fallback of a slash run rebuilds the run's layers
without the slash SEA; the model-picker SEA under it was loaded already,
and loading its file again runs its import a second time.
``task_runner._reload_layers`` re-implemented the layer walk of
``load_layers`` to avoid that; both loaders now take the SEAs already
loaded (``loaded``) and execute only the files not among them.
Real SEA files that tally every import; nothing is mocked.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.sorcar.sea_apply import load_layers
from kiss.agents.sorcar.sea_commands import sea_layers

_COUNTING_SEA = (
    "from kiss.agents.seas.base.base_sea import BaseSea\n"
    "with open({tally!r}, 'a') as tally:\n"
    "    tally.write({name!r} + '\\n')\n"
    "class Sea(BaseSea):\n"
    "    pass\n"
)


def _sea(directory: Path, name: str, tally: Path) -> Path:
    """Write a SEA *name* that appends its name to *tally* each time its file runs."""
    path = directory / f"{name}_sea.py"
    path.write_text(_COUNTING_SEA.format(tally=str(tally), name=name), encoding="utf-8")
    return path


def _imports(tally: Path) -> list[str]:
    return sorted(tally.read_text(encoding="utf-8").split()) if tally.exists() else []


def test_load_layers_executes_only_the_files_not_already_loaded(tmp_path: Path) -> None:
    tally = tmp_path / "tally.txt"
    picker, slash, prior = (_sea(tmp_path, name, tally) for name in ("picker", "slash", "prior"))
    layers = load_layers({"seaPath": str(slash)}, base=picker)
    assert _imports(tally) == ["picker", "slash"]

    # The fallback with no seaPath keeps the picker's SEA alone: reused.
    (alone,) = load_layers({}, base=picker, loaded=layers)
    assert alone is layers[0]
    # With the tab's prior SEA: the picker's reused, the prior one loaded.
    picker_layer, prior_layer = load_layers({"seaPath": str(prior)}, base=picker, loaded=layers)
    assert picker_layer is layers[0] and prior_layer.path == prior
    assert _imports(tally) == ["picker", "prior", "slash"]
    # No SEA and no picker: the bare root layer, whatever was loaded.
    (bare,) = load_layers({}, loaded=layers)
    assert type(bare) is BaseSea


def test_sea_layers_without_loaded_seas_executes_every_file(tmp_path: Path) -> None:
    tally = tmp_path / "tally.txt"
    picker, slash = (_sea(tmp_path, name, tally) for name in ("picker", "slash"))
    first = sea_layers(slash, base=picker)
    second = sea_layers(slash, base=picker, loaded=())
    assert [sea.path for sea in second] == [sea.path for sea in first]
    assert first[0] is not second[0]
    assert _imports(tally) == ["picker", "picker", "slash", "slash"]
    (same,) = sea_layers(slash, base=slash, loaded=[first[1]])
    assert same is first[1]
