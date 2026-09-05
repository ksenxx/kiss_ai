# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``update_models.py --model-info`` option.

The option points the updater at an arbitrary ``MODEL_INFO.json`` catalog
— most importantly the user-local ``~/.kiss/MODEL_INFO.json`` behind the
settings panel's "Update Models" button — instead of the repo's bundled
``src/kiss/core/models/MODEL_INFO.json`` default.  A non-default target:

* is the only file rewritten (the repo catalog and ``README.md`` stay
  byte-identical),
* is seeded from the bundled catalog when it does not exist yet (so a
  brand-new target starts as a full copy, not an empty table),
* is left untouched by ``--dry-run``.

Every test runs the real script as a subprocess (``--scrub-only``: the
offline mode that needs no vendor API keys and no network).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_SCRIPT = _REPO / "src" / "kiss" / "scripts" / "update_models.py"
_BUNDLED = _REPO / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"
_README = _REPO / "README.md"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--scrub-only", *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(_REPO),
    )
    assert proc.returncode == 0, proc.stderr
    return proc


def _sample_catalog() -> dict[str, dict[str, object]]:
    return {
        "minimax-m2": {
            "context_length": 1000,
            "input_price_per_1M": 1.0,
            "output_price_per_1M": 2.0,
        },
        "keep/me": {
            "context_length": 2000,
            "input_price_per_1M": 3.0,
            "output_price_per_1M": 4.0,
        },
    }


def test_model_info_option_updates_only_the_target(tmp_path: Path) -> None:
    """A scrub run against a custom catalog leaves the repo files alone."""
    target = tmp_path / "MODEL_INFO.json"
    target.write_text(json.dumps(_sample_catalog()), encoding="utf-8")
    bundled_before = _BUNDLED.read_bytes()
    readme_before = _README.read_bytes()

    proc = _run("--model-info", str(target))

    data = json.loads(target.read_text(encoding="utf-8"))
    assert "minimax-m2" not in data, "excluded provider not scrubbed"
    assert "keep/me" in data, "unrelated entry lost"
    assert _BUNDLED.read_bytes() == bundled_before, "repo catalog touched"
    assert _README.read_bytes() == readme_before, "README touched"
    assert "README left untouched" in proc.stdout


def test_missing_target_is_seeded_from_the_bundled_catalog(
    tmp_path: Path,
) -> None:
    """A brand-new target starts as a full copy of the bundled catalog."""
    target = tmp_path / "kiss-home" / "MODEL_INFO.json"

    proc = _run("--model-info", str(target))

    assert f"Seeded {target}" in proc.stdout
    data = json.loads(target.read_text(encoding="utf-8"))
    bundled = json.loads(_BUNDLED.read_text(encoding="utf-8"))
    assert set(data) == set(bundled), (
        "seeded target must carry every bundled entry, not just the ones "
        "this run touched"
    )


def test_dry_run_leaves_the_target_untouched(tmp_path: Path) -> None:
    """``--dry-run`` neither rewrites nor seeds the target."""
    target = tmp_path / "MODEL_INFO.json"
    payload = json.dumps(_sample_catalog())
    target.write_text(payload, encoding="utf-8")

    _run("--model-info", str(target), "--dry-run")
    assert target.read_text(encoding="utf-8") == payload

    missing = tmp_path / "missing" / "MODEL_INFO.json"
    _run("--model-info", str(missing), "--dry-run")
    assert not missing.exists(), "--dry-run must not seed a new target"


def test_default_target_is_the_bundled_repo_catalog() -> None:
    """Without ``--model-info`` the script targets the repo catalog."""
    import os

    # An inherited KISS_WORKDIR pointing at a different checkout would
    # change the default path rendered into --help; drop it so the
    # default is derived from this repo (the subprocess cwd).
    env = {k: v for k, v in os.environ.items() if k != "KISS_WORKDIR"}
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(_REPO),
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    help_text = " ".join(proc.stdout.split())
    assert "--model-info PATH" in help_text
    assert str(_BUNDLED) in help_text
