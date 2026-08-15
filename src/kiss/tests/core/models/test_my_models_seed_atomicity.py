# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: seeding ``~/.kiss/MY_MODELS.json`` must be atomic.

Finding **F2** of ``tmp/audit/01-core-models-a.md``: ``_seed_my_models_file``
did a check-then-act ``Path.write_text``, which opens the target with
``"w"`` — the file becomes visible **at zero length** before the default
content is written.  A concurrent importer that reads it in that window
gets ``""``, ``_read_my_models`` swallows the ``JSONDecodeError`` and
returns ``{}``, and every user-defined model silently disappears for
that process (``model("my-org/my-custom-model")`` then raises
``Unknown model name``, or ``calculate_cost`` kills the task's
accounting).

``src/kiss/server/user_assets.py`` already solved exactly this for
``MY_TASK_TEMPLATES.md`` / ``MY_INJECTION.md`` by staging the default in
a sibling temp file and hard-linking it into place; ``_seed_my_models_file``
was a second, weaker implementation of the same behaviour.

Test strategy — no mocks, patches or test doubles: ``USER_MY_MODELS_PATH``
is computed from ``Path.home()`` at import time, so the only way to keep
a test off the real ``~/.kiss`` is to run it in **real child processes**
with ``HOME`` pointed at a temp directory.  One child hammers the real
``_seed_my_models_file`` while three others poll the real file, counting
how often they observe a file that exists but does not parse.  Pre-fix
roughly one read in five is torn; post-fix the count must be exactly 0.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from kiss.core.models.model_info import _seed_file_atomically

_DURATION_S = 3.0
_READERS = 3

_SEEDER = """
import sys, time
from kiss.core.models.model_info import USER_MY_MODELS_PATH, _seed_my_models_file

deadline = time.monotonic() + float(sys.argv[1])
rounds = 0
while time.monotonic() < deadline:
    USER_MY_MODELS_PATH.unlink(missing_ok=True)
    _seed_my_models_file()
    rounds += 1
print(rounds)
"""

_READER = """
import json, sys, time
from kiss.core.models.model_info import USER_MY_MODELS_PATH

deadline = time.monotonic() + float(sys.argv[1])
seen = torn = 0
while time.monotonic() < deadline:
    try:
        text = USER_MY_MODELS_PATH.read_text(encoding="utf-8")
    except OSError:
        continue
    seen += 1
    try:
        json.loads(text)
    except json.JSONDecodeError:
        torn += 1
print(json.dumps({"seen": seen, "torn": torn}))
"""


def _child(source: str, home: Path) -> subprocess.Popen[str]:
    """Start a real child process rooted at a throwaway ``HOME``."""
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["KISS_HOME"] = str(home / ".kiss")
    return subprocess.Popen(
        [sys.executable, "-c", source, str(_DURATION_S)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


class TestMyModelsSeedIsAtomic:
    """A concurrent reader must never observe a half-seeded file."""

    def test_seed_never_publishes_an_empty_file(self, tmp_path: Path) -> None:
        """Hammer the real seeder while real readers poll the real file."""
        home = tmp_path / "home"
        (home / ".kiss").mkdir(parents=True)

        seeder = _child(_SEEDER, home)
        readers = [_child(_READER, home) for _ in range(_READERS)]
        seeder_out, seeder_err = seeder.communicate(timeout=_DURATION_S + 60)
        results = []
        for reader in readers:
            out, err = reader.communicate(timeout=_DURATION_S + 60)
            assert reader.returncode == 0, err
            results.append(json.loads(out))

        assert seeder.returncode == 0, seeder_err
        assert int(seeder_out) > 100, "the seeder barely ran; test is not loading"
        assert sum(r["seen"] for r in results) > 100, "readers never saw the file"
        torn = sum(r["torn"] for r in results)
        assert torn == 0, (
            f"{torn} reads observed an existing but unparseable "
            f"MY_MODELS.json — the seed is not atomic"
        )

    def test_seed_writes_the_documented_default(self, tmp_path: Path) -> None:
        """The seeded file must be the full inline default, and survive."""
        home = tmp_path / "home"
        (home / ".kiss").mkdir(parents=True)
        source = """
import json
from kiss.core.models.model_info import (
    MY_MODELS_DEFAULT_CONTENT,
    USER_MY_MODELS_PATH,
    _read_my_models,
    _seed_my_models_file,
)

_seed_my_models_file()
first = USER_MY_MODELS_PATH.read_text(encoding="utf-8")
USER_MY_MODELS_PATH.write_text(
    json.dumps({"mine/custom": {"context_length": 8, "input_price_per_1M": 1.0,
                                "output_price_per_1M": 2.0}}),
    encoding="utf-8",
)
_seed_my_models_file()
print(json.dumps({
    "seeded_default": first == MY_MODELS_DEFAULT_CONTENT,
    "user_entry_survived": list(_read_my_models()),
}))
"""
        child = _child(source, home)
        out, err = child.communicate(timeout=60)
        assert child.returncode == 0, err
        result = json.loads(out)

        assert result["seeded_default"] is True
        assert result["user_entry_survived"] == ["mine/custom"]

    def test_unwritable_home_is_survivable(self, tmp_path: Path) -> None:
        """A read-only ``~/.kiss`` must not break importing the catalog."""
        home = tmp_path / "home"
        (home / ".kiss").mkdir(parents=True)
        (home / ".kiss").chmod(0o500)
        try:
            source = """
import json
from kiss.core.models.model_info import USER_MY_MODELS_PATH, _read_my_models
print(json.dumps({"models": list(_read_my_models()),
                  "exists": USER_MY_MODELS_PATH.exists()}))
"""
            child = _child(source, home)
            out, err = child.communicate(timeout=60)
            assert child.returncode == 0, err
            assert json.loads(out) == {"models": [], "exists": False}
        finally:
            (home / ".kiss").chmod(0o700)


class TestAtomicSeedHelper:
    """The general seeding primitive, driven directly against a temp dir."""

    _PAYLOAD = "x" * 50_000

    def test_racing_seeders_all_publish_complete_content(
        self, tmp_path: Path,
    ) -> None:
        """Real threads racing on one path must never truncate it."""
        errors: list[BaseException] = []
        torn: list[int] = []

        def seed(path: Path, start: threading.Barrier) -> None:
            start.wait(timeout=10)
            try:
                _seed_file_atomically(path, self._PAYLOAD)
            except BaseException as exc:  # noqa: BLE001 — reported below
                errors.append(exc)

        for round_index in range(200):
            path = tmp_path / f"asset-{round_index}.txt"
            start = threading.Barrier(3)
            threads = [
                threading.Thread(target=seed, args=(path, start), daemon=True)
                for _ in range(3)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            if path.read_text(encoding="utf-8") != self._PAYLOAD:
                torn.append(round_index)

        assert errors == []
        assert torn == [], f"{len(torn)} rounds published incomplete content"
        assert list(tmp_path.glob(".asset-*")) == [], "temp files left behind"

    def test_a_target_that_appears_after_the_check_wins(
        self, tmp_path: Path,
    ) -> None:
        """The link losing to an existing target must not be an error.

        A dangling symlink is the deterministic form of that race:
        ``Path.exists()`` follows the link and reports ``False``, but
        ``os.link`` still fails with ``EEXIST`` because the symlink
        itself occupies the name.
        """
        path = tmp_path / "asset.txt"
        target = tmp_path / "absent-target.txt"
        path.symlink_to(target)
        assert path.exists() is False

        _seed_file_atomically(path, self._PAYLOAD)

        assert path.is_symlink()
        assert not target.exists(), "the seed clobbered a name that was taken"
        assert list(tmp_path.glob(".asset-*")) == [], "temp files left behind"

    def test_existing_file_is_never_overwritten(self, tmp_path: Path) -> None:
        """User edits survive every later seed attempt."""
        path = tmp_path / "asset.txt"
        path.write_text("mine", encoding="utf-8")

        _seed_file_atomically(path, self._PAYLOAD)

        assert path.read_text(encoding="utf-8") == "mine"

    def test_unwritable_parent_raises_oserror(self, tmp_path: Path) -> None:
        """The caller decides what to do — here the parent is read-only."""
        parent = tmp_path / "ro"
        parent.mkdir()
        parent.chmod(0o500)
        try:
            with pytest.raises(OSError):
                _seed_file_atomically(parent / "asset.txt", self._PAYLOAD)
        finally:
            parent.chmod(0o700)


if __name__ == "__main__":  # pragma: no cover - manual run
    raise SystemExit(pytest.main([__file__]))
