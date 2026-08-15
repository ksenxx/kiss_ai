# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``MODEL_INFO.json`` write/read contract.

Two findings from ``tmp/audit/01-core-models-a.md``:

* **F3** — ``update_models._write_model_info_json`` rewrote the ~200 KB
  catalog with a plain truncating ``Path.write_text`` while
  ``model_info._load_model_info`` did an unguarded ``json.loads`` **at
  import time**.  A kiss process starting while the script was writing
  read ``""`` (or a half-written prefix) and died with a raw
  ``JSONDecodeError`` traceback out of ``import kiss.core.models.model_info``
  instead of running.  The window is milliseconds, but ``run_parallel``
  can start dozens of processes per second against a live checkout.
* **F4** — the script's module docstring and ``apply_updates_to_file``'s
  docstring both promised that ``~/.kiss/MODEL_INFO.json`` "is also
  refreshed"; no code path has written that file since the copy was
  deliberately removed (``src/kiss/tests/test_install_no_model_info_copy.py``
  forbids re-adding it).  The regression below pins the real behaviour so
  nobody "fixes" the stale prose by re-introducing the copy.

No mocks, patches or test doubles: real threads, real temp files, and the
real production functions on both sides of the file.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import _read_model_info_json
from kiss.scripts.update_models import _write_model_info_json

_ROUNDS = 200


def _catalog(size: int = 600) -> dict[str, dict[str, Any]]:
    """Build a catalog payload the size of the real shipped one."""
    return {
        f"vendor/model-{i:04d}": {
            "context_length": 200000,
            "input_price_per_1M": 1.25,
            "output_price_per_1M": 10.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "comment": "padding so the payload matches the real catalog's size",
        }
        for i in range(size)
    }


class TestWriterNeverPublishesATornFile:
    """F3, writer half: a concurrent reader must always see valid JSON."""

    def test_concurrent_reader_never_sees_partial_json(
        self, tmp_path: Path,
    ) -> None:
        """Rewrite the catalog repeatedly while a real thread reads it."""
        path = tmp_path / "MODEL_INFO.json"
        data = _catalog()
        _write_model_info_json(path, dict(data))
        assert len(path.read_text(encoding="utf-8")) > 100_000

        done = threading.Event()
        failures: list[str] = []
        reads = [0]

        def reader() -> None:
            while not done.is_set():
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                reads[0] += 1
                try:
                    json.loads(text)
                except json.JSONDecodeError as exc:
                    failures.append(f"{exc} (len={len(text)})")

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            for _ in range(_ROUNDS):
                _write_model_info_json(path, dict(data))
        finally:
            done.set()
            thread.join(timeout=10)

        assert reads[0] > 10, "the reader thread never got to run"
        assert not failures, (
            f"{len(failures)} torn reads of MODEL_INFO.json during "
            f"{_ROUNDS} rewrites, e.g. {failures[0]}"
        )

    def test_write_leaves_no_temp_files_behind(self, tmp_path: Path) -> None:
        """An atomic write must not litter the package directory."""
        path = tmp_path / "MODEL_INFO.json"
        _write_model_info_json(path, _catalog(3))
        _write_model_info_json(path, _catalog(4))

        assert [p.name for p in tmp_path.iterdir()] == ["MODEL_INFO.json"]

    def test_failed_publish_cleans_up_and_propagates(
        self, tmp_path: Path,
    ) -> None:
        """A publish that cannot land must not leave a stage file behind."""
        path = tmp_path / "MODEL_INFO.json"
        # A non-empty directory in the target's place: the staged file is
        # written fine, but os.replace cannot put it there.
        path.mkdir()
        (path / "occupied").write_text("x", encoding="utf-8")

        with pytest.raises(OSError):
            _write_model_info_json(path, _catalog(3))

        assert sorted(p.name for p in tmp_path.iterdir()) == ["MODEL_INFO.json"]

    def test_written_catalog_round_trips_sorted_and_capped(
        self, tmp_path: Path,
    ) -> None:
        """The atomic write must preserve the documented output shape."""
        path = tmp_path / "MODEL_INFO.json"
        data = _catalog(3)
        data["vendor/huge"] = {
            "context_length": 1000000,
            "input_price_per_1M": 1.0,
            "output_price_per_1M": 2.0,
        }
        _write_model_info_json(path, data)

        text = path.read_text(encoding="utf-8")
        raw = json.loads(text)
        assert text.endswith("}\n")
        assert list(raw) == sorted(raw)
        assert raw["vendor/huge"]["context_length"] == 500000


class TestReaderToleratesAConcurrentRewrite:
    """F3, reader half: import must not die on a transient torn file."""

    def test_read_retries_through_a_truncation_window(
        self, tmp_path: Path,
    ) -> None:
        """A file truncated by another writer must be read once restored."""
        path = tmp_path / "MODEL_INFO.json"
        data = _catalog(5)
        _write_model_info_json(path, dict(data))
        good = path.read_text(encoding="utf-8")

        # Exactly the window F3 describes: another tool has truncated the
        # catalog and has not written its payload yet.
        path.write_text("", encoding="utf-8")

        def restore() -> None:
            time.sleep(0.15)
            path.write_text(good, encoding="utf-8")

        thread = threading.Thread(target=restore, daemon=True)
        thread.start()
        try:
            raw = _read_model_info_json(path)
        finally:
            thread.join(timeout=10)

        assert set(raw) == set(data)

    def test_permanently_broken_catalog_raises_a_clear_error(
        self, tmp_path: Path,
    ) -> None:
        """A catalog that never becomes valid must fail with a named error."""
        path = tmp_path / "MODEL_INFO.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(KISSError) as excinfo:
            _read_model_info_json(path)

        assert str(path) in str(excinfo.value)

    def test_missing_catalog_raises_a_clear_error(self, tmp_path: Path) -> None:
        """An absent catalog must also surface as a KISSError."""
        with pytest.raises(KISSError):
            _read_model_info_json(tmp_path / "absent.json")

    @pytest.mark.parametrize("payload", ["null", "[]", '"a catalog"', "42"])
    def test_valid_json_of_the_wrong_shape_raises_a_clear_error(
        self, tmp_path: Path, payload: str,
    ) -> None:
        """A catalog that parses but is not a table must still be classified.

        An editor or an external catalog updater can atomically leave
        behind syntactically valid JSON of the wrong top-level shape.
        That is the same "the catalog is unusable" condition as a
        truncated file, and it happens at **import time**, so it must
        name the offending path instead of escaping as an unclassified
        ``TypeError`` from ``dict(...)``.
        """
        path = tmp_path / "MODEL_INFO.json"
        path.write_text(payload, encoding="utf-8")

        with pytest.raises(KISSError) as excinfo:
            _read_model_info_json(path)

        assert str(path) in str(excinfo.value)


class TestNoUserLocalCatalogCopy:
    """F4: the documented ``~/.kiss/MODEL_INFO.json`` sync must not exist.

    The defect itself is in the prose — the code has not written that file
    since the copy was removed on purpose.  This pins the behaviour so the
    stale docstrings cannot be "fixed" by re-adding the write, which
    ``test_install_no_model_info_copy.py`` also forbids for the installer.
    """

    def test_writing_the_catalog_does_not_touch_the_user_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A real write to the repo catalog leaves ``~/.kiss`` alone."""
        home = tmp_path / "home"
        (home / ".kiss").mkdir(parents=True)
        user_copy = home / ".kiss" / "MODEL_INFO.json"
        user_copy.write_text('{"sentinel/model": {}}', encoding="utf-8")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("KISS_HOME", str(home / ".kiss"))

        _write_model_info_json(tmp_path / "MODEL_INFO.json", _catalog(3))

        assert user_copy.read_text(encoding="utf-8") == '{"sentinel/model": {}}'


if __name__ == "__main__":  # pragma: no cover - manual run
    raise SystemExit(pytest.main([__file__]))
