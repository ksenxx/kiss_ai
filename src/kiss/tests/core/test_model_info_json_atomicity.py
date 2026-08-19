# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``MODEL_INFO.json`` read contract (F3, reader half).

Split out of ``tests/scripts/test_model_info_json_atomicity.py``: these
methods exercise only ``kiss.core.models.model_info._read_model_info_json``
and ``kiss.core.kiss_error.KISSError``, so they belong in ``tests/core``
per the placement invariants.  The writer-side tests (and the reader test
that needs the real writer) stay in ``tests/scripts`` because they call
``kiss.scripts.update_models._write_model_info_json``.

``model_info._load_model_info`` does an unguarded catalog read **at
import time**; a permanently unusable catalog (bad syntax, missing file,
or valid JSON of the wrong top-level shape) must fail with a ``KISSError``
that names the offending path instead of a raw traceback.

No mocks, patches or test doubles: real temp files and the real
production reader.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import _read_model_info_json


class TestReaderToleratesAConcurrentRewrite:
    """F3, reader half: import must not die on a transient torn file."""

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


if __name__ == "__main__":  # pragma: no cover - manual run
    raise SystemExit(pytest.main([__file__]))
