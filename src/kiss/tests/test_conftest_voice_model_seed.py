# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""The session ``KISS_HOME`` reuses the developer's downloaded voice models.

``conftest._seed_voice_models`` symlinks every entry of the real
``~/.kiss/models`` into the temporary test home so the voice suites run
offline instead of re-downloading 40-80 MB per pytest process (a TLS
timeout on that download failed ``test_wake_word_mic_browser`` under a
parallel full run on 2026-09-28 and 2026-09-29).
"""

import os
from pathlib import Path

import pytest

from kiss.server import voice_wake
from kiss.tests.conftest import _seed_voice_models


def test_real_cache_entries_are_linked_individually(tmp_path: Path) -> None:
    """Each model dir and archive becomes a symlink; lock files are skipped."""
    real_home = tmp_path / "real"
    real_models = real_home / "models"
    (real_models / voice_wake.MODEL_NAME).mkdir(parents=True)
    (real_models / voice_wake.MODEL_NAME / "README").write_text("model\n")
    archive = real_models / f"{voice_wake.MODEL_NAME}.tar.gz"
    archive.write_bytes(b"archive")
    (real_models / f".{voice_wake.MODEL_NAME}.lock").write_text("")
    test_home = tmp_path / "test_home"
    test_home.mkdir()

    _seed_voice_models(str(test_home), real_home)

    seeded = test_home / "models"
    assert sorted(p.name for p in seeded.iterdir()) == [
        voice_wake.MODEL_NAME,
        archive.name,
    ]
    assert all(p.is_symlink() for p in seeded.iterdir())
    # The product's own checks see a complete model and a cached archive.
    assert voice_wake._ensure_downloaded_model(seeded, voice_wake.MODEL_NAME) == (
        seeded / voice_wake.MODEL_NAME
    )
    assert (seeded / archive.name).is_file()
    assert (seeded / archive.name).read_bytes() == b"archive"
    # Replacing the seeded archive rewrites the link, not the real cache.
    tmp = seeded / "new.tmp"
    tmp.write_bytes(b"fresh")
    tmp.replace(seeded / archive.name)
    assert archive.read_bytes() == b"archive"
    assert not (seeded / archive.name).is_symlink()


def test_relative_real_home_links_resolve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A relative pre-override ``KISS_HOME`` still yields working links."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rel_home" / "models" / voice_wake.MODEL_NAME).mkdir(parents=True)
    test_home = tmp_path / "test_home"
    test_home.mkdir()

    _seed_voice_models(str(test_home), Path("rel_home"))

    link = test_home / "models" / voice_wake.MODEL_NAME
    assert link.is_symlink()
    assert link.is_dir()
    assert Path(os.readlink(link)).is_absolute()


def test_missing_real_cache_seeds_nothing(tmp_path: Path) -> None:
    """A developer without downloaded models gets an untouched test home."""
    test_home = tmp_path / "test_home"
    test_home.mkdir()

    _seed_voice_models(str(test_home), tmp_path / "no_such_home")

    assert list(test_home.iterdir()) == []
