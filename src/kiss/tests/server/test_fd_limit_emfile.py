"""End-to-end tests for the RLIMIT_NOFILE fix for EMFILE (Errno 24).

The kiss-web daemon on macOS inherits a soft open-file limit of 256,
which it exhausts under load, surfacing as ``OSError: [Errno 24] Too
many open files`` when e.g. saving a trajectory YAML.  These tests
lower the soft limit to 256, reproduce the EMFILE failure, and verify
``_raise_open_file_limit()`` raises the limit so the same workload
succeeds.  Real ``resource`` syscalls and real files are used — no
mocks.
"""

import resource

import pytest

from kiss.server.web_server import _raise_open_file_limit


@pytest.fixture()
def restore_nofile_limit():
    """Snapshot RLIMIT_NOFILE and restore it after the test."""
    original = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        yield original
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, original)


def _open_files_until_emfile(tmp_path, limit):
    """Open files until EMFILE is raised; return the open handles.

    Fails the test if more than ``limit`` files could be opened
    without an EMFILE, since that means the soft limit is not in
    effect.
    """
    handles = []
    try:
        for i in range(limit + 8):
            handles.append((tmp_path / f"f{i}").open("w"))
    except OSError as exc:
        assert exc.errno == 24, f"expected EMFILE, got {exc}"
        return handles
    pytest.fail("EMFILE was never raised below the soft limit")


def test_raise_open_file_limit_lifts_256_soft_limit(restore_nofile_limit):
    """The helper raises a 256 soft limit toward the hard limit."""
    _soft, hard = restore_nofile_limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
    assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == 256

    _raise_open_file_limit()

    new_soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    assert new_soft > 256, f"soft limit not raised: {new_soft}"


def test_raise_open_file_limit_is_noop_when_already_high(restore_nofile_limit):
    """The helper never lowers an already-sufficient soft limit."""
    _raise_open_file_limit()
    before = resource.getrlimit(resource.RLIMIT_NOFILE)[0]

    _raise_open_file_limit()

    assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == before


def test_emfile_reproduced_then_fixed(tmp_path, restore_nofile_limit):
    """EMFILE reproduces under a 256 limit and disappears after the fix.

    Reproduces the exact reported failure mode: writing one more file
    (a trajectory YAML) fails with ``[Errno 24] Too many open files``
    when the soft limit is 256, and succeeds after
    ``_raise_open_file_limit()``.
    """
    _soft, hard = restore_nofile_limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))

    handles = _open_files_until_emfile(tmp_path, 256)
    try:
        trajectory = tmp_path / "trajectory_Sorcar_VS_Code_Session-0.yaml"
        with pytest.raises(OSError) as excinfo:
            with trajectory.open("w") as f:
                f.write("steps: []\n")
        assert excinfo.value.errno == 24

        _raise_open_file_limit()

        with trajectory.open("w") as f:
            f.write("steps: []\n")
        assert trajectory.read_text() == "steps: []\n"
    finally:
        for h in handles:
            h.close()
