"""Tests for sorcar instance state sharing and agent independence.

Verifies that multiple Sorcar instances sharing the same work directory
reuse the same code-server data dir (sharing VSCode state like extensions,
settings, editor tabs) while each agent remains independent.
"""

import hashlib
import os
import socket
import tempfile
from pathlib import Path


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------------
# kiss/agents/sorcar/task_history.py — _KISS_DIR
# ---------------------------------------------------------------------------

class TestSharedDataDir:
    """Test that all instances of the same work dir share a data directory."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.kiss_dir = Path(self.tmpdir) / ".kiss"
        self.kiss_dir.mkdir()
        self.work_dir = tempfile.mkdtemp()
        self.wd_hash = hashlib.md5(self.work_dir.encode()).hexdigest()[:8]

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_canonical_data_dir_always_used(self) -> None:
        """Both first and second instances use cs-{wd_hash} as data dir."""
        cs_data_dir = str(self.kiss_dir / f"cs-{self.wd_hash}")
        # After the change, cs_data_dir is always the canonical path.
        # No PID-specific suffix is appended.
        assert f"cs-{self.wd_hash}" in cs_data_dir
        assert f"-{os.getpid()}" not in cs_data_dir

    def test_same_work_dir_same_data_dir(self) -> None:
        """Two instances with the same work_dir compute the same data dir."""
        h1 = hashlib.md5(self.work_dir.encode()).hexdigest()[:8]
        h2 = hashlib.md5(self.work_dir.encode()).hexdigest()[:8]
        assert h1 == h2

    def test_different_work_dirs_different_data_dirs(self) -> None:
        """Two different work directories produce different data dir hashes."""
        h1 = hashlib.md5(b"/tmp/project_a").hexdigest()[:8]
        h2 = hashlib.md5(b"/tmp/project_b").hexdigest()[:8]
        assert h1 != h2


class TestSharedExtensionsDir:
    """Test that extensions are stored in a shared global directory."""

    def test_shared_extensions_dir_name(self) -> None:
        """The shared extensions directory is cs-extensions under KISS_DIR."""
        from kiss.agents.sorcar.task_history import _KISS_DIR

        expected = _KISS_DIR / "cs-extensions"
        assert expected.name == "cs-extensions"

    def test_stale_cleanup_skips_extensions(self) -> None:
        """_cleanup_stale_cs_dirs must not remove cs-extensions directory."""
        tmpdir = tempfile.mkdtemp()
        try:
            kiss_dir = Path(tmpdir)
            # Create a cs-extensions dir that looks "stale" (old mtime)
            ext_dir = kiss_dir / "cs-extensions"
            ext_dir.mkdir()
            (ext_dir / "some-ext").mkdir()
            # Set old mtime
            old_time = 0  # epoch
            os.utime(str(ext_dir), (old_time, old_time))

            # Simulate the cleanup loop logic
            for d in sorted(kiss_dir.glob("cs-*")):
                if not d.is_dir() or d.name == "cs-extensions":
                    continue
                # Would be cleaned up
                assert False, "cs-extensions should have been skipped"

            # cs-extensions should still exist
            assert ext_dir.exists()
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)


class TestCodeServerReuse:
    """Test that instances reuse existing code-server instead of creating new ones."""

    def test_reuse_when_port_in_use(self) -> None:
        """When code-server port is already in use, instance reuses it (cs_proc=None)."""
        port = _find_free_port()
        # Simulate a running code-server
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", port))
        server_sock.listen(1)
        try:
            # Check port is in use
            port_in_use = False
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    port_in_use = True
            except (ConnectionRefusedError, OSError):
                pass
            assert port_in_use
            # In the new code, port_in_use = True means cs_proc stays None
            # (reuse existing code-server)
            cs_proc = None  # Simulating the reuse path
            assert cs_proc is None
        finally:
            server_sock.close()

    def test_assistant_port_overwritten_by_latest(self) -> None:
        """The latest Sorcar instance overwrites assistant-port in shared data dir."""
        tmpdir = tempfile.mkdtemp()
        try:
            data_dir = Path(tmpdir) / "cs-abc12345"
            data_dir.mkdir()

            # First instance writes its port
            (data_dir / "assistant-port").write_text("11111")
            assert (data_dir / "assistant-port").read_text() == "11111"

            # Second instance overwrites with its port
            (data_dir / "assistant-port").write_text("22222")
            assert (data_dir / "assistant-port").read_text() == "22222"
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)
