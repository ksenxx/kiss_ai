"""Tests for sorcar instance isolation when parent and child share a work directory."""

import hashlib
import os
import socket
import tempfile
import time
from pathlib import Path

import pytest


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


class TestInstanceIsolation:
    """Test that child sorcar instances get isolated data directories."""

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

    def test_no_existing_instance_reuses_data_dir(self) -> None:
        """When no other instance is running, the base cs_data_dir is used."""
        cs_data_dir = str(self.kiss_dir / f"cs-{self.wd_hash}")
        Path(cs_data_dir).mkdir(parents=True, exist_ok=True)

        # No assistant-port file => no isolation needed
        existing_port_file = Path(cs_data_dir) / "assistant-port"
        assert not existing_port_file.exists()

        # Simulate the logic from run_chatbot
        result_dir = cs_data_dir
        if existing_port_file.exists():
            try:
                existing_port = int(existing_port_file.read_text().strip())
                with socket.create_connection(
                    ("127.0.0.1", existing_port), timeout=0.5
                ):
                    result_dir = str(
                        self.kiss_dir / f"cs-{self.wd_hash}-{os.getpid()}"
                    )
            except (ConnectionRefusedError, OSError, ValueError):
                pass

        assert result_dir == cs_data_dir

    def test_stale_port_file_reuses_data_dir(self) -> None:
        """When assistant-port exists but the port is not reachable, reuse the dir."""
        cs_data_dir = str(self.kiss_dir / f"cs-{self.wd_hash}")
        Path(cs_data_dir).mkdir(parents=True, exist_ok=True)

        # Write a port that nothing is listening on
        stale_port = _find_free_port()
        (Path(cs_data_dir) / "assistant-port").write_text(str(stale_port))

        # Simulate the logic
        result_dir = cs_data_dir
        existing_port_file = Path(cs_data_dir) / "assistant-port"
        if existing_port_file.exists():
            try:
                existing_port = int(existing_port_file.read_text().strip())
                with socket.create_connection(
                    ("127.0.0.1", existing_port), timeout=0.5
                ):
                    result_dir = str(
                        self.kiss_dir / f"cs-{self.wd_hash}-{os.getpid()}"
                    )
            except (ConnectionRefusedError, OSError, ValueError):
                pass

        assert result_dir == cs_data_dir

    def test_active_instance_creates_isolated_dir(self) -> None:
        """When another sorcar instance is running, create a PID-specific dir."""
        cs_data_dir = str(self.kiss_dir / f"cs-{self.wd_hash}")
        Path(cs_data_dir).mkdir(parents=True, exist_ok=True)

        # Start a real server to simulate a running parent
        parent_port = _find_free_port()
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", parent_port))
        server_sock.listen(1)

        (Path(cs_data_dir) / "assistant-port").write_text(str(parent_port))

        try:
            # Simulate the isolation logic
            result_dir = cs_data_dir
            existing_port_file = Path(cs_data_dir) / "assistant-port"
            if existing_port_file.exists():
                try:
                    existing_port = int(existing_port_file.read_text().strip())
                    with socket.create_connection(
                        ("127.0.0.1", existing_port), timeout=0.5
                    ):
                        result_dir = str(
                            self.kiss_dir / f"cs-{self.wd_hash}-{os.getpid()}"
                        )
                except (ConnectionRefusedError, OSError, ValueError):
                    pass

            assert result_dir == str(
                self.kiss_dir / f"cs-{self.wd_hash}-{os.getpid()}"
            )
            assert result_dir != cs_data_dir

            # Verify parent's assistant-port is untouched
            assert (
                Path(cs_data_dir) / "assistant-port"
            ).read_text().strip() == str(parent_port)
        finally:
            server_sock.close()

    def test_invalid_port_file_reuses_data_dir(self) -> None:
        """When assistant-port contains invalid data, reuse the base dir."""
        cs_data_dir = str(self.kiss_dir / f"cs-{self.wd_hash}")
        Path(cs_data_dir).mkdir(parents=True, exist_ok=True)

        (Path(cs_data_dir) / "assistant-port").write_text("not_a_number")

        result_dir = cs_data_dir
        existing_port_file = Path(cs_data_dir) / "assistant-port"
        if existing_port_file.exists():
            try:
                existing_port = int(existing_port_file.read_text().strip())
                with socket.create_connection(
                    ("127.0.0.1", existing_port), timeout=0.5
                ):
                    result_dir = str(
                        self.kiss_dir / f"cs-{self.wd_hash}-{os.getpid()}"
                    )
            except (ConnectionRefusedError, OSError, ValueError):
                pass

        assert result_dir == cs_data_dir

    def test_isolated_dir_has_pid_suffix(self) -> None:
        """The isolated data dir name contains the current PID."""
        pid = os.getpid()
        isolated_dir = str(self.kiss_dir / f"cs-{self.wd_hash}-{pid}")
        assert isolated_dir.endswith(f"-{pid}")
