# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Large ``DockerTools`` payloads and the Docker client's release (S4-2, S4-8).

``DockerTools.Write``/``Edit`` ship their payload inside the one
``bash -c`` argument of a ``docker exec``, which Linux caps at 128 KiB
(``MAX_ARG_STRLEN``): a 98 KB write used to fail with ``Argument list
too long``.  The payload now travels in argv-sized pieces, so these
tests push 1 MiB through a real container and read it back, through
both the Python edit and the Perl fallback of an image without Python.

``DockerManager.close`` used to leave the ``docker.from_env()`` client
and its pooled socket to the daemon open for the life of the process;
it now closes the client, which this module checks on the SDK's own
connection pool.
"""

import hashlib
import unittest
from typing import Any

from kiss.agents.sorcar.docker_manager import DockerManager
from kiss.agents.sorcar.docker_tools import DockerTools
from kiss.tests.agents.sorcar.docker_test_containers import is_docker_available

ONE_MIB = 1024 * 1024


def _lines(total_bytes: int, tag: str) -> str:
    """Text of about *total_bytes* bytes: numbered lines carrying *tag* and non-ASCII."""
    out: list[str] = []
    size = 0
    while size < total_bytes:
        line = f"{len(out):08d} {tag} ünïcödé line with 'quotes' and \"doubles\" and $vars\n"
        out.append(line)
        size += len(line.encode())
    return "".join(out)


def _pools(env: DockerManager) -> int:
    """Connection pools the SDK client holds to the daemon (0 once closed)."""
    adapter: Any = env.client.api.get_adapter("http+docker://localhost")
    return len(adapter.pools)


def _sha256_in(env: DockerManager, path: str) -> str:
    """The SHA-256 the container computes for *path*."""
    return env.Bash(f"sha256sum {path}", "hash").split()[0]


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestLargePayloads(unittest.TestCase):
    """Writes and edits far above the argv limit, in a Python image."""

    env: DockerManager
    tools: DockerTools

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = DockerManager("python:3.11-slim")
        cls.env.open()
        cls.tools = DockerTools(cls.env.Bash)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_write_one_mib_and_read_back(self) -> None:
        """A 1 MiB Write lands byte-for-byte, and Read windows it."""
        content = _lines(ONE_MIB, "big")
        self.assertGreaterEqual(len(content.encode()), ONE_MIB)
        result = self.tools.Write("/tmp/large/one_mib.txt", content)
        self.assertIn("Successfully wrote", result)
        self.assertEqual(
            _sha256_in(self.env, "/tmp/large/one_mib.txt"),
            hashlib.sha256(content.encode()).hexdigest(),
        )
        last = content.splitlines()[-1]
        total = len(content.splitlines())
        window = self.tools.Read("/tmp/large/one_mib.txt", max_lines=3, start_line=total)
        self.assertIn(last, window)

    def test_overwrite_with_smaller_content_truncates(self) -> None:
        """A multi-piece write followed by a short one leaves only the short one."""
        self.tools.Write("/tmp/large/shrink.txt", _lines(300_000, "first"))
        self.tools.Write("/tmp/large/shrink.txt", "tiny\n")
        self.assertEqual(self.env.Bash("cat /tmp/large/shrink.txt", "cat").split("\n")[0], "tiny")

    def test_edit_with_large_strings(self) -> None:
        """Edit with a 150 KB old_string and a 200 KB new_string, replace_all."""
        old = _lines(150_000, "old")
        head, tail = "HEAD\n", "TAIL\n"
        self.tools.Write("/tmp/large/edit.txt", head + old + "MIDDLE\n" + old + tail)
        new = _lines(200_000, "new")
        result = self.tools.Edit("/tmp/large/edit.txt", old, new, replace_all=True)
        self.assertIn("Successfully replaced 2 occurrence(s)", result)
        expected = head + new + "MIDDLE\n" + new + tail
        self.assertEqual(
            _sha256_in(self.env, "/tmp/large/edit.txt"),
            hashlib.sha256(expected.encode()).hexdigest(),
        )
        # The payload file of the edit is gone, whatever the outcome.
        self.assertIn("none", self.env.Bash("ls /tmp/.kiss-edit-* 2>/dev/null || echo none", "ls"))
        missing = self.tools.Edit("/tmp/large/edit.txt", old, "x")
        self.assertIn("String not found", missing)
        self.assertIn("none", self.env.Bash("ls /tmp/.kiss-edit-* 2>/dev/null || echo none", "ls"))

    def test_edit_with_nul_bytes(self) -> None:
        """Strings holding NUL bytes are edited like any other."""
        self.tools.Write("/tmp/large/nul.txt", "a\x00b\x00c\n")
        result = self.tools.Edit("/tmp/large/nul.txt", "\x00b\x00", "\x00B\x00")
        self.assertIn("Successfully replaced", result)
        self.assertEqual(
            _sha256_in(self.env, "/tmp/large/nul.txt"),
            hashlib.sha256(b"a\x00B\x00c\n").hexdigest(),
        )


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestLargePayloadsWithoutPython(unittest.TestCase):
    """The Perl fallback of Edit, in an image with no Python at all."""

    env: DockerManager
    tools: DockerTools

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = DockerManager("debian:13")
        cls.env.open()
        cls.tools = DockerTools(cls.env.Bash)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_no_python(self) -> None:
        """The image really lacks Python, so Edit takes the Perl path."""
        found = self.env.Bash("command -v python3 || command -v python", "which")
        self.assertIn("[exit code: 1]", found)

    def test_perl_edit_with_large_strings_and_nul(self) -> None:
        """A 150 KB replacement, and one holding NUL bytes, through Perl."""
        old = _lines(150_000, "old")
        self.tools.Write("/tmp/perl.txt", "x\n" + old + "y\n")
        new = _lines(120_000, "new") + "\x00\n"
        result = self.tools.Edit("/tmp/perl.txt", old, new)
        self.assertIn("Successfully replaced 1 occurrence(s)", result)
        self.assertEqual(
            _sha256_in(self.env, "/tmp/perl.txt"),
            hashlib.sha256(("x\n" + new + "y\n").encode()).hexdigest(),
        )
        self.assertIn("none", self.env.Bash("ls /tmp/.kiss-edit-* 2>/dev/null || echo none", "ls"))
        self.assertIn("String not found", self.tools.Edit("/tmp/perl.txt", "absent", "z"))
        self.assertIn("not unique", self.tools.Edit("/tmp/perl.txt", "\n", "z"))
        self.assertIn("File not found", self.tools.Edit("/tmp/missing.txt", "a", "b"))


@unittest.skipUnless(is_docker_available(), "Docker daemon is not running")
class TestClientReleased(unittest.TestCase):
    """``close`` releases the SDK client's pooled connections to the daemon."""

    def test_close_empties_the_connection_pool(self) -> None:
        env = DockerManager("python:3.11-slim")
        self.addCleanup(env.close)
        env.open()
        self.assertIn("ok", env.Bash("echo ok", "ping"))
        self.assertGreater(_pools(env), 0)
        env.close()
        self.assertEqual(_pools(env), 0)
        # The manager stays usable: the SDK opens a new connection on demand.
        env.open()
        self.assertIn("again", env.Bash("echo again", "ping"))
        env.close()
        self.assertEqual(_pools(env), 0)

    def test_close_without_container_still_releases_the_client(self) -> None:
        env = DockerManager("python:3.11-slim")
        env.client.ping()
        self.assertGreater(_pools(env), 0)
        env.close()
        self.assertEqual(_pools(env), 0)
