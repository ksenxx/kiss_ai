# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Docker Bash/Read/Write/Edit guards, without a docker daemon.

``DockerTools`` is defined against *any* ``bash_fn(command, description)``,
so the exact shell scripts it generates can be executed by a real local
``/bin/bash`` subprocess.  That covers the F2 guards (Write success
detection, Read's ``max_lines``/empty-file handling, Edit's empty-needle
refusal) on machines with no docker daemon, using real files in real temp
directories and a real timeout — no mocks, patches or doubles.

The F3 truncation helper and the F4 incremental decoder are exercised
directly with real byte sequences, because those are the pieces of
``DockerManager`` that carry the fix.

Audit 2026-11-07 (A08): the Write marker counts only as a whole line
(fable-9); a dash-prefixed relative path is a file operand (fable-13);
both Edit editors publish a complete new file by rename, keeping the
mode, writing a hard-linked file in place and a symlink's target through
the link (astra-3), and the Perl one retries an LF needle against a CRLF
file as the Python one does (astra-15 / fable-12) and reports a failed
write (astra-5); a raised executor error still discards the payload or
staging file (astra-14); a symlink's Write stages beside the file it
points to (astra-4).  The Perl editor is reached through a bash whose
``command -v`` finds no python.
"""

import functools
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from collections.abc import Callable
from pathlib import Path

from kiss.agents.sorcar.docker_manager import (
    MAX_OUTPUT_CHARS,
    PendingRewrite,
    _new_utf8_decoder,
    _rewrite_instead,
    _with_exit_code,
    probe_command,
)
from kiss.agents.sorcar.docker_tools import _CHUNK_BYTES, DockerTools
from kiss.agents.sorcar.useful_tools import _truncate_output
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_error import KISSError
from kiss.core.tool_verdict import ToolRedirect


def host_bash(
    command: str, description: str, timeout_seconds: float = 30, cwd: str | None = None,
) -> str:
    """Run *command* in the real local bash, shaped like ``DockerManager.Bash``.

    Args:
        command: The shell script to run.
        description: Ignored (kept for signature parity).
        timeout_seconds: Real wall-clock timeout; on expiry the exact
            timeout string ``DockerManager.Bash`` returns is produced.
        cwd: The directory the command runs in (the container's working
            directory); the process's own when ``None``.

    Returns:
        The combined output plus the ``[exit code: N]`` marker on failure,
        or the timeout error.
    """
    del description
    try:
        proc = subprocess.run(
            ["/bin/bash", "-c", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout_seconds}s"
    output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    return _with_exit_code(output, proc.returncode)


def slow_host_bash(command: str, description: str) -> str:
    """Run *command* behind a real 3 s sleep with a 1 s timeout."""
    return host_bash(f"sleep 3; {command}", description, timeout_seconds=1)


def host_bash_without_python(command: str, description: str, cwd: str | None = None) -> str:
    """Run *command* in a bash whose ``command -v`` finds nothing: Edit falls back to Perl."""
    return host_bash(f"command() {{ return 1; }}\n{command}", description, cwd=cwd)


def size_limited_bash(command: str, description: str, with_python: bool = True) -> str:
    """Run *command* where no file may grow past 1 KiB (``EFBIG`` instead of ``SIGXFSZ``)."""
    bash = host_bash if with_python else host_bash_without_python
    return bash(f"trap '' XFSZ; ulimit -f 1\n{command}", description)


def recording_bash(
    commands: list[str], failing: str, command: str, description: str,
) -> str:
    """Record *command*; raise ``KISSError`` for the one *failing* names (script or description)."""
    commands.append(command)
    if failing in command or description.endswith(failing):
        raise KISSError("one transient executor failure")
    return host_bash(command, description)


def peeking_bash(
    seen: list[str], directories: list[Path], command: str, description: str,
) -> str:
    """Run *command*; after the first of two pieces, record the files of *directories*."""
    result = host_bash(command, description)
    if description.endswith("(part 1/2)"):
        seen.extend(str(p) for d in directories for p in sorted(d.iterdir()))
    return result


def pausing_bash(
    events: list[str], entered: threading.Event, release: threading.Event,
    command: str, description: str,
) -> str:
    """Run *command*; the first editor script waits for *release* once it set *entered*.

    A command that touches the edited file (an editor script, the
    publishing piece of a Write) is recorded in *events* as
    ``"edit start"`` / ``"edit end"`` or ``"write start"`` / ``"write end"``;
    a payload piece (``.kiss-edit-``) is not.
    """
    if '"$PYTHON" -c' in command:
        kind = "edit"
    elif "mv -fT" in command and ".kiss-edit-" not in command:
        kind = "write"
    else:
        return host_bash(command, description)
    events.append(f"{kind} start")
    if kind == "edit" and not entered.is_set():
        entered.set()
        assert release.wait(30), "the paused editor was never released"
    result = host_bash(command, description)
    events.append(f"{kind} end")
    return result


def switching_bash(
    commands: list[str], link: Path, to: Path, command: str, description: str,
) -> str:
    """Record and run *command*; once the first of two pieces landed, point *link* at *to*."""
    commands.append(command)
    result = host_bash(command, description)
    if description.endswith("(part 1/2)"):
        link.unlink()
        link.symlink_to(to)
    return result


def host_bash_with_a_missing_locale(command: str, description: str) -> str:
    """Run *command* as a container without its locale does: bash warns first, on the one stream."""
    return host_bash(f"exec 2>&1\nexport LC_ALL=xx_XX.UTF-8\n{command}", description)


def planting_bash(target: Path, victim: Path, command: str, description: str) -> str:
    """Run *command* without python; the Perl editor finds a link to *victim* planted for it.

    The link waits at ``.<name>.<pid>.tmp`` beside *target*, the name the
    Perl editor once staged the new text under (``exec`` keeps the pid).
    """
    if command.startswith("perl "):
        link = shlex.quote(f"{target.parent}/.{target.name}.") + "$$.tmp"
        command = f"ln -s {shlex.quote(str(victim))} {link}\nexec {command}"
    return host_bash_without_python(command, description)


def _read_bytes(path: str) -> bytes:
    """The bytes of the local file *path* (the container's ``read_bytes`` of the guards)."""
    return Path(path).read_bytes()


def _read_bytes_then_rewrite(rewritten: bytes, path: str) -> bytes:
    """The bytes of *path*, which holds *rewritten* once they are read: a writer in between."""
    data = Path(path).read_bytes()
    Path(path).write_bytes(rewritten)
    return data


def _forget(pending: PendingRewrite | None) -> None:
    """The guards' ``remember`` of a test that makes no append."""
    del pending


def _link_probe(path: str) -> bool:
    """The guards' ``link`` probe of *path*, run by the local bash and read as the managers do."""
    return subprocess.run(
        ["/bin/bash", "-c", probe_command(path, "link")], check=False,
    ).returncode != 1


def _collect(results: list[str], call: Callable[..., str], *args: str) -> None:
    """Thread body: append ``call(*args)`` to *results*."""
    results.append(call(*args))


#: The Write/Edit scripts publish a file with ``mv -T`` and ``stat -c``
#: of the GNU/busybox coreutils in the Linux containers they run in
#: (Git bash on Windows has them too); BSD ``mv`` (macOS) rejects ``-T``,
#: so a host bash there cannot run them.
gnu_coreutils_only = unittest.skipIf(
    sys.platform == "darwin", "the store scripts need GNU/busybox mv -T and stat -c"
)


@unittest.skipIf(sys.platform == "win32", "needs a POSIX shell")
class TestDockerToolsWithoutDocker(unittest.TestCase):
    """F2 — the generated scripts, run by a real local bash."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="kiss_g_tools_")
        self.dir = Path(self._tmp.name)
        self.tools = DockerTools(host_bash)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_write_that_timed_out_is_not_reported_as_success(self) -> None:
        """The timeout return carries no exit-code marker; Write must still fail."""
        target = self.dir / "never.txt"
        result = DockerTools(slow_host_bash).Write(str(target), "payload")
        self.assertNotIn("Successfully wrote", result)
        self.assertIn("timed out", result)
        self.assertFalse(target.exists())

    @gnu_coreutils_only
    def test_write_success_reports_success_and_writes_bytes(self) -> None:
        """The happy path really creates the file with the exact content."""
        target = self.dir / "sub" / "ok.txt"
        result = self.tools.Write(str(target), "hello world")
        self.assertIn("Successfully wrote 11 characters", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "hello world")

    def test_write_failure_reports_the_exit_code(self) -> None:
        """A write into a non-directory fails loudly."""
        blocker = self.dir / "blocker"
        blocker.write_text("x", encoding="utf-8")
        result = self.tools.Write(str(blocker / "nested.txt"), "fail")
        self.assertNotIn("Successfully wrote", result)
        self.assertIn("exit code:", result)

    @gnu_coreutils_only
    def test_write_marker_is_not_echoed_to_the_model(self) -> None:
        """The success sentinel never leaks into the returned message."""
        target = self.dir / "clean.txt"
        result = self.tools.Write(str(target), "content")
        self.assertNotIn("KISS_WRITE_OK", result)

    def test_read_rejects_max_lines_below_one(self) -> None:
        """``max_lines=0`` errors instead of silently reading nothing."""
        target = self.dir / "lines.txt"
        target.write_text("a\nb\nc\n", encoding="utf-8")
        self.assertIn(
            "max_lines must be >= 1", self.tools.Read(str(target), max_lines=0),
        )
        self.assertIn(
            "max_lines must be >= 1", self.tools.Read(str(target), max_lines=-5),
        )

    def test_read_rejects_start_line_below_one(self) -> None:
        """The pre-existing start_line guard is unchanged."""
        target = self.dir / "lines.txt"
        target.write_text("a\n", encoding="utf-8")
        self.assertIn(
            "start_line must be >= 1", self.tools.Read(str(target), start_line=0),
        )

    def test_read_empty_file_returns_sentinel(self) -> None:
        """An empty file is not indistinguishable from an empty result."""
        target = self.dir / "empty.txt"
        target.write_text("", encoding="utf-8")
        self.assertEqual(self.tools.Read(str(target)).strip(), "(file is empty)")

    def test_read_non_empty_file_is_unaffected(self) -> None:
        """The empty-file branch does not swallow real content."""
        target = self.dir / "content.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        result = self.tools.Read(str(target))
        self.assertIn("alpha", result)
        self.assertIn("beta", result)

    def test_read_missing_file_reports_not_found(self) -> None:
        """A missing file still reports the not-found error."""
        result = self.tools.Read(str(self.dir / "nope.txt"))
        self.assertIn("Error: File not found", result)

    def test_read_truncates_and_windows(self) -> None:
        """max_lines/start_line windowing still works after the guards."""
        target = self.dir / "many.txt"
        target.write_text("".join(f"line{i}\n" for i in range(10)), encoding="utf-8")
        result = self.tools.Read(str(target), max_lines=2, start_line=3)
        self.assertIn("line2", result)
        self.assertIn("line3", result)
        self.assertNotIn("line4", result)
        self.assertIn("truncated: 6 more lines", result)

    def test_edit_rejects_empty_old_string(self) -> None:
        """replace_all with an empty needle must not corrupt the file."""
        target = self.dir / "edit.txt"
        target.write_text("aaa", encoding="utf-8")
        result = self.tools.Edit(str(target), "", "X", replace_all=True)
        self.assertIn("old_string must not be empty", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "aaa")

    def test_edit_rejects_empty_old_string_without_replace_all(self) -> None:
        """The guard applies to the single-replacement mode too."""
        target = self.dir / "edit.txt"
        target.write_text("aaa", encoding="utf-8")
        self.assertIn(
            "old_string must not be empty", self.tools.Edit(str(target), "", "X"),
        )
        self.assertEqual(target.read_text(encoding="utf-8"), "aaa")

    @gnu_coreutils_only
    def test_edit_still_replaces(self) -> None:
        """The happy path is unchanged by the new guard."""
        target = self.dir / "edit.txt"
        target.write_text("aaa bbb", encoding="utf-8")
        result = self.tools.Edit(str(target), "bbb", "ccc")
        self.assertIn("Successfully replaced", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "aaa ccc")

    @gnu_coreutils_only
    def test_write_marker_counts_only_as_a_whole_line(self) -> None:
        """A failed publication whose diagnostic names the marker is no success (A08 fable-9)."""
        target = self.dir / "__KISS_WRITE_OK__"
        target.mkdir()
        result = self.tools.Write(str(target), "not stored")
        self.assertNotIn("Successfully wrote", result)
        self.assertIn("exit code:", result)
        self.assertTrue(target.is_dir())
        self.assertEqual(list(self.dir.iterdir()), [target])  # the staging file is discarded

    @gnu_coreutils_only
    def test_dash_prefixed_relative_paths_are_file_operands(self) -> None:
        """``-sample.txt`` in the working directory is a file, not options (A08 fable-13)."""
        for bash in (host_bash, host_bash_without_python):
            tools = DockerTools(functools.partial(bash, cwd=str(self.dir)))
            self.assertIn("Successfully wrote 4 characters to -sample.txt",
                          tools.Write("-sample.txt", "old\n"))
            self.assertEqual((self.dir / "-sample.txt").read_text(encoding="utf-8"), "old\n")
            self.assertEqual(tools.Read("-sample.txt").strip(), "old")
            self.assertIn("Successfully replaced 1", tools.Edit("-sample.txt", "old", "new"))
            self.assertEqual((self.dir / "-sample.txt").read_text(encoding="utf-8"), "new\n")
            (self.dir / "-sample.txt").unlink()

    @gnu_coreutils_only
    def test_both_editors_retry_lf_needles_on_crlf_files(self) -> None:
        """The Perl fallback matches the Python editor on CRLF files (A08 astra-15 / fable-12).

        A one-line needle is found as it is in a CRLF file; the lines its
        new text adds take the file's line ends all the same, as the host
        Edit writes them (they were written with a bare LF), unless the
        edit is ``exact`` (S11 re-fix round 2).
        """
        for tools in (self.tools, DockerTools(host_bash_without_python)):
            target = self.dir / "crlf.txt"
            target.write_bytes(b"old\r\nsecond\r\n")
            result = tools.Edit(str(target), "old\nsecond", "new\nlast")
            self.assertIn("Successfully replaced 1 occurrence", result)
            self.assertEqual(target.read_bytes(), b"new\r\nlast\r\n")
            self.assertIn("Successfully", tools.Edit(str(target), "last", "more\nlast"))
            self.assertEqual(target.read_bytes(), b"new\r\nmore\r\nlast\r\n")
            self.assertIn("Successfully", tools.Edit(str(target), "more", "a\nb", exact=True))
            self.assertEqual(target.read_bytes(), b"new\r\na\nb\r\nlast\r\n")

    @gnu_coreutils_only
    def test_both_editors_publish_a_new_file_keeping_mode_and_links(self) -> None:
        """The edited text is renamed over the file, never written into it (A08 astra-3).

        As the host ``Edit``: the mode is kept, a hard-linked file is
        written in place (its other names see the text), a symlink stays
        and its target changes.  The inode changing proves the file was
        never opened for writing, so no reader ever saw it truncated.
        """
        for tools in (self.tools, DockerTools(host_bash_without_python)):
            target = self.dir / "mode.txt"
            target.write_text("aaa bbb", encoding="utf-8")
            target.chmod(0o640)
            before = target.stat().st_ino
            self.assertIn("Successfully replaced", tools.Edit(str(target), "bbb", "ccc"))
            self.assertEqual(target.read_text(encoding="utf-8"), "aaa ccc")
            self.assertNotEqual(target.stat().st_ino, before)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
            real = self.dir / "real.txt"
            real.write_text("aaa bbb", encoding="utf-8")
            link = self.dir / "link.txt"
            link.symlink_to(real)
            self.assertIn("Successfully replaced", tools.Edit(str(link), "bbb", "ccc"))
            self.assertTrue(link.is_symlink())
            self.assertEqual(real.read_text(encoding="utf-8"), "aaa ccc")
            one = self.dir / "one.txt"
            one.write_text("aaa bbb", encoding="utf-8")
            two = self.dir / "two.txt"
            os.link(one, two)
            self.assertIn("Successfully replaced", tools.Edit(str(one), "bbb", "ccc"))
            self.assertEqual(two.read_text(encoding="utf-8"), "aaa ccc")
            self.assertEqual(one.stat().st_ino, two.stat().st_ino)
            leftovers = [p.name for p in self.dir.iterdir() if p.name.endswith(".tmp")]
            self.assertEqual(leftovers, [])
            for path in (target, real, link, one, two):
                path.unlink()

    @gnu_coreutils_only
    def test_a_read_only_file_is_refused_and_a_sed_on_it_is_no_edit(self) -> None:
        """A file this user may not write is not replaced by a rename (S11 re-fix rounds 2, 3).

        The rename takes the directory's permission only: Write and both
        editors replaced a 0444 file and reported success, where the host
        tools and the shell's ``>`` answer "Permission denied".  The
        ``exact`` Edit a shell rewrite is applied as still replaced it
        (round 3).  A ``sed -i`` does replace such a file: the guards'
        probe now says so and it is not made as an Edit, as on the host;
        a python rewrite, itself refused, still is.
        """
        if os.geteuid() == 0:
            self.skipTest("root may write a read-only file")
        target = self.dir / "locked.txt"
        target.write_text("old\n", encoding="utf-8")
        target.chmod(0o444)
        for bash in (host_bash, host_bash_without_python):
            tools = DockerTools(bash)
            for refused in (
                tools.Edit(str(target), "old", "new"),
                tools.Edit(str(target), "old", "new", exact=True),
                tools.Write(str(target), "new\n"),
                tools.Write(str(target), "x" * 100_000),
            ):
                self.assertTrue(refused.startswith("Error"), refused)
                self.assertIn("Permission denied", refused)
            self.assertEqual(target.read_text(encoding="utf-8"), "old\n")
            self.assertEqual([p.name for p in self.dir.iterdir()], ["locked.txt"])
        sed = f"sed -i 's/old/new/' {target}"
        python = (
            f"python3 - <<'EOF'\np = '{target}'\ns = open(p).read()\n"
            "s = s.replace('old', 'new')\nopen(p, 'w').write(s)\nEOF"
        )
        for command, mode, made_as_edit in (
            (sed, 0o444, False), (python, 0o444, True), (sed, 0o644, True),
        ):
            target.chmod(mode)
            try:
                made = _rewrite_instead(  # None: the command runs as written
                    command, str(self.dir), os.path.exists, os.path.isdir, _link_probe,
                    _read_bytes, "rewrite", 30, 1000, _forget,
                )
            except ToolRedirect as redirect:
                made = redirect.name
            self.assertEqual(made, "Edit" if made_as_edit else None, (command, oct(mode)))

    @gnu_coreutils_only
    def test_a_path_that_is_not_a_regular_file_is_refused(self) -> None:
        """A FIFO or a directory is neither edited nor replaced (S11 re-fix round 2).

        Both editors opened the path they were given: a FIFO nobody
        writes to blocked the Edit until its timeout and a directory
        came back as a traceback, where the host Edit answers "File not
        found".  Write renamed its staged file over a FIFO (over
        ``/dev/null`` in a container) and reported success, where the
        host Write refuses.
        """
        fifo, folder = self.dir / "fifo", self.dir / "folder"
        os.mkfifo(fifo)
        folder.mkdir()
        for path in (fifo, folder):
            for bash in (host_bash, host_bash_without_python):
                self.assertEqual(
                    DockerTools(bash).Edit(str(path), "a", "b"),
                    f"Error: File not found: {path}\n\n[exit code: 1]",
                )
            for content in ("one piece", "x" * 100_000):
                refused = self.tools.Write(str(path), content)
                self.assertTrue(refused.startswith(f"Error: {path} exists and is not"), refused)
        self.assertTrue(stat.S_ISFIFO(fifo.stat().st_mode))
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["fifo", "folder"])

    @gnu_coreutils_only
    def test_a_file_name_of_the_full_length_is_written_and_edited(self) -> None:
        """A 255-byte name leaves no room for a longer sibling (S11 re-fix round 2).

        The staged copy was named after the file (``<name>.kiss-partial-<hex>``,
        ``.<name>.<hex>.tmp``): Write and the Python editor answered
        "File name too long" and the Perl editor wrote the file in place.
        The changed inode proves the new text was staged and renamed.
        """
        target = self.dir / ("n" * 255)
        self.assertIn("Successfully wrote", self.tools.Write(str(target), "old\n"))
        for bash in (host_bash, host_bash_without_python):
            target.write_text("old\n", encoding="utf-8")
            before = target.stat().st_ino
            edited = DockerTools(bash).Edit(str(target), "old", "new")
            self.assertIn("Successfully replaced 1", edited)
            self.assertEqual(target.read_text(encoding="utf-8"), "new\n")
            self.assertNotEqual(target.stat().st_ino, before)
        self.assertEqual([p.name for p in self.dir.iterdir()], [target.name])

    @gnu_coreutils_only
    def test_the_perl_editor_stages_under_a_name_nobody_planted(self) -> None:
        """A link waiting at a guessable staging name is not written through (S11 re-fix 2).

        The Perl editor staged the new text at ``.<name>.<pid>.tmp``,
        opened for writing: a symlink planted there sent the text into
        the file it pointed to and was then renamed over the edited file.
        """
        target, victim = self.dir / "perl.txt", self.dir / "victim.txt"
        target.write_text("old value\n", encoding="utf-8")
        target.chmod(0o640)
        victim.write_text("victim original\n", encoding="utf-8")
        tools = DockerTools(functools.partial(planting_bash, target, victim))
        self.assertIn("Successfully replaced 1", tools.Edit(str(target), "old", "new"))
        self.assertFalse(target.is_symlink())
        self.assertEqual(target.read_text(encoding="utf-8"), "new value\n")
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
        self.assertEqual(victim.read_text(encoding="utf-8"), "victim original\n")

    @gnu_coreutils_only
    def test_a_write_or_an_edit_that_failed_opens_with_error(self) -> None:
        """A failure reads as one to whoever asks whether the call went through (S11 re-fix).

        The output of the command that failed came back as it was (a
        ``mkdir`` diagnostic, a traceback), and by the tools' convention
        only a result opening with ``Error`` is a failure: the agent ran
        the rest of a compound command after it and the digest replayed
        the write.  An error the editors word themselves stays as it is.
        """
        blocker = self.dir / "blocker"
        blocker.write_text("x", encoding="utf-8")
        unmade = self.tools.Write(str(blocker / "notes.md"), "never written")
        self.assertTrue(unmade.startswith("Error: "), unmade)
        self.assertIn("exit code:", unmade)
        latin1 = self.dir / "latin1.txt"
        latin1.write_bytes(b"one\xff\n")
        undecoded = self.tools.Edit(str(latin1), "one", "uno")
        self.assertTrue(undecoded.startswith("Error: Traceback"), undecoded)
        self.assertIn("UnicodeDecodeError", undecoded)
        self.assertEqual(latin1.read_bytes(), b"one\xff\n")
        missing = self.tools.Edit(str(self.dir / "absent.txt"), "one", "uno")
        self.assertTrue(missing.startswith("Error: File not found"), missing)
        # The strings of an Edit that could not be stored (no file may pass 1 KiB here).
        small = self.dir / "small.txt"
        small.write_text("one\n", encoding="utf-8")
        unstored = DockerTools(size_limited_bash).Edit(str(small), "one", "x" * 4000)
        self.assertTrue(unstored.startswith("Error: "), unstored)
        self.assertIn("exit code:", unstored)
        self.assertEqual(small.read_text(encoding="utf-8"), "one\n")

    @gnu_coreutils_only
    def test_a_shell_warning_before_the_outcome_does_not_change_it(self) -> None:
        """An edit that landed stays one behind a shell's warnings (S11 re-fix).

        The bash of a container whose locale is not installed warns
        before every command's own output.  The outcome was read off the
        start of the result: a batch stopped after its first edit and
        reported an error for an edit that was in the file.
        """
        tools = DockerTools(host_bash_with_a_missing_locale)
        target = self.dir / "noisy.txt"
        target.write_text("one\ntwo\n", encoding="utf-8")
        single = tools.Edit(str(target), "one", "uno")
        self.assertIn("warning: setlocale", single.splitlines()[0])
        self.assertIn("Successfully replaced 1", single)
        self.assertFalse(single.startswith("Error"), single)
        batch = tools.Edit(str(target), edits=json.dumps([
            {"old_string": "uno", "new_string": "eins"},
            {"old_string": "two", "new_string": "zwei"},
        ]))
        self.assertFalse(batch.startswith("Error"), batch)
        self.assertEqual(target.read_text(encoding="utf-8"), "eins\nzwei\n")
        unmade = tools.Edit(str(target), "absent", "x")
        self.assertTrue(unmade.startswith("Error: "), unmade)
        self.assertIn("String not found in file", unmade)

    @gnu_coreutils_only
    def test_both_editors_replace_the_matched_lines_where_they_are(self) -> None:
        """The whitespace-tolerant match is replaced at its place (S11 re-fix 6).

        The matched lines went to the editors as text to find: a longer
        line holding the same text made the one match "appear 2 times".
        The place is counted in bytes (the "\u00e9" before it is two); when
        the file changed since it was fetched (other bytes at the place,
        or the file ending before it), the text is searched for as
        before.
        """
        target = self.dir / "span.txt"
        for bash in (host_bash, host_bash_without_python):
            for eol in ("\n", "\r\n"):
                target.write_bytes(f"pr\u00e9fix  one  two{eol}  one  two{eol}".encode())
                tools = DockerTools(bash, _read_bytes)
                result = tools.Edit(str(target), " one two", " changed\n more")
                self.assertIn("Successfully replaced 1", result)
                self.assertEqual(
                    target.read_bytes(),
                    f"pr\u00e9fix  one  two{eol}  changed{eol}  more{eol}".encode(),
                )
            for rewritten in ("added first\n\u00e9\n  one  two\n", "  one  two\n"):
                target.write_text("\u00e9 a longer first line\n  one  two\n", encoding="utf-8")
                tools = DockerTools(
                    bash, functools.partial(_read_bytes_then_rewrite, rewritten.encode()),
                )
                result = tools.Edit(str(target), " one two", " changed")
                self.assertIn("Successfully replaced 1", result)
                self.assertEqual(
                    target.read_text(encoding="utf-8"),
                    rewritten.replace("  one  two", "  changed"),
                )

    @gnu_coreutils_only
    def test_write_keeps_a_newline_at_the_end_of_a_path(self) -> None:
        """A path ending in a newline names its own file, not the one without it (S11 re-fix 1).

        A command substitution drops the trailing newlines of what it
        captures: the resolved target lost its own, so the Write of
        ``name\\n`` replaced ``name`` and reported success, and the same
        cut sent the parent directory and the discard elsewhere.
        """
        plain, odd = self.dir / "target", self.dir / "target\n"
        plain.write_text("plain", encoding="utf-8")
        for content in ("one piece", "x" * 100_000):
            self.assertIn("Successfully wrote", self.tools.Write(str(odd), content))
            self.assertEqual(odd.read_text(encoding="utf-8"), content)
            self.assertEqual(plain.read_text(encoding="utf-8"), "plain")
        nested = self.dir / "sub\n" / "file.txt"
        self.assertIn("Successfully wrote", self.tools.Write(str(nested), "in"))
        self.assertEqual(nested.read_text(encoding="utf-8"), "in")
        # A store that fails (the target is a directory) discards its staging file.
        folder = self.dir / "folder\n"
        folder.mkdir()
        for content in ("one piece", "x" * 100_000):
            self.assertNotIn("Successfully wrote", self.tools.Write(str(folder), content))
        self.assertEqual(
            sorted(p.name for p in self.dir.iterdir()),
            ["folder\n", "sub\n", "target", "target\n"],
        )

    @gnu_coreutils_only
    def test_both_editors_report_a_failed_write_and_keep_the_file(self) -> None:
        """A write the file system refuses is no success, and the file is intact (A08 astra-5)."""
        target = self.dir / "big.txt"
        content = "a" * 2000 + "END"
        for with_python in (True, False):
            target.write_text(content, encoding="utf-8")
            tools = DockerTools(functools.partial(size_limited_bash, with_python=with_python))
            result = tools.Edit(str(target), "END", "FIN")
            self.assertNotIn("Successfully", result)
            self.assertEqual(target.read_text(encoding="utf-8"), content)
            self.assertEqual([p.name for p in self.dir.iterdir()], ["big.txt"])

    @gnu_coreutils_only
    def test_a_raised_executor_error_still_discards_the_payload_and_staging(self) -> None:
        """A transient executor failure leaves no payload or staging file behind (A08 astra-14)."""
        target = self.dir / "edit.txt"
        target.write_text("aaa bbb", encoding="utf-8")
        commands: list[str] = []
        tools = DockerTools(functools.partial(recording_bash, commands, '"$PYTHON" -c'))
        with self.assertRaises(KISSError):
            tools.Edit(str(target), "bbb", "ccc")
        self.assertTrue(commands[-1].startswith("rm -f /tmp/.kiss-edit-"), commands[-1])
        self.assertFalse(Path(commands[-1].removeprefix("rm -f ")).exists())
        self.assertEqual(target.read_text(encoding="utf-8"), "aaa bbb")
        commands.clear()
        tools = DockerTools(functools.partial(recording_bash, commands, "(part 2/2)"))
        with self.assertRaises(KISSError):
            tools.Write(str(self.dir / "big.txt"), "x" * 100_000)
        self.assertTrue(commands[-1].startswith("rm -f "), commands[-1])
        self.assertEqual([p.name for p in self.dir.iterdir()], ["edit.txt"])

    @gnu_coreutils_only
    def test_edits_and_writes_of_one_process_do_not_interleave(self) -> None:
        """A second Edit or Write waits for a first Edit's read-modify-publish (astra-2 / fable-1).

        As the host ``Edit`` and ``Write`` under its ``_FILE_WRITE_LOCK``:
        the tool instances of a task and its sub-agents (threads of one
        process attached to one container) would otherwise undo each
        other's changes, both reporting success, when one's write landed
        between another's read and publication of the same file.  The
        first editor script is held on an Event; the second call's
        write must not start while it is held.
        """
        target = self.dir / "shared.txt"
        for second in ("Edit", "Write"):
            target.write_text("first=old\nsecond=old\n", encoding="utf-8")
            events: list[str] = []
            entered, release = threading.Event(), threading.Event()
            tools = DockerTools(functools.partial(pausing_bash, events, entered, release))
            results: list[str] = []
            first = threading.Thread(
                target=_collect, args=(results, tools.Edit, str(target), "first=old", "first=new"),
            )
            first.start()
            self.assertTrue(entered.wait(10), "the first editor never ran")
            args = (
                (tools.Edit, str(target), "second=old", "second=new") if second == "Edit"
                else (tools.Write, str(target), "first=new\nsecond=new\n")
            )
            other = threading.Thread(target=_collect, args=(results, *args))
            other.start()
            other.join(1.0)
            self.assertEqual(events, ["edit start"], f"{second} ran during a paused Edit")
            release.set()
            first.join(10)
            other.join(10)
            self.assertFalse(first.is_alive() or other.is_alive())
            kind = second.lower()
            self.assertEqual(events, ["edit start", "edit end", f"{kind} start", f"{kind} end"])
            self.assertTrue(all("Successfully" in r for r in results), results)
            self.assertEqual(target.read_text(encoding="utf-8"), "first=new\nsecond=new\n")

    @gnu_coreutils_only
    def test_write_stages_beside_the_file_a_symlink_points_to(self) -> None:
        """The partial file lives beside the resolved target (A08 astra-4: one file system)."""
        other = self.dir / "other"
        other.mkdir()
        real = other / "real.txt"
        real.write_text("previous", encoding="utf-8")
        link = self.dir / "link.txt"
        link.symlink_to(real)
        seen: list[str] = []
        tools = DockerTools(functools.partial(peeking_bash, seen, [self.dir, other]))
        self.assertIn("Successfully wrote", tools.Write(str(link), "x" * 100_000))
        partials = [p for p in seen if ".kiss-partial-" in p]
        self.assertEqual(len(partials), 1, seen)
        self.assertEqual(Path(partials[0]).parent, other, partials[0])
        self.assertTrue(link.is_symlink())
        self.assertEqual(real.read_text(encoding="utf-8"), "x" * 100_000)
        self.assertEqual(sorted(p.name for p in other.iterdir()), ["real.txt"])

    @gnu_coreutils_only
    def test_write_through_a_symlink_repointed_between_pieces_lands_whole(self) -> None:
        """The file a link pointed to when the Write began gets every piece (S3/F2).

        Each piece resolved the link anew: repointed between two pieces,
        the later ones staged beside the new target, which was published
        with a suffix of the data as the whole, while the first pieces
        stayed beside the old target, out of the discard's reach.  As the
        ``cat > link`` the Write stands for, which keeps the file it
        opened, the first command's resolution binds all of them.
        """
        first, second, link = self.dir / "first", self.dir / "second", self.dir / "link"
        first.write_text("first-original", encoding="utf-8")
        second.write_text("second-original", encoding="utf-8")
        link.symlink_to(first)
        commands: list[str] = []
        tools = DockerTools(functools.partial(switching_bash, commands, link, second))
        content = "A" * _CHUNK_BYTES + "B" * 23
        self.assertIn("Successfully wrote", tools.Write(str(link), content))
        self.assertEqual(first.read_text(encoding="utf-8"), content)
        self.assertEqual(second.read_text(encoding="utf-8"), "second-original")
        self.assertEqual(os.readlink(link), str(second))
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["first", "link", "second"])
        noted = re.compile(r"/tmp/\.kiss-partial-\w+\.note")
        notes = {m.group() for c in commands for m in noted.finditer(c)}
        self.assertTrue(notes and not any(Path(n).exists() for n in notes), notes)
        # A file in ``/tmp`` is staged in the directory of the notes: under
        # the note's very name, its pieces overwrote the path noted there
        # and the next command read them as one (S11 re-fix round 2).
        handle, beside = tempfile.mkstemp(dir="/tmp", prefix="kiss-note-")
        os.close(handle)
        self.addCleanup(os.unlink, beside)
        self.assertIn("Successfully wrote", self.tools.Write(beside, content))
        self.assertEqual(Path(beside).read_text(encoding="utf-8"), content)
        # A one-piece store (no note to read) still discards its staging
        # file where the target resolves.
        (self.dir / "adir").mkdir()
        self.assertNotIn("Successfully", tools.Write(str(self.dir / "adir"), "short"))
        self.assertEqual(
            sorted(p.name for p in self.dir.iterdir()), ["adir", "first", "link", "second"],
        )

    @gnu_coreutils_only
    def test_write_keeps_a_hard_linked_file_on_its_inode(self) -> None:
        """A Write of a file with other hard links goes through their shared inode (S3/F3).

        As the host Write and both Edit editors: the rename cut the file
        from its other names, which kept the old text; so did the Write
        a ``cat > file`` here-document copy is redirected to, while the
        copy itself writes through the inode.  A file with one name is
        still replaced by rename (a new inode), keeping its mode.
        """
        one, two = self.dir / "one.txt", self.dir / "two.txt"
        one.write_text("before\n", encoding="utf-8")
        one.chmod(0o640)
        os.link(one, two)
        inode = one.stat().st_ino
        with self.assertRaises(ToolRedirect) as caught:
            _rewrite_instead(
                f"cat > {one} <<'EOF'\nafter\nEOF", str(self.dir), os.path.exists,
                os.path.isdir, os.path.islink, _read_bytes, "copy", 30, 1000, _forget,
            )
        self.assertEqual(caught.exception.name, "Write")
        self.assertIn("Successfully wrote", self.tools.Write(**caught.exception.arguments))
        self.assertEqual((two.read_text(encoding="utf-8"), two.stat().st_ino), ("after\n", inode))
        big = "x" * 100_000
        self.assertIn("Successfully wrote", self.tools.Write(str(one), big))
        self.assertEqual((two.read_text(encoding="utf-8"), one.stat().st_ino), (big, inode))
        self.assertEqual(stat.S_IMODE(one.stat().st_mode), 0o640)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["one.txt", "two.txt"])
        two.unlink()
        self.assertIn("Successfully wrote", self.tools.Write(str(one), "alone\n"))
        self.assertNotEqual(one.stat().st_ino, inode)
        self.assertEqual(stat.S_IMODE(one.stat().st_mode), 0o640)


class TestDockerManagerHelpers(unittest.TestCase):
    """F3 + F4 — the helpers that carry the manager's fixes."""

    def test_decoder_handles_split_multibyte_sequence(self) -> None:
        """A character split across two frames decodes once both arrive."""
        decoder = _new_utf8_decoder()
        raw = "\u00e9".encode()
        self.assertEqual(decoder.decode(raw[:1]), "")
        self.assertEqual(decoder.decode(raw[1:]), "\u00e9")

    def test_decoder_replaces_invalid_bytes(self) -> None:
        """Binary output is degraded, never fatal."""
        decoder = _new_utf8_decoder()
        text = decoder.decode(b"\xff\xfe hello")
        self.assertIn("hello", text)
        self.assertIn("\ufffd", text)

    def test_decoder_flush_emits_dangling_partial(self) -> None:
        """A truncated sequence at end of stream becomes U+FFFD, not an error."""
        decoder = _new_utf8_decoder()
        self.assertEqual(decoder.decode("\u00e9".encode()[:1]), "")
        self.assertEqual(decoder.decode(b"", True), "\ufffd")

    def test_decoders_are_independent(self) -> None:
        """stdout and stderr must not share decoder state."""
        first, second = _new_utf8_decoder(), _new_utf8_decoder()
        raw = "\u00e9".encode()
        self.assertEqual(first.decode(raw[:1]), "")
        self.assertEqual(second.decode(b"x"), "x")
        self.assertEqual(first.decode(raw[1:]), "\u00e9")

    def test_truncation_cap_matches_useful_tools(self) -> None:
        """The docker default cap is the configured default UsefulTools uses."""
        self.assertEqual(MAX_OUTPUT_CHARS, DEFAULT_CONFIG.tool_output_default_chars)
        self.assertEqual(MAX_OUTPUT_CHARS, 8000)
        truncated = _truncate_output("x" * 200000, MAX_OUTPUT_CHARS)
        self.assertLessEqual(len(truncated), MAX_OUTPUT_CHARS)
        self.assertIn("truncated", truncated)

    def test_with_exit_code_marks_only_failures(self) -> None:
        """The exit-code marker is appended exactly when the command failed."""
        self.assertEqual(_with_exit_code("out", 0), "out")
        self.assertEqual(_with_exit_code("out", 3), "out\n[exit code: 3]")
        self.assertEqual(_with_exit_code("", 3), "[exit code: 3]")
        self.assertEqual(_with_exit_code("", 0), "")


if __name__ == "__main__":
    unittest.main()
