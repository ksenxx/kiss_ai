# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sweep 2026-10-26 (sorcar-tools F1/F2): Docker tools after a Stop and the rewrite guard.

F1: ``DockerTools.Edit(edits=...)`` only stopped the batch on an inner
result starting with ``Error``.  A task Stop makes ``DockerManager.Bash``
return ``Killed: the task was stopped.`` (with the exit-code line after
it), which the loop recorded as ``edit 1: ...`` and then went on with the
next edit: every remaining edit still stored its payload through an exec
that was cancelled at once, each costing a worker thread and a bounded
reaper wait, and the report listed them all as results.  The batch now
returns the Stop result as is after the first stopped edit.

F2: ``DockerManager.Bash`` and ``run_commands_parallel`` applied only the
destructive/install guards; the host ``Bash`` also refuses the shell
rewrites ``Edit`` should make (``sed -i`` with a multi-line script, a
``python3 -`` heredoc that reads, replaces and writes a file back).  The
container tools now apply the same ``rewrite_guard``.  Since 2026-10-31
a rewrite that spells out one exact ``Write``/``Edit`` (a literal
here-document copy, a python ``.replace``, a one-substitution ``sed -i``)
is raised by ``Bash`` as that call instead
(:class:`~kiss.core.tool_verdict.ToolRedirect`, which the agent makes
through its own tool path: hook, guard and events see it as if the
model had issued it; ``Bash`` itself touches nothing), while
``run_commands_parallel``, which cannot redirect, still refuses it.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar.docker_manager import STOPPED_MESSAGE, DockerManager, PendingEdit
from kiss.agents.sorcar.docker_tools import DockerTools
from kiss.core.tool_verdict import ALLOW, ToolRedirect, Verdict, refuse
from kiss.tests.agents.sorcar.docker_test_containers import IMAGE, is_docker_available
from kiss.tests.agents.sorcar.test_docker_parity_2026_10_31 import _Harness

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running"),
]

_SED_REWRITE = "sed -i 's/a/b\\nc/' /tmp/some_file.txt"
_HEREDOC_REWRITE = (
    "python3 - <<'EOF'\n"
    "p = '/tmp/some_file.txt'\n"
    "s = open(p).read()\n"
    "s = s.replace('a', 'b')\n"
    "open(p, 'w').write(s)\n"
    "EOF"
)
# A cat here-document over the existing file after a ``cd`` (``/`` is the
# default workdir): the guard asks the container whether ``/`` is a directory.
_CAT_HEREDOC_REWRITE = "cd / && cat > tmp/some_file.txt <<'EOF'\nb\nEOF"
# The exact call each rewrite of the seeded ``a\n`` stands for (an Edit matching
# verbatim only, as sed and ``str.replace`` do: ``exact``).
_REWRITE_CALLS = {
    _SED_REWRITE: (
        "Edit",
        {"file_path": "/tmp/some_file.txt", "old_string": "a", "new_string": "b\nc",
         "replace_all": False, "exact": True},
    ),
    _HEREDOC_REWRITE: (
        "Edit",
        {"file_path": "/tmp/some_file.txt", "old_string": "a", "new_string": "b",
         "replace_all": True, "exact": True},
    ),
    _CAT_HEREDOC_REWRITE: ("Write", {"file_path": "/tmp/some_file.txt", "content": "b\n"}),
}
# Python's executables moved aside inside the container (and back): the
# Edit then runs its Perl fallback, as in an image without Python.
_PYTHON_OFF = (
    "for p in /usr/local/bin/python /usr/local/bin/python3; do mv \"$p\" \"$p.off\"; done"
    " && ! command -v python3 && ! command -v python"
)
_PYTHON_ON = "for p in /usr/local/bin/python /usr/local/bin/python3; do mv \"$p.off\" \"$p\"; done"
_CRLF = b"a\r\nb\r\nc\r\nd\r\n"
_SEED_CRLF = "printf 'a\\r\\nb\\r\\nc\\r\\nd\\r\\n' > /tmp/race.txt"


def _raw(env: DockerManager, command: str) -> str:
    """Run *command* in the container as is (no guard, no redirect); it must exit 0."""
    assert env.container is not None
    result = env.container.exec_run(["/bin/bash", "-c", command])
    output = result.output
    assert isinstance(output, bytes)  # not streamed: ``exec_run`` returns the whole output
    assert result.exit_code == 0, output
    return output.decode()


@pytest.fixture(scope="module")
def env() -> Iterator[DockerManager]:
    """A manager on a fresh container of :data:`IMAGE`."""
    manager = DockerManager(IMAGE)
    manager.open()
    try:
        yield manager
    finally:
        manager.close()


def test_stopped_edit_batch_ends_at_the_first_stopped_edit(
    env: DockerManager,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The batch returns the Stop result as is; the later edits never run."""
    tools = DockerTools(env.Bash)
    env.stop_event = threading.Event()
    try:
        env.Bash("printf 'one\\ntwo\\nthree\\n' > /tmp/batch_stop.txt", "seed")
        capsys.readouterr()
        env.stop_event.set()
        result = tools.Edit(
            "/tmp/batch_stop.txt",
            edits=json.dumps([
                {"old_string": "one", "new_string": "uno"},
                {"old_string": "two", "new_string": "dos"},
                {"old_string": "three", "new_string": "tres"},
            ]),
        )
    finally:
        env.stop_event = None
    assert STOPPED_MESSAGE in result, result
    assert not result.startswith("edit 1:"), result
    assert "edit 2" not in result and "edit 3" not in result, result
    # Only the first edit's payload store reached the container.
    assert capsys.readouterr().out.count("Edit /tmp/batch_stop.txt") == 1
    assert env.Bash("cat /tmp/batch_stop.txt", "cat").startswith("one\ntwo\nthree")


def test_failed_edit_batch_still_reports_the_failing_edit(env: DockerManager) -> None:
    """An ordinary failure keeps its ``Error: edit k`` report (unchanged contract)."""
    tools = DockerTools(env.Bash)
    env.Bash("printf 'one\\ntwo\\n' > /tmp/batch_fail.txt", "seed")
    result = tools.Edit(
        "/tmp/batch_fail.txt",
        edits=json.dumps([
            {"old_string": "one", "new_string": "uno"},
            {"old_string": "missing", "new_string": "x"},
        ]),
    )
    assert result.startswith("Error: edit 2: edits 1-1 applied; "), result
    assert env.Bash("cat /tmp/batch_fail.txt", "cat").startswith("uno\ntwo")


@pytest.mark.parametrize("command", [_SED_REWRITE, _HEREDOC_REWRITE, _CAT_HEREDOC_REWRITE])
def test_docker_bash_makes_shell_rewrites_as_the_edit_or_write(
    env: DockerManager, command: str,
) -> None:
    """The container ``Bash`` raises the exact ``Edit``/``Write`` a rewrite stands for."""
    env.Bash("printf 'a\\n' > /tmp/some_file.txt", "seed")
    with pytest.raises(ToolRedirect) as info:
        env.Bash(command, "rewrite")
    name, arguments = _REWRITE_CALLS[command]
    assert (info.value.name, info.value.arguments) == (name, arguments)
    assert info.value.followup is None
    assert info.value.note.startswith(f"Applied as {name}("), info.value.note
    # The call is the agent's to make: Bash itself left the file alone.
    assert env.Bash("cat /tmp/some_file.txt", "cat").startswith("a\n")


@pytest.mark.parametrize(
    ("command", "name", "args", "content"),
    [
        (
            _SED_REWRITE, "Edit",
            {"old_string": "a", "new_string": "b\nc", "replace_all": False, "exact": True},
            b"b\nc\n",
        ),
        (
            _HEREDOC_REWRITE, "Edit",
            {"old_string": "a", "new_string": "b", "replace_all": True, "exact": True},
            b"b\n",
        ),
        (_CAT_HEREDOC_REWRITE, "Write", {"content": "b\n"}, b"b\n"),
    ],
)
def test_docker_bash_makes_shell_rewrites_as_edit_or_write(
    env: DockerManager, command: str, name: str, args: dict[str, Any], content: bytes,
) -> None:
    """The container ``Bash`` makes the Edit/Write a rewrite stands for through the agent."""
    h = _Harness(env)
    env.Bash("printf 'a\\n' > /tmp/some_file.txt", "seed")
    expected = (name, {"file_path": "/tmp/some_file.txt", **args})
    # The hook sees the redirected call and may refuse it: nothing changes then.
    h.verdict = refuse("not yet")
    result = h.call("Bash", command=command, description="rewrite")
    assert result.startswith(f"Applied as {name}(") and result.endswith(
        f"REFUSED: the {name} call did NOT run and had no effect. not yet"
    ), result
    assert h.hook_calls[-1] == expected and h.tool_calls() == ["Bash", name]
    assert env.read_bytes("/tmp/some_file.txt") == b"a\n"
    # Allowed, the call lands and the rewrite's result opens with what it was applied as.
    h.verdict = ALLOW
    result = h.call("Bash", command=command, description="rewrite")
    assert result.startswith(f"Applied as {name}(") and "Successfully" in result, result
    assert h.hook_calls[-1] == expected and h.tool_calls() == ["Bash", name, "Bash", name]
    assert env.read_bytes("/tmp/some_file.txt") == content


def test_docker_redirected_edit_is_applied_verbatim_only(env: DockerManager) -> None:
    """The Edit a ``sed -i`` is redirected to matches verbatim, whatever the file holds by then.

    The pair is settled from the file's text when the command is
    converted and the agent makes the Edit afterwards, its hook first:
    a writer landing in between (here the hook itself, writing through
    the container) turned a sed matching nothing into an Edit landing
    through the whitespace-tolerant retry, and a python ``.replace`` over
    a file turned CRLF into a batch landing through the CRLF retry of the
    container's Python script and of its Perl fallback (audit 2026-11-08,
    S3/F4; round 2).  The model's own Edit of the same strings keeps both
    retries; a batch carries ``exact`` to each of its edits.
    """
    h = _Harness(env)
    env.Bash("printf 'a b\\n' > /tmp/race.txt && printf 'a   b\\n' > /tmp/race_oracle.txt", "seed")

    def hook_writing_first(name: str, args: dict[str, Any]) -> Verdict:
        if name == "Edit":
            env.Bash("printf 'a   b\\n' > /tmp/race.txt", "another writer")
        return h.hook(name, args)

    h.agent.tool_call_hook = hook_writing_first
    out = h.call("Bash", command="sed -i 's/a b/X/' /tmp/race.txt", description="sed")
    assert out.startswith(
        'Applied as Edit(file_path="/tmp/race.txt", old_string="a b", new_string="X"), '
        "whose result follows:\nError: String not found in file"
    ), out
    assert h.hook_calls[-1] == ("Edit", {
        "file_path": "/tmp/race.txt", "old_string": "a b", "new_string": "X",
        "replace_all": False, "exact": True,
    })
    assert env.read_bytes("/tmp/race.txt") == b"a   b\n"
    # sed itself over that text changes nothing either.
    oracle = env.Bash("sed -i 's/a b/X/' /tmp/race_oracle.txt && cat /tmp/race_oracle.txt", "sed")
    assert oracle.startswith("a   b\n"), oracle
    # The model's own Edit keeps the retry; its exact batch does not.
    h.agent.tool_call_hook = h.hook
    edits = json.dumps([{"old_string": "a b", "new_string": "X"}])
    out = h.call("Edit", file_path="/tmp/race.txt", edits=edits, exact=True)
    assert out.startswith("Error: edit 1: String not found in file"), out
    assert env.read_bytes("/tmp/race.txt") == b"a   b\n"
    out = h.call("Edit", file_path="/tmp/race.txt", edits=edits)
    assert out.startswith("edit 1: Successfully replaced 1 occurrence(s)"), out
    assert env.read_bytes("/tmp/race.txt") == b"X\n"
    # Two ``.replace`` calls over a file that turns CRLF before the call is made.
    program = (
        "s = open('/tmp/race.txt', newline='').read()\n"
        "s = s.replace('a\\nb', 'A\\nB').replace('c\\nd', 'C\\nD')\n"
        "open('/tmp/race.txt', 'w', newline='').write(s)\n"
    )
    command = f"python3 - <<'EOF'\n{program}EOF"
    edits = json.dumps([
        {"old_string": "a\nb", "new_string": "A\nB", "replace_all": True},
        {"old_string": "c\nd", "new_string": "C\nD", "replace_all": True},
    ])

    def hook_turning_crlf_first(name: str, args: dict[str, Any]) -> Verdict:
        if name == "Edit":
            env.Bash(_SEED_CRLF, "another writer")
        return h.hook(name, args)

    env.Bash("printf 'a\\nb\\nc\\nd\\n' > /tmp/race.txt", "seed")
    h.agent.tool_call_hook = hook_turning_crlf_first
    out = h.call("Bash", command=command, description="two literal replacements")
    assert h.hook_calls[-1] == (
        "Edit", {"file_path": "/tmp/race.txt", "edits": edits, "exact": True},
    )
    assert "whose result follows:\nError: edit 1: String not found in file" in out, out
    assert env.read_bytes("/tmp/race.txt") == _CRLF
    # The program itself over that text changes nothing either.
    _raw(env, _SEED_CRLF.replace("race.txt", "race_oracle.txt"))
    _raw(env, command.replace("race.txt", "race_oracle.txt"))
    assert env.read_bytes("/tmp/race_oracle.txt") == _CRLF
    # Direct exact calls, one pair and a batch, in the container's Python and
    # in its Perl fallback; the model's own batch keeps the CRLF retry in both.
    h.agent.tool_call_hook = h.hook
    single = {"old_string": "a\nb", "new_string": "A\nB", "replace_all": True}
    for backend in ("python", "perl"):
        if backend == "perl":
            _raw(env, _PYTHON_OFF)
        try:
            for call in (single, {"edits": edits}):
                env.Bash(_SEED_CRLF, "seed")
                out = h.call("Edit", file_path="/tmp/race.txt", exact=True, **call)
                assert "String not found in file" in out, (backend, out)
                assert env.read_bytes("/tmp/race.txt") == _CRLF, backend
            out = h.call("Edit", file_path="/tmp/race.txt", edits=edits)
            assert out.startswith("edit 1: Successfully replaced 1 occurrence(s)"), (backend, out)
            assert env.read_bytes("/tmp/race.txt") == b"A\r\nB\r\nC\r\nD\r\n", backend
        finally:
            if backend == "perl":
                _raw(env, _PYTHON_ON)


def test_docker_redirected_edit_is_refused_once_the_file_holds_another_text(
    env: DockerManager,
) -> None:
    """The Edit a rewrite is redirected to is made only while it is what the command stands for.

    Its pairs are settled from the file's text when the command is
    converted.  Another writer of the container landing before the Edit
    (here from the hook, past this manager) may leave a text the command
    changes otherwise: a line holding a per-line ``sed`` pattern twice,
    where the ``replace_all`` settled for two lines replaced both, and a
    string a python ``.replace`` did not find, which the Edit left out.
    The Edit is refused, as the host's is, and writes nothing; the
    command issued again does what it does by itself.  So is the Edit
    of a file nobody can read any more.  Only the Edit made for the
    command is held to it.
    """
    h = _Harness(env)
    sed = "sed -i 's/a/X/' /tmp/stale.txt"
    refused = (
        "whose result follows:\nError: /tmp/stale.txt changed since the {} was read as this "
        "Edit, which the command no longer stands for: nothing was written. Re-issue the command."
    )
    program = (
        "s = open('/tmp/stale.txt', newline='').read()\n"
        "s = s.replace('one', '1').replace('two', '2')\n"
        "open('/tmp/stale.txt', 'w', newline='').write(s)\n"
    )
    writes: list[str] = []

    def hook_after_another_writer(name: str, args: dict[str, Any]) -> Verdict:
        if name == "Edit" and writes:
            _raw(env, writes.pop())
        return h.hook(name, args)

    h.agent.tool_call_hook = hook_after_another_writer
    python = f"python3 - <<'EOF'\n{program}EOF"
    both = json.dumps([
        {"old_string": "one", "new_string": "1", "replace_all": True},
        {"old_string": "two", "new_string": "2", "replace_all": True},
    ])
    for command, what, seeded, changed, edit in (
        (sed, "`sed -i` call", "a\\na\\n", "a a\\n",
         {"old_string": "a", "new_string": "X", "replace_all": True}),
        (python, "python heredoc", "one\\n", "one\\ntwo\\n",
         {"old_string": "one", "new_string": "1", "replace_all": True}),
        # Two replacements settled, of which the file then takes one.
        (python, "python heredoc", "one\\ntwo\\n", "one\\n", {"edits": both}),
    ):
        _raw(env, f"printf '{seeded}' > /tmp/stale.txt; printf '{changed}' > /tmp/stale_oracle.txt")
        writes.append(f"printf '{changed}' > /tmp/stale.txt")
        out = h.call("Bash", command=command, description="rewrite")
        assert h.hook_calls[-1] == ("Edit", {"file_path": "/tmp/stale.txt", **edit, "exact": True})
        assert out.endswith(refused.format(what)), out
        assert _raw(env, "cat /tmp/stale.txt") == _raw(env, "cat /tmp/stale_oracle.txt")
        h.call("Bash", command=command, description="rewrite")
        _raw(env, command.replace("stale.txt", "stale_oracle.txt"))
        assert _raw(env, "cat /tmp/stale.txt") == _raw(env, "cat /tmp/stale_oracle.txt"), what
    # A file removed meanwhile; the command issued again finds none.
    _raw(env, "printf 'a\\na\\n' > /tmp/stale.txt")
    writes.append("rm /tmp/stale.txt")
    out = h.call("Bash", command=sed, description="rewrite")
    assert out.endswith(refused.format("`sed -i` call")), out
    out = h.call("Bash", command=sed, description="rewrite")
    assert "whose result follows:\nError: File not found: /tmp/stale.txt" in out, out
    # The Edit the hook refused leaves the next one, another Edit of the file, as it is.
    _raw(env, "printf 'a\\na\\n' > /tmp/stale.txt")
    h.verdict = refuse("not yet")
    h.call("Bash", command=sed, description="rewrite")
    assert isinstance(env.pending_append, PendingEdit)
    h.verdict = ALLOW
    out = h.call("Edit", file_path="/tmp/stale.txt", old_string="a\na", new_string="b")
    assert out.startswith("Successfully replaced 1 occurrence(s)"), out
    assert env.pending_append is None and _raw(env, "cat /tmp/stale.txt") == "b\n"


#: A ``.pth`` file of the container python's site-packages (the shell names the directory).
_PTH = "\"$(python3 -c 'import site; print(site.getsitepackages()[0])')/shadow.pth\""


def test_docker_python_rewrite_is_no_edit_where_another_module_shadows_its_import(
    env: DockerManager,
) -> None:
    """A python rewrite is made as an Edit only where its imports are the library's.

    Python searches the directory a program read from stdin runs in, and
    the entries of ``PYTHONPATH``, before its own library: with a
    ``json`` module there ``import json`` fails in bash and the file
    stays as it was, while the Edit the program stands for was made
    (cross finding X6 of 2026-11-12; the host asks ``_host_shadows``).
    The container is asked, in the environment the command gets; an
    answer nobody got (the task is stopped) stands for no Edit either.
    It is the interpreter that is asked where it finds the modules
    (round 2, B7 #28): a reading of the directory and ``PYTHONPATH``
    saw neither a directory python's own startup puts first (a
    ``.pth``) nor a module the library's ``json`` itself imports.
    """
    h = _Harness(env)
    assert env.container is not None
    program = ("import json\ns = open('/tmp/shadow/a.py').read()\ns = s.replace('x', 'y')\n"
               "open('/tmp/shadow/a.py', 'w').write(s)\n")
    rewrite = f"python3 - <<'EOF'\n{program}EOF"
    command = f"cd /tmp/shadow && {rewrite}"
    refused = "REFUSED: the python heredoc did NOT run"
    _raw(env, "mkdir -p /tmp/shadow/lib && printf 'x = 1\\n' > /tmp/shadow/a.py")
    module = "printf 'raise RuntimeError(1)\\n' > json.py"
    # How the module gets there, and the environment of the command.
    layouts: tuple[tuple[str, dict[str, str]], ...] = (
        (module, {}),  # in the directory the program runs in
        ("mv json.py lib/", {"PYTHONPATH": "lib"}),  # in a PYTHONPATH directory, named from it
        # A package whose __init__ is an extension module.
        ("rm lib/json.py && mkdir json && : > json/__init__.so", {}),
        # In an archive on PYTHONPATH.
        (f"rm -r json && {module} && python3 -m zipfile -c lib.zip json.py && rm json.py",
         {"PYTHONPATH": "lib.zip"}),
        (":", {"PYTHONPATH": "lib.zip/"}),  # the same archive, as python takes it too
        # In a directory of the archive.
        (f"mkdir pkg && {module} && mv json.py pkg/ && python3 -m zipfile -c lib.zip pkg"
         " && rm -r pkg", {"PYTHONPATH": "lib.zip/pkg"}),
        # Named as python reads it: `..` drops the name before it, whatever is there.
        (":", {"PYTHONPATH": "missing/../lib.zip/pkg"}),
        # In a directory an executable line of a site-packages .pth puts first.
        (f"mkdir -p /tmp/pthlib && {module} && mv json.py /tmp/pthlib/ && printf "
         f"'import sys; sys.path.insert(0, \"/tmp/pthlib\")\\n' > {_PTH}", {}),
        # A module of the directory that the library's json imports (through re).
        ("printf 'raise RuntimeError(1)\\n' > enum.py", {}),
    )
    for setup, environment in layouts:
        _raw(env, f"cd /tmp/shadow && {setup}")
        env.exec_env = environment
        try:
            in_bash = env.container.exec_run(["/bin/bash", "-c", command], environment=environment)
            assert in_bash.exit_code == 1, setup
            out = h.call("Bash", command=command, description="shadowed")
            assert out.startswith(refused), (setup, out)
        finally:
            env.exec_env = {}
            _raw(env, f"rm -f {_PTH}")
    _raw(env, "rm /tmp/shadow/lib.zip /tmp/shadow/enum.py")
    assert h.hook_calls[-1][0] == "Bash" and _raw(env, "cat /tmp/shadow/a.py") == "x = 1\n"
    # An answer nobody got: the directory may hold any module.
    env.stop_event = threading.Event()
    env.stop_event.set()
    try:
        assert h.call("Bash", command=rewrite, description="stopped").startswith(refused)
    finally:
        env.stop_event = None
    # Where nothing shadows the import, the program is made as its Edit.
    out = h.call("Bash", command=command, description="not shadowed")
    assert out.startswith("Applied as Edit("), out
    assert _raw(env, "cat /tmp/shadow/a.py") == "y = 1\n"


def test_docker_python_rewrite_is_no_edit_where_bash_finds_no_python(env: DockerManager) -> None:
    """A python rewrite is made as an Edit only where bash finds its interpreter.

    Without one the command fails as written (status 127) and the file
    stays as it was, while the Edit it stands for would be made (by the
    editor's Perl fallback); the container is asked.  One found through
    a relative ``PATH`` entry counts as not found: the entry names
    another directory after the command's ``cd``.
    """
    h = _Harness(env)
    assert env.container is not None
    program = ("s = open('/tmp/nopy.py').read()\ns = s.replace('x', 'y')\n"
               "open('/tmp/nopy.py', 'w').write(s)\n")
    command = f"python3 - <<'EOF'\n{program}EOF"
    _raw(env, "printf 'x = 1\\n' > /tmp/nopy.py")
    _raw(env, _PYTHON_OFF)
    try:
        in_bash = env.container.exec_run(["/bin/bash", "-c", command])
        assert in_bash.exit_code == 127, in_bash.output
        out = h.call("Bash", command=command, description="no python3")
    finally:
        _raw(env, _PYTHON_ON)
    assert out.startswith("REFUSED: the python heredoc did NOT run"), out
    # A relative PATH entry names another directory after the command's cd.
    relbin = f"{env.workdir}/relbin"
    _raw(env, f"mkdir -p {relbin} && ln -sf /usr/local/bin/python3 {relbin}/")
    env.exec_env = environment = {"PATH": "relbin:/usr/bin:/bin"}
    try:
        in_bash = env.container.exec_run(
            ["/bin/bash", "-c", f"cd /tmp && {command}"], environment=environment,
            workdir=env.workdir,
        )
        assert in_bash.exit_code == 127, in_bash.output
        out = h.call("Bash", command=f"cd /tmp && {command}", description="relative PATH")
    finally:
        env.exec_env = {}
    assert out.startswith("REFUSED: the python heredoc did NOT run"), out
    assert _raw(env, "cat /tmp/nopy.py") == "x = 1\n"
    out = h.call("Bash", command=command, description="python3 found")
    assert out.startswith("Applied as Edit("), out
    assert _raw(env, "cat /tmp/nopy.py") == "y = 1\n"


def test_docker_run_commands_parallel_refuses_shell_rewrites(env: DockerManager) -> None:
    """The fan-out refuses a rewrite per command and still runs the others."""
    env.Bash("printf 'a\\n' > /tmp/some_file.txt", "seed")
    report = env.run_commands_parallel(
        json.dumps([_SED_REWRITE, _CAT_HEREDOC_REWRITE, "echo fine"]))
    assert report.count("REFUSED:") == 2, report
    assert "fine" in report, report
    assert env.Bash("cat /tmp/some_file.txt", "cat").startswith("a\n")


def test_docker_bash_heredoc_into_a_new_file_is_a_write_and_an_append_reads_it(
    env: DockerManager,
) -> None:
    """A here-document into a file the container lacks is its ``Write``; an append adds to it."""
    env.Bash("rm -f /tmp/heredoc_new.txt", "clean")
    with pytest.raises(ToolRedirect) as info:
        env.Bash("cat > /tmp/heredoc_new.txt <<'EOF'\nnew\nEOF", "create")
    assert info.value.name == "Write"
    assert info.value.arguments == {"file_path": "/tmp/heredoc_new.txt", "content": "new\n"}
    assert env.Bash("test -e /tmp/heredoc_new.txt; echo $?", "check").startswith("1")
    env.Bash("printf 'new\\n' > /tmp/heredoc_new.txt", "seed")
    with pytest.raises(ToolRedirect) as info:
        env.Bash("cat >> /tmp/heredoc_new.txt <<'EOF'\nmore\nEOF", "append")
    assert info.value.name == "Write"
    assert info.value.arguments == {"file_path": "/tmp/heredoc_new.txt", "content": "new\nmore\n"}
    assert info.value.note.startswith("Appended through Write("), info.value.note
    content = env.Bash("cat /tmp/heredoc_new.txt", "cat")
    assert content.startswith("new\n") and "more" not in content, content


def test_docker_bash_heredoc_may_create_a_new_file(env: DockerManager) -> None:
    """A here-document into a file the container lacks creates it; an append goes after it."""
    h = _Harness(env)
    env.Bash("rm -f /tmp/heredoc_new.txt", "clean")
    create = "cat > /tmp/heredoc_new.txt <<'EOF'\nnew\nEOF"
    created = h.call("Bash", command=create, description="create")
    assert created == (
        'Applied as Write(file_path="/tmp/heredoc_new.txt", content="new\\n"), whose result '
        "follows:\nSuccessfully wrote 4 characters to /tmp/heredoc_new.txt"
    ), created
    append = "cat >> /tmp/heredoc_new.txt <<'EOF'\nmore\nEOF"
    again = h.call("Bash", command=append, description="append")
    assert again.startswith(
        'Appended through Write(file_path="/tmp/heredoc_new.txt", content="new\\nmore\\n")'
    ), again
    assert h.hook_calls[-1] == (
        "Write", {"file_path": "/tmp/heredoc_new.txt", "content": "new\nmore\n"},
    )
    assert env.read_bytes("/tmp/heredoc_new.txt") == b"new\nmore\n"
