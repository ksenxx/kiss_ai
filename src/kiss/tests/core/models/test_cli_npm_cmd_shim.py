# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A CLI reached through an npm ``.cmd`` shim gets its arguments intact.

On Windows ``shutil.which("claude")`` returns the batch shim npm installs,
and running a batch file through ``Popen`` hands the argument list to
``cmd.exe``, which cuts every argument at its first newline: the
multi-line ``--append-system-prompt`` text lost everything after its
first line (and the arguments after it), so ``cc/*`` models never saw the
KISS tool protocol.  :func:`_unwrap_npm_cmd_shim` starts the shim's real
target instead.  These tests build shims in the exact shape npm writes;
the JavaScript one runs through :class:`_CLIProcess` on every platform
(``node`` executes the target wherever the test runs), the native one
can only be resolved, since its ``.exe`` is not a program.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

from kiss.core.models.model import _CLIProcess, _unwrap_npm_cmd_shim

requires_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

_ECHO_JS = (
    "process.stdin.on('data', () => {});\n"
    "process.stdin.on('end', () => {\n"
    "  process.stdout.write(JSON.stringify(process.argv.slice(2)) + '\\n');\n"
    "});\n"
)

# What npm's cmd-shim writes for a JavaScript ``bin`` entry (codex.cmd).
_NODE_SHIM = (
    "@ECHO off\r\nGOTO start\r\n:find_dp0\r\nSET dp0=%~dp0\r\nEXIT /b\r\n"
    ":start\r\nSETLOCAL\r\nCALL :find_dp0\r\n\r\n"
    'IF EXIST "%dp0%\\node.exe" (\r\n  SET "_prog=%dp0%\\node.exe"\r\n) ELSE (\r\n'
    '  SET "_prog=node"\r\n  SET PATHEXT=%PATHEXT:;.JS;=;%\r\n)\r\n\r\n'
    'endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"  '
    '"%dp0%\\node_modules\\@openai\\codex\\bin\\codex.js" %*\r\n'
)

# What it writes for a native ``bin`` entry (claude.cmd since the CLI ships
# a platform binary).
_EXE_SHIM = (
    "@ECHO off\r\nGOTO start\r\n:find_dp0\r\nSET dp0=%~dp0\r\nEXIT /b\r\n"
    ":start\r\nSETLOCAL\r\nCALL :find_dp0\r\n"
    '"%dp0%\\node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe"   %*\r\n'
)

_SYSTEM_PROMPT = 'line one\nline two with "quotes" & %PATH% | more\n{"tool_calls": []}'


def _write_node_shim(bin_dir: Path, flags: str = "") -> Path:
    """Write ``codex.cmd`` plus the JavaScript target it names under *bin_dir*.

    *flags* are the interpreter flags of a ``#!/usr/bin/env -S node <flags>``
    shebang, which npm puts between ``"%_prog%"`` and the target.
    """
    target = bin_dir / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
    target.parent.mkdir(parents=True)
    target.write_text(_ECHO_JS, encoding="utf-8")
    shim = bin_dir / "codex.cmd"
    shim.write_bytes(_NODE_SHIM.replace('"%_prog%"  ', f'"%_prog%" {flags} ').encode())
    return shim


@requires_node
def test_arguments_survive_an_npm_node_shim(tmp_path: Path) -> None:
    """Every argument, newlines and all, reaches the program behind the shim."""
    shim = _write_node_shim(tmp_path)
    args = [str(shim), "--append-system-prompt", _SYSTEM_PROMPT, "--model", "x"]
    with _CLIProcess(args, "probe", 60) as proc:
        proc.send_prompt("hi")
        lines = list(proc.lines())
        proc.raise_for_exit()
    assert json.loads(lines[-1]) == args[1:]


@requires_node
def test_interpreter_flags_and_a_bundled_node_are_kept(tmp_path: Path) -> None:
    """``node --no-warnings target.js`` runs with the node.exe beside the shim."""
    shim = _write_node_shim(tmp_path, "--no-warnings")
    target = str(tmp_path / "node_modules" / "@openai" / "codex" / "bin" / "codex.js")
    node = shutil.which("node")
    assert node is not None
    assert _unwrap_npm_cmd_shim([str(shim), "a"]) == [node, "--no-warnings", target, "a"]
    with _CLIProcess([str(shim), "--model", "x"], "probe", 60) as proc:
        proc.send_prompt("hi")
        lines = list(proc.lines())
        proc.raise_for_exit()
    assert json.loads(lines[-1]) == ["--model", "x"]
    # npm's shim prefers a node.exe installed beside it over the PATH one.
    sibling = tmp_path / "node.exe"
    sibling.write_bytes(b"MZ")
    assert _unwrap_npm_cmd_shim([str(shim)]) == [str(sibling), "--no-warnings", target]


def test_a_native_shim_resolves_to_its_binary(tmp_path: Path) -> None:
    """A shim around a platform binary runs that binary, without ``node``."""
    target = tmp_path / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"MZ")
    shim = tmp_path / "claude.CMD"
    shim.write_bytes(_EXE_SHIM.encode())
    assert _unwrap_npm_cmd_shim([str(shim), "--print"]) == [str(target), "--print"]


def test_only_real_npm_shims_are_unwrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anything that is not an npm shim with a present target runs as is."""
    plain = [sys.executable, "-c", "pass"]
    assert _unwrap_npm_cmd_shim(plain) is plain
    # A .cmd of another shape (the test suite's own interpreter shim).
    other = tmp_path / "tool.cmd"
    other.write_text('@"python" "%~dp0tool" %*\n', encoding="utf-8")
    assert _unwrap_npm_cmd_shim([str(other), "a"]) == [str(other), "a"]
    # An npm shim whose target was uninstalled.
    dangling = tmp_path / "gone.cmd"
    dangling.write_bytes(_EXE_SHIM.encode())
    assert _unwrap_npm_cmd_shim([str(dangling)]) == [str(dangling)]
    # A directory named like a shim cannot be read.
    folder = tmp_path / "dir.cmd"
    folder.mkdir()
    assert _unwrap_npm_cmd_shim([str(folder)]) == [str(folder)]
    # A JavaScript shim on a machine without node is left to the shim.
    shim = _write_node_shim(tmp_path / "npm")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert _unwrap_npm_cmd_shim([str(shim), "a"]) == [str(shim), "a"]
