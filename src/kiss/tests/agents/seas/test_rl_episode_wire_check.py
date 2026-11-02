# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``examples/rl_episode/wire_check.py``.

The script is run as the subprocess it is meant to be, with the scripted
local OpenAI-compatible server standing in for the live API behind
``--upstream``: the recording proxy, the headless run of the episode
(``scripts/run_sea.py --json`` as a subprocess of the script) and the
byte-for-byte comparison of every recorded system message with the SEA's
``SYSTEM_PROMPT`` are all real.  The ``https://api.openai.com`` default
needs a key and costs money (that is the live run the script exists for):
the suite drives its proxy side, the bearer-token override, in process
against the scripted server and checks that a missing key stops the script
before any traffic.  Not covered: the guard against a request with zero or
several system messages or a non-string one, which the framework never
produces (it composes exactly one string system message) and which only a
fake adapter could reach.
"""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from kiss.agents.sorcar.sea_settings import execute_python_file
from kiss.tests.agents.seas.sea_contract import system_message
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    ScriptedHandler,
    serve,
    text_body,
    tool_call_body,
)

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = ROOT / "examples" / "rl_episode" / "wire_check.py"
SEA_PATH = ROOT / "examples" / "rl_episode" / "rl_episode_sea.py"
PROMPT = str(execute_python_file(SEA_PATH)["SYSTEM_PROMPT"])
RECORDING_PROXY = execute_python_file(SCRIPT)["RecordingProxy"]
EPISODE, SECRET = "episode-7", 45
SOLVED = [50, 25, 37, SECRET]


class HeaderRecordingHandler(ScriptedHandler):
    """The scripted model that also keeps every request header pair it receives."""

    headers_seen: list[list[tuple[str, str]]] = []

    def do_POST(self) -> None:  # noqa: N802
        """Record the header pairs (names lower-cased), then answer with the next scripted body."""
        self.headers_seen.append([(name.lower(), value) for name, value in self.headers.items()])
        super().do_POST()


def _values(headers: list[tuple[str, str]], name: str) -> list[str]:
    """Every value of the header *name* in one recorded request, in order."""
    return [value for header, value in headers if header == name]


def _guess(value: int) -> bytes:
    return tool_call_body("guess", {"value": value}, 500)


def _origin(url: str) -> str:
    """The scripted server's origin: ``serve()`` yields ``.../v1``, the proxy forwards the path."""
    return url.removesuffix("/v1")


def _env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = os.environ.copy() | {"KISS_HOME": str(home)}
    env.pop("KISS_SORCAR_LOCAL", None)
    return env


def _run(
    tmp_path: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], env=env or _env(tmp_path), cwd=tmp_path,
        capture_output=True, text=True, timeout=300, check=False,
    )


def test_every_recorded_system_message_is_the_prompt_byte_for_byte(tmp_path: Path) -> None:
    record = tmp_path / "wire.jsonl"
    with serve([_guess(v) for v in SOLVED]) as (url, upstream_requests):
        proc = _run(
            tmp_path, EPISODE, "-m", MODEL, "--upstream", _origin(url), "--record", str(record)
        )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[0].startswith(f"episode {EPISODE!r} on {MODEL}: http://127.0.0.1:")
    assert lines[0].endswith(f"/v1 -> {_origin(url)}")
    assert "run: exit 0, success True, 4 steps," in lines[1]
    expected = PROMPT.encode("utf-8")
    assert f"SYSTEM_PROMPT: {len(expected)} bytes sha256 " in proc.stdout
    assert sum(line.startswith("ok   request") for line in lines) == 4
    assert lines[-1] == "4 requests recorded, 4 system messages equal SYSTEM_PROMPT byte for byte"
    # The record is what the proxy forwarded: the upstream saw exactly these bodies.
    recorded = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
    assert [request["path"] for request in recorded] == ["/v1/chat/completions"] * 4
    assert [request["body"] for request in recorded] == upstream_requests
    assert all(system_message(request["body"]) == PROMPT for request in recorded)


def test_the_proxy_forwards_the_client_headers_to_another_upstream(tmp_path: Path) -> None:
    # The openai SDK derives OpenAI-Organization / OpenAI-Project from these variables; a
    # proxy that kept only a few named headers would drop them (review finding, 2026-10-31).
    HeaderRecordingHandler.headers_seen.clear()
    env = _env(tmp_path) | {"OPENAI_ORG_ID": "org-wire-check", "OPENAI_PROJECT_ID": "proj-wire"}
    with serve([_guess(v) for v in SOLVED], HeaderRecordingHandler) as (url, _):
        proc = _run(tmp_path, EPISODE, "-m", MODEL, "--upstream", _origin(url), env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    seen = HeaderRecordingHandler.headers_seen
    assert len(seen) == 4
    for headers in seen:
        assert _values(headers, "openai-organization") == ["org-wire-check"]
        assert _values(headers, "openai-project") == ["proj-wire"]
        assert _values(headers, "content-type") == ["application/json"]
        assert _values(headers, "user-agent")[0].startswith("OpenAI/Python")
        # What the proxy sets itself: the upstream's host, an identity body, and (for an
        # upstream other than OpenAI) the client's own bearer token, the SEA's keyless one.
        assert _values(headers, "host") == [_origin(url).removeprefix("http://")]
        assert _values(headers, "accept-encoding") == ["identity"]
        assert _values(headers, "authorization") == ["Bearer no-key"]


def test_the_upstream_token_replaces_the_client_token_whatever_its_spelling() -> None:
    # The OpenAI path of the proxy (a configured bearer token) driven in process, so that no
    # key and no paid request are needed: the client's Authorization header is replaced, not
    # duplicated, however the client spells it (review finding, 2026-10-31).
    HeaderRecordingHandler.headers_seen.clear()
    with serve([text_body("ok")], HeaderRecordingHandler) as (url, upstream_requests):
        proxy = RECORDING_PROXY(_origin(url), "Bearer real-key")
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        try:
            for spelling in ("Authorization", "authorization"):
                client = http.client.HTTPConnection("127.0.0.1", proxy.server_port, timeout=30)
                client.request(
                    "POST", "/v1/chat/completions", b'{"messages": []}',
                    {spelling: "Bearer no-key", "Accept-Encoding": "gzip", "X-Trace": spelling},
                )
                response = client.getresponse()
                assert (response.status, json.loads(response.read())) == (
                    200, json.loads(text_body("ok"))
                ), "the scripted answer came back through the proxy"
                client.close()
        finally:
            proxy.shutdown()
            proxy.server_close()
            thread.join(timeout=30)
    assert [request["body"] for request in proxy.requests] == upstream_requests
    assert upstream_requests == [{"messages": []}] * 2
    seen = HeaderRecordingHandler.headers_seen
    assert [_values(headers, "x-trace") for headers in seen] == [
        ["Authorization"], ["authorization"]
    ]
    for headers in seen:
        assert _values(headers, "authorization") == ["Bearer real-key"]
        assert _values(headers, "accept-encoding") == ["identity"]


def test_a_prompt_that_differs_by_one_byte_fails_the_check(tmp_path: Path) -> None:
    # A copy of the SEA whose system_prompt() sends one byte more than its SYSTEM_PROMPT.
    source = SEA_PATH.read_text(encoding="utf-8")
    assert source.count("return SYSTEM_PROMPT\n") == 1
    sea_dir = tmp_path / "rl_episode"
    sea_dir.mkdir()
    sea_copy = sea_dir / "rl_episode_sea.py"
    sea_copy.write_text(
        source.replace("return SYSTEM_PROMPT\n", "return SYSTEM_PROMPT + ' '\n"), encoding="utf-8"
    )
    with serve([_guess(v) for v in SOLVED]) as (url, _):
        proc = _run(
            tmp_path, EPISODE, "-m", MODEL, "--upstream", _origin(url), "--sea", str(sea_copy)
        )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    offset = len(PROMPT.encode("utf-8"))
    assert "run: exit 0, success True, 4 steps," in proc.stdout, "the episode itself succeeded"
    assert proc.stdout.count(f"system message {offset + 1} bytes sha256 ") == 4
    assert proc.stdout.count(
        f"!= SYSTEM_PROMPT: first difference at byte {offset}: wire b' ' vs prompt b''"
    ) == 4
    assert proc.stdout.splitlines()[-1] == (
        "4 requests recorded, 0 system messages equal SYSTEM_PROMPT byte for byte: CHECK FAILED"
    )


def test_a_failed_episode_fails_the_check_even_when_every_prompt_matched(tmp_path: Path) -> None:
    with serve([_guess(1)]) as (url, _):  # the last body repeats: never right, the turn cap ends it
        proc = _run(tmp_path, EPISODE, "-m", MODEL, "--upstream", _origin(url))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "run: exit 1, success False, 10 steps," in proc.stdout
    assert proc.stdout.count("== SYSTEM_PROMPT") == 10
    assert proc.stdout.splitlines()[-1] == (
        "10 requests recorded, 10 system messages equal SYSTEM_PROMPT byte for byte: CHECK FAILED"
    )


def test_without_a_key_the_openai_default_stops_before_any_traffic(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.pop("OPENAI_API_KEY", None)
    proc = _run(tmp_path, EPISODE, env=env)
    assert proc.returncode == 2
    assert "OPENAI_API_KEY is not set" in proc.stderr
    assert proc.stdout == ""
