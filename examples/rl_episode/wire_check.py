#!/usr/bin/env python
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Check on the wire that a live model reads the episode's ``SYSTEM_PROMPT`` byte for byte.

    OPENAI_API_KEY=... uv run python examples/rl_episode/wire_check.py [episode] [-m MODEL]
                                     [--upstream ORIGIN] [--sea FILE] [--record FILE]

A recording proxy in this process stores every request body and forwards
the request unchanged to the upstream API: ``https://api.openai.com``
unless ``--upstream`` names another OpenAI-compatible origin, such as a
trainer's policy server.  For OpenAI the real ``OPENAI_API_KEY`` replaces
the keyless token the SEA sends to a local URL; another upstream gets the
client's own ``Authorization`` header.  The episode runs exactly as a
trainer runs it: ``scripts/run_sea.py --json`` as a subprocess with
``RL_POLICY_URL`` pointing at the proxy and ``RL_POLICY_MODEL`` naming the
model.  Afterwards every recorded request must carry exactly one system
message whose UTF-8 bytes equal those of the SEA file's ``SYSTEM_PROMPT``;
the lengths and SHA-256 digests of both are printed and, on a mismatch,
the offset of the first differing byte.  Exit status 0 only when the
episode succeeded and every request passed.  ``--record`` writes the
recorded requests as JSON lines for inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from kiss.agents.sorcar.sea_settings import execute_python_file

HERE = Path(__file__).resolve().parent
RUN_SEA = HERE.parents[1] / "scripts" / "run_sea.py"
OPENAI = "https://api.openai.com"
REQUEST_HEADERS_REPLACED = {"host", "connection", "content-length", "transfer-encoding"}
"""Request headers ``http.client`` sets itself; with the proxy's own overrides (identity encoding,
the upstream's bearer token) they replace the client's case-insensitively, every other request
header is forwarded as the client sent it."""
RESPONSE_HEADERS_DROPPED = {
    "connection", "content-encoding", "content-length", "transfer-encoding",
}
"""Response headers the proxy owns: the client reads an identity body until the connection closes
(HTTP/1.0)."""


class RecordingProxy(ThreadingHTTPServer):
    """A local HTTP server that records each POST body and forwards the request to *upstream*.

    Args:
        upstream: The origin (``scheme://host[:port]``) that answers; the
            request path is forwarded unchanged.
        authorization: The ``Authorization`` header value sent upstream in
            place of the client's, or ``""`` to pass the client's through.
    """

    def __init__(self, upstream: str, authorization: str = "") -> None:
        super().__init__(("127.0.0.1", 0), _Forwarder)
        self.upstream = urlsplit(upstream)
        self.authorization = authorization
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    @property
    def url(self) -> str:
        """The OpenAI-compatible API root a client points at (``RL_POLICY_URL``)."""
        return f"http://127.0.0.1:{self.server_port}/v1"


class _Forwarder(BaseHTTPRequestHandler):
    """Record one POST, forward it and relay the answer as it streams in."""

    def do_POST(self) -> None:  # noqa: N802
        proxy = cast(RecordingProxy, self.server)
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        with proxy.lock:
            proxy.requests.append({"path": self.path, "body": json.loads(body)})
        overrides = {"Accept-Encoding": "identity"}
        if proxy.authorization:
            overrides["Authorization"] = proxy.authorization
        replaced = REQUEST_HEADERS_REPLACED | {name.lower() for name in overrides}
        headers = {
            name: value for name, value in self.headers.items() if name.lower() not in replaced
        } | overrides
        secure = proxy.upstream.scheme == "https"
        connect = http.client.HTTPSConnection if secure else http.client.HTTPConnection
        connection = connect(proxy.upstream.netloc, timeout=600)
        connection.request("POST", self.path, body, headers)
        response = connection.getresponse()
        self.send_response(response.status)
        for name, value in response.getheaders():
            if name.lower() not in RESPONSE_HEADERS_DROPPED:
                self.send_header(name, value)
        self.end_headers()
        while chunk := response.read1(4096):
            self.wfile.write(chunk)
            self.wfile.flush()
        connection.close()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the access log."""


def digest(data: bytes) -> str:
    """Return the first 16 hex digits of the SHA-256 of *data*."""
    return hashlib.sha256(data).hexdigest()[:16]


def compare(index: int, request: dict[str, Any], expected: bytes) -> str:
    """Return the report line of one recorded *request*.

    Args:
        index: The request's 1-based position in the episode.
        request: ``{"path": ..., "body": <chat-completions request>}``.
        expected: The UTF-8 bytes of the SEA's ``SYSTEM_PROMPT``.

    Returns:
        A line starting with ``ok`` when the request carries exactly one
        system message equal to *expected* byte for byte, else one
        starting with ``FAIL`` that says why (count, type, or the offset
        and a window of the first difference).
    """
    contents = [m["content"] for m in request["body"]["messages"] if m["role"] == "system"]
    if len(contents) != 1 or not isinstance(contents[0], str):
        kinds = [type(c).__name__ for c in contents]
        return f"FAIL request {index}: expected one system message string, found {kinds}"
    got = contents[0].encode("utf-8")
    label = f"request {index} POST {request['path']}: system message {len(got)} bytes"
    if got == expected:
        return f"ok   {label} sha256 {digest(got)} == SYSTEM_PROMPT"
    pairs = zip(got, expected, strict=False)
    offset = next((i for i, (a, b) in enumerate(pairs) if a != b), min(len(got), len(expected)))
    return (
        f"FAIL {label} sha256 {digest(got)} != SYSTEM_PROMPT: first difference at byte {offset}: "
        f"wire {got[offset:offset + 40]!r} vs prompt {expected[offset:offset + 40]!r}"
    )


def run_episode(
    sea: dict[str, Any], sea_path: Path, episode: str, model: str, policy_url: str
) -> subprocess.CompletedProcess[str]:
    """Run one *episode* of the SEA at *sea_path* headless, with the policy at *policy_url*.

    Args:
        sea: The SEA file's namespace (its ``POLICY_URL_ENV`` / ``POLICY_MODEL_ENV`` names).
        sea_path: The SEA file ``scripts/run_sea.py`` loads.
        episode: The task text; it seeds the secret.
        model: The model name the policy server answers for.
        policy_url: The OpenAI-compatible API root (the proxy's :attr:`RecordingProxy.url`).

    Returns:
        The completed ``scripts/run_sea.py --json --no-trajectories`` process.
    """
    env = os.environ | {sea["POLICY_URL_ENV"]: policy_url, sea["POLICY_MODEL_ENV"]: model}
    command = [sys.executable, str(RUN_SEA), str(sea_path), episode, "--json", "--no-trajectories"]
    return subprocess.run(command, env=env, capture_output=True, text=True, check=False)


def main(argv: list[str] | None = None) -> int:
    """Run the proxied episode and the byte-for-byte check; return the exit status."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "episode", nargs="?", default="episode-3", help="episode id (seeds the secret)"
    )
    parser.add_argument("-m", "--model", default="gpt-5-mini", help="the model (RL_POLICY_MODEL)")
    parser.add_argument(
        "--upstream", default=OPENAI,
        help="origin (scheme://host[:port]) answering the forwarded requests; default %(default)s",
    )
    parser.add_argument(
        "--sea", type=Path, default=HERE / "rl_episode_sea.py", help="the SEA file to run"
    )
    parser.add_argument(
        "--record", type=Path, help="write the recorded requests to this JSON-lines file"
    )
    args = parser.parse_args(argv)

    sea = execute_python_file(args.sea)
    expected = str(sea["SYSTEM_PROMPT"]).encode("utf-8")
    authorization = ""
    if urlsplit(args.upstream).hostname == urlsplit(OPENAI).hostname:
        if not os.environ.get("OPENAI_API_KEY"):
            parser.error("OPENAI_API_KEY is not set: the proxy cannot sign requests to OpenAI")
        authorization = f"Bearer {os.environ['OPENAI_API_KEY']}"
    proxy = RecordingProxy(args.upstream, authorization)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    print(f"episode {args.episode!r} on {args.model}: {proxy.url} -> {args.upstream}", flush=True)
    try:
        proc = run_episode(sea, args.sea, args.episode, args.model, proxy.url)
    finally:
        proxy.shutdown()
        proxy.server_close()
        thread.join(timeout=30)
    if args.record:
        lines = [json.dumps(request) + "\n" for request in proxy.requests]
        args.record.write_text("".join(lines), encoding="utf-8")

    try:
        outcome = json.loads(proc.stdout)
    except ValueError:
        outcome = {}
    sys.stderr.write(proc.stderr)
    print(
        f"run: exit {proc.returncode}, success {outcome.get('success')}, "
        f"{outcome.get('total_steps')} steps, ${outcome.get('budget_used') or 0:.4f}, "
        f"error {outcome.get('error', '')!r}"
    )
    print(f"result: {outcome.get('result', '')}")
    print(f"SYSTEM_PROMPT: {len(expected)} bytes sha256 {digest(expected)}")
    reports = [compare(i, request, expected) for i, request in enumerate(proxy.requests, 1)]
    print("\n".join(reports))
    equal = sum(report.startswith("ok") for report in reports)
    passed = proc.returncode == 0 and len(reports) > 0 and equal == len(reports)
    verdict = "" if passed else ": CHECK FAILED"
    print(
        f"{len(reports)} requests recorded, {equal} system messages equal SYSTEM_PROMPT "
        f"byte for byte{verdict}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
