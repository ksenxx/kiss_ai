# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the redaction of the key store's secrets in tool output.

Every test has a key store of its own (``KISS_HOME`` points at a temporary
directory) that holds invented values.  A real ``SorcarAgent`` runs against
the scripted Anthropic stand-in and calls the real ``Bash`` tool, real
``UsefulTools`` and ``DockerManager`` run real commands, and the reader reads
the real store file.  Nothing is mocked.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import docker
import pytest

from kiss.agents.sorcar.docker_manager import ATTACH_PREFIX, DockerManager
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.useful_tools import _READ_CHARS, UsefulTools
from kiss.core import secret_redaction
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.printer import Printer
from kiss.core.secret_redaction import StreamRedactor, redact_secrets
from kiss.core.tool_verdict import ALLOW, Verdict, refuse
from kiss.core.user_config import api_keys_env_path
from kiss.tests.agents.sorcar.docker_test_containers import IMAGE, is_docker_available
from kiss.tests.conftest import posix_only
from kiss.tests.core import local_anthropic_server as anthropic_server

pytestmark = posix_only("the commands are bash")

KEY = "sk-ant-fake-0123456789abcdef"
LATER = "later-fake-token-0123"
STORE = f"export ANTHROPIC_API_KEY={KEY}\nexport AWS_REGION=us-east-1\n"
REDACTED_STORE = "export ANTHROPIC_API_KEY=ANTHROPIC_API_KEY\nexport AWS_REGION=us-east-1\n"


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The key store of a home of the test's own: one invented key and the region."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    path = api_keys_env_path()
    path.parent.mkdir()
    path.write_text(STORE)
    return path


def test_every_long_value_is_replaced_whatever_its_variable_is_called(store: Path) -> None:
    """Only the region and a value under 8 characters stay: a name without a word such as
    KEY (the client id, the lower-case webhook) hides its value all the same.  A value two
    names hold takes the first in sorted order, a value that contains another is replaced
    whole, and an unchanged store is not read again."""
    store.write_text(
        STORE
        + "export KISS_GITHUB_CLIENT_ID=Iv1.0123456789abcdef\n"
        "export SHORT_TOKEN=abc1234\n"
        "export B_TOKEN=shared-value-123\n"
        "export A_TOKEN=shared-value-123\n"
        "export PREFIX_KEY=abcd1234\n"
        "export LONGER_KEY=abcd1234-rest-of-it\n"
        "export my_webhook_url='https://hooks.example/T000/B000'\n"
    )
    text = (
        f"key={KEY} region=us-east-1 client=Iv1.0123456789abcdef short=abc1234 "
        "shared=shared-value-123 prefix=abcd1234 longer=abcd1234-rest-of-it "
        "hook=https://hooks.example/T000/B000\n"
    )
    assert redact_secrets(text) == (
        "key=ANTHROPIC_API_KEY region=us-east-1 client=KISS_GITHUB_CLIENT_ID short=abc1234 "
        "shared=A_TOKEN prefix=PREFIX_KEY longer=LONGER_KEY hook=my_webhook_url\n"
    )
    assert secret_redaction._secrets() is secret_redaction._secrets()


def test_a_store_never_read_holds_no_secrets_and_what_was_read_stays_secret(store: Path) -> None:
    """Nothing is replaced or held back while the store is missing or unreadable from the
    start.  Once read, a value stays a secret for the process: a command that began before
    its key was rotated, or before the store went missing, still prints the old one."""
    text = f"key={KEY} sk-ant"
    store.unlink()
    assert redact_secrets(text) == text
    shown: list[str] = []
    StreamRedactor(shown.append).feed(text)
    assert shown == [text]
    store.mkdir()  # found, but not readable as a file
    assert redact_secrets(text) == text
    store.rmdir()
    store.write_text(STORE)
    assert redact_secrets(text) == "key=ANTHROPIC_API_KEY sk-ant"
    # Rotated between two pieces of one command's output, the first of them held back.
    shown.clear()
    stream = StreamRedactor(shown.append)
    stream.feed(f"old={KEY[:9]}")
    store.write_text(f"export ANTHROPIC_API_KEY={LATER}\n")
    stream.feed(f"{KEY[9:]} new={LATER}\n")
    stream.flush()
    assert shown == ["old=", "ANTHROPIC_API_KEY new=ANTHROPIC_API_KEY\n"]
    both = "key=ANTHROPIC_API_KEY sk-ant ANTHROPIC_API_KEY"
    for unreadable in (store.unlink, store.mkdir):
        unreadable()
        assert redact_secrets(f"{text} {LATER}") == both


def _redact_until(done: threading.Event) -> None:
    while not done.is_set():
        redact_secrets("output")


def test_readers_at_once_forget_no_secret(store: Path) -> None:
    """Tool calls that find a new store at the same moment keep what each other has read: a
    reader that was slow to read does not put back only the keys known when it began."""
    keys = [f"rotated-fake-key-{number:04d}" for number in range(30)]
    done = threading.Event()
    readers = [threading.Thread(target=_redact_until, args=(done,)) for _ in range(4)]
    for reader in readers:
        reader.start()
    try:
        for key in keys:  # saved as the settings are: a new file replaces the store
            fresh = store.with_name("fresh")
            fresh.write_text(f"export ANTHROPIC_API_KEY={key}\n")
            fresh.replace(store)
            assert redact_secrets(key) == "ANTHROPIC_API_KEY"
    finally:
        done.set()
        for reader in readers:
            reader.join()
    assert [key for key in keys if redact_secrets(key) != "ANTHROPIC_API_KEY"] == []


def test_a_stream_holds_back_only_what_may_open_a_secret(store: Path) -> None:
    """Pieces are shown as they arrive, except an end that is the beginning of a secret:
    it waits for the next piece or the end of the command.  However the output is cut,
    what is shown adds up to the redaction of the whole."""
    store.write_text(
        STORE
        + "export PREFIX_KEY=abcd1234\n"
        "export LONGER_KEY=abcd1234-rest-of-it\n"
        "export TAIL_SECRET=xyz-ends-in-abcd\n"
    )
    shown: list[str] = []
    stream = StreamRedactor(shown.append)
    for piece in (
        "an ordinary line\n",
        "no newline at the end",
        f" key={KEY[:9]}",  # may open the key
        f"{KEY[9:]} next sk-",  # the key is whole; "sk-" may open it again
        "ip\n",  # it did not
        "p=abcd1234",  # a whole secret that the longer one begins with
        "-rest-of-it q=abcd1234",  # it became the longer one; the next may too
        " t=xyz-ends-in-abcd",  # it did not; a whole secret does not wait for its "abcd"
        " end sk-ant",  # waits for the end of the command
    ):
        stream.feed(piece)
    stream.flush()
    assert shown == [
        "an ordinary line\n",
        "no newline at the end",
        " key=",
        "ANTHROPIC_API_KEY next ",
        "sk-ip\n",
        "p=",
        "LONGER_KEY q=",
        "PREFIX_KEY t=TAIL_SECRET",
        " end ",
        "sk-ant",
    ]
    text = f"a={KEY} b=abcd1234-rest-of-it c=abcd1234 d=xyz-ends-in-abcd1234 sk-"
    for cut in range(len(text) + 1):
        shown.clear()
        stream = StreamRedactor(shown.append)
        stream.feed(text[:cut])
        stream.feed(text[cut:])
        stream.flush()
        assert "".join(shown) == redact_secrets(text), cut
    # A key saved while the command runs is replaced in the output that follows.
    shown.clear()
    stream = StreamRedactor(shown.append)
    stream.feed("before ")
    with store.open("a") as file:
        file.write(f"export LATER_TOKEN={LATER}\n")
    stream.feed(f"{LATER} after")
    stream.flush()
    assert shown == ["before ", "LATER_TOKEN after"]


class _Events(Printer):
    """A real ``Printer`` that keeps every event, as ``history.db`` does."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:  # noqa: A002
        self.events.append({"type": type, "content": content, **kwargs})

    def token_callback(self, token: str) -> None:
        pass

    def reset(self) -> None:
        pass


def save_key() -> str:
    """Add a key to the key store, as a save from the Settings panel does mid-run."""
    with api_keys_env_path().open("a") as file:
        file.write(f"export LATER_TOKEN={LATER}\n")
    return "saved"


def fail_with_keys(shift: int) -> str:
    """Fail with a message longer than a result's cut that is copies of the key.

    Args:
        shift: How many characters stand before the first copy.
    """
    raise ValueError("x" * shift + " ".join([KEY] * 300))


def _refuse_exit_7(name: str, args: dict[str, Any]) -> Verdict:
    """Refuse ``Bash("exit 7")`` with a text that quotes the key, as a gate quotes a verifier."""
    if args.get("command") == "exit 7":
        return refuse(f"REFUSED: the verifier printed {KEY}")
    return ALLOW


def _bash(command: str, call_id: str, **arguments: Any) -> dict[str, Any]:
    arguments.update(command=command, description="run")
    return anthropic_server.tool_use_message("Bash", arguments, 1000, call_id)


def _result_sent(request: dict[str, Any]) -> str:
    """The tool result a request to the provider reports."""
    return str(request["messages"][-1]["content"][0]["content"])


def test_a_secret_a_tool_prints_is_neither_stored_nor_sent(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real agent run.  The events ``history.db`` stores and the requests the provider
    receives name the variable where a tool printed its value: in a result, in live
    output cut in two pieces, for a key saved mid-run, in a refusal that quotes one
    and in the error of a tool that failed, which is cut to its head and tail (two
    shifts, so that one of the cuts falls well inside a copy of the key).
    No command or tool argument holds a value, so no request may hold one anywhere.
    The first command runs as a job whose log is followed (the default timeout is above
    the detach grace), the second on a pipe read in pieces of ``_READ_CHARS``."""
    monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
    work = tmp_path / "work"
    work.mkdir()
    pad = _READ_CHARS - 6
    script = [
        _bash('cat "$KISS_HOME/api_keys.env"', "toolu_1"),
        anthropic_server.tool_use_message("save_key", {}, 1000, "toolu_2"),
        # One line the tool reads in two pieces: the first ends with 6 characters of the key.
        _bash(
            f'. "$KISS_HOME/api_keys.env"; printf "%{pad}s%s\\n" "" "$LATER_TOKEN"',
            "toolu_3",
            timeout_seconds=60,
        ),
        _bash("exit 7", "toolu_4"),
        anthropic_server.tool_use_message("fail_with_keys", {"shift": 0}, 1000, "toolu_5"),
        anthropic_server.tool_use_message("fail_with_keys", {"shift": 10}, 1000, "toolu_6"),
        anthropic_server.tool_use_message(
            "finish", {"success": True, "summary_in_html": "<p>done</p>"}, 1000, "toolu_7"
        ),
    ]
    printer = _Events()
    with anthropic_server.serve(script) as (url, requests):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
        SorcarAgent("redaction").run(
            model_name=anthropic_server.MODEL,
            prompt_template="Print the key store, then finish.",
            tools=[save_key, fail_with_keys],
            tool_call_hook=_refuse_exit_7,
            work_dir=str(work),
            printer=printer,
            web_tools=False,
            use_memory=False,
            max_steps=8,
            max_budget=5.0,
            max_sub_sessions=1,
            verbose=False,
        )
    streamed = [e["content"] for e in printer.events if e["type"] == "bash_stream"]
    results = [e["content"] for e in printer.events if e["type"] == "tool_result"]
    steps = [r for r in requests if r.get("stream")]
    assert "".join(streamed[:-2]) == REDACTED_STORE
    assert streamed[-2:] == [" " * pad, "LATER_TOKEN\n"]
    assert results[0].startswith(REDACTED_STORE)
    assert "LATER_TOKEN\n" in results[2]
    assert "the verifier printed ANTHROPIC_API_KEY" in results[3]
    assert len(steps) == 7
    for failure in results[4:6]:
        assert failure.startswith("Failed to call fail_with_keys")
        assert "ANTHROPIC_API_KEY" in failure
        assert not [at for at in range(len(KEY) - 3) if KEY[at : at + 4] in failure]
    assert _result_sent(steps[1]).startswith(REDACTED_STORE)
    assert "LATER_TOKEN\n" in _result_sent(steps[3])
    assert "the verifier printed ANTHROPIC_API_KEY" in _result_sent(steps[4])
    for text in streamed + results + [json.dumps(step) for step in steps]:
        assert KEY not in text and LATER not in text


def test_output_read_from_a_job_log_is_redacted_across_two_reads(
    store: Path, tmp_path: Path
) -> None:
    """A command run as a job is shown from its growing log: a key the log holds only
    after two reads is shown by name, and the end held back when the command exits is
    shown then.  Unwatched, the command runs the same."""
    pieces: list[str] = []

    def on_piece(piece: str) -> None:
        pieces.append(piece)
        (tmp_path / "release").touch()

    command = (
        f"printf 'k={KEY[:12]}'; while [ ! -e release ]; do sleep 0.01; done; "
        f"printf '{KEY[12:]}\\nend sk-ant'"
    )
    watched = UsefulTools(stream_callback=on_piece, work_dir=str(tmp_path))
    watched.Bash(command, "split key", timeout_seconds=30, on_timeout="background")
    assert pieces == ["k=", "ANTHROPIC_API_KEY\nend ", "sk-ant"]
    unwatched = UsefulTools(work_dir=str(tmp_path))
    out = unwatched.Bash("printf 'k=sk-ant'", "end", timeout_seconds=30, on_timeout="background")
    assert out == "k=sk-ant"


def test_a_result_cut_to_its_budget_keeps_no_part_of_a_secret(store: Path, tmp_path: Path) -> None:
    """A result longer than its budget keeps its head and its tail.  The key is replaced
    before the cut, so neither end holds a part of it, which the agent's pass over the
    finished result would not recognise."""
    command = (
        '. "$KISS_HOME/api_keys.env"; '
        'for i in $(seq 300); do printf "%s " "$ANTHROPIC_API_KEY"; done'
    )
    out = UsefulTools(work_dir=str(tmp_path)).Bash(
        command, "many keys", timeout_seconds=30, max_output_chars=600
    )
    head, marker, tail = re.split(r"(\n\n\.\.\. \[truncated .*\] \.\.\.\n\n)", out)
    assert marker and head and tail and len(out) <= 600
    # A part of the key would bring lower-case letters, digits or hyphens.
    assert set(head + tail) <= set("ANTHROPIC_API_KEY ")


@pytest.mark.slow
@pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running")
def test_container_output_is_redacted_within_each_stream(store: Path) -> None:
    """stdout and stderr arrive interleaved: a key stdout prints around a stderr line is
    shown by name and the end held back is shown when the command ends.  After a Stop
    nothing is shown: neither the output still queued nor the end held back."""
    container = docker.from_env().containers.run(IMAGE, command="sleep infinity", detach=True)
    pieces: list[str] = []
    stop = threading.Event()

    def stop_at(piece: str) -> None:
        pieces.append(piece)
        stop.set()
        time.sleep(0.5)  # the rest of the output queues up meanwhile

    try:
        with DockerManager(f"{ATTACH_PREFIX}{container.id}") as manager:
            manager.stream_callback = pieces.append
            manager.Bash(
                f"printf 'k={KEY[:12]}'; sleep 1; echo warning >&2; sleep 1; "
                f"printf '{KEY[12:]}\\nend sk-ant'",
                "split key",
            )
            assert "".join(pieces) == "k=warning\nANTHROPIC_API_KEY\nend sk-ant"
            pieces.clear()
            manager.stop_event = stop
            manager.stream_callback = stop_at
            manager.Bash("printf 'k=sk-ant'; yes x | head -c 2000000; sleep 30", "stopped")
            assert len(pieces) == 1 and pieces[0].startswith("k="), pieces
    finally:
        container.remove(force=True)
