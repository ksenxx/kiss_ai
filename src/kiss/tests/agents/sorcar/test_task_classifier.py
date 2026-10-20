# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for pre-run task classification.

``kiss.agents.sorcar.task_classifier`` runs one NON-AGENTIC KISSAgent
generation on the run's own model BEFORE a Sorcar agent starts a task.
The structured JSON verdict ``{"is_simple": ...}`` selects the system
prompt (``SYSTEM_LITE.md`` for simple tasks, ``SYSTEM.md`` otherwise).
Worktree isolation is no longer a classifier decision: the agent calls
``create_worktree`` itself before its first tracked-file edit.

The classification tests call real LLMs (a cheap model) — no mocks.
The verdict-cache tests use the real local OpenAI-compatible stand-in
endpoint from ``parallel_agent_harness`` instead.

The wiring tests drive real ``SorcarAgent`` runs inside an isolated
KISS_HOME.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _ClassifierSpend
from kiss.agents.sorcar.task_classifier import (
    _VERDICT_JSON_SCHEMA,
    CLASSIFIER_CACHE_FILENAME,
    CLASSIFIER_CACHE_MAX_ENTRIES,
    CLASSIFIER_MAX_TOKENS,
    CLASSIFIER_STALL_TIMEOUT_SECONDS,
    TaskClassification,
    _cache_key,
    _classifier_model_config,
    _parse_verdict,
    _structured_output_config,
    cached_classification,
    classification_enabled,
    classify_task,
    clear_classification_cache,
    remember_classification,
)
from kiss.core.base import SYSTEM_PROMPT, SYSTEM_PROMPT_LITE
from kiss.server.json_printer import JsonPrinter
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
)

MODEL = "claude-haiku-4-5"

# Real-LLM tests follow the repo's live_api convention: marked, and
# skipped outright when the provider key is absent.
requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; classifier live tests need it",
)
live_api = pytest.mark.live_api

_SIMPLE_TASK = (
    "What is 2 + 2? Answer in text only. This does not involve any "
    "software development, any file edits, or any internet search."
)

_DEV_TASK = (
    "Software development task: edit the file src/utils.py in this "
    "repository to fix the off-by-one bug in the pagination helper, "
    "and update its unit test. This requires editing source files."
)

_DISABLE_ENV = "KISS_DISABLE_TASK_CLASSIFIER"


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """Isolated KISS_HOME + scratch repo, with the LLM classifier enabled.

    The root conftest disables the classifier suite-wide via
    ``KISS_DISABLE_TASK_CLASSIFIER=1``; these are the classifier's own
    tests, so the kill switch is lifted and restored afterwards.  The
    decisions classifier is pinned off (``classify_with_decisions``) so
    every test here exercises the LLM path whatever keys the developer
    has; ``test_task_classifier_decisions.py`` covers the other route.
    """
    saved = os.environ.get(_DISABLE_ENV)
    os.environ[_DISABLE_ENV] = "0"
    isolated = IsolatedKissHome("kiss-task-classifier-")
    isolated.write_config(classify_with_decisions=False)
    # The verdict cache lives in KISS_HOME; drop the in-process mirror
    # so a memo from an earlier test's home never serves this one.
    clear_classification_cache()
    try:
        yield isolated
    finally:
        clear_classification_cache()
        if saved is None:
            os.environ.pop(_DISABLE_ENV, None)
        else:
            os.environ[_DISABLE_ENV] = saved
        isolated.cleanup()


# ---------------------------------------------------------------------------
# classify_task — real LLM
# ---------------------------------------------------------------------------


@live_api
@requires_anthropic
def test_classify_simple_task(env: IsolatedKissHome) -> None:
    """A pure-arithmetic task is simple."""
    outcome = classify_task(task=_SIMPLE_TASK, model_name=MODEL)
    assert outcome.classification is not None
    assert outcome.classification.is_simple is True
    # The attempt's real usage is reported for budget folding, and the
    # non-agentic classification is exactly one generation.
    assert outcome.budget_used > 0.0
    assert outcome.tokens_used > 0
    assert outcome.steps == 1


@live_api
@requires_anthropic
def test_classify_development_task(env: IsolatedKissHome) -> None:
    """A file-editing coding task is not simple."""
    outcome = classify_task(task=_DEV_TASK, model_name=MODEL)
    assert outcome.classification is not None
    assert outcome.classification.is_simple is False


@live_api
@requires_anthropic
def test_classify_truncates_huge_task(env: IsolatedKissHome) -> None:
    """A giant prompt is truncated before classification, bounding the
    call's input cost, and the gist still classifies correctly."""
    huge_task = _DEV_TASK + "\n" + ("filler line about the codebase\n" * 5000)
    assert len(huge_task) > 100_000
    outcome = classify_task(task=huge_task, model_name=MODEL)
    assert outcome.classification is not None
    assert outcome.classification.is_simple is False
    # ~20k chars of task survive; the input can't be the full 150k chars.
    assert outcome.tokens_used < 15_000


@live_api
@requires_anthropic
def test_classify_adaptive_thinking_model_takes_one_attempt(
    env: IsolatedKissHome,
) -> None:
    """``claude-fable-5`` classifies in ONE generation.

    The adaptive-thinking generation rejects ``thinking.type=disabled``
    with HTTP 400, so the former structured attempt always failed and
    every run paid a wasted round trip before the plain fallback.  With
    the plain prompt as the only attempt the verdict costs one step.
    """
    outcome = classify_task(task=_SIMPLE_TASK, model_name="claude-fable-5")
    assert outcome.classification is not None
    assert outcome.classification.is_simple is True
    assert outcome.steps == 1


def test_classify_task_bad_model_fails_soft(env: IsolatedKissHome) -> None:
    """An unknown model yields classification=None, not an exception."""
    outcome = classify_task(
        task="anything", model_name="no-such-model-xyz",
    )
    assert outcome.classification is None
    assert outcome.budget_used == 0.0


# ---------------------------------------------------------------------------
# Verdict cache — real local OpenAI-compatible endpoint, no LLM
# ---------------------------------------------------------------------------


def _verdict_of(entry: dict[str, Any]) -> bool:
    """The ``is_simple`` verdict of one stored cache entry."""
    verdict = entry["is_simple"]
    assert isinstance(verdict, bool)
    return verdict


def _entry(is_simple: bool, age_s: float = 0.0) -> dict[str, Any]:
    """A cache file entry written *age_s* seconds ago."""
    return {"is_simple": is_simple, "ts": time.time() - age_s}


def _verdict_body(is_simple: bool) -> dict[str, Any]:
    """A chat-completions body whose assistant text is the verdict JSON."""
    return {
        "id": "chatcmpl-kiss-classifier",
        "object": "chat.completion",
        "created": 0,
        "model": STANDIN_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"is_simple": is_simple}),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 12,
            "total_tokens": 52,
        },
    }


class _CountingClassifierEndpoint:
    """A stand-in model endpoint that counts the classifier's requests."""

    def __init__(self, is_simple: bool = True) -> None:
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._verdict = is_simple
        self.server = StandInModelServer(self._respond)

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.requests.append(request)
        return _verdict_body(self._verdict)

    @property
    def model_config(self) -> dict[str, Any]:
        return self.server.model_config

    def stop(self) -> None:
        self.server.stop()


@pytest.fixture
def endpoint() -> Iterator[_CountingClassifierEndpoint]:
    """A counting stand-in endpoint, stopped after the test."""
    standin = _CountingClassifierEndpoint()
    try:
        yield standin
    finally:
        standin.stop()


def test_repeated_task_is_served_from_cache(
    env: IsolatedKissHome, endpoint: _CountingClassifierEndpoint,
) -> None:
    """The second classification of the same prompt makes no model call,
    reports zero usage, and the memo survives in KISS_HOME."""
    first = classify_task(
        task="say hello", model_name=STANDIN_MODEL,
        model_config=endpoint.model_config,
    )
    assert first.classification == TaskClassification(is_simple=True)
    assert first.steps == 1
    assert len(endpoint.requests) == 1

    second = classify_task(
        task="say hello", model_name=STANDIN_MODEL,
        model_config=endpoint.model_config,
    )
    assert second.classification == first.classification
    assert second.steps == 0
    assert second.tokens_used == 0
    assert second.budget_used == 0.0
    assert len(endpoint.requests) == 1, "a cache hit must not call the model"

    cache_file = env.kiss_home / CLASSIFIER_CACHE_FILENAME
    assert cache_file.is_file()
    stored = json.loads(cache_file.read_text(encoding="utf-8"))
    assert [_verdict_of(v) for v in stored.values()] == [True]
    (entry,) = stored.values()
    assert abs(entry["ts"] - time.time()) < 60


def test_cache_misses_on_different_task_model_or_endpoint(
    env: IsolatedKissHome, endpoint: _CountingClassifierEndpoint,
) -> None:
    """Only the exact (model, endpoint, task) triple hits."""
    config = endpoint.model_config
    classify_task(task="alpha", model_name=STANDIN_MODEL, model_config=config)
    assert len(endpoint.requests) == 1
    # Different task text.
    classify_task(task="alpha ", model_name=STANDIN_MODEL, model_config=config)
    assert len(endpoint.requests) == 2
    # Same task, different endpoint (a second stand-in on another port).
    other = _CountingClassifierEndpoint(is_simple=False)
    try:
        outcome = classify_task(
            task="alpha", model_name=STANDIN_MODEL,
            model_config=other.model_config,
        )
        assert len(other.requests) == 1
        assert outcome.classification == TaskClassification(is_simple=False)
    finally:
        other.stop()
    # The original triple still hits its own memo, not the other's.
    hit = classify_task(task="alpha", model_name=STANDIN_MODEL, model_config=config)
    assert hit.steps == 0
    assert hit.classification == TaskClassification(is_simple=True)
    assert len(endpoint.requests) == 2
    # Different model name misses the first memo too.
    assert cached_classification("alpha", "gpt-4.1-mini", config) is None


def test_cache_persists_across_process_state_reset(
    env: IsolatedKissHome, endpoint: _CountingClassifierEndpoint,
) -> None:
    """A memo written by one process is read back by the next.

    Dropping the in-process mirror (what a daemon restart does) and
    classifying again must reload the memo from the file.
    """
    classify_task(
        task="restart me", model_name=STANDIN_MODEL,
        model_config=endpoint.model_config,
    )
    assert len(endpoint.requests) == 1
    # Forget the in-memory mirror only; the file stays.
    from kiss.agents.sorcar import task_classifier as tc_module

    with tc_module._cache_lock:
        tc_module._cache = None
        tc_module._cache_loaded_from = None
    again = classify_task(
        task="restart me", model_name=STANDIN_MODEL,
        model_config=endpoint.model_config,
    )
    assert again.steps == 0
    assert len(endpoint.requests) == 1


def test_unparseable_verdict_is_not_cached_and_falls_back(
    env: IsolatedKissHome,
) -> None:
    """A model that answers prose gets the plain retry, yields no verdict,
    and leaves no memo (only genuine verdicts are memoised)."""
    def prose(_request: dict[str, Any]) -> dict[str, Any]:
        body = _verdict_body(True)
        body["choices"][0]["message"]["content"] = "I would rather not say."
        return body

    server = StandInModelServer(prose)
    try:
        outcome = classify_task(
            task="mumble", model_name=STANDIN_MODEL, model_config=server.model_config,
        )
        assert outcome.classification is None
        # OpenAI-compatible: structured attempt, then the plain fallback.
        assert outcome.steps == 2
        assert cached_classification("mumble", STANDIN_MODEL, server.model_config) is None
        assert not (env.kiss_home / CLASSIFIER_CACHE_FILENAME).exists()
    finally:
        server.stop()


def test_failed_classification_is_not_cached(env: IsolatedKissHome) -> None:
    """Only real verdicts are memoised; a failure leaves no memo."""
    outcome = classify_task(task="anything", model_name="no-such-model-xyz")
    assert outcome.classification is None
    assert cached_classification("anything", "no-such-model-xyz") is None
    assert not (env.kiss_home / CLASSIFIER_CACHE_FILENAME).exists()


def test_malformed_cache_file_is_ignored_and_rewritten(
    env: IsolatedKissHome, endpoint: _CountingClassifierEndpoint,
) -> None:
    """Garbage, non-object JSON and non-boolean entries never poison a
    verdict; the next memo rewrites the file cleanly."""
    cache_file = env.kiss_home / CLASSIFIER_CACHE_FILENAME
    cache_file.write_text("not json at all", encoding="utf-8")
    assert cached_classification("x", STANDIN_MODEL) is None

    clear_classification_cache()
    cache_file.write_text(json.dumps(["a", "list"]), encoding="utf-8")
    assert cached_classification("x", STANDIN_MODEL) is None

    clear_classification_cache()
    cache_file.write_text(
        json.dumps({
            "junk-key": {**_entry(True), "is_simple": "yes"},
            "other-junk": "not a dict",
            "int-not-bool": {**_entry(True), "is_simple": 1},
            "missing-key": {"ts": time.time()},
            "no-timestamp": {"is_simple": True},
            "bad-timestamp": {"is_simple": True, "ts": "now"},
            "kept": _entry(False),
        }),
        encoding="utf-8",
    )
    classify_task(
        task="clean slate", model_name=STANDIN_MODEL,
        model_config=endpoint.model_config,
    )
    stored = json.loads(cache_file.read_text(encoding="utf-8"))
    assert list(stored) == [
        "kept", _cache_key("clean slate", STANDIN_MODEL, endpoint.model_config),
    ], "invalid entries are dropped on load, valid ones and the new memo are written"


def test_cache_is_bounded_oldest_first(
    env: IsolatedKissHome, endpoint: _CountingClassifierEndpoint,
) -> None:
    """The file never grows past the cap; the oldest memo goes first."""
    cache_file = env.kiss_home / CLASSIFIER_CACHE_FILENAME
    full = {
        f"key-{i:05d}": _entry(True)
        for i in range(CLASSIFIER_CACHE_MAX_ENTRIES)
    }
    cache_file.write_text(json.dumps(full), encoding="utf-8")
    classify_task(
        task="one more", model_name=STANDIN_MODEL,
        model_config=endpoint.model_config,
    )
    stored = json.loads(cache_file.read_text(encoding="utf-8"))
    assert len(stored) == CLASSIFIER_CACHE_MAX_ENTRIES
    assert "key-00000" not in stored
    assert "key-00001" in stored
    assert list(stored)[-1] == _cache_key(
        "one more", STANDIN_MODEL, endpoint.model_config,
    )


def test_remembering_refreshes_an_existing_memo(env: IsolatedKissHome) -> None:
    """Re-remembering a key moves it to the newest slot and updates it."""
    remember_classification(
        "t", "m", None, TaskClassification(is_simple=True),
    )
    remember_classification(
        "u", "m", None, TaskClassification(is_simple=True),
    )
    remember_classification(
        "t", "m", None, TaskClassification(is_simple=False),
    )
    stored = json.loads(
        (env.kiss_home / CLASSIFIER_CACHE_FILENAME).read_text(encoding="utf-8"),
    )
    assert list(stored) == [_cache_key("u", "m", None), _cache_key("t", "m", None)]
    assert cached_classification("t", "m") == TaskClassification(is_simple=False)


def test_cache_key_changes_with_prompt_prefix() -> None:
    """The key embeds the classifier prompt so a criteria change
    retires every old memo; the model and the WHOLE model config take
    part too, since credentials, headers and provider routing can send
    the same model name to a different backend."""
    from kiss.agents.sorcar import task_classifier as tc_module

    base = _cache_key("t", "m", None)
    assert base == _cache_key("t", "m", {})
    assert base != _cache_key("t", "m", {"api_key": "other-account"})
    assert base != _cache_key("t", "m", {"extra_headers": {"X-Route": "b"}})
    assert base != _cache_key("t", "m", {"base_url": "http://x/v1"})
    assert base != _cache_key("t", "n", None)
    assert base != _cache_key("t ", "m", None)
    # Key order inside the config is irrelevant.
    assert _cache_key("t", "m", {"a": 1, "b": 2}) == _cache_key("t", "m", {"b": 2, "a": 1})
    expected = hashlib.sha256(
        tc_module._CLASSIFIER_PROMPT_PREFIX.encode("utf-8") + b"\0"
        + b"m" + b"\0" + b"{}" + b"\0" + b"t",
    ).hexdigest()
    assert base == expected
    # The file never holds the config values themselves, only digests.
    remember_classification(
        "the task text itself", "m", {"api_key": "sk-secret-value"},
        TaskClassification(is_simple=True),
    )
    text = tc_module._cache_path().read_text(encoding="utf-8")
    assert "sk-secret-value" not in text
    assert "the task text itself" not in text


def test_cache_key_covers_api_key_and_header_routing(
    env: IsolatedKissHome, endpoint: _CountingClassifierEndpoint,
) -> None:
    """Same model, same endpoint, different credentials or headers: the
    second request must reach the model and may get its own verdict."""
    config_a = endpoint.model_config
    first = classify_task(task="route", model_name=STANDIN_MODEL, model_config=config_a)
    assert first.classification == TaskClassification(is_simple=True)
    other = _CountingClassifierEndpoint(is_simple=False)
    try:
        config_b = {**other.model_config, "api_key": "kiss-other-account"}
        second = classify_task(task="route", model_name=STANDIN_MODEL, model_config=config_b)
        assert len(other.requests) == 1
        assert second.classification == TaskClassification(is_simple=False)
        config_c = {**config_a, "extra_headers": {"X-Route": "b"}}
        third = classify_task(task="route", model_name=STANDIN_MODEL, model_config=config_c)
        assert third.steps == 1
        assert len(endpoint.requests) == 2
    finally:
        other.stop()


def test_expired_memo_is_re_asked(
    env: IsolatedKissHome, endpoint: _CountingClassifierEndpoint,
) -> None:
    """A memo older than the TTL is ignored and replaced by a fresh call."""
    from kiss.agents.sorcar.task_classifier import CLASSIFIER_CACHE_TTL_SECONDS

    config = endpoint.model_config
    key = _cache_key("old news", STANDIN_MODEL, config)
    cache_file = env.kiss_home / CLASSIFIER_CACHE_FILENAME
    cache_file.write_text(
        json.dumps({
            key: _entry(False, age_s=CLASSIFIER_CACHE_TTL_SECONDS + 60),
            "fresh-other": _entry(True, age_s=CLASSIFIER_CACHE_TTL_SECONDS - 60),
        }),
        encoding="utf-8",
    )
    assert cached_classification("old news", STANDIN_MODEL, config) is None
    outcome = classify_task(task="old news", model_name=STANDIN_MODEL, model_config=config)
    assert outcome.steps == 1
    assert outcome.classification == TaskClassification(is_simple=True)
    stored = json.loads(cache_file.read_text(encoding="utf-8"))
    assert _verdict_of(stored[key]) is True, "the stale memo was replaced"
    assert "fresh-other" in stored, "an unexpired memo survives the rewrite"
    # A memo that expires while the mirror is loaded is not served either.
    from kiss.agents.sorcar import task_classifier as tc_module

    with tc_module._cache_lock:
        assert tc_module._cache is not None
        tc_module._cache[key]["ts"] = time.time() - CLASSIFIER_CACHE_TTL_SECONDS - 1
    assert cached_classification("old news", STANDIN_MODEL, config) is None


def test_memos_from_another_process_are_merged_on_write(
    env: IsolatedKissHome, endpoint: _CountingClassifierEndpoint,
) -> None:
    """A memo another process wrote after this process loaded the file
    survives this process's next write; for a shared key the newer
    write wins."""
    config = endpoint.model_config
    classify_task(task="mine", model_name=STANDIN_MODEL, model_config=config)
    cache_file = env.kiss_home / CLASSIFIER_CACHE_FILENAME
    # "Another process" rewrites the file: adds one memo and a NEWER
    # verdict for our key.
    mine_key = _cache_key("mine", STANDIN_MODEL, config)
    theirs_key = _cache_key("theirs", STANDIN_MODEL, config)
    cache_file.write_text(
        json.dumps({
            theirs_key: _entry(False),
            mine_key: {**_entry(False), "ts": time.time() + 5},
        }),
        encoding="utf-8",
    )
    remember_classification(
        "third", STANDIN_MODEL, config,
        TaskClassification(is_simple=True),
    )
    stored = json.loads(cache_file.read_text(encoding="utf-8"))
    assert _verdict_of(stored[theirs_key]) is False
    assert _verdict_of(stored[mine_key]) is False, "newer on-disk memo wins"
    assert _verdict_of(stored[_cache_key("third", STANDIN_MODEL, config)]) is True
    assert cached_classification("theirs", STANDIN_MODEL, config) == TaskClassification(
        is_simple=False,
    )


def test_clear_cache_removes_file_and_memory(env: IsolatedKissHome) -> None:
    """``clear_classification_cache`` forgets everything, twice is fine."""
    remember_classification(
        "t", "m", None, TaskClassification(is_simple=True),
    )
    cache_file = env.kiss_home / CLASSIFIER_CACHE_FILENAME
    assert cache_file.is_file()
    clear_classification_cache()
    assert not cache_file.exists()
    assert cached_classification("t", "m") is None
    clear_classification_cache()


def test_unwritable_cache_dir_keeps_verdict(env: IsolatedKissHome) -> None:
    """A cache that cannot be written only loses the memo.

    The cache file's parent is replaced by a regular FILE, so the
    ``mkdir``/``write_text`` inside ``remember_classification`` fails
    with a real ``OSError``; the verdict itself is unaffected and the
    in-memory mirror still serves the next lookup in this process.
    """
    from kiss.agents.sorcar import task_classifier as tc_module

    blocked_home = env.kiss_home / "blocked"
    blocked_home.write_text("i am a file, not a directory", encoding="utf-8")
    saved = os.environ["KISS_HOME"]
    os.environ["KISS_HOME"] = str(blocked_home)
    try:
        clear_classification_cache()
        assert tc_module._cache_path().parent == blocked_home
        remember_classification(
            "t", "m", None,
            TaskClassification(is_simple=True),
        )
        assert cached_classification("t", "m") == TaskClassification(is_simple=True)
        assert not tc_module._cache_path().exists()
    finally:
        os.environ["KISS_HOME"] = saved
        clear_classification_cache()


# ---------------------------------------------------------------------------
# Structured-output verdict parsing — no LLM
# ---------------------------------------------------------------------------


def test_parse_verdict_accepts_bare_json() -> None:
    """The demanded schema — one bare JSON object — parses directly."""
    assert _parse_verdict('{"is_simple": true}') == TaskClassification(is_simple=True)
    # String-typed booleans from lax models are interpreted by content.
    assert _parse_verdict('{"is_simple": "yes"}') == TaskClassification(is_simple=True)
    # Extra keys a lax model adds are ignored.
    assert _parse_verdict(
        '{"is_simple": false, "is_development": true}'
    ) == TaskClassification(is_simple=False)


def test_parse_verdict_extracts_json_from_fences_and_prose() -> None:
    """Code fences and surrounding prose do not defeat extraction."""
    fenced = (
        "```json\n"
        '{"is_simple": false}\n'
        "```"
    )
    assert _parse_verdict(fenced) == TaskClassification(is_simple=False)
    prose = (
        "Here is my classification of the task:\n"
        '{"is_simple": true}\n'
        "Let me know if you need anything else."
    )
    assert _parse_verdict(prose) == TaskClassification(is_simple=True)
    # A leading brace that is not JSON does not stop the scan.
    noisy = (
        "{broken json} then the real verdict "
        '{"is_simple": false}'
    )
    assert _parse_verdict(noisy) == TaskClassification(is_simple=False)
    # An earlier valid JSON object without the verdict keys is skipped.
    decoy = (
        '{"note": "thinking"} '
        '{"is_simple": true}'
    )
    assert _parse_verdict(decoy) == TaskClassification(is_simple=True)


def test_parse_verdict_rejects_junk() -> None:
    """Prose, non-object JSON, and partial objects all yield None."""
    assert _parse_verdict("I think this task is simple.") is None
    assert _parse_verdict("[1, 2]") is None
    assert _parse_verdict('{"is_development": true}') is None
    assert _parse_verdict("") is None
    assert _parse_verdict(None) is None  # type: ignore[arg-type]


def test_parse_verdict_rejects_non_boolean_values() -> None:
    """Truthy junk must not silently select the lite prompt."""
    assert _parse_verdict('{"is_simple": []}') is None
    assert _parse_verdict('{"is_simple": {"x": 1}}') is None
    assert _parse_verdict('{"is_simple": 2}') is None
    assert _parse_verdict('{"is_simple": null}') is None
    assert _parse_verdict('{"is_simple": 1.0}') is None
    assert _parse_verdict('{"is_simple": "maybe"}') is None
    # A later well-formed verdict still wins over an earlier junk one.
    assert _parse_verdict(
        '{"is_simple": null} {"is_simple": false}'
    ) == TaskClassification(is_simple=False)


def test_classifier_model_config_enforces_output_cap() -> None:
    """The classifier's output cap wins over every caller spelling."""
    config = _classifier_model_config(None)
    assert config["max_tokens"] == CLASSIFIER_MAX_TOKENS
    assert config["stream_stall_timeout"] == CLASSIFIER_STALL_TIMEOUT_SECONDS
    original = {
        "extra_headers": {"x": "y"},
        "max_tokens": 64000,
        "max_completion_tokens": 64000,
        "max_output_tokens": 64000,
        "stream_stall_timeout": 3600.0,
        "thinking": {"type": "enabled", "budget_tokens": 10000},
    }
    config = _classifier_model_config(original)
    assert config["max_tokens"] == CLASSIFIER_MAX_TOKENS
    assert "max_completion_tokens" not in config
    assert "max_output_tokens" not in config
    assert config["extra_headers"] == {"x": "y"}
    # The classifier's stall timeout wins over the caller's patience
    # budget, which sizes the main run.
    assert config["stream_stall_timeout"] == CLASSIFIER_STALL_TIMEOUT_SECONDS
    # The main run's thinking budget cannot fit inside the classifier's
    # output cap (Anthropic rejects budget_tokens >= max_tokens), so the
    # caller's thinking configuration is dropped.
    assert "thinking" not in config
    # The caller's dict is untouched.
    assert original["max_tokens"] == 64000
    assert original["max_output_tokens"] == 64000
    assert original["thinking"] == {"type": "enabled", "budget_tokens": 10000}


def test_structured_output_config_per_provider() -> None:
    """Each provider gets its own schema-enforcement knob; unknown
    providers get none and rely on the prompt alone."""
    base = {"max_tokens": CLASSIFIER_MAX_TOKENS}

    # Anthropic gets the plain prompt only: the ``output_format``
    # grammar measurably slows the call (+~1 s, +~280 input tokens), and
    # a forced ``thinking.type=disabled`` is rejected (HTTP 400) by the
    # adaptive-thinking generation, which used to cost every fable-5 /
    # opus-5 run a wasted round trip before the plain fallback.
    for anthropic_model in (
        "claude-haiku-4-5", "claude-fable-5", "claude-opus-5",
        "anthropic/claude-haiku-4-5",
    ):
        assert _structured_output_config(base, anthropic_model) is None

    gemini = _structured_output_config(base, "gemini-2.5-flash")
    assert gemini is not None
    assert gemini["response_mime_type"] == "application/json"
    assert gemini["response_json_schema"] == _VERDICT_JSON_SCHEMA

    openai = _structured_output_config(base, "gpt-5.2")
    assert openai is not None
    assert openai["response_format"]["type"] == "json_schema"
    assert (
        openai["response_format"]["json_schema"]["schema"]
        == _VERDICT_JSON_SCHEMA
    )

    # Harbor-style provider prefixes route like the model factory does.
    prefixed = _structured_output_config(base, "google/gemini-2.5-flash")
    assert prefixed is not None
    assert "response_json_schema" in prefixed

    assert _structured_output_config(base, "no-such-provider/x") is None
    # The base config is never mutated.
    assert base == {"max_tokens": CLASSIFIER_MAX_TOKENS}


# ---------------------------------------------------------------------------
# classification_enabled — env kill switch and config key
# ---------------------------------------------------------------------------


def test_classification_enabled_env_kill_switch(
    env: IsolatedKissHome,
) -> None:
    """KISS_DISABLE_TASK_CLASSIFIER=1 wins over any config value."""
    env.write_config(classify_tasks=True)
    os.environ[_DISABLE_ENV] = "1"
    assert classification_enabled() is False


def test_classification_enabled_follows_config(
    env: IsolatedKissHome,
) -> None:
    """With the kill switch lifted, the classify_tasks key decides."""
    assert classification_enabled() is True  # default (key absent)
    env.write_config(classify_tasks=False)
    assert classification_enabled() is False
    env.write_config(classify_tasks=True)
    assert classification_enabled() is True


def test_classification_enabled_per_run_override(
    env: IsolatedKissHome,
) -> None:
    """A per-run boolean override beats the config; None follows it."""
    env.write_config(classify_tasks=True)
    assert classification_enabled(override=False) is False
    assert classification_enabled(override=None) is True
    env.write_config(classify_tasks=False)
    assert classification_enabled(override=True) is True
    assert classification_enabled(override=None) is False


def test_classification_env_kill_switch_beats_override(
    env: IsolatedKissHome,
) -> None:
    """KISS_DISABLE_TASK_CLASSIFIER=1 wins even over override=True."""
    env.write_config(classify_tasks=True)
    os.environ[_DISABLE_ENV] = "1"
    assert classification_enabled(override=True) is False


# ---------------------------------------------------------------------------
# SorcarAgent._classify_task_once — caching and the disabled path
# ---------------------------------------------------------------------------


def test_classify_once_returns_none_when_disabled(
    env: IsolatedKissHome,
) -> None:
    """A disabled classifier attempts nothing and reports no verdict."""
    env.write_config(classify_tasks=False)
    agent = SorcarAgent("clf-disabled")
    assert agent._classify_task_once(MODEL, _SIMPLE_TASK, None) is None
    assert agent._classification_attempted is True
    # The attempt is cached: later calls do not re-consult the config.
    env.write_config(classify_tasks=True)
    assert agent._classify_task_once(MODEL, _SIMPLE_TASK, None) is None


def test_classify_for_run_disabled_by_override(
    env: IsolatedKissHome,
) -> None:
    """classify_task_for_run(enabled=False) skips classification even
    with the config on, and the disabled seed survives into the run:
    a later in-run classification attempt returns None without
    re-consulting the (enabled) config."""
    env.write_config(classify_tasks=True)
    agent = SorcarAgent("clf-run-override-off")
    assert (
        agent.classify_task_for_run(
            model_name=MODEL, task=_DEV_TASK, enabled=False,
        )
        is None
    )
    assert agent._classification_attempted is True
    assert agent._classification_preseeded is True
    # No classification ran, so no classifier spend was published.
    assert agent._classifier_spend is None
    # The run reuses the seed instead of re-classifying.
    assert agent._classify_task_once(MODEL, _DEV_TASK, None) is None


def test_classify_once_returns_cached_verdict(
    env: IsolatedKissHome,
) -> None:
    """A verdict already computed for this run is returned as-is."""
    agent = SorcarAgent("clf-cached")
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(is_simple=False)
    verdict = agent._classify_task_once(MODEL, _SIMPLE_TASK, None)
    assert verdict == TaskClassification(is_simple=False)


def test_fold_classifier_usage_banks_and_resets(
    env: IsolatedKissHome,
) -> None:
    """Classifier spend lands in the run totals exactly once."""
    agent = SorcarAgent("clf-fold")
    agent._classifier_spend = _ClassifierSpend("classifier:t", 0.25, 123, 2)
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(True)
    agent._fold_classifier_usage()
    assert agent.budget_used == pytest.approx(0.25)
    assert agent.total_tokens_used == 123
    assert agent.total_steps == 2
    assert agent._classifier_spend is None
    assert agent._classification_attempted is False
    assert agent._task_classification is None
    # Folding again is a no-op: the counters were reset.
    agent._fold_classifier_usage()
    assert agent.budget_used == pytest.approx(0.25)


class _CapturePrinter(JsonPrinter):
    """A real ``JsonPrinter`` whose broadcast events are collected."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []

    def broadcast(self, event: dict) -> None:  # type: ignore[override]
        self.events.append(dict(event))


def _run_with_preloaded_classifier_spend(
    env: IsolatedKissHome, printer: _CapturePrinter,
) -> tuple[SorcarAgent, BaseException | None]:
    """Run a zero-session ``SorcarAgent`` whose classifier already spent $0.25.

    Returns:
        The agent and the exception its run raised, if any.
    """
    agent = SorcarAgent("clf-fold-event")
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(True)
    agent._classifier_spend = _ClassifierSpend("classifier:e", 0.25, 123, 2)
    raised: BaseException | None = None
    try:
        agent.run(
            model_name=MODEL,
            prompt_template="noop",
            max_sub_sessions=0,
            max_budget=5.0,
            append_basic_tools=False,
            use_memory=False,
            web_tools=False,
            work_dir=str(env.repo),
            printer=printer,
            verbose=False,
        )
    except BaseException as e:  # noqa: BLE001 – the outcome is under test
        raised = e
    return agent, raised


def test_fold_emits_the_cumulative_cost_for_the_ui(
    env: IsolatedKissHome,
) -> None:
    """The fold runs after the run's last event, so the run must publish
    the new total: the UI's "Cost:" otherwise omits the classifier's
    spend while the persisted task cost includes it."""
    printer = _CapturePrinter()
    agent, _raised = _run_with_preloaded_classifier_spend(env, printer)
    event = printer.events[-1]
    assert event["type"] == "usage_info"
    assert event["cost"] == "$0.2500"
    assert event["total_tokens"] == 123
    assert event["total_steps"] == 2
    assert "Budget: $0.2500" in event["text"]
    assert agent.usage_snapshot() == (0.25, 123, 2)


def test_stopped_run_still_cleans_up_before_the_fold_event(
    env: IsolatedKissHome,
) -> None:
    """Printing raises the task's stop when it is set; the run's cleanup
    must already be done by then and the spend must stay banked."""
    printer = _CapturePrinter()
    printer._thread_local.stop_event = threading.Event()
    printer._thread_local.stop_event.set()
    agent, raised = _run_with_preloaded_classifier_spend(env, printer)
    assert isinstance(raised, KeyboardInterrupt)
    assert agent.usage_snapshot() == (0.25, 123, 2)
    assert agent._classification_attempted is False
    assert agent.pre_step_hook is None
    assert agent._ask_user_question_callback is None


def test_zero_classifier_spend_emits_no_extra_event(env: IsolatedKissHome) -> None:
    agent = SorcarAgent("clf-fold-zero")
    agent._classifier_spend = _ClassifierSpend("classifier:z", 0.0, 0, 0)
    agent._fold_classifier_usage()
    assert agent._classifier_spend is None
    assert agent.usage_snapshot() == (0.0, 0, 0)
    assert agent._usage_ledger.records == []


# ---------------------------------------------------------------------------
# System prompt selection — full SorcarAgent runs (real LLM)
# ---------------------------------------------------------------------------


def _run_minimal_sorcar(agent: SorcarAgent, task: str, **kwargs: Any) -> str:
    """Run *agent* on *task* with the smallest possible tool surface."""
    return agent.run(
        model_name=MODEL,
        prompt_template=task,
        append_basic_tools=False,
        web_tools=False,
        is_parallel=False,
        max_steps=8,
        max_budget=2.0,
        verbose=False,
        **kwargs,
    )


@live_api
@requires_anthropic
def test_simple_task_runs_with_lite_prompt(env: IsolatedKissHome) -> None:
    """A simple task's run uses SYSTEM_LITE.md as its base prompt."""
    agent = SorcarAgent("clf-lite-prompt")
    result = _run_minimal_sorcar(
        agent, _SIMPLE_TASK + " Call the finish tool with the answer.",
    )
    assert yaml.safe_load(result)["success"] is True, result
    assert agent.system_prompt.startswith(SYSTEM_PROMPT_LITE)
    assert not agent.system_prompt.startswith(SYSTEM_PROMPT)
    # The classifier's own spend was folded into the run totals.
    assert agent.budget_used > 0.0
    # Classification state was cleared for the next run.
    assert agent._classification_attempted is False
    assert agent._task_classification is None


@live_api
@requires_anthropic
def test_non_simple_verdict_keeps_full_prompt(
    env: IsolatedKissHome,
) -> None:
    """An is_simple=False verdict keeps the full SYSTEM.md prompt."""
    agent = SorcarAgent("clf-full-prompt")
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(is_simple=False)
    # The full prompt tells the model to report success only for a checked
    # task, so the instruction names the success value it must pass.
    result = _run_minimal_sorcar(
        agent, "Call the finish tool with success=true and the summary DONE.",
    )
    assert yaml.safe_load(result)["success"] is True, result
    assert agent.system_prompt.startswith(SYSTEM_PROMPT)


@live_api
@requires_anthropic
def test_disabled_classifier_keeps_full_prompt(
    env: IsolatedKissHome,
) -> None:
    """With classify_tasks off, every task keeps the full prompt."""
    env.write_config(classify_tasks=False)
    agent = SorcarAgent("clf-off-prompt")
    result = _run_minimal_sorcar(
        agent, _SIMPLE_TASK + " Call the finish tool with the answer.",
    )
    assert yaml.safe_load(result)["success"] is True, result
    assert agent.system_prompt.startswith(SYSTEM_PROMPT)


@live_api
@requires_anthropic
def test_base_system_prompt_override_beats_verdict(
    env: IsolatedKissHome,
) -> None:
    """A caller-supplied base prompt wins over the lite selection."""
    agent = SorcarAgent("clf-custom-prompt")
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(is_simple=True)
    custom = "You are a terse assistant. Use the finish tool to answer."
    result = _run_minimal_sorcar(
        agent,
        "Call the finish tool with the word DONE.",
        base_system_prompt=custom,
    )
    assert yaml.safe_load(result)["success"] is True, result
    assert agent.system_prompt.startswith(custom)


# ---------------------------------------------------------------------------
# Review-driven fixes: CLI models, pre-seeding, substitution, stale usage
# ---------------------------------------------------------------------------


def test_reset_drops_stale_unfolded_usage(env: IsolatedKissHome) -> None:
    """A crashed run's never-folded classifier spend is dropped, not
    misattributed to the next unrelated run."""
    agent = SorcarAgent("clf-stale-usage")
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(True)
    agent._classifier_spend = _ClassifierSpend("classifier:stale", 0.5, 42, 1)
    agent._reset_task_classification()
    assert agent._classification_attempted is False
    assert agent._task_classification is None
    assert agent._classifier_spend is None


def test_reset_keeps_preseeded_verdict(env: IsolatedKissHome) -> None:
    """A verdict pre-seeded by an external driver survives the
    per-run reset so every subtask of a submission reuses it."""
    agent = SorcarAgent("clf-preseed-keep")
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(False)
    agent._classification_preseeded = True
    agent._reset_task_classification()
    assert agent._task_classification == TaskClassification(False)
    assert agent._classification_attempted is True


@live_api
@requires_anthropic
def test_classify_task_for_run_preseeds_prompt_selection(
    env: IsolatedKissHome,
) -> None:
    """The server-style pre-classification drives the run's prompt
    selection; the run reuses the seeded verdict instead of
    re-classifying the (contradictory) prompt it is given."""
    agent = SorcarAgent("clf-preseed-run")
    verdict = agent.classify_task_for_run(model_name=MODEL, task=_DEV_TASK)
    assert verdict is not None
    assert verdict.is_simple is False
    # The run's own prompt is a SIMPLE task: if the run re-classified
    # it, the lite prompt would be selected.  The seeded non-simple
    # verdict must win and keep the full prompt.
    result = _run_minimal_sorcar(
        agent, _SIMPLE_TASK + " Call the finish tool with the answer.",
    )
    assert yaml.safe_load(result)["success"] is True, result
    assert agent.system_prompt.startswith(SYSTEM_PROMPT)
    assert not agent.system_prompt.startswith(SYSTEM_PROMPT_LITE)
    # The next server-style classification replaces the seed.
    env.write_config(classify_tasks=False)
    assert agent.classify_task_for_run(model_name=MODEL, task=_DEV_TASK) is None
    assert agent._task_classification is None


@live_api
@requires_anthropic
def test_classify_task_for_run_enabled_override_beats_config(
    env: IsolatedKissHome,
) -> None:
    """enabled=True classifies for real even with classify_tasks off.

    The per-run ``classify_tasks`` parameter of
    ``kiss.server.sorcar.run`` (wire field ``classifyTasks``) must win
    over the persisted "Classify tasks before running" setting in both
    directions; the forcing direction needs a real verdict to prove
    the classification actually ran.
    """
    env.write_config(classify_tasks=False)
    agent = SorcarAgent("clf-run-override-on")
    verdict = agent.classify_task_for_run(
        model_name=MODEL, task=_DEV_TASK, enabled=True,
    )
    assert verdict is not None
    assert verdict.is_simple is False
    spend = agent._classifier_spend
    assert spend is not None and spend.budget > 0.0


@live_api
@requires_anthropic
def test_classifier_sees_substituted_template_arguments(
    env: IsolatedKissHome,
) -> None:
    """{placeholder} templates are substituted before classification."""
    agent = SorcarAgent("clf-substitution")
    verdict = agent._classify_task_once(
        MODEL, "{task}", None, arguments={"task": _DEV_TASK},
    )
    assert verdict is not None
    assert verdict.is_simple is False
