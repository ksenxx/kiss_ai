# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for pre-run task classification.

``kiss.agents.sorcar.task_classifier`` runs one NON-AGENTIC KISSAgent
generation on the run's own model BEFORE a Sorcar agent starts a task.
The structured JSON verdict ``{"is_simple": ..., "is_development": ...}``
selects the system prompt (``SYSTEM_LITE.md`` for simple tasks,
``SYSTEM.md`` otherwise) and decides worktree isolation for that run
(``is_development`` becomes the effective ``use_worktree``) without
touching the persisted ``is_worktree`` setting.  Tasks that only
request git operations are never development.

The classification tests call real LLMs (a cheap model) — no mocks.
The verdict-cache tests use the real local OpenAI-compatible stand-in
endpoint from ``parallel_agent_harness`` instead.

Unreachable without test doubles, documented per the testing policy
rather than mocked: the ``except Exception`` in
``task_classifier._model_runs_task_to_completion`` — the registry
lookup it guards (``model_info.model_runs_task_to_completion``) returns
``False`` for every unknown name and raises for none of the inputs a
caller can produce.
The wiring tests drive real ``SorcarAgent`` / ``WorktreeSorcarAgent``
runs inside an isolated KISS_HOME with a scratch git repo; where a full
agent run is unnecessary, the run is stopped early by a real server
printer whose task-allocation hook raises (the same technique as
``test_audit0903_worktree_run_wrapping``).
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

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
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
    classification_will_call_model,
    classify_task,
    clear_classification_cache,
    remember_classification,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.base import SYSTEM_PROMPT, SYSTEM_PROMPT_LITE
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
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

_GIT_TASK = (
    "Run git operations only: check `git status`, squash-merge the "
    "branch kiss/wt-123 into main with `git merge --squash`, resolve "
    "any merge conflicts, commit the result, and delete the branch. "
    "Do not write any new code."
)

_DISABLE_ENV = "KISS_DISABLE_TASK_CLASSIFIER"


class _RaisingPrinter(CapturePrinter):
    """A real server printer whose task-allocation hook raises.

    Stops an agent run right after classification and the worktree
    decision, so worktree-gating tests spend exactly one LLM call (the
    classifier's) instead of a whole agent run.
    """

    def __init__(self, exc: BaseException) -> None:
        super().__init__()
        self._exc = exc

    def agent_task_allocated(
        self, agent: Any, task_id: Any, chat_id: str = ""
    ) -> None:
        super().agent_task_allocated(agent, task_id, chat_id)
        raise self._exc


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """Isolated KISS_HOME + scratch repo, with the classifier enabled.

    The root conftest disables the classifier suite-wide via
    ``KISS_DISABLE_TASK_CLASSIFIER=1``; these are the classifier's own
    tests, so the kill switch is lifted and restored afterwards.
    """
    saved = os.environ.get(_DISABLE_ENV)
    os.environ[_DISABLE_ENV] = "0"
    isolated = IsolatedKissHome("kiss-task-classifier-")
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
    """A pure-arithmetic task is simple and not development."""
    outcome = classify_task(task=_SIMPLE_TASK, model_name=MODEL)
    assert outcome.classification is not None
    assert outcome.classification.is_simple is True
    assert outcome.classification.is_development is False
    # The attempt's real usage is reported for budget folding, and the
    # non-agentic classification is exactly one generation.
    assert outcome.budget_used > 0.0
    assert outcome.tokens_used > 0
    assert outcome.steps == 1


@live_api
@requires_anthropic
def test_classify_development_task(env: IsolatedKissHome) -> None:
    """A file-editing coding task is development and not simple."""
    outcome = classify_task(task=_DEV_TASK, model_name=MODEL)
    assert outcome.classification is not None
    assert outcome.classification.is_simple is False
    assert outcome.classification.is_development is True


@live_api
@requires_anthropic
def test_classify_git_task_is_not_development(
    env: IsolatedKissHome,
) -> None:
    """A task that only requests git operations is never development."""
    outcome = classify_task(task=_GIT_TASK, model_name=MODEL)
    assert outcome.classification is not None
    assert outcome.classification.is_development is False


@live_api
@requires_anthropic
def test_classify_truncates_huge_task(env: IsolatedKissHome) -> None:
    """A giant prompt is truncated before classification, bounding the
    call's input cost, and the gist still classifies correctly."""
    huge_task = _DEV_TASK + "\n" + ("filler line about the codebase\n" * 5000)
    assert len(huge_task) > 100_000
    outcome = classify_task(task=huge_task, model_name=MODEL)
    assert outcome.classification is not None
    assert outcome.classification.is_development is True
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
    assert outcome.classification.is_development is False
    assert outcome.steps == 1


def test_classify_task_bad_model_fails_soft() -> None:
    """An unknown model yields classification=None, not an exception."""
    outcome = classify_task(
        task="anything", model_name="no-such-model-xyz",
    )
    assert outcome.classification is None
    assert outcome.budget_used == 0.0


# ---------------------------------------------------------------------------
# Verdict cache — real local OpenAI-compatible endpoint, no LLM
# ---------------------------------------------------------------------------


def _verdict_of(entry: dict[str, Any]) -> tuple[bool, bool]:
    """The ``(is_simple, is_development)`` pair of one stored cache entry."""
    return (entry["is_simple"], entry["is_development"])


def _entry(is_simple: bool, is_development: bool, age_s: float = 0.0) -> dict[str, Any]:
    """A cache file entry written *age_s* seconds ago."""
    return {
        "is_simple": is_simple,
        "is_development": is_development,
        "ts": time.time() - age_s,
    }


def _verdict_body(is_simple: bool, is_development: bool) -> dict[str, Any]:
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
                    "content": json.dumps(
                        {"is_simple": is_simple, "is_development": is_development},
                    ),
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

    def __init__(self, is_simple: bool = True, is_development: bool = False) -> None:
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._verdict = (is_simple, is_development)
        self.server = StandInModelServer(self._respond)

    def _respond(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.requests.append(request)
        return _verdict_body(*self._verdict)

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
    assert first.classification == TaskClassification(
        is_simple=True, is_development=False,
    )
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
    assert [_verdict_of(v) for v in stored.values()] == [(True, False)]
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
    other = _CountingClassifierEndpoint(is_simple=False, is_development=True)
    try:
        outcome = classify_task(
            task="alpha", model_name=STANDIN_MODEL,
            model_config=other.model_config,
        )
        assert len(other.requests) == 1
        assert outcome.classification == TaskClassification(
            is_simple=False, is_development=True,
        )
    finally:
        other.stop()
    # The original triple still hits its own memo, not the other's.
    hit = classify_task(task="alpha", model_name=STANDIN_MODEL, model_config=config)
    assert hit.steps == 0
    assert hit.classification == TaskClassification(
        is_simple=True, is_development=False,
    )
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


def test_classification_will_call_model_mirrors_the_gates(
    env: IsolatedKissHome, endpoint: _CountingClassifierEndpoint,
) -> None:
    """The overlap hint is true exactly when a round trip is imminent."""
    config = endpoint.model_config
    assert classification_will_call_model("fresh", STANDIN_MODEL, config) is True
    # Disabled by override, env kill switch, or config.
    assert classification_will_call_model(
        "fresh", STANDIN_MODEL, config, enabled_override=False,
    ) is False
    os.environ[_DISABLE_ENV] = "1"
    try:
        assert classification_will_call_model("fresh", STANDIN_MODEL, config) is False
    finally:
        os.environ[_DISABLE_ENV] = "0"
    env.write_config(classify_tasks=False)
    assert classification_will_call_model("fresh", STANDIN_MODEL, config) is False
    assert classification_will_call_model(
        "fresh", STANDIN_MODEL, config, enabled_override=True,
    ) is True
    env.write_config(classify_tasks=True)
    # Run-to-completion CLI models are never classified.
    assert classification_will_call_model("fresh", "cc/opus", config) is False
    assert classification_will_call_model("fresh", "codex/default", config) is False
    # A memoised verdict needs no call; truncation is applied before
    # the lookup, so an over-long prompt keys the same as its prefix.
    classify_task(task="fresh", model_name=STANDIN_MODEL, model_config=config)
    assert classification_will_call_model("fresh", STANDIN_MODEL, config) is False
    huge = "x" * 30_000
    classify_task(task=huge, model_name=STANDIN_MODEL, model_config=config)
    assert classification_will_call_model(huge, STANDIN_MODEL, config) is False
    assert classification_will_call_model(
        huge[:20_000] + "y" * 10_000, STANDIN_MODEL, config,
    ) is False
    assert len(endpoint.requests) == 2


def test_unparseable_verdict_is_not_cached_and_falls_back(
    env: IsolatedKissHome,
) -> None:
    """A model that answers prose gets the plain retry, yields no verdict,
    and leaves no memo (only genuine verdicts are memoised)."""
    def prose(_request: dict[str, Any]) -> dict[str, Any]:
        body = _verdict_body(True, False)
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
            "junk-key": {**_entry(True, False), "is_simple": "yes"},
            "other-junk": "not a dict",
            "int-not-bool": {**_entry(True, False), "is_development": 1},
            "missing-key": {"is_simple": True, "ts": time.time()},
            "no-timestamp": {"is_simple": True, "is_development": True},
            "bad-timestamp": {"is_simple": True, "is_development": True, "ts": "now"},
            "kept": _entry(False, True),
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
        f"key-{i:05d}": _entry(True, False)
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
        "t", "m", None, TaskClassification(is_simple=True, is_development=False),
    )
    remember_classification(
        "u", "m", None, TaskClassification(is_simple=True, is_development=True),
    )
    remember_classification(
        "t", "m", None, TaskClassification(is_simple=False, is_development=True),
    )
    stored = json.loads(
        (env.kiss_home / CLASSIFIER_CACHE_FILENAME).read_text(encoding="utf-8"),
    )
    assert list(stored) == [_cache_key("u", "m", None), _cache_key("t", "m", None)]
    assert cached_classification("t", "m") == TaskClassification(
        is_simple=False, is_development=True,
    )


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
        TaskClassification(is_simple=True, is_development=False),
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
    assert first.classification == TaskClassification(is_simple=True, is_development=False)
    other = _CountingClassifierEndpoint(is_simple=False, is_development=True)
    try:
        config_b = {**other.model_config, "api_key": "kiss-other-account"}
        second = classify_task(task="route", model_name=STANDIN_MODEL, model_config=config_b)
        assert len(other.requests) == 1
        assert second.classification == TaskClassification(
            is_simple=False, is_development=True,
        )
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
            key: _entry(False, True, age_s=CLASSIFIER_CACHE_TTL_SECONDS + 60),
            "fresh-other": _entry(True, True, age_s=CLASSIFIER_CACHE_TTL_SECONDS - 60),
        }),
        encoding="utf-8",
    )
    assert cached_classification("old news", STANDIN_MODEL, config) is None
    outcome = classify_task(task="old news", model_name=STANDIN_MODEL, model_config=config)
    assert outcome.steps == 1
    assert outcome.classification == TaskClassification(is_simple=True, is_development=False)
    stored = json.loads(cache_file.read_text(encoding="utf-8"))
    assert _verdict_of(stored[key]) == (True, False), "the stale memo was replaced"
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
            theirs_key: _entry(False, True),
            mine_key: {**_entry(False, True), "ts": time.time() + 5},
        }),
        encoding="utf-8",
    )
    remember_classification(
        "third", STANDIN_MODEL, config,
        TaskClassification(is_simple=True, is_development=True),
    )
    stored = json.loads(cache_file.read_text(encoding="utf-8"))
    assert _verdict_of(stored[theirs_key]) == (False, True)
    assert _verdict_of(stored[mine_key]) == (False, True), "newer on-disk memo wins"
    assert _verdict_of(stored[_cache_key("third", STANDIN_MODEL, config)]) == (True, True)
    assert cached_classification("theirs", STANDIN_MODEL, config) == TaskClassification(
        is_simple=False, is_development=True,
    )


def test_clear_cache_removes_file_and_memory(env: IsolatedKissHome) -> None:
    """``clear_classification_cache`` forgets everything, twice is fine."""
    remember_classification(
        "t", "m", None, TaskClassification(is_simple=True, is_development=False),
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
            TaskClassification(is_simple=True, is_development=False),
        )
        assert cached_classification("t", "m") == TaskClassification(
            is_simple=True, is_development=False,
        )
        assert not tc_module._cache_path().exists()
    finally:
        os.environ["KISS_HOME"] = saved
        clear_classification_cache()


# ---------------------------------------------------------------------------
# Structured-output verdict parsing — no LLM
# ---------------------------------------------------------------------------


def test_parse_verdict_accepts_bare_json() -> None:
    """The demanded schema — one bare JSON object — parses directly."""
    assert _parse_verdict(
        '{"is_simple": true, "is_development": false}'
    ) == TaskClassification(is_simple=True, is_development=False)
    # String-typed booleans from lax models are interpreted by content.
    assert _parse_verdict(
        '{"is_simple": "yes", "is_development": "no"}'
    ) == TaskClassification(is_simple=True, is_development=False)


def test_parse_verdict_extracts_json_from_fences_and_prose() -> None:
    """Code fences and surrounding prose do not defeat extraction."""
    fenced = (
        "```json\n"
        '{"is_simple": false, "is_development": true}\n'
        "```"
    )
    assert _parse_verdict(fenced) == TaskClassification(
        is_simple=False, is_development=True,
    )
    prose = (
        "Here is my classification of the task:\n"
        '{"is_simple": true, "is_development": false}\n'
        "Let me know if you need anything else."
    )
    assert _parse_verdict(prose) == TaskClassification(
        is_simple=True, is_development=False,
    )
    # A leading brace that is not JSON does not stop the scan.
    noisy = (
        "{broken json} then the real verdict "
        '{"is_simple": false, "is_development": false}'
    )
    assert _parse_verdict(noisy) == TaskClassification(
        is_simple=False, is_development=False,
    )
    # An earlier valid JSON object without the verdict keys is skipped.
    decoy = (
        '{"note": "thinking"} '
        '{"is_simple": true, "is_development": true}'
    )
    assert _parse_verdict(decoy) == TaskClassification(
        is_simple=True, is_development=True,
    )


def test_parse_verdict_rejects_junk() -> None:
    """Prose, non-object JSON, and partial objects all yield None."""
    assert _parse_verdict("I think this task is simple.") is None
    assert _parse_verdict("[1, 2]") is None
    assert _parse_verdict('{"is_simple": true}') is None
    assert _parse_verdict("") is None
    assert _parse_verdict(None) is None  # type: ignore[arg-type]


def test_parse_verdict_rejects_non_boolean_values() -> None:
    """Truthy junk must not silently flip worktree isolation."""
    assert _parse_verdict('{"is_simple": [], "is_development": {"x": 1}}') is None
    assert _parse_verdict('{"is_simple": 2, "is_development": null}') is None
    assert _parse_verdict('{"is_simple": 1.0, "is_development": true}') is None
    assert _parse_verdict('{"is_simple": "maybe", "is_development": true}') is None
    # A later well-formed verdict still wins over an earlier junk one.
    assert _parse_verdict(
        '{"is_simple": null, "is_development": 3} '
        '{"is_simple": false, "is_development": true}'
    ) == TaskClassification(is_simple=False, is_development=True)


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
    # No classification ran, so no classifier spend was banked.
    assert agent._classifier_budget_used == 0.0
    assert agent._classifier_tokens_used == 0
    assert agent._classifier_steps == 0
    # The run reuses the seed instead of re-classifying.
    assert agent._classify_task_once(MODEL, _DEV_TASK, None) is None


def test_classify_once_returns_cached_verdict(
    env: IsolatedKissHome,
) -> None:
    """A verdict already computed for this run is returned as-is."""
    agent = SorcarAgent("clf-cached")
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(
        is_simple=False, is_development=True,
    )
    verdict = agent._classify_task_once(MODEL, _SIMPLE_TASK, None)
    assert verdict == TaskClassification(is_simple=False, is_development=True)


def test_fold_classifier_usage_banks_and_resets(
    env: IsolatedKissHome,
) -> None:
    """Classifier spend lands in the run totals exactly once."""
    agent = SorcarAgent("clf-fold")
    agent._classifier_budget_used = 0.25
    agent._classifier_tokens_used = 123
    agent._classifier_steps = 2
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(True, False)
    agent._fold_classifier_usage()
    assert agent.budget_used == pytest.approx(0.25)
    assert agent.total_tokens_used == 123
    assert agent.total_steps == 2
    assert agent._classifier_budget_used == 0.0
    assert agent._classifier_tokens_used == 0
    assert agent._classifier_steps == 0
    assert agent._classification_attempted is False
    assert agent._task_classification is None
    # Folding again is a no-op: the counters were reset.
    agent._fold_classifier_usage()
    assert agent.budget_used == pytest.approx(0.25)


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
    assert yaml.safe_load(result)["success"] is True
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
    agent._task_classification = TaskClassification(
        is_simple=False, is_development=True,
    )
    result = _run_minimal_sorcar(
        agent, "Call the finish tool with the word DONE.",
    )
    assert yaml.safe_load(result)["success"] is True
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
    assert yaml.safe_load(result)["success"] is True
    assert agent.system_prompt.startswith(SYSTEM_PROMPT)


@live_api
@requires_anthropic
def test_base_system_prompt_override_beats_verdict(
    env: IsolatedKissHome,
) -> None:
    """A caller-supplied base prompt wins over the lite selection."""
    agent = SorcarAgent("clf-custom-prompt")
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(
        is_simple=True, is_development=False,
    )
    custom = "You are a terse assistant. Use the finish tool to answer."
    result = _run_minimal_sorcar(
        agent,
        "Call the finish tool with the word DONE.",
        base_system_prompt=custom,
    )
    assert yaml.safe_load(result)["success"] is True
    assert agent.system_prompt.startswith(custom)


# ---------------------------------------------------------------------------
# Worktree gating — WorktreeSorcarAgent runs stopped after the decision
# ---------------------------------------------------------------------------


@live_api
@requires_anthropic
def test_non_development_task_skips_worktree(
    env: IsolatedKissHome,
) -> None:
    """is_development=False turns worktree isolation off for the run."""
    agent = WorktreeSorcarAgent("clf-wt-skip")
    printer = _RaisingPrinter(RuntimeError("stop-after-decision"))
    result = agent.run(
        prompt_template=_SIMPLE_TASK,
        model_name=MODEL,
        work_dir=str(env.repo),
        printer=printer,
    )
    payload = yaml.safe_load(result)
    assert payload["success"] is False
    assert "stop-after-decision" in payload["summary"]
    assert agent._wt is None
    assert printer.events_of_type("worktree_created") == []


@live_api
@requires_anthropic
def test_development_task_forces_worktree(env: IsolatedKissHome) -> None:
    """is_development=True turns worktree isolation on for the run,
    even when the caller asked for none."""
    agent = WorktreeSorcarAgent("clf-wt-force")
    printer = _RaisingPrinter(RuntimeError("stop-after-decision"))
    result = agent.run(
        prompt_template=_DEV_TASK,
        model_name=MODEL,
        use_worktree=False,
        work_dir=str(env.repo),
        printer=printer,
    )
    payload = yaml.safe_load(result)
    assert payload["success"] is False
    assert len(printer.events_of_type("worktree_created")) == 1
    assert agent._wt is not None
    # The persisted setting was never touched by the runtime override.
    from kiss.core.vscode_config import load_config

    assert load_config()["is_worktree"] is True
    agent.discard()
    assert agent._wt is None


@live_api
@requires_anthropic
def test_disabled_classifier_keeps_callers_worktree_choice(
    env: IsolatedKissHome,
) -> None:
    """With the classifier off, use_worktree=False is respected."""
    env.write_config(classify_tasks=False)
    agent = WorktreeSorcarAgent("clf-wt-off")
    printer = _RaisingPrinter(RuntimeError("stop-after-decision"))
    result = agent.run(
        prompt_template=_DEV_TASK,
        model_name=MODEL,
        use_worktree=False,
        work_dir=str(env.repo),
        printer=printer,
    )
    assert yaml.safe_load(result)["success"] is False
    assert agent._wt is None
    assert printer.events_of_type("worktree_created") == []


# ---------------------------------------------------------------------------
# Review-driven fixes: CLI models, pre-seeding, substitution, stale usage
# ---------------------------------------------------------------------------


def test_classify_task_skips_run_to_completion_models() -> None:
    """cc/* models could execute the embedded task with native tools,
    so classification is skipped for them without any spend."""
    outcome = classify_task(task=_DEV_TASK, model_name="cc/claude-opus-4-6")
    assert outcome.classification is None
    assert outcome.budget_used == 0.0
    assert outcome.tokens_used == 0
    assert outcome.steps == 0


def test_reset_drops_stale_unfolded_usage(env: IsolatedKissHome) -> None:
    """A crashed run's never-folded classifier spend is dropped, not
    misattributed to the next unrelated run."""
    agent = SorcarAgent("clf-stale-usage")
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(True, False)
    agent._classifier_budget_used = 0.5
    agent._classifier_tokens_used = 42
    agent._classifier_steps = 1
    agent._reset_task_classification()
    assert agent._classification_attempted is False
    assert agent._task_classification is None
    assert agent._classifier_budget_used == 0.0
    assert agent._classifier_tokens_used == 0
    assert agent._classifier_steps == 0


def test_reset_keeps_preseeded_verdict(env: IsolatedKissHome) -> None:
    """A verdict pre-seeded by an external driver survives the
    per-run reset so every subtask of a submission reuses it."""
    agent = SorcarAgent("clf-preseed-keep")
    agent._classification_attempted = True
    agent._task_classification = TaskClassification(False, True)
    agent._classification_preseeded = True
    agent._reset_task_classification()
    assert agent._task_classification == TaskClassification(False, True)
    assert agent._classification_attempted is True


@live_api
@requires_anthropic
def test_classify_task_for_run_preseeds_worktree_decision(
    env: IsolatedKissHome,
) -> None:
    """The server-style pre-classification drives the run's worktree
    mode; the run reuses the seeded verdict instead of re-classifying
    the (contradictory) prompt it is given."""
    agent = WorktreeSorcarAgent("clf-preseed-run")
    verdict = agent.classify_task_for_run(model_name=MODEL, task=_DEV_TASK)
    assert verdict is not None
    assert verdict.is_development is True
    printer = _RaisingPrinter(RuntimeError("stop-after-decision"))
    # The run's own prompt is a NON-development task: if the run
    # re-classified it, no worktree would be created.  The seeded
    # development verdict must win.
    result = agent.run(
        prompt_template=_SIMPLE_TASK,
        model_name=MODEL,
        work_dir=str(env.repo),
        printer=printer,
    )
    assert yaml.safe_load(result)["success"] is False
    assert len(printer.events_of_type("worktree_created")) == 1
    assert agent._wt is not None
    agent.discard()
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
    assert verdict.is_development is True
    assert agent._classifier_budget_used > 0.0


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
    assert verdict.is_development is True
