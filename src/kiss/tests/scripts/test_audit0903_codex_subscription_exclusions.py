# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The catalog updater must not resurrect subscription-incompatible codex models.

``codex/gpt-5.2``, ``codex/gpt-daybreak-blue-latest`` and
``codex/gpt-daybreak-red-latest`` were verified live (2026-09-02) to be
rejected by the Codex CLI under a ChatGPT subscription (HTTP 400), and
were removed from ``MODEL_INFO.json``.  But the knowledge lived only in
the guard test — ``update_models.py`` kept adding any slug present in
the CLI's upstream ``models.json`` — so the next catalog refresh
(commit 778fa15ef on this branch) resurrected all three.  This is the
duplicate-knowledge redundancy: two places knew which slugs are
allowed, and they drifted.

The fix gives ``update_models.py`` a single named
``SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS`` set consulted by BOTH the
candidate-adder and the deprecation scan, and the guard test imports it.

These tests drive the real updater functions on real catalog dicts —
no mocks and no network (the slug set is passed in, exactly as
``main()`` passes the fetched one).
"""

from __future__ import annotations

from kiss.scripts.update_models import (
    SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS,
    _add_codex_candidates,
    find_deprecated_models,
)


def test_incompatible_slugs_never_added_as_candidates() -> None:
    """A refresh with the bad slugs upstream must not re-add them."""
    upstream = set(SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS) | {"gpt-5.5"}
    new_models: list[dict] = []
    _add_codex_candidates(upstream, current={}, openrouter={},
                          new_models=new_models)
    names = {m["name"] for m in new_models}
    assert "codex/gpt-5.5" in names
    for slug in SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS:
        assert f"codex/{slug}" not in names, (
            f"codex/{slug} is rejected by the Codex CLI on ChatGPT "
            "subscriptions and must never be re-added by a refresh"
        )


def test_incompatible_entries_reported_deprecated() -> None:
    """Existing bad entries are flagged even when upstream still lists them."""
    current = {
        f"codex/{slug}": {"context_length": 400000}
        for slug in SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS
    }
    current["codex/gpt-5.5"] = {"context_length": 400000}
    deprecated = find_deprecated_models(
        current, openrouter={}, anthropic={}, gemini={}, openai={},
        codex_slugs=set(SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS) | {"gpt-5.5"},
    )
    flagged = {d["name"] for d in deprecated}
    for slug in SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS:
        assert f"codex/{slug}" in flagged
    assert "codex/gpt-5.5" not in flagged


def test_guard_test_shares_the_single_source_of_truth() -> None:
    """The MODEL_INFO guard derives its list from the updater's set."""
    from kiss.tests.core.models.test_codex_model import (
        _SUBSCRIPTION_INCOMPATIBLE_NAMES,
    )

    assert set(_SUBSCRIPTION_INCOMPATIBLE_NAMES) == {
        f"codex/{slug}" for slug in SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS
    }
