from kiss.agents.sorcar.sorcar import _clip_autocomplete_suggestion

# ---------------------------------------------------------------------------
# kiss/agents/sorcar/sorcar.py — _clip_autocomplete_suggestion
# ---------------------------------------------------------------------------

def test_clip_autocomplete_suggestion_keeps_only_few_words() -> None:
    suggestion = _clip_autocomplete_suggestion(
        "fix", " the failing test in parser now"
    )
    assert suggestion == "the failing test in"


def test_clip_autocomplete_suggestion_rejects_sentence_like_completion() -> None:
    suggestion = _clip_autocomplete_suggestion(
        "fix", " the failing test in parser now and then update docs too"
    )
    assert suggestion == ""


def test_clip_autocomplete_suggestion_rejects_punctuation_boundary() -> None:
    assert _clip_autocomplete_suggestion("fix", " the failing test. Then update docs") == ""


def test_clip_autocomplete_suggestion_strips_repeated_query_prefix() -> None:
    assert _clip_autocomplete_suggestion("hello", "hello world again") == "world again"


def test_clip_autocomplete_suggestion_rejects_empty_result() -> None:
    assert _clip_autocomplete_suggestion("hello", "   ") == ""
