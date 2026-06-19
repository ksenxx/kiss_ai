# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for xhigh reasoning-effort detection in update_models.py.

These verify that the data-driven xhigh flag (``supports_xhigh_reasoning_effort``
on ``ModelInfo``) is populated end-to-end by update_models.py:

1. ``_make_entry_line`` emits ``xhigh=True`` in the generated ``_mi(...)`` line
   when the flag is set, and omits it otherwise.
2. ``test_xhigh_reasoning_effort`` short-circuits to ``False`` for backends
   that don't accept ``reasoning_effort`` (codex/*, claude-*, gemini-*) and
   for variants known to reject it (``-pro``, ``-chat-latest``, ``-image``)
   and for unrelated prefixes (Together, OpenRouter non-OpenAI), so we never
   waste an API call on them.
3. ``get_current_model_info`` surfaces the ``xhigh`` flag for every model in
   ``MODEL_INFO`` and matches each model's
   ``supports_xhigh_reasoning_effort`` field.
"""

from kiss.core.models.model_info import MODEL_INFO
from kiss.scripts.update_models import (
    _make_entry_line,
    get_current_model_info,
)
from kiss.scripts.update_models import (
    test_xhigh_reasoning_effort as _probe_xhigh,
)


def test_make_entry_line_emits_xhigh_true():
    """A new gpt-5.5-style entry with xhigh=True must include xhigh=True."""
    line = _make_entry_line(
        "gpt-99",
        ctx=1050000,
        inp=5.00,
        out=30.00,
        xhigh=True,
        comment="NEW",
    )
    assert '"gpt-99": _mi(1050000, 5.00, 30.00, xhigh=True),' in line
    assert "# NEW" in line


def test_make_entry_line_omits_xhigh_false():
    """Without xhigh=True the line must not mention xhigh at all."""
    line = _make_entry_line(
        "gpt-foo",
        ctx=200000,
        inp=1.00,
        out=2.00,
    )
    assert "xhigh" not in line
    assert '"gpt-foo": _mi(200000, 1.00, 2.00),' in line


def test_make_entry_line_combines_xhigh_with_other_flags():
    """xhigh=True must compose with fc=False, emb=True, gen=False in order."""
    line = _make_entry_line(
        "gpt-bar",
        ctx=128000,
        inp=1.00,
        out=2.00,
        fc=False,
        xhigh=True,
    )
    assert "fc=False, xhigh=True" in line


def test_xhigh_test_skipped_for_codex_models():
    """codex/* models route through the Codex CLI, not Chat Completions."""
    assert _probe_xhigh("codex/gpt-5.5") is False


def test_xhigh_test_skipped_for_claude_models():
    """Anthropic Claude does not accept reasoning_effort."""
    assert _probe_xhigh("claude-3-5-sonnet-20241022") is False


def test_xhigh_test_skipped_for_gemini_models():
    """Google Gemini does not accept reasoning_effort."""
    assert _probe_xhigh("gemini-2.5-pro") is False


def test_xhigh_test_skipped_for_pro_variants():
    """-pro variants reject reasoning_effort outright on OpenAI."""
    assert _probe_xhigh("gpt-5.5-pro") is False


def test_xhigh_test_skipped_for_chat_latest_variants():
    """-chat-latest variants don't accept reasoning_effort."""
    assert _probe_xhigh("gpt-5-chat-latest") is False


def test_xhigh_test_skipped_for_image_variants():
    """Image variants don't accept reasoning_effort."""
    assert _probe_xhigh("gpt-image-1") is False


def test_xhigh_test_skipped_for_together_models():
    """Together / non-OpenAI OpenRouter models are not OpenAI Chat Completions."""
    assert _probe_xhigh("meta-llama/Llama-3.3-70B-Instruct-Turbo") is False
    assert _probe_xhigh("openrouter/anthropic/claude-sonnet-4.5") is False
    assert _probe_xhigh("openrouter/google/gemini-2.5-pro") is False


def test_xhigh_test_skipped_for_embedding_models():
    """text-embedding-* models are not generation models and reject the param."""
    assert _probe_xhigh("text-embedding-3-large") is False


def test_get_current_model_info_exposes_xhigh_flag():
    """Every model in MODEL_INFO must surface its xhigh flag verbatim."""
    snapshot = get_current_model_info()
    for name, info in MODEL_INFO.items():
        assert "xhigh" in snapshot[name]
        assert snapshot[name]["xhigh"] is info.supports_xhigh_reasoning_effort


def test_get_current_model_info_marks_gpt_5_5_as_xhigh():
    """gpt-5.5 is the canonical xhigh-capable model and must be flagged."""
    snapshot = get_current_model_info()
    assert snapshot["gpt-5.5"]["xhigh"] is True


def test_get_current_model_info_does_not_mark_gpt_4o_as_xhigh():
    """Pre-5.5 models must remain unflagged so we don't break them."""
    snapshot = get_current_model_info()
    assert snapshot["gpt-4o"]["xhigh"] is False
