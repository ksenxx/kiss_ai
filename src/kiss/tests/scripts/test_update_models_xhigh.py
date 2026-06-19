# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for thinking-level (reasoning_effort) detection in update_models.py.

These verify that the data-driven ``thinking`` field on ``ModelInfo`` is
populated end-to-end by update_models.py:

1. ``_make_entry_line`` emits ``thinking="xhigh"`` in the generated
   ``_mi(...)`` line when the level is set, and omits it otherwise.
2. ``detect_thinking_level`` short-circuits to ``None`` for backends that
   don't accept ``reasoning_effort`` (codex/*, claude-*, gemini-*) and for
   variants known to reject it (``-pro``, ``-chat-latest``, ``-image``) and
   for unrelated prefixes (Together, OpenRouter non-OpenAI), so we never
   waste an API call on them.
3. ``get_current_model_info`` surfaces the ``thinking`` level for every
   model in ``MODEL_INFO`` and matches each model's ``thinking`` field.
"""

from kiss.core.models.model_info import MODEL_INFO
from kiss.scripts.update_models import _make_entry_line, get_current_model_info
from kiss.scripts.update_models import detect_thinking_level as _probe


def test_make_entry_line_emits_thinking_xhigh():
    """A new gpt-5.5-style entry with thinking='xhigh' must include it."""
    line = _make_entry_line(
        "gpt-99",
        ctx=1050000,
        inp=5.00,
        out=30.00,
        thinking="xhigh",
        comment="NEW",
    )
    assert '"gpt-99": _mi(1050000, 5.00, 30.00, thinking="xhigh"),' in line
    assert "# NEW" in line


def test_make_entry_line_omits_thinking_when_none():
    """Without a thinking level the line must not mention thinking at all."""
    line = _make_entry_line(
        "gpt-foo",
        ctx=200000,
        inp=1.00,
        out=2.00,
    )
    assert "thinking" not in line
    assert '"gpt-foo": _mi(200000, 1.00, 2.00),' in line


def test_make_entry_line_combines_thinking_with_other_flags():
    """thinking='xhigh' must compose with fc=False in the right order."""
    line = _make_entry_line(
        "gpt-bar",
        ctx=128000,
        inp=1.00,
        out=2.00,
        fc=False,
        thinking="xhigh",
    )
    assert 'fc=False, thinking="xhigh"' in line


def test_thinking_probe_skipped_for_codex_models():
    """codex/* models route through the Codex CLI, not Chat Completions."""
    assert _probe("codex/gpt-5.5") is None


def test_thinking_probe_skipped_for_claude_models():
    """Anthropic Claude does not accept reasoning_effort."""
    assert _probe("claude-3-5-sonnet-20241022") is None


def test_thinking_probe_skipped_for_gemini_models():
    """Google Gemini does not accept reasoning_effort."""
    assert _probe("gemini-2.5-pro") is None


def test_thinking_probe_skipped_for_pro_variants():
    """-pro variants reject reasoning_effort outright on OpenAI."""
    assert _probe("gpt-5.5-pro") is None


def test_thinking_probe_skipped_for_chat_latest_variants():
    """-chat-latest variants don't accept reasoning_effort."""
    assert _probe("gpt-5-chat-latest") is None


def test_thinking_probe_skipped_for_image_variants():
    """Image variants don't accept reasoning_effort."""
    assert _probe("gpt-image-1") is None


def test_thinking_probe_skipped_for_together_models():
    """Together / non-OpenAI OpenRouter models are not OpenAI Chat Completions."""
    assert _probe("meta-llama/Llama-3.3-70B-Instruct-Turbo") is None
    assert _probe("openrouter/anthropic/claude-sonnet-4.5") is None
    assert _probe("openrouter/google/gemini-2.5-pro") is None


def test_thinking_probe_skipped_for_embedding_models():
    """text-embedding-* models are not generation models and reject the param."""
    assert _probe("text-embedding-3-large") is None


def test_get_current_model_info_exposes_thinking_field():
    """Every model in MODEL_INFO must surface its thinking level verbatim."""
    snapshot = get_current_model_info()
    for name, info in MODEL_INFO.items():
        assert "thinking" in snapshot[name]
        assert snapshot[name]["thinking"] == info.thinking


def test_get_current_model_info_marks_gpt_5_5_thinking_xhigh():
    """gpt-5.5 is the canonical xhigh-capable model and must be flagged."""
    snapshot = get_current_model_info()
    assert snapshot["gpt-5.5"]["thinking"] == "xhigh"


def test_get_current_model_info_gpt_4o_thinking_is_none():
    """Pre-5.5 models must remain unflagged so we don't break them."""
    snapshot = get_current_model_info()
    assert snapshot["gpt-4o"]["thinking"] is None
