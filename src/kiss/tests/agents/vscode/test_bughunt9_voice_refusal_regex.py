# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests for the STT refusal-hallucination detector.

``looks_like_stt_refusal`` guards dictation against gpt-audio replies
that ANSWER the transcription prompt instead of transcribing ("Please
provide the audio, and I will transcribe and translate it
accordingly.").  Such refusals drop the two-line format, so they carry
no language tag; only tag-less replies are ever inspected.

Bug reproduced here: the alternation
``\\bi\\s+(?:will|can|'ll)\\s+(?:transcribe|translate)\\b`` in
``_STT_REFUSAL_RE`` required whitespace between "i" and "'ll", so the
contraction form "I'll transcribe the audio for you." was NEVER
flagged and got forwarded verbatim as the user's dictated command,
while the long form "I will transcribe..." was caught.  The fix moves
the whitespace inside the long-form branches:
``\\bi(?:\\s+will|\\s+can|'ll)\\s+(?:transcribe|translate)\\b``.
"""

from __future__ import annotations

from kiss.server.voice_wake import looks_like_stt_refusal


def test_contraction_ill_transcribe_is_flagged() -> None:
    """"I'll transcribe ..." (no language tag) must be a refusal.

    This is the reproduction: pre-fix the regex demanded whitespace
    between "i" and "'ll", so this hallucination sailed through and
    was submitted as the user's dictated command.
    """
    assert looks_like_stt_refusal("I'll transcribe the audio for you.", None)


def test_contraction_ill_translate_is_flagged() -> None:
    """The translate variant of the contraction must also be caught."""
    assert looks_like_stt_refusal("I'll translate the recording now.", None)


def test_contraction_case_insensitive() -> None:
    """Mixed-case contractions must be caught (regex is IGNORECASE)."""
    assert looks_like_stt_refusal("i'll Transcribe it for you.", None)


def test_long_form_i_will_transcribe_still_flagged() -> None:
    """Regression guard: the long form kept matching before the fix."""
    assert looks_like_stt_refusal("I will transcribe the audio for you.", None)


def test_long_form_i_can_transcribe_still_flagged() -> None:
    """Regression guard: "I can transcribe" kept matching before the fix."""
    assert looks_like_stt_refusal(
        "I can transcribe it once you provide the audio.", None
    )


def test_provide_audio_refusal_still_flagged() -> None:
    """Regression guard: the canonical observed refusal keeps matching."""
    assert looks_like_stt_refusal(
        "Please provide the audio, and I will transcribe and "
        "translate it accordingly.",
        None,
    )


def test_language_tag_never_flagged() -> None:
    """A reply WITH a language tag is genuine dictation, never a refusal."""
    assert not looks_like_stt_refusal(
        "I'll transcribe the audio for you.", "en"
    )
    assert not looks_like_stt_refusal(
        "I will transcribe the audio for you.", "en"
    )


def test_genuine_dictation_about_audio_not_flagged() -> None:
    """Dictation that merely mentions audio must not be swallowed."""
    assert not looks_like_stt_refusal(
        "please provide the audio files to the team by friday", None
    )


def test_ill_without_stt_verb_not_flagged() -> None:
    """"I'll" followed by a non-STT verb must not trip the detector."""
    assert not looks_like_stt_refusal(
        "I'll take care of the deployment tomorrow.", None
    )
