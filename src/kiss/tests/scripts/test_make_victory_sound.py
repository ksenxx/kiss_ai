# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The victory jingle renderer (``kiss.scripts.make_victory_sound``).

``render()`` is checked on the signal it produces; ``main()``'s
encoding branch needs the ``lameenc`` extension, which is a one-off
tool of the script and not installed in the project environment, so
only its refusal is exercised here (a stub module in ``sys.modules``
would be a test double).  The clip the page ships,
``media/victory.mp3``, is checked by ``test_media_cache_bust``.
"""

from __future__ import annotations

import importlib.util
import math

import pytest

from kiss.scripts import make_victory_sound as jingle


def _rms(samples: list[float]) -> float:
    return math.sqrt(sum(x * x for x in samples) / len(samples))


def test_render_is_a_short_articulated_clip_at_the_notification_level() -> None:
    """The render is a ~0.5 s clip of finite audio peaking at PEAK, silent
    before the first note, and the arpeggio's notes are separated by real
    gaps."""
    samples = jingle.render()
    last_note_end = max(start + length for _, start, length, _ in jingle.NOTES)
    assert len(samples) == int(jingle.RATE * (last_note_end + 0.2))
    assert len(samples) / jingle.RATE < 0.6
    assert all(math.isfinite(x) for x in samples)
    assert max(abs(x) for x in samples) == pytest.approx(jingle.PEAK, abs=1e-9)
    assert samples[0] == 0.0
    sec = jingle.RATE
    # The first note (0-50 ms) sounds; the end of the 30 ms gap before
    # the second note (50-80 ms) is at least 9 dB quieter than the
    # note's body, so the arpeggio is heard as separate notes.
    body = _rms(samples[int(0.015 * sec) : int(0.045 * sec)])
    gap = _rms(samples[int(0.07 * sec) : int(0.08 * sec)])
    assert body > 0.1
    assert gap < body / 3
    # The closing chord (160-340 ms) sounds, then the tail decays to nothing.
    chord = _rms(samples[int(0.2 * sec) : int(0.3 * sec)])
    tail = _rms(samples[-int(0.05 * sec) :])
    assert chord > 0.1
    assert tail < chord / 20


def test_envelope_shapes_attack_sustain_and_release() -> None:
    """Attack ramps from 0 over 12 ms, the body sits near the sustain, a
    short note stops within tens of ms of its end and the chord rings on."""
    assert jingle.envelope(0.0, 0.1) == 0.0
    assert jingle.envelope(0.006, 0.1) == pytest.approx(0.5, abs=0.01)
    assert 0.85 <= jingle.envelope(0.05, 0.1) <= 1.0
    assert jingle.envelope(0.1 + 0.06, 0.1) < 0.05
    assert jingle.envelope(0.18 + 0.02, 0.18) > 0.5


def test_tone_has_the_fundamental_at_a_fixed_pitch() -> None:
    """A tone starts at zero phase and repeats one period later."""
    assert jingle.tone(440.0, 0.0) == 0.0
    assert jingle.tone(440.0, 1 / 440.0) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(
    importlib.util.find_spec("lameenc") is not None,
    reason="lameenc is installed here: the refusal branch cannot be reached",
)
def test_main_refuses_without_the_encoder(capsys: pytest.CaptureFixture[str]) -> None:
    """Without lameenc the script names the command to run and writes nothing."""
    before = jingle.OUT.read_bytes()
    assert jingle.main() == 1
    assert "uv run --with lameenc" in capsys.readouterr().err
    assert jingle.OUT.read_bytes() == before
