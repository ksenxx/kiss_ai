# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Render the webapp's victory jingle, ``media/victory.mp3``.

The chat page plays the clip when a task finishes (``playVictorySound``
in ``media/main.js``).  It is synthesized here rather than recorded: a
half-second brass-like flourish, two quick notes up a C-major arpeggio
landing on a brief chord, so it carries no licence and can be
re-rendered with other notes or another timbre by editing :data:`NOTES` and :func:`tone`.
Run from the repo root::

    uv run --with lameenc python -m kiss.scripts.make_victory_sound

``lameenc`` (the LAME mp3 encoder) is a one-off tool of this script,
not a project dependency.  The render is deterministic, so the same
source gives the same bytes.
"""

from __future__ import annotations

import array
import importlib
import math
import sys
from pathlib import Path

RATE = 44100
OUT = Path(__file__).resolve().parents[1] / "webapp" / "media" / "victory.mp3"

C4, E4, C5, E5, G5, C6 = 261.63, 329.63, 523.25, 659.25, 783.99, 1046.5

# (frequency, start, length, level): the notes in seconds.  Two quick
# steps up the arpeggio, then the fifth over a C-major chord with an
# octave "sparkle" on top.
NOTES: list[tuple[float, float, float, float]] = [
    (C5, 0.00, 0.05, 0.9),
    (E5, 0.08, 0.05, 0.9),
    (G5, 0.16, 0.18, 1.0),
    (C4, 0.16, 0.18, 0.45),
    (E4, 0.16, 0.18, 0.35),
    (C5, 0.16, 0.18, 0.30),
    (C6, 0.17, 0.17, 0.12),
]

# Peak level of the written clip: a notification, not a fanfare at full
# scale (the "working on it" ack peaks at 0.55).
PEAK = 0.6

HARMONICS = 8


def envelope(t: float, length: float) -> float:
    """Return the amplitude of a note *t* seconds after its onset.

    A 12 ms attack, a slight settle into the sustain, and an exponential
    release past *length*: the closing chord rings briefly, the quick
    notes stop almost at once so the arpeggio stays crisp.

    Args:
        t: Seconds since the note started.
        length: The note's sounding length in seconds.
    """
    attack = min(1.0, t / 0.012)
    settle = 0.85 + 0.15 * math.exp(-t / 0.06)
    tau = 0.04 if length >= 0.15 else 0.015
    release = 1.0 if t <= length else math.exp(-(t - length) / tau)
    return attack * settle * release


def tone(freq: float, t: float) -> float:
    """Return one sample of a brass-like tone at *freq*, *t* seconds in.

    Eight harmonics rolling off as ``1/n^1.3`` (a softened sawtooth).

    Args:
        freq: The fundamental in Hz.
        t: Seconds since the note started.
    """
    phase = 2 * math.pi * freq * t
    return sum(math.sin(n * phase) / math.pow(n, 1.3) for n in range(1, HARMONICS + 1))


def render() -> list[float]:
    """Return the whole jingle as float samples peaking at :data:`PEAK`, at :data:`RATE`."""
    tail = 0.2
    total = int(RATE * (max(s + ln for _, s, ln, _ in NOTES) + tail))
    dry = [0.0] * total
    for freq, start, length, level in NOTES:
        first = int(start * RATE)
        last = min(total, int((start + length + tail) * RATE))
        for i in range(first, last):
            t = (i - first) / RATE
            dry[i] += level * envelope(t, length) * tone(freq, t)
    # A touch of room: two early reflections and one feedback comb.
    comb = int(0.053 * RATE)
    early = [(int(0.031 * RATE), 0.5), (int(0.047 * RATE), 0.35)]
    wet = [0.0] * total
    for i in range(total):
        wet[i] = dry[i] + (0.3 * wet[i - comb] if i >= comb else 0.0)
    out = []
    for i in range(total):
        echo = sum(g * dry[i - d] for d, g in early if i >= d)
        out.append(dry[i] + 0.15 * (wet[i] - dry[i]) + 0.25 * echo)
    peak = max(abs(x) for x in out)
    return [PEAK * math.tanh(1.2 * x / peak) / math.tanh(1.2) for x in out]


def main() -> int:
    """Render the jingle and write it to :data:`OUT` as a 64 kbps mono mp3."""
    try:
        lameenc = importlib.import_module("lameenc")
    except ImportError:
        print("lameenc is not installed: run with `uv run --with lameenc`", file=sys.stderr)
        return 1
    pcm = array.array("h", (int(round(x * 32000)) for x in render()))
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(64)
    encoder.set_in_sample_rate(RATE)
    encoder.set_channels(1)
    encoder.set_quality(2)
    data = bytes(encoder.encode(pcm.tobytes())) + bytes(encoder.flush())
    OUT.write_bytes(data)
    print(f"wrote {OUT} ({len(data)} bytes, {len(pcm) / RATE:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
