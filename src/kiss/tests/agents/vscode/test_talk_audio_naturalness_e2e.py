# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Real speaker→microphone e2e tests for natural ``talk`` delivery.

These tests exercise the WHOLE emotive-speech pipeline with real
audio hardware — no mocks, no synthetic buffers:

1. The REAL production webview (``media/main.js`` running in jsdom
   via node) converts a ``talk`` event into per-sentence utterances
   with emotion-shaped ``rate``/``pitch`` prosody.
2. Each utterance is rendered aloud through the device SPEAKERS with
   the macOS ``say`` engine, mapping the utterance's Web-Speech
   ``rate``/``pitch`` multipliers onto ``[[rate N]]``/``[[pbas N]]``
   embedded speech commands.
3. The device MICROPHONE records the playback (sounddevice /
   PortAudio input stream).
4. The recording is analyzed acoustically (frame RMS energy and an
   autocorrelation pitch tracker) to verify the delivery actually
   *sounds* natural: emotions audibly move the voice pitch, sentence
   boundaries produce audible pauses, and an expressive reply carries
   real pitch movement instead of a flat robotic monotone.

The suite skips itself when the acoustic loop is unavailable (no
input device, not macOS, or the calibration playback is inaudible to
the microphone — muted speakers, headphones, etc.).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[5]
VSCODE_DIR = PROJECT_ROOT / "src" / "kiss" / "agents" / "vscode"

SAMPLE_RATE = 44100
F0_MIN_HZ = 70.0
F0_MAX_HZ = 400.0

# Web Speech rate/pitch are multipliers around 1.0; the macOS ``say``
# engine takes absolute words-per-minute and an absolute pitch base
# (``pbas``, in MIDI-like semitone units).  175 wpm / pbas 47 are the
# engine's ordinary neutral settings.
NEUTRAL_WORDS_PER_MINUTE = 175.0
NEUTRAL_PITCH_BASE = 47.0

# Node driver: runs the REAL production ``media/main.js`` in jsdom
# (exactly like the webview JS tests), dispatches one ``talk`` event,
# and prints the resulting utterances (text/rate/pitch) as JSON.
NODE_PROSODY_DRIVER = r"""
'use strict';
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = process.argv[2];
const language = process.argv[3];
const text = process.argv[4];
const emotion = process.argv[5];

let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  url: 'https://localhost/',
});
const win = dom.window;
win.Element.prototype.scrollIntoView = function () {};
win.Element.prototype.scrollTo = function () {};
win.HTMLElement.prototype.scrollTo = function () {};
win.requestAnimationFrame = function (cb) { cb(); return 0; };
win.acquireVsCodeApi = function () {
  let state;
  return {postMessage: () => {}, getState: () => state,
          setState: s => { state = s; }};
};

const spoken = [];
win.SpeechSynthesisUtterance = function (t) { this.text = t; this.lang = ''; };
win.speechSynthesis = {getVoices: () => [], speak: u => spoken.push(u)};

win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

win.dispatchEvent(new win.MessageEvent('message', {data: {
  type: 'talk', language, text, emotion,
  tabId: win._demoApi.getActiveTabId(),
}}));

console.log(JSON.stringify(spoken.map(
    u => ({text: u.text, rate: u.rate, pitch: u.pitch}))));
"""


def production_talk_prosody(
    text: str, emotion: str = "", language: str = "en-US"
) -> list[dict]:
    """Run *text* through the real webview talk pipeline.

    Executes the production ``media/main.js`` in jsdom via node and
    returns the utterances the webview would hand to the Web Speech
    API: ``[{"text": ..., "rate": ..., "pitch": ...}, ...]``.
    """
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("node binary not found on PATH")
    with tempfile.NamedTemporaryFile(
        "w", suffix=".js", delete=False
    ) as fh:
        fh.write(NODE_PROSODY_DRIVER)
        driver = fh.name
    try:
        proc = subprocess.run(
            [node, driver, str(VSCODE_DIR / "media"), language, text,
             emotion],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=VSCODE_DIR,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
                 "NODE_PATH": str(VSCODE_DIR / "node_modules")},
        )
    finally:
        Path(driver).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise AssertionError(
            f"prosody driver failed: {proc.stderr}\n{proc.stdout}"
        )
    utterances = json.loads(proc.stdout.strip().splitlines()[-1])
    assert isinstance(utterances, list)
    return utterances


def pick_tune_capable_voice() -> str | None:
    """Find a ``say`` voice that honors ``[[pbas]]`` TUNE commands.

    Modern neural macOS voices silently ignore embedded TUNE
    commands, so pitch assertions would measure nothing.  Each
    candidate classic voice is probed offline by synthesizing the
    same words at two pitch bases; a voice that honors ``[[pbas]]``
    produces different audio for the two.  Returns the first such
    voice name, or ``None`` when none is installed.
    """
    for voice in ("Alex", "Fred", "Victoria", "Samantha"):
        with tempfile.TemporaryDirectory() as tmp:
            low = Path(tmp) / "low.aiff"
            high = Path(tmp) / "high.aiff"
            try:
                for pbas, out in ((40, low), (56, high)):
                    subprocess.run(
                        ["say", "-v", voice, "-o", str(out),
                         f"[[pbas {pbas}]] testing one two three"],
                        check=True,
                        capture_output=True,
                        timeout=60,
                    )
            except (subprocess.CalledProcessError,
                    subprocess.TimeoutExpired):
                continue
            if low.read_bytes() != high.read_bytes():
                return voice
    return None


def say_aloud(utterances: list[dict], voice: str) -> None:
    """Speak *utterances* through the speakers with real prosody.

    Maps each utterance's Web-Speech ``rate``/``pitch`` multipliers
    onto the macOS ``say`` engine's embedded ``[[rate N]]`` (words
    per minute) and ``[[pbas N]]`` (pitch base, semitone-like)
    commands, then synthesizes and plays each sentence in order with
    the TUNE-capable *voice*.
    """
    for utt in utterances:
        wpm = round(NEUTRAL_WORDS_PER_MINUTE * float(utt["rate"]))
        pbas = round(NEUTRAL_PITCH_BASE * float(utt["pitch"]))
        pbas = min(65, max(30, pbas))
        subprocess.run(
            ["say", "-v", voice,
             f"[[rate {wpm}]] [[pbas {pbas}]] {utt['text']}"],
            check=True,
            timeout=120,
        )


def record_while(action) -> np.ndarray:
    """Record the default microphone while *action*() runs.

    Returns the mono float32 recording, including ~0.4s of ambient
    lead-in before the action (usable as the noise floor) and ~0.4s
    of tail.
    """
    import sounddevice as sd

    chunks: list[np.ndarray] = []

    def on_audio(indata, _frames, _time, _status) -> None:
        chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, callback=on_audio
    ):
        time.sleep(0.4)
        action()
        time.sleep(0.4)
    return np.concatenate(chunks)[:, 0]


def frame_rms(audio: np.ndarray) -> np.ndarray:
    """Return the per-frame RMS energy track of *audio* (20ms hop)."""
    frame = int(0.04 * SAMPLE_RATE)
    hop = int(0.02 * SAMPLE_RATE)
    out = []
    for start in range(0, len(audio) - frame, hop):
        window = audio[start:start + frame]
        out.append(float(np.sqrt(np.mean(window * window))))
    return np.asarray(out)

def f0_track(audio: np.ndarray) -> np.ndarray:
    """Estimate the voiced-frame pitch track of *audio* in Hz.

    Plain autocorrelation pitch tracker over 40ms frames (20ms hop):
    frames whose energy is below a third of the loud-speech level or
    whose best autocorrelation peak in the 70–400Hz lag range is weak
    (< 0.35 of the zero-lag energy, i.e. unvoiced/noise) are skipped.
    Frames whose best peak lands exactly on the minimum-lag boundary
    are also skipped: a boundary argmax means the true periodicity
    lies at or above F0_MAX (out of the search range) — typical of
    broadband/ambient noise, not trackable voiced speech, whose peak
    is an interior maximum.
    """
    frame = int(0.04 * SAMPLE_RATE)
    hop = int(0.02 * SAMPLE_RATE)
    lag_lo = int(SAMPLE_RATE / F0_MAX_HZ)
    lag_hi = int(SAMPLE_RATE / F0_MIN_HZ)
    energies = frame_rms(audio)
    loud = float(np.percentile(energies, 95))
    f0s = []
    for index, start in enumerate(range(0, len(audio) - frame, hop)):
        if energies[index] < loud / 3.0:
            continue
        window = audio[start:start + frame].astype(np.float64)
        window = window - window.mean()
        auto = np.correlate(window, window, "full")[frame - 1:]
        if auto[0] <= 0:
            continue
        lag = lag_lo + int(np.argmax(auto[lag_lo:lag_hi]))
        if lag == lag_lo:
            continue
        if auto[lag] / auto[0] < 0.35:
            continue
        f0s.append(SAMPLE_RATE / lag)
    return np.asarray(f0s)


def count_pauses(audio: np.ndarray, noise_floor: float) -> int:
    """Count audible pauses (>=150ms of near-silence) inside speech.

    A pause is a run of consecutive quiet frames (RMS below 3x the
    ambient *noise_floor*) at least 150ms long, strictly between the
    first and last loud frames of the recording.
    """
    energies = frame_rms(audio)
    quiet = energies < max(3.0 * noise_floor, 1e-6)
    loud_indexes = np.flatnonzero(~quiet)
    if len(loud_indexes) == 0:
        return 0
    first, last = int(loud_indexes[0]), int(loud_indexes[-1])
    min_frames = int(0.15 / 0.02)
    pauses = 0
    run = 0
    for index in range(first, last + 1):
        if quiet[index]:
            run += 1
        else:
            if run >= min_frames:
                pauses += 1
            run = 0
    return pauses


class TestTalkAudioNaturalnessE2E(unittest.TestCase):
    """Speak through real speakers, record with the real microphone,
    and verify the delivery is audibly natural and emotive."""

    noise_floor: float
    voice: str

    @classmethod
    def setUpClass(cls) -> None:
        """Calibrate the speaker→microphone loop or skip the suite."""
        if sys.platform != "darwin":
            raise unittest.SkipTest("macOS `say` engine required")
        try:
            import sounddevice as sd

            sd.query_devices(kind="input")
        except Exception as exc:  # noqa: BLE001 - any device error
            raise unittest.SkipTest(f"no audio input device: {exc}")
        voice = pick_tune_capable_voice()
        if voice is None:
            raise unittest.SkipTest(
                "no installed `say` voice honors [[pbas]] TUNE "
                "commands, so pitch cannot be verified acoustically"
            )
        cls.voice = voice
        recording = record_while(
            lambda: subprocess.run(
                ["say", "-v", voice,
                 "calibration check one two three"],
                check=True,
                timeout=60,
            )
        )
        lead_in = recording[: int(0.3 * SAMPLE_RATE)]
        cls.noise_floor = float(np.sqrt(np.mean(lead_in * lead_in)))
        speech_rms = float(np.percentile(frame_rms(recording), 95))
        if speech_rms < 3.0 * cls.noise_floor or speech_rms < 1e-4:
            raise unittest.SkipTest(
                "speaker playback is inaudible to the microphone "
                f"(speech RMS {speech_rms:.6f}, "
                f"noise floor {cls.noise_floor:.6f}) — speakers muted "
                "or headphones in use"
            )

    def test_emotion_audibly_moves_the_voice_pitch(self) -> None:
        """'excited' speech is measurably higher-pitched than 'sad'.

        The same sentence goes through the production talk pipeline
        twice — once with emotion 'excited', once with 'sad' — and
        both renditions are played aloud and recorded.  The recorded
        pitch (median f0) of the excited rendition must be clearly
        above the sad one, proving the emotion parameter changes the
        actual sound coming out of the speakers, not just numbers.
        """
        sentence = "The whole test suite is green and the work is done."
        excited = production_talk_prosody(sentence, "excited")
        sad = production_talk_prosody(sentence, "sad")
        self.assertEqual(len(excited), 1)
        self.assertEqual(len(sad), 1)
        self.assertGreater(excited[0]["pitch"], sad[0]["pitch"])

        excited_audio = record_while(
            lambda: say_aloud(excited, self.voice)
        )
        sad_audio = record_while(lambda: say_aloud(sad, self.voice))
        excited_f0 = f0_track(excited_audio)
        sad_f0 = f0_track(sad_audio)
        self.assertGreater(len(excited_f0), 5, "excited speech voiced")
        self.assertGreater(len(sad_f0), 5, "sad speech voiced")
        excited_pitch = float(np.median(excited_f0))
        sad_pitch = float(np.median(sad_f0))
        self.assertGreater(
            excited_pitch,
            sad_pitch * 1.15,
            f"excited median f0 {excited_pitch:.1f}Hz must be clearly "
            f"above sad median f0 {sad_pitch:.1f}Hz",
        )

    def test_expressive_reply_is_not_a_robotic_monotone(self) -> None:
        """A multi-sentence reply has audible pauses and pitch motion.

        An expressive reply (statement, question, exclamation,
        trailing ellipsis) goes through the production pipeline,
        which must split it into several utterances with differing
        prosody.  The recorded playback must contain (a) at least one
        audible inter-sentence pause (a natural breath) and (b) more
        pitch movement (f0 spread) than a flat monotone rendition of
        the same words — i.e. it must not sound robotic.
        """
        text = (
            "Alright, the refactor is finished. "
            "Want me to run the whole suite now? "
            "It only takes a minute! "
            "Then we can relax..."
        )
        utterances = production_talk_prosody(text, "")
        self.assertGreaterEqual(
            len(utterances), 4, "one utterance per sentence"
        )
        rates = {u["rate"] for u in utterances}
        pitches = {u["pitch"] for u in utterances}
        self.assertTrue(
            len(rates) > 1 or len(pitches) > 1,
            "per-sentence prosody must vary across the reply",
        )

        expressive_audio = record_while(
            lambda: say_aloud(utterances, self.voice)
        )
        monotone_words = (
            "Alright the refactor is finished "
            "want me to run the whole suite now "
            "it only takes a minute "
            "then we can relax"
        )
        monotone = [{"text": monotone_words, "rate": 1.0, "pitch": 1.0}]
        monotone_audio = record_while(
            lambda: say_aloud(monotone, self.voice)
        )

        self.assertGreaterEqual(
            count_pauses(expressive_audio, self.noise_floor),
            1,
            "sentence boundaries must be audible as pauses",
        )

        expressive_f0 = f0_track(expressive_audio)
        monotone_f0 = f0_track(monotone_audio)
        self.assertGreater(len(expressive_f0), 10)
        self.assertGreater(len(monotone_f0), 10)
        expressive_spread = float(
            np.percentile(expressive_f0, 90) -
            np.percentile(expressive_f0, 10)
        )
        monotone_spread = float(
            np.percentile(monotone_f0, 90) -
            np.percentile(monotone_f0, 10)
        )
        self.assertGreater(
            expressive_spread,
            monotone_spread,
            f"expressive pitch spread {expressive_spread:.1f}Hz must "
            f"exceed the monotone spread {monotone_spread:.1f}Hz",
        )


if __name__ == "__main__":
    unittest.main()
