# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `harness` module fixture is imported from a
#   sibling test module and reused as a parameter)
"""E2E tests: dictation, the composer's mic (``#dictate-btn``).

A REAL browser (Playwright Chromium) visits the chat page of a REAL
:class:`RemoteAccessServer`, so the dictation code of ``media/voice.js``
runs on the browser's own microphone and audio graph, in the page's own
style sheets.  The browser's microphone is Chromium's fake device: the
user blocks it in the first test, it plays a tone in the second and, in
the third, a Spanish sentence read aloud by the model that speaks for
the ``talk`` tool, which the daemon's own transcription model answers.
A tone has no words, so the second test's transcript is a fixed reply
of a local stand-in for the transcription API.

What a real microphone cannot be made to do on demand (a permission
prompt nobody answers, a device unplugged in mid-sentence, a transcript
that never comes) is driven with a scripted microphone in
``src/kiss/webapp/test/dictation.test.js``.
"""

from __future__ import annotations

import base64
import io
import re
import wave
from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright

from kiss.core.speech_synthesis import (
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    TTS_SYSTEM_PROMPT,
    TalkSynthesisAgent,
    audio_timeout_seconds,
)
from kiss.server.speech import SAMPLE_RATE
from kiss.tests.conftest import requires_openai_api_key
from kiss.tests.server.test_explorer_scm_commands import (
    harness,  # noqa: F401  (module fixture used by param name)
)
from kiss.tests.server.test_voice_dual_check import CannedOpenAiServer, _tone_pcm
from kiss.tests.webapp.test_workspace_sections import _open_page

# A statement: handed a request or a question, the reader may answer it
# instead of reading it aloud.  The transcriber may mishear one word of
# a synthetic voice, so two of the three English words are asked for.
SPANISH = "Mañana voy a comprar pan y leche."
ENGLISH_WORDS = ("tomorrow", "bread", "milk")
DRAFT = "Translate: [END]"
CURSOR = len("Translate: ")

# The colour of the mic's icon, and the colour that the palette's token
# (`--error`, `--red`, ...) stands for where the mic is.
_MIC_COLOUR_JS = """token => {
  const probe = document.createElement('span');
  probe.style.color = `var(${token})`;
  document.body.appendChild(probe);
  const expected = getComputedStyle(probe).color;
  probe.remove();
  return [getComputedStyle(document.getElementById('dictate-btn')).color, expected];
}"""

# Keep what the mic looked like each time its state changed: its classes
# and the colour of its icon.
_WATCH_MIC_JS = """() => {
  const mic = document.getElementById('dictate-btn');
  window.__micStates = [];
  new MutationObserver(() => {
    window.__micStates.push([mic.className, getComputedStyle(mic).color]);
  }).observe(mic, {attributes: true, attributeFilter: ['class']});
}"""


@pytest.fixture(scope="module")
def chromium():
    """The Chromium launcher of one Playwright for every test of this module."""
    with sync_playwright() as playwright:
        yield playwright.chromium


def _launch(chromium, microphone: str):
    """Launch a headless Chromium with a microphone.

    *microphone* is the answer to the page's request for it: ``deny``,
    or the path of the WAV file that the microphone plays once, from
    the moment it opens, before it goes silent.
    """
    answer = "=deny" if microphone == "deny" else ""
    plays = [] if answer else [f"--use-file-for-fake-audio-capture={microphone}%noloop"]
    return chromium.launch(
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--use-fake-device-for-media-stream",
            f"--use-fake-ui-for-media-stream{answer}",
            "--mute-audio",
            *plays,
        ],
    )


def _mic_wav(path: Path, pcm: bytes, rate: int) -> None:
    """Write the WAV at *path*: *pcm* (mono, 16-bit) with silence around it.

    One second before it and four after: past the end of its file the
    fake microphone goes on repeating the last few milliseconds, which
    must be silent.
    """
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(bytes(2 * rate) + pcm + bytes(8 * rate))


def _read_aloud(script: str) -> tuple[bytes, int]:
    """Return *script* read aloud, as mono 16-bit PCM and its sample rate.

    The reader is the model that speaks for the ``talk`` tool.
    """
    agent = TalkSynthesisAgent("dictation-test-speech")
    agent.run(
        model_name=DEFAULT_TTS_MODEL,
        prompt_template="Script:\n{script}",
        arguments={"script": script},
        system_prompt=TTS_SYSTEM_PROMPT.format(tone="in a clear, natural tone"),
        is_agentic=False,
        model_config={
            "modalities": ["text", "audio"],
            "audio": {"voice": DEFAULT_TTS_VOICE, "format": "wav"},
            "timeout": audio_timeout_seconds(),
        },
        verbose=False,
    )
    reading = base64.b64decode(getattr(agent.model, "last_audio_data", None) or "")
    with wave.open(io.BytesIO(reading), "rb") as spoken:
        assert (spoken.getnchannels(), spoken.getsampwidth()) == (1, 2)
        return spoken.readframes(spoken.getnframes()), spoken.getframerate()


def _mic_colours(page, state: str) -> list[str]:
    """Return the colours the watched mic had while its classes held *state*."""
    return [
        colour
        for classes, colour in page.evaluate("window.__micStates")
        if state in classes.split()
    ]


def test_a_blocked_microphone_turns_the_mic_off_in_the_error_colour(chromium, harness):
    """The user has blocked the microphone for the site.

    The mic goes back to off, says why in its tooltip and in a
    notification, and shows the palette's error colour with the pointer
    on it and away from it.  The remote skin's colours for the
    composer's buttons used to outrank the error colour, so a failed
    mic looked like one never clicked.
    """
    browser = _launch(chromium, "deny")
    try:
        _context, page, sent = _open_page(browser, harness)
        mic = page.locator("#dictate-btn")
        mic.click()
        expect(mic).to_have_class(re.compile(r"\bdictate-error\b"))
        expect(mic).to_have_attribute("aria-pressed", "false")
        expect(mic).to_have_attribute(
            "data-tooltip",
            re.compile(r"^Dictation error: Microphone access was denied\."),
        )
        expect(page.locator(".kiss-notification")).to_contain_text(
            "Microphone access was denied.",
        )
        mic_colour, error = page.evaluate(_MIC_COLOUR_JS, "--error")
        assert mic_colour == error, "under the pointer"
        page.mouse.move(5, 300)
        mic_colour, error = page.evaluate(_MIC_COLOUR_JS, "--error")
        assert mic_colour == error, "with the pointer away"
        assert not [f for f in sent if f.get("type") == "voiceTranscribe"]
    finally:
        browser.close()


def test_the_mic_turned_off_in_mid_sentence_types_what_it_heard(
    chromium, harness, tmp_path, monkeypatch,
):
    """The mic is turned off while a sound still comes in.

    The mic is red while it hears the sound.  What it heard up to the
    click goes to the daemon, marked as dictation, and the mic (off,
    with the pointer still on it) is amber until the transcript is
    back, which is typed although the mic is off.
    """
    canned = CannedOpenAiServer("en\nOpen the kitchen window.")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{canned.port}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-kiss-dictation-test")
    _mic_wav(tmp_path / "tone.wav", _tone_pcm(20.0), SAMPLE_RATE)
    browser = _launch(chromium, str(tmp_path / "tone.wav"))
    try:
        _context, page, sent = _open_page(browser, harness)
        page.evaluate(_WATCH_MIC_JS)
        mic = page.locator("#dictate-btn")
        mic.click()
        expect(mic).to_have_class(re.compile(r"\bdictate-hearing\b"), timeout=30_000)
        page.wait_for_timeout(1500)
        mic.click()
        expect(mic).to_have_attribute("aria-pressed", "false")
        expect(page.locator("#task-input")).to_have_value(
            "Open the kitchen window.", timeout=120_000,
        )
        expect(mic).to_have_class("toggle-btn")

        _mic, red = page.evaluate(_MIC_COLOUR_JS, "--red")
        _mic, amber = page.evaluate(_MIC_COLOUR_JS, "--warning")
        assert red != amber
        assert set(_mic_colours(page, "dictate-hearing")) == {red}
        states = page.evaluate("window.__micStates")
        off = [(classes, colour) for classes, colour in states if "active" not in classes.split()]
        assert ("toggle-btn dictate-transcribing", amber) in off, off

        (request,) = [f for f in sent if f.get("type") == "voiceTranscribe"]
        assert request["dictation"] is True and "wakePrefixed" not in request
        heard_seconds = len(base64.b64decode(request["audio"])) / (2 * SAMPLE_RATE)
        assert 0.5 < heard_seconds < 15, "cut at the click, not at the tone's end"
        assert len(canned.request_bodies) == 1, "one request to the transcription API"
    finally:
        browser.close()
        canned.close()


@pytest.mark.live_api
@requires_openai_api_key
def test_spoken_spanish_is_typed_in_english_at_the_cursor(chromium, harness, tmp_path):
    """Someone turns the mic on and says a Spanish sentence.

    No wake word is said.  The pause after the sentence ends it; the mic
    is amber while the transcript is on its way.  The sentence then
    stands in English where the cursor was, with the text around it
    kept, and is not sent.
    """
    speech, rate = _read_aloud(SPANISH)
    spoken_seconds = len(speech) / (2 * rate)
    _mic_wav(tmp_path / "speech.wav", speech, rate)
    browser = _launch(chromium, str(tmp_path / "speech.wav"))
    try:
        _context, page, sent = _open_page(browser, harness)
        page.fill("#task-input", DRAFT)
        page.evaluate(
            "at => document.getElementById('task-input').setSelectionRange(at, at)",
            CURSOR,
        )
        page.evaluate(_WATCH_MIC_JS)
        mic = page.locator("#dictate-btn")
        mic.click()
        expect(mic).to_have_attribute("aria-pressed", "true")
        page.wait_for_function(
            "draft => document.getElementById('task-input').value !== draft",
            arg=DRAFT,
            timeout=120_000,
        )
        typed = page.input_value("#task-input")
        rest = " " + DRAFT[CURSOR:]
        assert typed.startswith(DRAFT[:CURSOR]) and typed.endswith(rest), typed
        english = typed[CURSOR : len(typed) - len(rest)]
        found = [word for word in ENGLISH_WORDS if re.search(rf"\b{word}\b", english, re.I)]
        assert len(found) >= 2, english
        assert not re.search(r"mañana|comprar|leche", english, re.I), english
        assert page.evaluate(
            "document.getElementById('task-input').selectionStart",
        ) == CURSOR + len(english), "the cursor follows what was typed"

        # One utterance went to the daemon: the sentence, from its first
        # sound to the pause that ended it.
        (request,) = [f for f in sent if f.get("type") == "voiceTranscribe"]
        assert request["dictation"] is True and "wakePrefixed" not in request
        heard_seconds = len(base64.b64decode(request["audio"])) / (2 * SAMPLE_RATE)
        assert spoken_seconds < heard_seconds < spoken_seconds + 4, heard_seconds
        assert not [f for f in sent if f.get("type") == "submit"]
        _mic, amber = page.evaluate(_MIC_COLOUR_JS, "--warning")
        assert amber in _mic_colours(page, "dictate-transcribing")

        mic.click()
        expect(mic).to_have_attribute("aria-pressed", "false")
        expect(mic).not_to_have_class(re.compile(r"\bactive\b"))
        assert page.input_value("#task-input") == typed
    finally:
        browser.close()
