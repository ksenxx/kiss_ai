// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// voice.js: the wake word fired at random during ordinary conversation.
//
// The browser recognizer runs Vosk with the two-word grammar
// ["abracadabra", "[unk]"], so every sound must be labelled one or the
// other.  Measured with the small-en-us model on synthesized speech
// (tmp/wake_partial_probe.py): mid-utterance PARTIAL hypotheses read
// "[unk] abracadabra" for sentences that never contain the word
// ("abstract algebra", "the data is in the dashboard and the branch is
// already merged", "sorry, I could not hear you", "abra"), while the
// FINAL decode of each settles on plain "[unk]".  voice.js used to wake
// on such a partial once the microphone had been quiet for 100 ms
// (default sensitivity 80), with no per-word confidence to check, so any
// pause between words could trigger it.
//
// The fix: only a final result wakes.  These tests drive the real
// chat.html + main.js + voice.js in JSDOM through the fake recognizer of
// voice_page_helpers and check that a wake-looking partial, with or
// without a quiet microphone, never starts a capture, while the final
// result still does.

const assert = require('assert');
const {makeRunner} = require('./ui_antipattern_harness');
const {
  makeVoicePage,
  micBtn,
  startPipeline,
  feed,
  lastProcessor,
  fireWake,
  fireTimer,
} = require('./voice_page_helpers');

const {test, report} = makeRunner();

function lastRecognizer(page) {
  return page.state.recognizers[page.state.recognizers.length - 1];
}

function triggered(page) {
  return micBtn(page.win).classList.contains('voice-triggered');
}

async function main() {
  await test('a wake-looking partial after a quiet mic does not wake', async () => {
    const page = makeVoicePage();
    await startPipeline(page);
    const rec = lastRecognizer(page);
    const proc = lastProcessor(page);
    page.clock.now += 3000; // past the 2 s wake cooldown
    // Speech, then over half a second of silence: the longest pause the
    // old partial gate ever asked for (400 ms at sensitivity 0).
    feed(proc, true, 2);
    feed(proc, false, 3);
    for (const partial of [
      'abracadabra',
      '[unk] abracadabra',
      '[unk] [unk] abracadabra',
    ]) {
      rec.handlers.partialresult({result: {partial}});
      assert.ok(
        !triggered(page),
        `the partial ${JSON.stringify(partial)} must not wake`,
      );
    }
    assert.strictEqual(
      page.posted.filter(m => m.type === 'voiceTranscribe').length,
      0,
      'no capture was started, so nothing is posted for transcription',
    );
  });

  await test('a wake-looking partial while the mic is loud does not wake', async () => {
    const page = makeVoicePage();
    await startPipeline(page);
    const rec = lastRecognizer(page);
    page.clock.now += 3000;
    feed(lastProcessor(page), true, 4);
    rec.handlers.partialresult({result: {partial: '[unk] abracadabra'}});
    assert.ok(!triggered(page), 'a partial mid-speech must not wake');
  });

  await test('a final result still wakes and starts the capture', async () => {
    const page = makeVoicePage();
    await startPipeline(page);
    const rec = lastRecognizer(page);
    // The partials a genuine wake produces on the way to its final
    // result are ignored; the final result is what wakes.
    rec.handlers.partialresult({result: {partial: '[unk] abracadabra'}});
    assert.ok(!triggered(page), 'the partial alone must not wake');
    fireWake(page);
    assert.ok(triggered(page), 'the final result must wake');
    // Half a second of speech and two seconds of silence end the
    // capture: its audio goes to the daemon for transcription.
    const proc = lastProcessor(page);
    feed(proc, true, 2);
    feed(proc, false, 8);
    assert.strictEqual(
      page.posted.filter(m => m.type === 'voiceTranscribe').length,
      1,
      'the woken round must post its audio',
    );
    // Retire the posted round's 60 s safety timer so the suite exits.
    fireTimer(page, 60000);
  });

  await test('an empty final result after the wake is harmless', async () => {
    // The pipeline no longer flushes the recognizer on a wake, but the
    // worker may still deliver an empty final result at any time: it
    // must neither wake nor disturb the capture in progress.
    const page = makeVoicePage();
    await startPipeline(page);
    const rec = lastRecognizer(page);
    rec.handlers.result({result: {text: ''}});
    assert.ok(!triggered(page), 'an empty result must not wake');
    fireWake(page);
    rec.handlers.result({result: {text: ''}});
    assert.ok(triggered(page), 'the capture in progress is kept');
    // Retire the wake's 45 s green-flash timer so the suite exits.
    fireTimer(page, 45000);
  });

  report('voiceWakeIgnoresPartials');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
