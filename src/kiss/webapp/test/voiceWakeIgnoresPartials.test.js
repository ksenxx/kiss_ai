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
//
// 2026-11-08: two wakes fired on Russian and Bengali background talk.
// Replayed through Vosk, the grammar recognizer FINALS on "abracadabra"
// for both (conf 0.53-1.0), while a grammar-free recognizer fed the same
// audio heard "which can remember" and "government".  A wake now needs
// both recognizers' finals to contain the word within 2.5 s of each
// other; the tests at the end pin that agreement rule.
//
// 2026-11-08 (later): the wake phrase became "Hey Sorcar".  Its grammar
// spells the name as "sore car" / "sir car" / "sar car" (Sorcar is not
// in the model's vocabulary).  The grammar-free recognizer cannot spell
// the name ("hey soccer", "hey saga", "hey saw car", "he's order"), so its
// final confirms the wake when it heard the word "hey" or ends in words
// within 3 edits of "heysorcar".

const assert = require('assert');
const {makeRunner} = require('./ui_antipattern_harness');
const {twoTabs, clickTab} = require('./simplify2_harness');
const {
  makeVoicePage: makeHelperPage,
  micBtn,
  startPipeline,
  feed,
  lastProcessor,
  lastRecognizer,
  lastFreeRecognizer,
  fireWake,
  wake,
  speak,
} = require('./voice_page_helpers');

const {test, report} = makeRunner();

// Every page built; closed after the run so the pipeline's armed safety
// timers (45 s wake flash, 60 s transcribing flash) do not keep node
// alive and a failed assertion cannot hang the suite.
const windows = [];

function makeVoicePage() {
  const page = makeHelperPage();
  windows.push(page.win);
  return page;
}

function triggered(page) {
  return micBtn(page.win).classList.contains('voice-triggered');
}

/** Deliver a wake-word final from the grammar recognizer, or the free one. */
function wakeFinal(page, fromGrammar) {
  const rec = fromGrammar ? lastRecognizer(page) : lastFreeRecognizer(page);
  rec.handlers.result({result: {text: fromGrammar ? 'hey sore car' : 'hey soccer'}});
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
      'hey sore car',
      '[unk] hey sore car',
      '[unk] [unk] hey sore car',
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
    rec.handlers.partialresult({result: {partial: '[unk] hey sore car'}});
    assert.ok(!triggered(page), 'a partial mid-speech must not wake');
  });

  await test('a final result still wakes and starts the capture', async () => {
    const page = makeVoicePage();
    await startPipeline(page);
    const rec = lastRecognizer(page);
    // The partials a genuine wake produces on the way to its final
    // result are ignored; the final result is what wakes.
    rec.handlers.partialresult({result: {partial: '[unk] hey sore car'}});
    assert.ok(!triggered(page), 'the partial alone must not wake');
    wake(page); // the final result wakes
    // Half a second of speech and two seconds of silence end the
    // capture: its audio goes to the daemon for transcription.
    speak(page);
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
  });

  await test('a grammar final alone does not wake; the free one confirms it', async () => {
    const page = makeVoicePage();
    await startPipeline(page);
    page.clock.now += 3000;
    lastRecognizer(page).handlers.result({result: {text: 'hey sore car'}});
    assert.ok(!triggered(page), 'the grammar recognizer alone must not wake');
    page.clock.now += 260; // one 4096-sample block later
    lastFreeRecognizer(page).handlers.result({
      result: {text: 'which can remember'},
    });
    assert.ok(!triggered(page), 'a free final without "hey" must not wake');
    lastFreeRecognizer(page).handlers.result({
      result: {text: 'they saw a car'},
    });
    assert.ok(!triggered(page), '"they" is not "hey"');
    lastFreeRecognizer(page).handlers.result({result: {text: "he's sorry"}});
    assert.ok(!triggered(page), '"he\'s sorry" is 4 edits from "heysorcar"');
    // What the free recognizer heard for "Hey Sorcar" played through
    // this Mac's speaker into its microphone: 3 edits away.
    lastFreeRecognizer(page).handlers.result({
      result: {text: "open it he's order"},
    });
    assert.ok(triggered(page), 'both recognizers agree: the page wakes');
    // A free result during the capture is ignored.
    lastFreeRecognizer(page).handlers.result({result: {text: 'hey soccer'}});
    speak(page);
  });

  await test('the free final may come first; stale agreement does not wake', async () => {
    const page = makeVoicePage();
    await startPipeline(page);
    page.clock.now += 3000;
    lastFreeRecognizer(page).handlers.result({
      result: {text: 'okay hey soccer'},
    });
    assert.ok(!triggered(page), 'the free recognizer alone must not wake');
    page.clock.now += 2600; // past the 2.5 s agreement window
    lastRecognizer(page).handlers.result({result: {text: 'hey sore car'}});
    assert.ok(!triggered(page), 'finals 2.6 s apart are not one wake');
    page.clock.now += 1000;
    lastFreeRecognizer(page).handlers.result({result: {text: 'hey soccer'}});
    assert.ok(triggered(page), 'a free final 1 s after the grammar one wakes');
  });

  await test('a command spoken before the second recognizer confirms the wake is posted', async () => {
    const page = makeVoicePage();
    await startPipeline(page);
    const proc = lastProcessor(page);
    page.clock.now += 3000;
    feed(proc, true, 4); // ~1 s of "hey sorcar"
    feed(proc, false, 4); // the endpointing silence
    wakeFinal(page, true); // the grammar recognizer hears it first
    // The user goes straight on: 0.77 s of command, then silence, all
    // before the free recognizer's final lands 1.5 s later.
    for (let i = 0; i < 6; i++) {
      page.clock.now += 256;
      feed(proc, i < 3, 1);
    }
    wakeFinal(page, false);
    assert.ok(triggered(page), 'the confirming final within 2.5 s wakes');
    feed(proc, false, 20); // nothing more is said
    const sent = page.posted.filter(m => m.type === 'voiceTranscribe');
    assert.strictEqual(
      sent.length,
      1,
      'the command spoken before the confirmation is posted',
    );
    assert.strictEqual(
      sent[0].wakeSamples,
      8 * 4096,
      'the preamble ends where the wake word was first heard',
    );
    const buf = Buffer.from(sent[0].audio, 'base64');
    const pcm = new Int16Array(buf.buffer, buf.byteOffset, buf.length / 2);
    let commandSamples = 0;
    for (let i = sent[0].wakeSamples; i < pcm.length; i++) {
      if (pcm[i] > 10000) commandSamples++;
    }
    assert.strictEqual(
      commandSamples,
      3 * 4096,
      'the command follows the preamble, where speaker identification hears it',
    );
  });

  await test('a wake word heard on one chat is not confirmed onto another', async () => {
    const page = makeVoicePage();
    const {first, second} = twoTabs(page.win);
    clickTab(page.win, first);
    await startPipeline(page);
    page.clock.now += 3000;
    wakeFinal(page, true); // heard while chat `first` is on screen
    clickTab(page.win, second); // the user moves on before the free final
    page.clock.now += 256;
    wakeFinal(page, false);
    assert.ok(
      !triggered(page),
      'words heard on one chat must not open a round on another',
    );
    // Said again on the chat now on screen, it takes both recognizers anew.
    page.clock.now += 256;
    wakeFinal(page, true);
    assert.ok(!triggered(page), 'the refused final confirms nothing later');
    page.clock.now += 256;
    wakeFinal(page, false);
    assert.ok(triggered(page), 'the wake word said on this chat wakes it');
  });

  // Whichever recognizer hears the word first, the audio the ring held
  // then is kept until the other confirms; an earlier final of the
  // confirming recognizer that is stale by then (2.6 s old) plays no part.
  async function lateConfirmation(grammarFirst) {
    const page = makeVoicePage();
    await startPipeline(page);
    const proc = lastProcessor(page);
    page.clock.now += 3000;
    wakeFinal(page, !grammarFirst);
    page.clock.now += 2600; // past the 2.5 s agreement window
    feed(proc, true, 4); // ~1 s of "hey sorcar"
    const out = feed(proc, false, 4); // the endpointing silence
    // Chrome swaps a sink that hears 30 s of exact zeros for a fake one
    // that runs the processor at under half speed, so the processor
    // always writes an inaudible non-zero level.
    assert.ok(
      out.every(v => v !== 0 && Math.abs(v) < 1e-6),
      'the processor output is never exact silence and never audible',
    );
    wakeFinal(page, grammarFirst);
    assert.ok(!triggered(page), 'the stale final does not confirm');
    for (let i = 0; i < 6; i++) {
      page.clock.now += 256; // 1.5 s more before the confirming final
      feed(proc, false, 1);
    }
    wakeFinal(page, !grammarFirst);
    assert.ok(triggered(page), 'the confirming final within 2.5 s wakes');
    speak(page);
    const sent = page.posted.find(m => m.type === 'voiceTranscribe');
    assert.strictEqual(sent.wakeAgreed, true, 'the server learns both recognizers agreed');
    const buf = Buffer.from(sent.audio, 'base64');
    const pcm = new Int16Array(buf.buffer, buf.byteOffset, buf.length / 2);
    let wordSamples = 0;
    for (let i = 0; i < sent.wakeSamples; i++) if (pcm[i] > 10000) wordSamples++;
    assert.strictEqual(
      wordSamples,
      4 * 4096,
      'the audio of the word heard first is the preamble of the confirmed wake',
    );
  }
  await test('a late confirmation still ships the wake word audio', () =>
    lateConfirmation(true));
  await test('so does a late confirmation of a free final that came first', () =>
    lateConfirmation(false));

  windows.forEach(win => win.close());
  report('voiceWakeIgnoresPartials');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
