// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The mic button's yellow "transcribing" flash is driven by the in-page
// pipeline: a captured round posts its audio and arms a 60 s safety
// timer; the daemon's voiceSpeech reply clears the flash. When the reply
// is lost the timer must clear the stale flash AND reset the round
// counter, otherwise every later utterance leaves the button blinking.
//
// jsdom has no speech engine, audio graph or microphone, so
// voice_page_helpers supplies the window.Vosk / AudioContext /
// mediaDevices the page expects and the tests feed real audio blocks
// through the real capture logic of a bare voice.js window.

const assert = require('assert');
const {makeRunner} = require('./ui_antipattern_harness');
const {send} = require('./simplify2_harness');
const {
  VOICE_CFG,
  voicePageState,
  stubVoiceSurface,
  makeBareVoiceWindow,
  micBtn,
  startPipeline,
  wake,
  speak,
  fireTimer,
} = require('./voice_page_helpers');
const {test, report} = makeRunner();

// Every window built; closed after the run so the armed safety timers
// do not keep node alive and a failed assertion cannot hang the suite.
const windows = [];

/** A bare voice.js window over the stubbed voice surface. */
function makeWindow() {
  const page = voicePageState();
  page.posted = [];
  page.win = makeBareVoiceWindow(VOICE_CFG, win => {
    stubVoiceSurface(win, page);
    win.addEventListener('kiss-voice-post', e => page.posted.push(e.detail));
  });
  windows.push(page.win);
  return page;
}

// Say the wake word, speak for half a second and fall silent for two:
// the round's audio is posted and the yellow transcribing flash is on.
function speakRound(page) {
  wake(page);
  speak(page);
  assert.ok(
    micBtn(page.win).classList.contains('voice-transcribing'),
    `the round must show the yellow transcribing flash: ${micBtn(page.win).className}`,
  );
}

async function main() {
  await test(
    'speech after a lost terminal event must not leave the yellow ' +
      'flash blinking (safety timeout self-heals the round counter)',
    async () => {
      const page = makeWindow();
      const {win, posted} = page;
      const btn = micBtn(win);
      const inp = win.document.getElementById('task-input');

      await startPipeline(page);
      speakRound(page); // round A: audio posted, reply never comes
      assert.strictEqual(
        posted.filter(m => m.type === 'voiceTranscribe').length,
        1,
      );

      fireTimer(page, 60000); // the yellow safety timer
      assert.ok(
        !btn.classList.contains('voice-transcribing'),
        'the safety timeout must clear the stale yellow flash',
      );

      speakRound(page); // round B
      send(win, {type: 'voiceSpeech', text: 'hello world'});

      assert.ok(
        !btn.classList.contains('voice-transcribing'),
        'yellow flash must clear once the spoken text was delivered',
      );
      assert.ok(!btn.classList.contains('voice-triggered'));
      assert.strictEqual(inp.value, 'hello world');
    },
  );

  await test('every later utterance also clears the flash after the self-heal', async () => {
    const page = makeWindow();
    const {win} = page;
    const btn = micBtn(win);

    await startPipeline(page);
    speakRound(page);
    fireTimer(page, 60000);

    for (let round = 1; round <= 3; round++) {
      speakRound(page);
      send(win, {type: 'voiceSpeech', text: `part${round}`});
      assert.ok(
        !btn.classList.contains('voice-transcribing'),
        `round ${round} must clear the yellow flash`,
      );
    }
  });

  await test('overlapping rounds still keep the flash for the newer round', async () => {
    const page = makeWindow();
    const {win} = page;
    const btn = micBtn(win);

    await startPipeline(page);
    speakRound(page);
    speakRound(page);

    send(win, {type: 'voiceSpeech', text: 'first'});
    assert.ok(
      btn.classList.contains('voice-transcribing'),
      'the newer in-flight round still owns the yellow flash',
    );

    send(win, {type: 'voiceSpeech', text: 'second'});
    assert.ok(!btn.classList.contains('voice-transcribing'));
  });

  windows.forEach(win => win.close());
  report('voiceMicFlashStuckYellow');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
