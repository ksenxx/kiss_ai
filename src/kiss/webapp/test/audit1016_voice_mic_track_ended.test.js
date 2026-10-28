// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// voice.js: the microphone can go away under a live pipeline (the
// headset is unplugged, the OS or another app takes the device, the
// site's permission is revoked).  The browser then fires `ended` on the
// audio track and delivers no more audio.  Before the fix nothing
// listened for that: the button kept saying "listening", every later
// click deferred to a pipeline that was dead, and the audio context and
// recogniser stayed allocated for the page lifetime.  The pipeline must
// come down like a failed start does: the mic turns off, the error
// explains why, and the next click asks for the microphone again.
//
// Harness: voice_page_helpers supplies the page's embedder surface
// (window.Vosk / AudioContext / mediaDevices with tracks the test can
// end); the real voice.js, main.js and DOM run.

const assert = require('assert');
const {
  makeVoicePage,
  micBtn,
  settle,
  startPipeline,
  fireWake,
} = require('./voice_page_helpers');

async function main() {
  const page = makeVoicePage({autoGrant: true});
  const {win, log, state} = page;
  const btn = micBtn(win);
  await startPipeline(page);
  assert.strictEqual(state.tracks.length, 1);
  assert.deepStrictEqual(log, ['ctx-open']);

  // The microphone goes away under the live pipeline.
  state.tracks[0].end();
  await settle();
  assert.ok(
    !btn.classList.contains('voice-listening'),
    'a pipeline whose microphone ended must not claim to listen',
  );
  assert.ok(
    btn.classList.contains('voice-error'),
    'the button shows the error state, got ' + btn.className,
  );
  assert.ok(
    /microphone/i.test(btn.getAttribute('data-tooltip') || ''),
    'the tooltip explains the microphone went away: ' +
      btn.getAttribute('data-tooltip'),
  );
  assert.deepStrictEqual(
    log,
    ['ctx-open', 'track-stopped', 'ctx-closed'],
    'the audio graph is released',
  );
  assert.strictEqual(
    win.localStorage.getItem('kissVoiceEnabled'),
    '0',
    'the trigger is remembered as off, like after a failed start',
  );

  // The next click asks for the microphone again and comes back live.
  await startPipeline(page);
  assert.strictEqual(state.tracks.length, 2, 'a fresh microphone request');
  assert.deepStrictEqual(log, [
    'ctx-open',
    'track-stopped',
    'ctx-closed',
    'ctx-open',
  ]);

  // Audit 2026-10-16 (webapp-other F1): the microphone goes away in the
  // middle of a wake-word capture.  The "Listening ..." overlay and the
  // red flash the wake turned on (for 45 s) must come down with the
  // pipeline, as they do when the user turns the mic off.
  const wrap = win.document.getElementById('input-text-wrap');
  fireWake(page);
  assert.ok(wrap.classList.contains('listening'), 'overlay on after wake');
  assert.ok(btn.classList.contains('voice-triggered'), 'flash on after wake');
  state.tracks[1].end();
  await settle();
  assert.ok(btn.classList.contains('voice-error'), 'error state shown');
  assert.ok(
    !wrap.classList.contains('listening'),
    'the Listening overlay is gone with the microphone',
  );
  assert.ok(
    !btn.classList.contains('voice-triggered'),
    'the wake flash is gone with the microphone: ' + btn.className,
  );
  await startPipeline(page);
  assert.strictEqual(state.tracks.length, 3, 'a fresh microphone request');

  // An `ended` from the track of a pipeline that was already torn down
  // (the user turned the mic off first) changes nothing.
  btn.click();
  await settle();
  assert.ok(btn.classList.contains('voice-off'));
  const after = log.slice();
  state.tracks[2].end();
  await settle();
  assert.ok(btn.classList.contains('voice-off'), 'still off');
  assert.deepStrictEqual(log, after, 'nothing torn down twice');

  console.log(
    '  ok - a microphone that ends under the live pipeline turns the trigger off with an error',
  );
  console.log('audit1016_voice_mic_track_ended: all passed');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
