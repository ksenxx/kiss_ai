// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// voice.js: the microphone goes away (track `ended`) while a round's
// audio is posted and its transcript still pending.  onMicEnded keeps
// that round's owner (its transcript still stands) but called
// flash(null) unconditionally, which cleared the 60 s "transcribing"
// timer that retires a round whose reply never comes.  With the timer
// gone the owner sat in the queue for good, and the NEXT round's
// transcript (dictated after the mic came back, in another chat) was
// paired with the stale owner and submitted into the wrong chat.
//
// Harness: voice_page_helpers (scriptable microphone whose tracks the
// test can end, recognizers, audio processors, recorded safety timers).

const assert = require('assert');
const {send} = require('./simplify2_harness');
const {
  makeVoicePage,
  micBtn,
  settle,
  startPipeline,
  wake,
  speak,
  fireTimer,
} = require('./voice_page_helpers');

const liveTimers = (page, ms) =>
  page.timers.filter(t => !t.cleared && t.ms === ms);

/** Wake, speak, fall silent: the round's audio is posted. */
function speakRound(page) {
  wake(page);
  speak(page);
}

async function main() {
  const page = makeVoicePage({autoGrant: true});
  const {win, posted, state} = page;
  const api = win._testApi;
  const first = api.getActiveTabId();
  api.createNewTab();
  const second = api.getActiveTabId();
  api.switchToTab(first); // chat A on screen

  await startPipeline(page);
  speakRound(page); // round A: audio posted, transcript pending
  assert.strictEqual(
    posted.filter(m => m.type === 'voiceTranscribe').length,
    1,
  );
  assert.strictEqual(
    liveTimers(page, 60000).length,
    1,
    'the 60 s transcribing timer is armed',
  );

  // The microphone goes away while round A transcribes.
  state.tracks[0].end();
  await settle();
  assert.ok(
    micBtn(win).classList.contains('voice-error'),
    'the trigger is off with the reason',
  );
  assert.ok(
    micBtn(win).classList.contains('voice-transcribing'),
    'the round still awaiting its transcript keeps the transcribing flash: ' +
      micBtn(win).className,
  );
  assert.strictEqual(
    liveTimers(page, 60000).length,
    1,
    'REGRESSION: the mic loss must keep (or re-arm) the 60 s timer that retires the round ' +
      'if its transcript never comes, got ' +
      liveTimers(page, 60000).length,
  );

  // Round A's reply never comes: the timer retires it.
  fireTimer(page, 60000);
  assert.ok(
    !micBtn(win).classList.contains('voice-transcribing'),
    'the flash is gone',
  );

  // The mic comes back; the user dictates in chat B.
  api.switchToTab(second);
  await startPipeline(page);
  speakRound(page); // round B
  const audio = posted.filter(m => m.type === 'voiceTranscribe');
  assert.strictEqual(audio.length, 2);
  posted.length = 0;
  // The daemon echoes round B's tag.
  send(win, {
    type: 'voiceSpeech',
    text: 'words spoken in chat B QB1',
    round: audio[1].round,
  });
  assert.deepStrictEqual(
    posted
      .filter(m => m.type === 'submit')
      .map(m => ({onB: m.tabId === second, prompt: m.prompt})),
    [{onB: true, prompt: 'words spoken in chat B QB1'}],
    "round B's words must reach chat B, not the chat of the round the mic loss stranded: " +
      JSON.stringify(posted),
  );
  win.close();
  console.log('  ok - a mic loss keeps the timer that retires a posted round');
  console.log('audit1018_voice_mic_ended_round_timer: all passed');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
