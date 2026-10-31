// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E page tests for three voice.js defects, on a chat page running the
// real main.js + voice.js over the scripted voice surface of
// voice_page_helpers.js.  Each test failed against the pre-fix code.
//
//  ROUND  A cancelled round's late transcript was typed into the next
//         round's conversation.  Turning the mic off cancels a round whose
//         audio is being transcribed; its reply, arriving after the next
//         wake, was paired with that new round although the new round was
//         still capturing (no reply can answer a round before its audio is
//         posted), so tab X's words were submitted into tab Y.  The page
//         now pairs an untagged reply only with a round whose audio was
//         posted, and tags every posted round so that a daemon echoing the
//         tag refuses the late reply even after the next round posted its
//         own audio.
//  SENS   The wake sensitivity of a sibling same-origin tab did not follow
//         a change made in another one (only auto-submit did), so the
//         slider showed a stale value and the wake detector kept using it.
//  VOSK   The vosk.js bundle load had no bound: a request that neither
//         loads nor errors (a stalled response, a service worker that never
//         answers) left the pipeline start pending, `busy` set for good,
//         and the mic wedged on "loading" with every re-enable deferred.

const assert = require('assert');
const {send, twoTabs, clickTab} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {
  VOICE_CFG,
  makeVoicePage,
  micBtn,
  settle,
  startPipeline,
  lastRecognizer,
  wake,
  speak,
  fireTimer,
} = require('./voice_page_helpers');

const {test, report} = makeRunner();

const SENSITIVITY_KEY = 'kissVoiceSensitivity';
// voice.js bounds the vosk.js load by its default 30 s model-load bound.
const LOAD_TIMEOUT_MS = 30000;

function submits(posted) {
  return posted.filter(m => m.type === 'submit');
}

function transcribes(posted) {
  return posted.filter(m => m.type === 'voiceTranscribe');
}

function taskInput(win) {
  return win.document.getElementById('task-input');
}

/** Another same-origin tab stored `value` as the wake sensitivity. */
function siblingSetsSensitivity(win, value) {
  win.localStorage.setItem(SENSITIVITY_KEY, value);
  const evt = new win.Event('storage');
  evt.key = SENSITIVITY_KEY;
  evt.newValue = value;
  win.dispatchEvent(evt);
}

/** Recognize a trailing-alias phrase, which wakes only at sensitivity >= 75. */
function hearTrailingAlias(page) {
  page.clock.now += 3000; // clear the 2 s wake cooldown
  lastRecognizer(page).handlers.result({result: {text: 'so abracadabra'}});
}

/**
 * Speak round A on tab `first`, turn the mic off (cancelling it while
 * its transcript is on its way), move to tab `second` and turn the mic
 * back on.
 */
async function cancelRoundThenMoveTab(page, first, second) {
  clickTab(page.win, first);
  await startPipeline(page);
  wake(page);
  speak(page);
  micBtn(page.win).click(); // mic off: round A is cancelled
  clickTab(page.win, second);
  await startPipeline(page);
}

async function main() {
  await test(
    "ROUND: a cancelled round's untagged reply arriving after the next " +
      'wake is refused, and the next round still answers on its tab',
    async () => {
      const page = makeVoicePage();
      const {win, posted} = page;
      const {first, second} = twoTabs(win);
      await cancelRoundThenMoveTab(page, first, second);
      wake(page); // round B begins on tab Y and is still capturing

      // Round A's reply comes back (from a daemon that does not echo
      // round tags).  Round B has posted nothing, so it cannot be B's.
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'round A words QW1'});
      assert.strictEqual(
        submits(posted).length,
        0,
        "round A's words must not be submitted into tab Y: " +
          JSON.stringify(posted),
      );
      assert.ok(
        !taskInput(win).value.includes('QW1'),
        "round A's words must not be drafted into tab Y's input",
      );
      assert.ok(
        micBtn(win).classList.contains('voice-triggered'),
        "round B's capture is still under way: its wake flash stays",
      );

      speak(page); // round B posts its audio
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'round B words QW2'});
      const subs = submits(posted);
      assert.strictEqual(subs.length, 1, JSON.stringify(posted));
      assert.ok(subs[0].prompt.includes('round B words QW2'));
      assert.strictEqual(subs[0].tabId, second);
      win.close();
    },
  );

  await test(
    "ROUND: with round tags echoed, a cancelled round's reply arriving " +
      'after the next round posted its audio is refused',
    async () => {
      const page = makeVoicePage();
      const {win, posted} = page;
      const {first, second} = twoTabs(win);
      await cancelRoundThenMoveTab(page, first, second);
      wake(page);
      speak(page); // round B posts its audio before A's reply lands

      const [roundA, roundB] = transcribes(posted).map(m => m.round);
      assert.strictEqual(typeof roundA, 'number', 'audio carries its round');
      assert.strictEqual(typeof roundB, 'number', 'audio carries its round');
      assert.notStrictEqual(roundA, roundB, 'each round has its own tag');

      // The daemon answers in order and echoes each round's tag.
      posted.length = 0;
      send(win, {
        type: 'voiceSpeech',
        text: 'round A words QW3',
        round: roundA,
      });
      assert.strictEqual(
        submits(posted).length,
        0,
        "the cancelled round's words must not reach tab Y: " +
          JSON.stringify(posted),
      );
      assert.ok(!taskInput(win).value.includes('QW3'));

      send(win, {
        type: 'voiceSpeech',
        text: 'round B words QW4',
        round: roundB,
      });
      const subs = submits(posted);
      assert.strictEqual(subs.length, 1, JSON.stringify(posted));
      assert.ok(subs[0].prompt.includes('round B words QW4'));
      assert.strictEqual(subs[0].tabId, second);
      win.close();
    },
  );

  await test(
    'SENS: a sensitivity change made in a sibling tab moves the slider ' +
      'and the live wake threshold',
    async () => {
      const page = makeVoicePage();
      const {win} = page;
      await startPipeline(page);
      const slider = win.document.getElementById('cfg-voice-sensitivity');
      const label = win.document.getElementById('cfg-voice-sensitivity-value');

      siblingSetsSensitivity(win, '10');
      assert.strictEqual(slider.value, '10', 'the slider follows the change');
      assert.strictEqual(label.textContent, '10', 'so does its label');
      hearTrailingAlias(page);
      assert.ok(
        !micBtn(win).classList.contains('voice-triggered'),
        'at sensitivity 10 a trailing alias must not wake',
      );

      siblingSetsSensitivity(win, '80');
      assert.strictEqual(slider.value, '80');
      hearTrailingAlias(page);
      assert.ok(
        micBtn(win).classList.contains('voice-triggered'),
        'back at 80 the trailing alias wakes again',
      );
      win.close();
    },
  );

  await test(
    'VOSK: a vosk.js request that never settles fails after the load ' +
      'bound, and the retry injects a fresh tag that can still load',
    async () => {
      const page = makeVoicePage({noVosk: true});
      const {win, mic, model} = page;
      const selector = `script[src=${JSON.stringify(VOICE_CFG.voskSrc)}]`;
      micBtn(win).click();
      const stalled = win.document.querySelector(selector);
      assert.ok(stalled, 'the vosk bundle must be injected');

      fireTimer(page, LOAD_TIMEOUT_MS); // the request never answered
      await settle();
      assert.ok(
        micBtn(win).classList.contains('voice-error'),
        `the stalled load must paint the error: ${micBtn(win).className}`,
      );
      assert.strictEqual(win.localStorage.getItem('kissVoiceEnabled'), '0');
      assert.strictEqual(
        win.document.querySelectorAll(selector).length,
        0,
        'the stalled tag is removed',
      );

      micBtn(win).click(); // the retry is honoured: busy was cleared
      assert.ok(
        micBtn(win).classList.contains('voice-loading'),
        `the retry must be loading: ${micBtn(win).className}`,
      );
      const retry = win.document.querySelector(selector);
      assert.ok(retry && retry !== stalled, 'the retry uses a fresh tag');
      win.Vosk = {createModel: () => Promise.resolve(model)};
      retry.onload();
      await settle();
      assert.strictEqual(mic.calls.length, 1, 'the retry reaches the mic');
      assert.ok(
        page.timers.every(t => t.cleared || t.ms !== LOAD_TIMEOUT_MS),
        'no load bound is left armed once the bundle loaded',
      );
      win.close();
    },
  );

  report('voicePipeline_sweep1030');
}

main();
