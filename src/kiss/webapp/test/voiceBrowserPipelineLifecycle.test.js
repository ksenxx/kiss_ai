// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E page tests for the async lifecycle of the in-page (browser-mic)
// voice pipeline in voice.js: cancellation while a start is in flight and
// toggle bursts around the microphone request.
//
// Defects being reproduced (each test failed against the pre-fix code):
//  1. Toggling OFF while the pipeline was starting (engine/model still
//     loading) left the button stuck in the loading state, and a pending
//     getUserMedia could still open the microphone.
//  2. A pipeline failure that settled AFTER the user toggled off painted
//     a stale red error over the user's explicit "off".
//  3. ON→OFF→ON bursts weren't tied to the start they began: a stale
//     microphone grant could leave two capture paths running.
//  4. A backgrounded browser tab must keep its microphone: the wake word
//     is listened for even when the tab is not the one on screen.
//
// jsdom provides no speech engine, audio graph, or microphone, so
// voice_page_helpers supplies the window.Vosk / AudioContext /
// mediaDevices the page expects from its embedder (the microphone grant
// and the model load are settled by the test).  Everything else (the
// real voice.js, main.js, DOM, message plumbing) is exercised for real.

const assert = require('assert');
const {makeRunner} = require('./ui_antipattern_harness');
const {makeVoicePage, micBtn, settle, VOICE_CFG} = require('./voice_page_helpers');
const {test, report} = makeRunner();

// Every page built; closed after the run so no armed timer keeps node
// alive and a failed assertion cannot hang the suite.
const windows = [];

/** A chat page with the mic off; see voice_page_helpers.makeVoicePage. */
function makePage(opts) {
  const page = makeVoicePage(opts);
  windows.push(page.win);
  return page;
}

function setPageVisibility(win, state) {
  Object.defineProperty(win.document, 'visibilityState', {
    configurable: true,
    get: () => state,
  });
  Object.defineProperty(win.document, 'hidden', {
    configurable: true,
    get: () => state === 'hidden',
  });
  win.document.dispatchEvent(new win.Event('visibilitychange'));
}

async function main() {
  await test('the mic button injects the vosk bundle and starts the pipeline directly', async () => {
    const {win, mic, model} = makePage({noVosk: true});
    micBtn(win).click();
    assert.ok(
      micBtn(win).classList.contains('voice-loading'),
      `the click must paint loading: ${micBtn(win).className}`,
    );
    const script = win.document.querySelector(
      `script[src=${JSON.stringify(VOICE_CFG.voskSrc)}]`,
    );
    assert.ok(script, 'the vosk bundle must be injected for in-page capture');
    assert.strictEqual(
      mic.calls.length,
      0,
      'no microphone is requested before the engine has loaded',
    );
    // The bundle arrives: the engine loads, then the microphone is asked.
    win.Vosk = {createModel: () => Promise.resolve(model)};
    script.onload();
    await settle();
    assert.strictEqual(mic.calls.length, 1, 'pipeline must request the mic');
    mic.calls[0].grant();
    await settle();
    assert.ok(
      micBtn(win).classList.contains('voice-listening'),
      `pipeline must be live: ${micBtn(win).className}`,
    );
    assert.strictEqual(win.localStorage.getItem('kissVoiceEnabled'), '1');
  });

  await test('a vosk bundle that fails to load leaves no dead script tag behind', async () => {
    const {win} = makePage({noVosk: true});
    const selector = `script[src=${JSON.stringify(VOICE_CFG.voskSrc)}]`;
    micBtn(win).click();
    const first = win.document.querySelector(selector);
    assert.ok(first, 'the vosk bundle must be injected');
    first.onerror();
    await settle();
    assert.strictEqual(
      win.document.querySelectorAll(selector).length,
      0,
      'the failed script tag must be removed from the document',
    );
    micBtn(win).click(); // off
    micBtn(win).click(); // retry
    const scripts = win.document.querySelectorAll(selector);
    assert.strictEqual(scripts.length, 1, 'the retry injects exactly one tag');
    assert.notStrictEqual(scripts[0], first, 'the retry uses a fresh tag');
  });

  await test('toggling off while the engine loads settles at off, never stuck loading, mic never opened', async () => {
    const page = makePage({pendingModel: true});
    const {win, mic, log} = page;
    micBtn(win).click(); // on: model pending
    await settle();
    assert.ok(
      micBtn(win).classList.contains('voice-loading'),
      `pipeline should be loading: ${micBtn(win).className}`,
    );
    micBtn(win).click(); // off while the model is still loading
    assert.ok(
      micBtn(win).classList.contains('voice-off'),
      `off click must paint off immediately: ${micBtn(win).className}`,
    );
    page.resolveModel(); // the superseded start now settles
    await settle();
    assert.ok(
      micBtn(win).classList.contains('voice-off'),
      `must stay off after the stale start settles: ${micBtn(win).className}`,
    );
    assert.strictEqual(
      mic.calls.length,
      0,
      'the superseded pipeline must never request the microphone',
    );
    assert.strictEqual(log.filter(e => e === 'ctx-open').length, 0);
  });

  await test('a microphone granted after the off click is released immediately', async () => {
    const {win, mic, log} = makePage();
    micBtn(win).click();
    await settle(); // model resolves, getUserMedia pending
    assert.strictEqual(mic.calls.length, 1, 'pipeline must request the mic');
    micBtn(win).click(); // off while getUserMedia is pending
    mic.calls[0].grant(); // the embedder grants it anyway
    await settle();
    assert.ok(
      log.includes('track-stopped'),
      `the late-granted stream must be stopped: ${JSON.stringify(log)}`,
    );
    assert.strictEqual(log.filter(e => e === 'ctx-open').length, 0);
    assert.ok(micBtn(win).classList.contains('voice-off'));
  });

  await test('a failure settling after the off click paints off, not a stale red error', async () => {
    const {win, mic} = makePage();
    micBtn(win).click();
    await settle();
    micBtn(win).click(); // off while getUserMedia is pending
    mic.calls[0].refuse('Requested device not found');
    await settle();
    const btn = micBtn(win);
    assert.ok(
      !btn.classList.contains('voice-error'),
      `a superseded failure must not paint an error: ${btn.className}`,
    );
    assert.ok(btn.classList.contains('voice-off'));
  });

  await test('a failure of an attempt the user still wants keeps the red error', async () => {
    const {win, mic} = makePage();
    micBtn(win).click();
    await settle();
    mic.calls[0].refuse('Requested device not found');
    await settle();
    const btn = micBtn(win);
    assert.ok(
      btn.classList.contains('voice-error'),
      `a real, current failure must stay visible: ${btn.className}`,
    );
    assert.strictEqual(
      win.localStorage.getItem('kissVoiceEnabled'),
      '0',
      'the failed attempt must clear the persisted intent',
    );
  });

  await test('ON-OFF-ON around a pending microphone: the stale grant is released and exactly one pipeline runs', async () => {
    const {win, mic, log} = makePage();
    micBtn(win).click(); // on: getUserMedia 1 pending after the load
    await settle();
    assert.strictEqual(mic.calls.length, 1);
    micBtn(win).click(); // off
    micBtn(win).click(); // on again: deferred behind the in-flight start
    assert.ok(micBtn(win).classList.contains('voice-loading'));
    mic.calls[0].grant(); // the stale request is granted
    await settle();
    assert.ok(
      log.includes('track-stopped'),
      `the stale stream must be released: ${JSON.stringify(log)}`,
    );
    assert.strictEqual(
      log.filter(e => e === 'ctx-open').length,
      0,
      'the stale grant must not open an audio graph',
    );
    assert.strictEqual(
      mic.calls.length,
      2,
      'exactly one restarted pipeline may request the mic',
    );
    mic.calls[1].grant();
    await settle();
    assert.ok(
      micBtn(win).classList.contains('voice-listening'),
      `the re-enabled pipeline must end live: ${micBtn(win).className}`,
    );
    assert.strictEqual(log.filter(e => e === 'ctx-open').length, 1);
  });

  await test('OFF then ON during an in-flight start restarts the pipeline exactly once', async () => {
    const page = makePage({pendingModel: true});
    const {win, mic} = page;
    micBtn(win).click(); // on: model pending
    await settle();
    micBtn(win).click(); // off during the load
    micBtn(win).click(); // on again during the same load
    assert.ok(micBtn(win).classList.contains('voice-loading'));
    page.resolveModel(); // the stale start settles and hands over
    await settle();
    assert.strictEqual(
      mic.calls.length,
      1,
      'exactly one restarted pipeline may request the mic',
    );
    mic.calls[0].grant();
    await settle();
    assert.ok(
      micBtn(win).classList.contains('voice-listening'),
      `the re-enabled pipeline must end live: ${micBtn(win).className}`,
    );
  });

  await test('turning the mic off releases the microphone and closes the audio graph', async () => {
    const {win, mic, log} = makePage();
    micBtn(win).click();
    await settle();
    mic.calls[0].grant();
    await settle();
    assert.ok(micBtn(win).classList.contains('voice-listening'));
    micBtn(win).click(); // off
    assert.ok(
      log.includes('track-stopped'),
      `turning off must stop the microphone: ${JSON.stringify(log)}`,
    );
    assert.ok(log.includes('ctx-closed'), 'turning off closes the graph');
    assert.ok(micBtn(win).classList.contains('voice-off'));
    assert.strictEqual(win.localStorage.getItem('kissVoiceEnabled'), '0');
  });

  await test('a hidden tab keeps listening for its wake word', async () => {
    const {win, mic, log} = makePage();
    micBtn(win).click();
    await settle();
    mic.calls[0].grant();
    await settle();
    assert.ok(micBtn(win).classList.contains('voice-listening'));
    setPageVisibility(win, 'hidden');
    assert.ok(
      !log.includes('track-stopped'),
      'a backgrounded tab must keep its microphone',
    );
    assert.ok(micBtn(win).classList.contains('voice-listening'));
    setPageVisibility(win, 'visible');
    await settle();
    assert.strictEqual(
      mic.calls.length,
      1,
      'showing the tab again must not open a second microphone',
    );
  });

  windows.forEach(win => win.close());
  report('voiceBrowserPipelineLifecycle');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
