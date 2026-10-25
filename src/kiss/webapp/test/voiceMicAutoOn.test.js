// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makeRunner} = require('./ui_antipattern_harness');
const {micBtn, makeBareVoiceWindow} = require('./voice_page_helpers');
const {test, report} = makeRunner();

const CFG = {
  voskSrc: '/media/vosk.js',
  modelUrl: 'https://models.example.com/model.tar.gz',
};

// The in-page pipeline's engine loads from a script tag the page injects;
// it never arrives in jsdom, so the mic parks in "loading" and nothing is
// posted to the daemon (only a captured round's audio ever is).
function makeWindow(storedEnabled) {
  const posted = [];
  const win = makeBareVoiceWindow(CFG, w => {
    if (storedEnabled !== undefined) {
      w.localStorage.setItem('kissVoiceEnabled', storedEnabled);
    }
    w.addEventListener('kiss-voice-post', e => posted.push(e.detail));
  });
  return {win, posted};
}

function engineInjected(win) {
  return !!win.document.querySelector('script[src="/media/vosk.js"]');
}

test('fresh storage (first install) keeps the mic OFF', () => {
  const {win, posted} = makeWindow();
  assert.strictEqual(posted.length, 0);
  assert.ok(micBtn(win).classList.contains('voice-off'));
  assert.ok(!engineInjected(win), 'no speech engine is loaded unasked');
});

test('explicit opt-in (stored "1") auto-starts the mic on load', () => {
  const {win, posted} = makeWindow('1');
  assert.strictEqual(posted.length, 0, 'nothing is posted to the daemon');
  assert.ok(micBtn(win).classList.contains('voice-loading'));
  assert.ok(engineInjected(win), 'the speech engine load has begun');
});

test('explicit user opt-out (stored "0") stays off', () => {
  const {win, posted} = makeWindow('0');
  assert.strictEqual(posted.length, 0);
  assert.ok(micBtn(win).classList.contains('voice-off'));
  assert.ok(!engineInjected(win));
});

test('turning the mic on is remembered for next launch', () => {
  const first = makeWindow();
  micBtn(first.win).click();
  assert.ok(micBtn(first.win).classList.contains('voice-loading'));
  assert.strictEqual(first.win.localStorage.getItem('kissVoiceEnabled'), '1');
  const second = makeWindow('1');
  assert.ok(micBtn(second.win).classList.contains('voice-loading'));
});

test('turning the mic off is remembered for next launch', () => {
  const first = makeWindow('1');
  micBtn(first.win).click();
  assert.ok(micBtn(first.win).classList.contains('voice-off'));
  assert.strictEqual(first.win.localStorage.getItem('kissVoiceEnabled'), '0');
  const second = makeWindow('0');
  assert.ok(micBtn(second.win).classList.contains('voice-off'));
});

report('voiceMicAutoOn');
