// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage: makeChatPage, send, stubRaf} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {installAudio} = require('./speech_helpers');

const B64 = 'SUQzBAAAAAAAAA==';

// Models a real HTMLMediaElement closely enough for playTalkAudio:
// `ended` fires asynchronously, after play() has returned and the
// stall sampler is installed, so completion clears that sampler
// instead of orphaning it; currentTime and pause() exist because the
// sampler (installed after play() returns) reads the one and retires a
// stalled clip with the other.
function playThenEnd(win, player) {
  player.currentTime = 0;
  player.pause = () => {};
  win.setTimeout(() => player.fire('ended'), 0);
  return Promise.resolve();
}

function makePage() {
  let played;
  const {win, posted} = makeChatPage({
    beforeScripts(win) {
      stubRaf(win);
      played = installAudio(win, player => playThenEnd(win, player));
    },
  });

  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'page must post ready with its tab id');
  return {win, posted, played, tabId: ready.tabId};
}

const runner = makeRunner();
const report = runner.report;

// Each test gets a fresh page and the page is closed afterwards, pass
// or fail, so its pending timers (the Audio stub's `ended`, main.js's
// talk stall sampler) cannot keep node alive.
function test(name, fn) {
  const page = makePage();
  try {
    runner.test(name, () => fn(page));
  } finally {
    page.win.close();
  }
}

test('a muted talk copy stays silent', page => {
  const {win, played, tabId} = page;
  send(win, {
    type: 'talk',
    language: 'en-US',
    text: 'played on the terminal',
    audioB64: B64,
    talkId: 't-muted-1',
    taskId: 7,
    tabId,
    muted: true,
  });
  assert.strictEqual(
    played.length,
    0,
    `muted copy must not play its clip, played: ${played.length}`,
  );
});

test('a muted copy does not block a later playable copy', page => {
  const {win, played, tabId} = page;
  send(win, {
    type: 'talk',
    language: 'en-US',
    text: 'first muted',
    audioB64: B64,
    talkId: 't-muted-2',
    taskId: 7,
    tabId,
    muted: true,
  });
  send(win, {
    type: 'talk',
    language: 'en-US',
    text: 'now audible',
    audioB64: B64,
    talkId: 't-muted-2',
    taskId: 7,
    tabId,
  });
  assert.strictEqual(played.length, 1, `played ${played.length} clips`);
  assert.strictEqual(played[0].src, 'data:audio/mpeg;base64,' + B64);
});

report('talkMutedCopySilent');
// Every page is closed, so nothing should be pending; the explicit exit
// guarantees a leaked handle can never turn a green run into a hang.
process.exit(0);
