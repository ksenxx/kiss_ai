// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test reproducing the "talk voice overlaps and breaks"
// bug: successive ``{type: 'talk'}`` events each created and played
// their own Audio element IMMEDIATELY, so two talk() calls spoke on
// top of each other, and the Web-Speech fallback could talk over a
// playing Audio clip.  The webview must serialize ALL talk playback
// through one FIFO queue: clip N+1 (audio or speech) starts only after
// clip N finishes ('ended'), fails ('error' / rejected play()), or its
// speech fallback finishes speaking.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/talkAudioOverlap.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/** Build a jsdom window running the production chat webview. */
function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win};
}

/**
 * Install a recording Audio constructor whose instances track play()
 * calls and can fire their 'ended'/'error' completion events the way
 * a real HTMLAudioElement does (onended/onerror properties).
 */
function installAudio(win, playResult) {
  const players = [];
  win.Audio = function Audio(src) {
    this.src = src;
    this.playCalls = 0;
    this.play = () => {
      this.playCalls++;
      return playResult === undefined ? Promise.resolve() : playResult;
    };
    this.fire = type => {
      const handler = this['on' + type];
      if (typeof handler === 'function') handler({type});
    };
    players.push(this);
  };
  return players;
}

/**
 * Install a recording Web Speech API on *win* (jsdom has none).
 * Utterances are recorded so tests can fire their onend/onerror the
 * way a real speech engine does when it finishes speaking.
 */
function installSpeech(win) {
  const spoken = [];
  win.SpeechSynthesisUtterance = function SpeechSynthesisUtterance(text) {
    this.text = text;
    this.lang = '';
    spoken.push(this);
  };
  win.speechSynthesis = {
    speak: () => {},
  };
  return spoken;
}

/** Finish a speech-fallback job by ending its last queued utterance. */
function endSpeech(spoken) {
  const last = spoken[spoken.length - 1];
  if (last && typeof last.onend === 'function') last.onend({type: 'end'});
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

const B64 = 'SUQzBAAAAAAAAA=='; // decodes to "ID3..." — an MP3 tag header

function talkEv(id, text, extra) {
  return Object.assign(
    {type: 'talk', language: 'en-US', text, talkId: id},
    extra || {},
  );
}

// --- Overlap reproduction: two audio clips must not play together ---

function testSecondAudioWaitsForFirstEnded() {
  const {win} = makeWebview();
  installSpeech(win);
  const players = installAudio(win, Promise.resolve());

  send(win, talkEv('q1', 'first clip', {audioB64: B64}));
  send(win, talkEv('q2', 'second clip', {audioB64: B64}));

  assert.strictEqual(players.length >= 1 && players[0].playCalls, 1,
    'first clip plays immediately');
  const playedEarly = players.length > 1 && players[1].playCalls > 0;
  assert.ok(!playedEarly,
    'second clip must NOT play before the first fires ended ' +
    '(overlapping audio)');

  players[0].fire('ended');
  assert.strictEqual(players.length, 2, 'second clip created after ended');
  assert.strictEqual(players[1].playCalls, 1,
    'second clip plays after the first ends');
  assert.strictEqual(players[1].src, 'data:audio/mpeg;base64,' + B64);
  console.log('PASS: second audio clip waits for the first to end');
}

function testThreeAudioClipsPlayInFifoOrder() {
  const {win} = makeWebview();
  installSpeech(win);
  const players = installAudio(win, Promise.resolve());

  send(win, talkEv('f1', 'one', {audioB64: B64}));
  send(win, talkEv('f2', 'two', {audioB64: B64}));
  send(win, talkEv('f3', 'three', {audioB64: B64}));

  assert.strictEqual(players.length, 1, 'only the first clip started');
  players[0].fire('ended');
  assert.strictEqual(players.length, 2, 'then the second');
  players[1].fire('ended');
  assert.strictEqual(players.length, 3, 'then the third');
  assert.strictEqual(players[2].playCalls, 1);
  console.log('PASS: three audio clips play strictly one after another');
}

// --- 'error' must advance the queue, not deadlock it ---

function testAudioErrorAdvancesQueue() {
  const {win} = makeWebview();
  installSpeech(win);
  const players = installAudio(win, Promise.resolve());

  send(win, talkEv('e1', 'broken clip', {audioB64: B64}));
  send(win, talkEv('e2', 'next clip', {audioB64: B64}));

  players[0].fire('error');
  assert.strictEqual(players.length, 2,
    'a decode/playback error must advance to the next clip');
  assert.strictEqual(players[1].playCalls, 1);
  console.log('PASS: audio error event advances the talk queue');
}

// --- Speech fallback must not talk over a playing audio clip ---

function testSpeechTalkWaitsForPlayingAudio() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  const players = installAudio(win, Promise.resolve());

  send(win, talkEv('m1', 'audio first', {audioB64: B64}));
  send(win, talkEv('m2', 'spoken second'));

  assert.strictEqual(spoken.length, 0,
    'speech must NOT start while the audio clip is still playing');

  players[0].fire('ended');
  assert.ok(spoken.length >= 1, 'speech starts after the audio ends');
  assert.strictEqual(spoken[0].text, 'spoken second');
  console.log('PASS: web-speech talk waits for the playing audio clip');
}

function testAudioWaitsForSpeechCompletion() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  const players = installAudio(win, Promise.resolve());

  send(win, talkEv('s1', 'spoken first'));
  send(win, talkEv('s2', 'audio second', {audioB64: B64}));

  assert.ok(spoken.length >= 1, 'speech job starts immediately');
  assert.strictEqual(players.length, 0,
    'audio must NOT start while the speech engine is still speaking');

  endSpeech(spoken);
  assert.strictEqual(players.length, 1, 'audio starts after speech ends');
  assert.strictEqual(players[0].playCalls, 1);
  console.log('PASS: audio clip waits for the speech job to finish');
}

function testSpeechUtteranceErrorAdvancesQueue() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  const players = installAudio(win, Promise.resolve());

  send(win, talkEv('u1', 'spoken first'));
  send(win, talkEv('u2', 'audio second', {audioB64: B64}));

  const last = spoken[spoken.length - 1];
  assert.strictEqual(typeof last.onerror, 'function',
    'speech job must complete on engine error too');
  last.onerror({type: 'error'});
  assert.strictEqual(players.length, 1,
    'a speech engine error must advance the talk queue');
  console.log('PASS: speech utterance error advances the talk queue');
}

// --- Rejected play() (autoplay policy): fallback then advance ---

async function testRejectedPlayFallsBackThenAdvances() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  const players = installAudio(
    win, Promise.reject(new Error('autoplay blocked')));

  send(win, talkEv('r1', 'blocked clip', {audioB64: B64}));
  send(win, talkEv('r2', 'queued clip', {audioB64: B64}));

  await new Promise(resolve => setTimeout(resolve, 0));
  assert.ok(spoken.length >= 1, 'rejected play() falls back to speech');
  assert.strictEqual(spoken[0].text, 'blocked clip');
  assert.strictEqual(players.length, 1,
    'the queued clip must wait for the fallback speech to finish');

  endSpeech(spoken);
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.strictEqual(players.length, 2,
    'queue advances after the fallback speech finishes');
  console.log('PASS: rejected play() falls back, then queue advances');
}

async function testLateErrorAfterRejectedPlayDoesNotCutFallback() {
  // Browsers fire BOTH a play() rejection and an 'error' event when a
  // clip cannot be decoded.  Once the fallback speech owns the talk,
  // the element's late 'error' must not advance the queue early and
  // start the next talk over the still-speaking fallback.
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  const players = installAudio(
    win, Promise.reject(new Error('decode failed')));

  send(win, talkEv('L1', 'undecodable clip', {audioB64: B64}));
  send(win, talkEv('L2', 'queued clip', {audioB64: B64}));

  await new Promise(resolve => setTimeout(resolve, 0));
  assert.ok(spoken.length >= 1, 'fallback speech started');

  players[0].fire('error'); // late media error after the rejection
  assert.strictEqual(players.length, 1,
    "the element's late error must not start the next talk while " +
    'the fallback speech is still speaking');

  endSpeech(spoken);
  assert.strictEqual(players.length, 2,
    'queue advances once the fallback speech actually finishes');
  console.log('PASS: late media error never cuts off the fallback speech');
}

function testAudioAbortAdvancesQueue() {
  // 'abort' is a terminal media event (fetch/decode aborted); it must
  // release the queue exactly like 'ended' and 'error'.
  const {win} = makeWebview();
  installSpeech(win);
  const players = installAudio(win, Promise.resolve());

  send(win, talkEv('A1', 'aborted clip', {audioB64: B64}));
  send(win, talkEv('A2', 'next clip', {audioB64: B64}));

  players[0].fire('abort');
  assert.strictEqual(players.length, 2,
    'an aborted clip must advance to the next talk');
  assert.strictEqual(players[1].playCalls, 1);
  console.log('PASS: audio abort event advances the talk queue');
}

// --- Jobs that produce no sound must complete immediately ---

function testUnspeakableTextDoesNotDeadlockQueue() {
  const {win} = makeWebview();
  installSpeech(win);
  const players = installAudio(win, Promise.resolve());

  // An emoji-only text cleans to nothing speakable → the job must
  // finish at once instead of holding the queue forever.
  send(win, talkEv('z1', '\u{1F642}'));
  send(win, talkEv('z2', 'real clip', {audioB64: B64}));

  assert.strictEqual(players.length, 1,
    'an unspeakable talk must not block the queue');
  assert.strictEqual(players[0].playCalls, 1);
  console.log('PASS: unspeakable talk text does not deadlock the queue');
}

function testMissingSpeechApiDoesNotDeadlockQueue() {
  const {win} = makeWebview();
  const players = installAudio(win, Promise.resolve());
  win.speechSynthesis = undefined; // device without Web Speech

  send(win, talkEv('n1', 'cannot speak me'));
  send(win, talkEv('n2', 'real clip', {audioB64: B64}));

  assert.strictEqual(players.length, 1,
    'a talk with no speech engine must not block the queue');
  console.log('PASS: missing Web Speech API does not deadlock the queue');
}

// --- Regression: single-talk behavior is unchanged ---

function testSingleAudioTalkStillPlaysImmediately() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  const players = installAudio(win, Promise.resolve());

  send(win, talkEv('one', 'hello', {audioB64: B64}));

  assert.strictEqual(players.length, 1);
  assert.strictEqual(players[0].playCalls, 1);
  assert.strictEqual(spoken.length, 0);
  console.log('PASS: a single audio talk still plays immediately');
}

async function main() {
  testSecondAudioWaitsForFirstEnded();
  testThreeAudioClipsPlayInFifoOrder();
  testAudioErrorAdvancesQueue();
  testSpeechTalkWaitsForPlayingAudio();
  testAudioWaitsForSpeechCompletion();
  testSpeechUtteranceErrorAdvancesQueue();
  await testRejectedPlayFallsBackThenAdvances();
  await testLateErrorAfterRejectedPlayDoesNotCutFallback();
  testAudioAbortAdvancesQueue();
  testUnspeakableTextDoesNotDeadlockQueue();
  testMissingSpeechApiDoesNotDeadlockQueue();
  testSingleAudioTalkStillPlaysImmediately();
  console.log('All talkAudioOverlap tests passed.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
