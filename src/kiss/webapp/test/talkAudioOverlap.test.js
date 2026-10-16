// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send, stubRaf} = require('./simplify2_harness');
const {installSpeech, assertNoSpeech} = require('./speech_helpers');

function installAudio(win, playResult) {
  const players = [];
  win.Audio = function Audio(src) {
    this.src = src;
    this.playCalls = 0;
    this.play = () => {
      this.playCalls++;
      const result =
        typeof playResult === 'function' ? playResult(this) : playResult;
      return result === undefined ? Promise.resolve() : result;
    };
    this.fire = type => {
      const handler = this['on' + type];
      if (typeof handler === 'function') handler({type});
    };
    players.push(this);
  };
  return players;
}

const B64 = 'SUQzBAAAAAAAAA==';

function talkEv(id, text, extra) {
  return Object.assign(
    {type: 'talk', language: 'en-US', text, talkId: id},
    extra || {},
  );
}

function testSecondAudioWaitsForFirstEnded() {
  const {win} = makePage({beforeScripts: stubRaf});
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
  const {win} = makePage({beforeScripts: stubRaf});
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

function testAudioErrorAdvancesQueue() {
  const {win} = makePage({beforeScripts: stubRaf});
  const spoken = installSpeech(win);
  const players = installAudio(win, Promise.resolve());

  send(win, talkEv('e1', 'broken clip', {audioB64: B64}));
  send(win, talkEv('e2', 'next clip', {audioB64: B64}));

  players[0].fire('error');
  assert.strictEqual(players.length, 2,
    'a decode/playback error must advance to the next clip');
  assert.strictEqual(players[1].playCalls, 1);
  assertNoSpeech(spoken, 'audio error event');
  console.log('PASS: audio error event advances the talk queue');
}

function testClipLessTalkIsSilentAndAdvancesQueue() {
  const {win} = makePage({beforeScripts: stubRaf});
  const spoken = installSpeech(win);
  const players = installAudio(win, Promise.resolve());

  send(win, talkEv('m1', 'audio first', {audioB64: B64}));
  send(win, talkEv('m2', 'no clip — must be skipped silently'));
  send(win, talkEv('m3', 'audio third', {audioB64: B64}));

  assert.strictEqual(players.length, 1,
    'the queued clips wait for the first to end');
  players[0].fire('ended');
  assert.strictEqual(players.length, 2,
    'a clip-less talk must complete at once and advance the queue');
  assert.strictEqual(players[1].src, 'data:audio/mpeg;base64,' + B64);
  assertNoSpeech(spoken, 'clip-less talk');
  console.log('PASS: clip-less talk stays silent and advances the queue');
}

function testLeadingClipLessTalkDoesNotDelayAudio() {
  const {win} = makePage({beforeScripts: stubRaf});
  const players = installAudio(win, Promise.resolve());

  send(win, talkEv('s1', 'no clip here'));
  send(win, talkEv('s2', 'audio second', {audioB64: B64}));

  assert.strictEqual(players.length, 1,
    'audio must start immediately after the silent clip-less talk');
  assert.strictEqual(players[0].playCalls, 1);
  console.log('PASS: a leading clip-less talk never delays the next clip');
}

async function testRejectedPlayDegradesSilentlyThenAdvances() {
  const {win} = makePage({beforeScripts: stubRaf});
  const spoken = installSpeech(win);
  const players = installAudio(
    win, () => Promise.reject(new Error('autoplay blocked')));

  send(win, talkEv('r1', 'blocked clip', {audioB64: B64}));
  send(win, talkEv('r2', 'queued clip', {audioB64: B64}));

  assert.strictEqual(players.length, 1,
    'the queued clip must wait for the blocked clip to settle');

  await new Promise(resolve => setTimeout(resolve, 0));
  assert.strictEqual(players.length, 2,
    'a rejected play() must complete silently and advance the queue');
  assert.strictEqual(players[1].playCalls, 1);
  assert.strictEqual(players[1].src, 'data:audio/mpeg;base64,' + B64);
  assertNoSpeech(spoken, 'rejected play()');
  console.log('PASS: rejected play() degrades to silence, queue advances');
}

async function testLateErrorAfterRejectedPlayDoesNotDoubleAdvance() {
  const {win} = makePage({beforeScripts: stubRaf});
  const spoken = installSpeech(win);
  const players = installAudio(
    win,
    player => players.indexOf(player) === 0
      ? Promise.reject(new Error('decode failed'))
      : Promise.resolve(),
  );

  send(win, talkEv('L1', 'undecodable clip', {audioB64: B64}));
  send(win, talkEv('L2', 'queued clip', {audioB64: B64}));
  send(win, talkEv('L3', 'third clip', {audioB64: B64}));

  await new Promise(resolve => setTimeout(resolve, 0));
  assert.strictEqual(players.length, 2,
    'the rejection advanced the queue to the second clip');

  players[0].fire('error');
  assert.strictEqual(players.length, 2,
    "the element's late error must not advance the queue again while " +
    'the second clip is still playing');

  players[1].fire('ended');
  assert.strictEqual(players.length, 3,
    'queue advances once the playing clip actually finishes');
  assertNoSpeech(spoken, 'late error after rejected play()');
  console.log('PASS: late media error never double-advances the queue');
}

function testAudioAbortAdvancesQueue() {
  const {win} = makePage({beforeScripts: stubRaf});
  const spoken = installSpeech(win);
  const players = installAudio(win, Promise.resolve());

  send(win, talkEv('A1', 'aborted clip', {audioB64: B64}));
  send(win, talkEv('A2', 'next clip', {audioB64: B64}));

  players[0].fire('abort');
  assert.strictEqual(players.length, 2,
    'an aborted clip must advance to the next talk');
  assert.strictEqual(players[1].playCalls, 1);
  assertNoSpeech(spoken, 'audio abort event');
  console.log('PASS: audio abort event advances the talk queue');
}

function testUnspeakableTextDoesNotDeadlockQueue() {
  const {win} = makePage({beforeScripts: stubRaf});
  const spoken = installSpeech(win);
  const players = installAudio(win, Promise.resolve());

  send(win, talkEv('z1', '\u{1F642}'));
  send(win, talkEv('z2', 'real clip', {audioB64: B64}));

  assert.strictEqual(players.length, 1,
    'an unspeakable talk must not block the queue');
  assert.strictEqual(players[0].playCalls, 1);
  assertNoSpeech(spoken, 'unspeakable talk text');
  console.log('PASS: unspeakable talk text does not deadlock the queue');
}

function testMissingAudioApiDegradesToSilence() {
  const {win} = makePage({beforeScripts: stubRaf});
  const spoken = installSpeech(win);
  win.Audio = undefined;

  send(win, talkEv('n1', 'cannot play me', {audioB64: B64}));
  send(win, talkEv('n2', 'me neither', {audioB64: B64}));

  const players = installAudio(win, Promise.resolve());
  send(win, talkEv('n3', 'now playable', {audioB64: B64}));
  assert.strictEqual(players.length, 1,
    'the queue must stay usable after silent degradation');
  assert.strictEqual(players[0].playCalls, 1);
  assertNoSpeech(spoken, 'missing Audio API');
  console.log('PASS: missing Audio API degrades to silence, no deadlock');
}

function testSingleAudioTalkStillPlaysImmediately() {
  const {win} = makePage({beforeScripts: stubRaf});
  const players = installAudio(win, Promise.resolve());

  send(win, talkEv('one', 'hello', {audioB64: B64}));

  assert.strictEqual(players.length, 1);
  assert.strictEqual(players[0].playCalls, 1);
  console.log('PASS: a single audio talk still plays immediately');
}

async function main() {
  testSecondAudioWaitsForFirstEnded();
  testThreeAudioClipsPlayInFifoOrder();
  testAudioErrorAdvancesQueue();
  testClipLessTalkIsSilentAndAdvancesQueue();
  testLeadingClipLessTalkDoesNotDelayAudio();
  await testRejectedPlayDegradesSilentlyThenAdvances();
  await testLateErrorAfterRejectedPlayDoesNotDoubleAdvance();
  testAudioAbortAdvancesQueue();
  testUnspeakableTextDoesNotDeadlockQueue();
  testMissingAudioApiDegradesToSilence();
  testSingleAudioTalkStillPlaysImmediately();
  console.log('All talkAudioOverlap tests passed.');
}

// The pages are left with a clip "playing": its stall sampler
// (main.js TALK_STALL_CHECK_MS interval) would keep node alive.
main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  },
);
