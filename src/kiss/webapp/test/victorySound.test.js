// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The victory jingle: the page plays media/victory.mp3 (the URL the
// daemon puts in window.__VOICE__.victoryAudioUrl) once when a task of
// its own finishes -- a live `task_done` for a tab it shows, the
// visible one or a background one, local or relayed from a registered
// machine -- and stays quiet for a sub-agent child's end, for an error
// or a stop, for a replayed transcript, for a tab it does not have, and
// when the Settings checkbox muted it (localStorage `kissVictorySound`).
//
// The real chat.html + panelCopy.js + api.js + main.js run in jsdom;
// `Audio` is the recording stub of speech_helpers (jsdom's own play()
// is unimplemented).

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {installAudio} = require('./speech_helpers');
const {spawnSubagent} = require('./subagent_tab_helpers');

const {send, readyTabId, stubRaf} = harness;
const {test, report} = makeRunner();

const URL = '/media/victory.mp3?v=abc123';
const KEY = 'kissVictorySound';

/**
 * A chat page with the victory clip configured and the Audio stub
 * installed.  `voice: false` leaves __VOICE__ unset, `audio: false`
 * removes the Audio constructor, `playResult` is what the stub's
 * play() returns, `url: null` opens the page on an opaque origin whose
 * localStorage throws.
 */
function makePage(opts) {
  const {voice = true, audio = true, playResult, url, stored} = opts || {};
  let played = [];
  const {win, posted} = harness.makePage({
    bodyClass: 'remote-chat',
    url,
    // Names the main.js instance for the coverage gate.
    sourceUrl: 'victory-main.js',
    beforeScripts: w => {
      stubRaf(w);
      if (voice) w.__VOICE__ = {victoryAudioUrl: URL};
      if (audio) played = installAudio(w, playResult);
      else w.Audio = undefined;
      if (stored !== undefined) w.localStorage.setItem(KEY, stored);
    },
  });
  const tabId = readyTabId(posted);
  send(win, {type: 'status', running: true, tabId, taskId: 'task-1'});
  return {win, posted, played, tabId};
}

function playedOnce(played, what) {
  assert.strictEqual(played.length, 1, what + ': ' + JSON.stringify(played));
  assert.strictEqual(played[0].src, URL, what + ': src');
  assert.strictEqual(played[0].playCalls, 1, what + ': play() calls');
}

test('task_done of the visible tab plays the jingle once', () => {
  const {win, played, tabId} = makePage();
  send(win, {type: 'task_done', success: true, tabId});
  playedOnce(played, 'visible tab');
});

test('task_done without a tab id (the visible tab) plays', () => {
  const {win, played} = makePage();
  send(win, {type: 'task_done'});
  playedOnce(played, 'no tabId');
});

test('a failed finish (success false) is still a finish: it plays', () => {
  const {win, played, tabId} = makePage();
  send(win, {type: 'task_done', success: false, tabId});
  playedOnce(played, 'success false');
});

test('task_done of a BACKGROUND tab of this page plays', () => {
  const {win, posted, played, tabId} = makePage();
  win._testApi.createNewTab();
  const newChat = posted.filter(m => m.type === 'newChat').pop();
  assert.ok(newChat && newChat.tabId !== tabId, 'a second tab opened');
  send(win, {type: 'task_done', success: true, tabId});
  playedOnce(played, 'background tab');
});

test('a sub-agent child\'s own task_done is silent; the parent\'s plays', () => {
  const page = makePage();
  const {win, played, tabId} = page;
  const subTabId = spawnSubagent(page, tabId, 'child-task', 'sub 1', 0);
  send(win, {type: 'status', running: true, tabId: subTabId, taskId: 'child-task'});
  send(win, {type: 'task_done', success: true, tabId: subTabId});
  assert.strictEqual(played.length, 0, 'child: ' + JSON.stringify(played));
  // The user's task has not finished: its tab is still running.
  const parent = win._testApi.openTabs().find(t => t.id === tabId);
  assert.strictEqual(parent.isRunning, true, 'the parent still runs');
  send(win, {type: 'task_done', success: true, tabId});
  playedOnce(played, 'parent after child');
});

test('task_error, task_stopped and task_interrupted are not a victory', () => {
  const {win, played, tabId} = makePage();
  send(win, {type: 'task_error', tabId});
  send(win, {type: 'task_stopped', tabId});
  send(win, {type: 'task_interrupted', tabId});
  assert.strictEqual(played.length, 0, JSON.stringify(played));
});

test('a replayed transcript ending in task_done is silent', () => {
  const {win, played, tabId} = makePage();
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: 'chat-1',
    task_id: '42',
    task: 'an old task',
    events: [
      {type: 'prompt', text: 'an old task'},
      {type: 'system_output', text: 'hello\n'},
      {type: 'result', text: 'done', success: true},
      {type: 'task_done', success: true},
    ],
  });
  assert.strictEqual(played.length, 0, JSON.stringify(played));
});

test('a tab this page does not show (closed by the user) still plays; an unknown sub-agent tab does not', () => {
  const {win, played} = makePage();
  // Closing a chat tab does not stop its task: its finish is one the
  // user is not looking at, the one worth hearing.
  send(win, {type: 'task_done', success: true, tabId: 'tab-closed-earlier'});
  playedOnce(played, 'closed tab');
  // A sub-agent tab of a closed parent is known by the daemon's id
  // convention <parent>__sub_<task>.
  send(win, {type: 'task_done', success: true, tabId: 'tab-closed-earlier__sub_42'});
  assert.strictEqual(played.length, 1, 'sub-agent id: ' + JSON.stringify(played));
  // A run_agent dispatch runs under a synthetic `api-…` tab no page
  // shows; the daemon marks its end `subagent: true`.
  send(win, {type: 'task_done', success: true, tabId: 'api-0f3a', subagent: true});
  assert.strictEqual(played.length, 1, 'dispatched sub-agent: ' + JSON.stringify(played));
});

test('the Settings checkbox mutes and unmutes, and persists', () => {
  const {win, played, tabId} = makePage({stored: 'off'});
  const box = win.document.getElementById('cfg-victory-sound');
  assert.ok(box, 'chat.html has the checkbox');
  assert.strictEqual(box.checked, false, 'a stored "off" unchecks the box');
  send(win, {type: 'task_done', success: true, tabId});
  assert.strictEqual(played.length, 0, 'muted: ' + JSON.stringify(played));

  box.checked = true;
  box.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.strictEqual(win.localStorage.getItem(KEY), 'on');
  send(win, {type: 'status', running: true, tabId, taskId: 'task-2'});
  send(win, {type: 'task_done', success: true, tabId});
  playedOnce(played, 'unmuted');

  box.checked = false;
  box.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.strictEqual(win.localStorage.getItem(KEY), 'off');
});

test('the checkbox starts checked by default and follows other pages', () => {
  const {win} = makePage();
  const box = win.document.getElementById('cfg-victory-sound');
  assert.strictEqual(box.checked, true, 'default on');
  // Another same-origin page unchecked its box.
  win.localStorage.setItem(KEY, 'off');
  win.dispatchEvent(new win.StorageEvent('storage', {key: KEY, newValue: 'off'}));
  assert.strictEqual(box.checked, false, 'follows the storage event');
  win.dispatchEvent(new win.StorageEvent('storage', {key: 'other', newValue: 'x'}));
  assert.strictEqual(box.checked, false, 'ignores other keys');
});

test('an opaque origin (localStorage throws) plays and keeps the box usable', () => {
  const {win, played, tabId} = makePage({url: null});
  assert.throws(() => win.localStorage.getItem(KEY), 'the fixture origin has no storage');
  const box = win.document.getElementById('cfg-victory-sound');
  assert.strictEqual(box.checked, true, 'default on without storage');
  box.checked = false;
  box.dispatchEvent(new win.Event('change', {bubbles: true}));
  send(win, {type: 'task_done', success: true, tabId});
  playedOnce(played, 'no storage');
});

test('no Audio constructor, or no clip configured: nothing to play, no error', () => {
  const noAudio = makePage({audio: false});
  send(noAudio.win, {type: 'task_done', success: true, tabId: noAudio.tabId});
  const noVoice = makePage({voice: false});
  send(noVoice.win, {type: 'task_done', success: true, tabId: noVoice.tabId});
  assert.strictEqual(noVoice.played.length, 0, JSON.stringify(noVoice.played));
});

function domError(name) {
  const err = new Error(name);
  err.name = name;
  return err;
}

function gesture(win, type) {
  win.document.dispatchEvent(new win.Event(type, {bubbles: true}));
}

const settle = () => new Promise(resolve => setImmediate(resolve));

const last = test('every play() the autoplay policy refuses sounds at the next gesture, in turn', async () => {
  const rejections = [];
  const onRejection = e => rejections.push(e);
  process.on('unhandledRejection', onRejection);
  try {
    // Refused until the page has had a gesture, then allowed.
    let allowed = false;
    const page = makePage({
      playResult: () =>
        allowed ? Promise.resolve() : Promise.reject(domError('NotAllowedError')),
    });
    const {win, played, tabId} = page;
    send(win, {type: 'task_done', success: true, tabId});
    send(win, {type: 'status', running: true, tabId, taskId: 'task-2'});
    send(win, {type: 'task_done', success: true, tabId});
    await settle();
    assert.strictEqual(played.length, 2, JSON.stringify(played));
    assert.deepStrictEqual(played.map(p => p.playCalls), [1, 1]);
    allowed = true;
    gesture(win, 'pointerdown');
    await settle();
    // The first refused finish sounds; the second waits for it to end.
    assert.deepStrictEqual(played.map(p => p.playCalls), [2, 1], 'first replay');
    played[0].fire('ended');
    await settle();
    assert.deepStrictEqual(played.map(p => p.playCalls), [2, 2], 'second in turn');
    gesture(win, 'keydown');
    gesture(win, 'pointerdown');
    await settle();
    assert.deepStrictEqual(played.map(p => p.playCalls), [2, 2], 'nothing left');
    // A refusal at the gesture too (still no activation) keeps it for the next one.
    allowed = false;
    send(win, {type: 'status', running: true, tabId, taskId: 'task-3'});
    send(win, {type: 'task_done', success: true, tabId});
    await settle();
    gesture(win, 'keydown');
    await settle();
    allowed = true;
    gesture(win, 'keydown');
    await settle();
    assert.deepStrictEqual(played.map(p => p.playCalls), [2, 2, 3], 'kept until allowed');
    // A queued jingle that fails to play (an error) does not block the next.
    allowed = false;
    send(win, {type: 'status', running: true, tabId, taskId: 'task-4'});
    send(win, {type: 'task_done', success: true, tabId});
    send(win, {type: 'status', running: true, tabId, taskId: 'task-5'});
    send(win, {type: 'task_done', success: true, tabId});
    await settle();
    allowed = true;
    gesture(win, 'pointerdown');
    await settle();
    assert.deepStrictEqual(played.map(p => p.playCalls), [2, 2, 3, 2, 1], 'fourth plays');
    played[3].fire('error');
    await settle();
    assert.deepStrictEqual(played.map(p => p.playCalls), [2, 2, 3, 2, 2], 'fifth follows');

    // Any other failure is dropped: an undecodable clip is not retried.
    const broken = makePage({
      playResult: () => Promise.reject(domError('NotSupportedError')),
    });
    send(broken.win, {type: 'task_done', success: true, tabId: broken.tabId});
    await settle();
    gesture(broken.win, 'pointerdown');
    await settle();
    playedOnce(broken.played, 'undecodable clip');
    // A play() that returns nothing (an old engine) is fine too.
    const bare = makePage({playResult: () => null});
    send(bare.win, {type: 'task_done', success: true, tabId: bare.tabId});
    playedOnce(bare.played, 'bare play');
    // And one that throws.
    const throwing = makePage({
      playResult: () => {
        throw new Error('boom');
      },
    });
    send(throwing.win, {type: 'task_done', success: true, tabId: throwing.tabId});
    gesture(throwing.win, 'pointerdown');
    playedOnce(throwing.played, 'throwing play');
    await settle();
    assert.deepStrictEqual(rejections, [], 'no unhandled rejection');
  } finally {
    process.off('unhandledRejection', onRejection);
  }
});

Promise.resolve(last).then(() => {
  report('victorySound');
  process.exit(0);
});
