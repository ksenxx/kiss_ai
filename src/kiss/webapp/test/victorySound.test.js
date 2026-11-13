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

const {send, readyTabId, stubRaf, flushMicrotasks: settle} = harness;
// Every page a test builds; the runner closes them when the test has ended.
const openWindows = [];
const {test, report} = makeRunner({windows: openWindows});

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
  openWindows.push(win);
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

test('the copy of a task end sent to a tab viewing the task ends that tab in silence', () => {
  const {win, played, tabId} = makePage();
  const viewing = () => win._testApi.openTabs().find(t => t.id === tabId);
  // The task was launched in a tab closed since: its end reaches every
  // page and sounds.  This page shows the task in another tab, which
  // the daemon sends a copy of the end, marked `viewer`.
  send(win, {type: 'task_done', success: true, tabId: 'tab-that-launched-it'});
  playedOnce(played, 'the end of the task');
  assert.strictEqual(viewing().isRunning, true, 'the viewing tab is not told yet');
  send(win, {type: 'task_done', success: true, tabId, viewer: true});
  playedOnce(played, 'BUG - a second jingle for one task end');
  assert.strictEqual(viewing().isRunning, false, 'BUG - the copy does not end the viewing tab');
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
  const errors = [];
  noAudio.win.addEventListener('error', e => {
    errors.push(String(e.error));
    e.preventDefault();
  });
  send(noAudio.win, {type: 'task_done', success: true, tabId: noAudio.tabId});
  assert.deepStrictEqual(errors, [], 'no Audio constructor: no error');
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

const last = test('every play() the autoplay policy refuses sounds at the next gesture, in turn', async () => {
  const rejections = [];
  const onRejection = e => rejections.push(e);
  process.on('unhandledRejection', onRejection);
  try {
    // Refused until the page has had a gesture, then allowed; `broken`
    // is a player whose clip the browser cannot play.
    let allowed = false;
    let broken = null;
    const page = makePage({
      playResult: player => {
        if (player === broken) return Promise.reject(domError('NotSupportedError'));
        return allowed ? Promise.resolve() : Promise.reject(domError('NotAllowedError'));
      },
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
    // A queued jingle whose clip fails to load (an error) is fetched
    // once more on a fresh player for the same clip, which the next
    // one waits for; a second failure gives up and the next follows.
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
    assert.strictEqual(played.length, 6, 'the failed clip is fetched again');
    assert.strictEqual(played[5].src, URL, 'retry: same clip');
    assert.deepStrictEqual(
      played.map(p => p.playCalls), [2, 2, 3, 2, 1, 1], 'retry plays, fifth waits',
    );
    played[5].fire('error');
    await settle();
    assert.strictEqual(played.length, 6, 'a second failure is not retried');
    assert.deepStrictEqual(played.map(p => p.playCalls), [2, 2, 3, 2, 2, 1], 'fifth follows');
    // A retry that plays to its end is that finish's one jingle.
    send(win, {type: 'status', running: true, tabId, taskId: 'task-6'});
    send(win, {type: 'task_done', success: true, tabId});
    send(win, {type: 'status', running: true, tabId, taskId: 'task-7'});
    send(win, {type: 'task_done', success: true, tabId});
    await settle();
    assert.strictEqual(played.length, 8, 'sixth and seventh');
    assert.deepStrictEqual(played.slice(6).map(p => p.playCalls), [1, 1], 'both played at once');
    played[6].fire('error');
    await settle();
    assert.strictEqual(played.length, 9, 'the sixth is retried');
    // A failed load reports twice (the error event and the rejected
    // play()): the second report changes nothing.
    played[6].fire('error');
    await settle();
    assert.strictEqual(played.length, 9, 'one retry per failure');
    played[8].fire('ended');
    await settle();
    assert.deepStrictEqual(played.slice(6).map(p => p.playCalls), [1, 1, 1], 'nothing queued');

    // A refused player whose clip then fails to load is replaced in the
    // deferred queue, in its place, by the second fetch, which the
    // gesture starts; one that cannot play then (a rejection other than
    // the policy's) is dropped and the finish after it sounds.
    allowed = false;
    send(win, {type: 'status', running: true, tabId, taskId: 'task-8'});
    send(win, {type: 'task_done', success: true, tabId});
    await settle();
    played[9].fire('error');
    await settle();
    assert.strictEqual(played.length, 11, 'the refused player is fetched again');
    assert.strictEqual(played[10].playCalls, 0, 'the second fetch waits for the gesture');
    send(win, {type: 'status', running: true, tabId, taskId: 'task-9'});
    send(win, {type: 'task_done', success: true, tabId});
    await settle();
    allowed = true;
    broken = played[10];
    gesture(win, 'pointerdown');
    await settle();
    assert.strictEqual(played.length, 12, 'a second fetch that fails is not fetched again');
    assert.deepStrictEqual(played.slice(9).map(p => p.playCalls), [1, 1, 2], 'ninth in turn');
    // A second fetch that is refused by the policy and then fails to
    // load leaves the queue without a player for that finish.
    allowed = false;
    broken = null;
    send(win, {type: 'status', running: true, tabId, taskId: 'task-10'});
    send(win, {type: 'task_done', success: true, tabId});
    await settle();
    played[12].fire('error');
    gesture(win, 'keydown');
    await settle();
    assert.strictEqual(played.length, 14, 'second fetch, refused');
    assert.strictEqual(played[13].playCalls, 1, 'refused at the gesture');
    played[13].fire('error');
    allowed = true;
    gesture(win, 'keydown');
    await settle();
    assert.strictEqual(played.length, 14, 'dropped: nothing fetched again');
    assert.deepStrictEqual(played.slice(12).map(p => p.playCalls), [1, 1], 'nothing to sound');
    // One waiting its turn behind the player the gesture started (the
    // deferred queue drained into that player's `next`) whose clip
    // fails to load meanwhile is replaced there too, so the second
    // fetch sounds in its turn and the one after it still follows.
    allowed = false;
    for (const task of ['task-11', 'task-12', 'task-13']) {
      send(win, {type: 'status', running: true, tabId, taskId: task});
      send(win, {type: 'task_done', success: true, tabId});
    }
    await settle();
    allowed = true;
    gesture(win, 'pointerdown');
    await settle();
    assert.deepStrictEqual(played.slice(14).map(p => p.playCalls), [2, 1, 1], 'first of three');
    played[15].fire('error');
    await settle();
    assert.strictEqual(played.length, 18, 'the waiting one is fetched again');
    assert.strictEqual(played[17].playCalls, 0, 'and waits its turn');
    played[14].fire('ended');
    await settle();
    assert.deepStrictEqual(
      played.slice(14).map(p => p.playCalls), [2, 1, 1, 1], 'the second fetch in its turn',
    );
    played[17].fire('ended');
    await settle();
    assert.deepStrictEqual(played.slice(14).map(p => p.playCalls), [2, 1, 2, 1], 'the third follows');
    // A second fetch that fails while it waits its turn is taken out of
    // the list, and the one after it follows when its turn comes.
    allowed = false;
    for (const task of ['task-14', 'task-15', 'task-16']) {
      send(win, {type: 'status', running: true, tabId, taskId: task});
      send(win, {type: 'task_done', success: true, tabId});
    }
    await settle();
    allowed = true;
    gesture(win, 'pointerdown');
    await settle();
    played[19].fire('error');
    await settle();
    assert.strictEqual(played.length, 22, 'the waiting one is fetched again');
    played[21].fire('error');
    await settle();
    assert.strictEqual(played.length, 22, 'a second fetch that fails is dropped');
    played[18].fire('ended');
    await settle();
    assert.deepStrictEqual(played.slice(18).map(p => p.playCalls), [2, 1, 2, 0], 'the third follows');

    // A clip the browser cannot play at all (play() rejects with another
    // error, no load error) is fetched once more, then dropped.
    const unplayable = makePage({
      playResult: () => Promise.reject(domError('NotSupportedError')),
    });
    send(unplayable.win, {type: 'task_done', success: true, tabId: unplayable.tabId});
    await settle();
    gesture(unplayable.win, 'pointerdown');
    await settle();
    assert.deepStrictEqual(
      unplayable.played.map(p => [p.src, p.playCalls]), [[URL, 1], [URL, 1]], 'unplayable clip',
    );
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

// A27_fable-4: a finish refused by the autoplay policy waits for the
// next gesture, but the user may mute the jingle meanwhile (the box
// here, or on another same-origin page): the deferred queue sounds
// nothing then, and is dropped with the ones queued after it, so an
// old finish never plays against the current setting.
const mutedLate = test('a deferred jingle muted before the gesture stays silent', async () => {
  await last;
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
  assert.deepStrictEqual(played.map(p => p.playCalls), [1, 1], 'both refused');
  // Muted on another page of this browser profile.
  win.localStorage.setItem(KEY, 'off');
  win.dispatchEvent(new win.StorageEvent('storage', {key: KEY, newValue: 'off'}));
  assert.strictEqual(win.document.getElementById('cfg-victory-sound').checked, false);
  allowed = true;
  gesture(win, 'pointerdown');
  await settle();
  assert.deepStrictEqual(
    played.map(p => p.playCalls),
    [1, 1],
    'BUG - the gesture sounded a jingle the user had muted',
  );
  // Unmuted again: the old finishes are gone, a new one plays.
  win.localStorage.setItem(KEY, 'on');
  gesture(win, 'pointerdown');
  await settle();
  assert.deepStrictEqual(played.map(p => p.playCalls), [1, 1], 'dropped for good');
  send(win, {type: 'status', running: true, tabId, taskId: 'task-3'});
  send(win, {type: 'task_done', success: true, tabId});
  await settle();
  assert.deepStrictEqual(played.map(p => p.playCalls), [1, 1, 1], 'a new finish plays');
  // Muted while a sequence sounds: the ones queued after it stay silent.
  win.localStorage.setItem(KEY, 'off');
  send(win, {type: 'status', running: true, tabId, taskId: 'task-4'});
  send(win, {type: 'task_done', success: true, tabId});
  await settle();
  assert.strictEqual(played.length, 3, 'a muted finish queues nothing');
  win.localStorage.setItem(KEY, 'on');
  allowed = false;
  send(win, {type: 'status', running: true, tabId, taskId: 'task-5'});
  send(win, {type: 'task_done', success: true, tabId});
  send(win, {type: 'status', running: true, tabId, taskId: 'task-6'});
  send(win, {type: 'task_done', success: true, tabId});
  await settle();
  allowed = true;
  gesture(win, 'pointerdown');
  await settle();
  assert.deepStrictEqual(played.map(p => p.playCalls), [1, 1, 1, 2, 1], 'the first sounds');
  win.localStorage.setItem(KEY, 'off');
  played[3].fire('ended');
  await settle();
  assert.deepStrictEqual(
    played.map(p => p.playCalls),
    [1, 1, 1, 2, 1],
    'the one after it is not sounded once muted',
  );
});

// Audit 2026-11-08 (W6 F3): the user agent paused the clip sounding
// (tab backgrounded, audio focus lost) and no `ended` follows -- the
// ones queued after it must still sound, as for playTalkAudio.
const interrupted = test('an interrupted jingle passes its turn; a natural end passes it once', async () => {
  await mutedLate;
  let allowed = false;
  const page = makePage({
    playResult: () =>
      allowed ? Promise.resolve() : Promise.reject(domError('NotAllowedError')),
  });
  const {win, played, tabId} = page;
  send(win, {type: 'task_done', success: true, tabId});
  send(win, {type: 'status', running: true, tabId, taskId: 'task-2'});
  send(win, {type: 'task_done', success: true, tabId});
  send(win, {type: 'status', running: true, tabId, taskId: 'task-3'});
  send(win, {type: 'task_done', success: true, tabId});
  await settle();
  allowed = true;
  gesture(win, 'pointerdown');
  await settle();
  assert.deepStrictEqual(played.map(p => p.playCalls), [2, 1, 1], 'the first sounds');
  // Interrupted: `pause` with no `ended` to come.
  played[0].fire('pause');
  await settle();
  assert.deepStrictEqual(
    played.map(p => p.playCalls),
    [2, 2, 1],
    'BUG - the finishes queued behind an interrupted jingle never sound',
  );
  assert.strictEqual(played[0].src, '', 'the interrupted player is retired');
  // A resume of the interrupted clip, or its late `ended`, is nothing.
  played[0].fire('ended');
  await settle();
  assert.deepStrictEqual(played.map(p => p.playCalls), [2, 2, 1], 'retired for good');
  // A natural end fires `pause` just before `ended`: one turn passes.
  played[1].fire('pause');
  played[1].fire('ended');
  await settle();
  assert.deepStrictEqual(played.map(p => p.playCalls), [2, 2, 2], 'the third sounds once');
  played[2].fire('abort');
  gesture(win, 'pointerdown');
  await settle();
  assert.deepStrictEqual(played.map(p => p.playCalls), [2, 2, 2], 'nothing left');
});

// Review 2026-11-08 (R11 W6/F3): in Chromium a play() whose clip has
// not loaded yet is a pending promise, and pausing the player (the
// user agent, or the retirement itself) fires `pause` and then rejects
// that promise with AbortError.  The `pause` passed the turn already:
// the rejection must not fail the retired player a second time, which
// fetched its emptied source again and passed the turn once more --
// the third jingle sounding over the second.  A rejection of an
// attempt that is still sounding is a failure as before.
const pendingPlay = test('the rejection of a retired player\'s pending play() changes nothing', async () => {
  await interrupted;
  let allowed = false;
  const page = makePage({
    playResult: player => {
      if (!allowed) return Promise.reject(domError('NotAllowedError'));
      return new Promise((resolve, reject) => {
        player.playPending = {resolve, reject};
      });
    },
  });
  const {win, played, tabId} = page;
  for (const task of ['task-1', 'task-2', 'task-3']) {
    send(win, {type: 'status', running: true, tabId, taskId: task});
    send(win, {type: 'task_done', success: true, tabId});
  }
  await settle();
  allowed = true;
  gesture(win, 'pointerdown');
  await settle();
  assert.deepStrictEqual(played.map(p => p.playCalls), [2, 1, 1], 'the first sounds');
  assert.ok(played[0].playPending, 'its play() is still pending');
  // Paused before the clip loaded: `pause` passes the turn, then the
  // pending play() rejects.
  played[0].fire('pause');
  await settle();
  assert.deepStrictEqual(played.map(p => p.playCalls), [2, 2, 1], 'the second sounds');
  assert.strictEqual(played[0].src, '', 'the interrupted player is retired');
  played[0].playPending.reject(domError('AbortError'));
  await settle();
  assert.strictEqual(
    played.length,
    3,
    'BUG - the rejection of the retired player\'s play() fetched its ' +
      'emptied source again: ' + JSON.stringify(played.map(p => p.src)),
  );
  assert.deepStrictEqual(
    played.map(p => p.playCalls),
    [2, 2, 1],
    'BUG - the rejection passed the turn a second time',
  );
  // The second plays on and ends: the third follows, once.
  played[1].playPending.resolve();
  played[1].fire('pause');
  played[1].fire('ended');
  await settle();
  assert.deepStrictEqual(played.map(p => p.playCalls), [2, 2, 2], 'the third in its turn');
  // A clip that cannot play (its pending play() rejects while it is
  // the one sounding) is still fetched once more.
  played[2].playPending.reject(domError('NotSupportedError'));
  await settle();
  assert.strictEqual(played.length, 4, 'the unplayable clip is fetched again');
  assert.strictEqual(played[3].src, URL, 'retry: same clip');
  assert.deepStrictEqual(played.map(p => p.playCalls), [2, 2, 2, 1], 'the retry sounds');
});

Promise.all([last, mutedLate, interrupted, pendingPlay]).then(() => {
  report('victorySound');
});
