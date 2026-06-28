// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for the per-tab work_dir invariant.
//
// Invariant (INVARIANTS.md → Tabs & chat webview):
//   When a tab has an associated chat-id of a REAL persisted task,
//   changing the working directory in the settings page MUST NOT
//   change the directory of that tab and chat id.
//
// The chat webview routes follow-up commands (submit, autocommitAction,
// worktreeAction, mergeAction, …) through ``workDirForTab(tabId)``.
// That helper used to return ``tab.workDir`` if known, else the
// daemon-global ``configWorkDir`` reported by the latest ``configData``.
//
// Bug: when the daemon binds a chat-id to a tab via a ``clear`` event
// (the normal path for a freshly-submitted task whose ``task_events``
// replay hasn't run because the task is brand new), ``tab.workDir``
// stays empty.  A later settings-panel change updates ``configWorkDir``
// — and the very next command issued from that tab is routed to the
// NEW directory, even though the bound chat-id still belongs to the
// original task.  Auto-commit/merge then operate on the wrong repo;
// follow-up submits inherit a wrong ``work_dir`` snapshot.
//
// Fix: pin ``tab.workDir`` from ``configWorkDir`` (when ``tab.workDir``
// is still empty) the moment a chat-id is bound to a tab — both on the
// ``clear`` event and on the ``task_events`` replay path — so a later
// settings change cannot shift that tab's effective work_dir.
//
// This test drives production media/chat.html + panelCopy.js + main.js
// in JSDOM.  Run directly with:
//
//     node src/kiss/agents/vscode/test/tabWorkDirSettingsInvariant.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

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

  const posted = [];
  let state;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function lastMsg(posted, type) {
  for (let i = posted.length - 1; i >= 0; i -= 1) {
    if (posted[i] && posted[i].type === type) return posted[i];
  }
  return null;
}

function initialTabId(posted) {
  const ready = posted.find(m => m && m.type === 'ready');
  return ready ? ready.tabId : '';
}

/**
 * Drive the chat webview through the exact event sequence the daemon
 * uses for a freshly-submitted task whose chat-id is bound via
 * ``clear``, then change the configured work_dir from the settings
 * panel, then trigger one autocommit prompt round-trip.  Returns the
 * ``workDir`` field of the ``autocommitAction`` message the webview
 * posts when the user clicks "Auto commit".
 */
function workDirOnAutocommitAfterSettingsChange(initialWd, newWd, bindKind) {
  const {win, posted} = makeWebview();
  const tabId = initialTabId(posted);
  assert.ok(tabId, 'main.js must announce the initial tab id');

  // (1) The daemon's startup ``configData`` reports the configured
  // ``work_dir`` for this connection.
  send(win, {
    type: 'configData',
    config: {work_dir: initialWd},
    apiKeys: {},
  });

  // (2) The daemon binds a REAL persisted chat-id to this tab.
  // Two production paths reach this state:
  //   * 'clear' is broadcast for a freshly-submitted task as soon as
  //     the agent persists it in the DB (no ``task_events`` replay).
  //   * 'task_events' is broadcast when the user reopens a chat from
  //     the history sidebar (extra MAY carry the task's persisted
  //     ``work_dir``; for older rows it does not).
  if (bindKind === 'clear') {
    send(win, {type: 'clear', tabId: tabId, chat_id: 'chat-real-task'});
  } else if (bindKind === 'task_events_no_extra_workdir') {
    send(win, {
      type: 'task_events',
      tabId: tabId,
      chat_id: 'chat-real-task',
      task_id: 42,
      task: 'Real persisted task',
      events: [],
      // ``extra`` here intentionally omits ``work_dir`` to model the
      // older persisted rows whose ``extra`` does not carry it.
      extra: JSON.stringify({startTs: 1_700_000_000_000}),
    });
  } else {
    throw new Error('unknown bindKind: ' + bindKind);
  }

  // (3) The user opens the settings panel and changes the configured
  //     work_dir.  This is the trigger that USED to leak into every
  //     tab via ``workDirForTab``'s ``configWorkDir`` fallback.
  send(win, {
    type: 'configData',
    config: {work_dir: newWd},
    apiKeys: {},
  });

  // (4) The daemon prompts the bound chat for autocommit.
  send(win, {
    type: 'autocommit_prompt',
    tabId: tabId,
    changedFiles: ['file.txt'],
  });

  const btn = win.document.querySelector('.wt-merge');
  assert.ok(btn, 'autocommit bar must render an "Auto commit" button');
  btn.click();

  const msg = lastMsg(posted, 'autocommitAction');
  assert.ok(msg, 'clicking Auto commit must post an autocommitAction');
  win.close();
  return msg.workDir;
}

function testInvariantHoldsAfterSettingsChange_ClearBind() {
  const wd = workDirOnAutocommitAfterSettingsChange(
    '/path/initial',
    '/path/new',
    'clear',
  );
  assert.strictEqual(
    wd,
    '/path/initial',
    'INVARIANT: after the user changes work_dir in settings, a tab ' +
      'whose backend chat-id was bound via a "clear" event MUST keep ' +
      'routing commands to the work_dir it had when the chat-id was ' +
      'bound — observed workDir = ' +
      JSON.stringify(wd),
  );
  console.log('  ok - clear-bound tab keeps its original work_dir after settings change');
}

function testInvariantHoldsAfterSettingsChange_TaskEventsBindNoExtraWorkdir() {
  const wd = workDirOnAutocommitAfterSettingsChange(
    '/path/initial',
    '/path/new',
    'task_events_no_extra_workdir',
  );
  assert.strictEqual(
    wd,
    '/path/initial',
    'INVARIANT: after the user changes work_dir in settings, a tab ' +
      'whose backend chat-id was bound via a "task_events" replay (and ' +
      'whose persisted "extra" carries no ``work_dir`` — older rows) ' +
      'MUST keep routing commands to the work_dir it had when the ' +
      'chat-id was bound — observed workDir = ' +
      JSON.stringify(wd),
  );
  console.log(
    '  ok - task_events-bound tab keeps its original work_dir after settings change',
  );
}

function testTaskEventsExtraWorkDirStillWinsOverConfig() {
  // Sanity: when ``extra.work_dir`` is present it MUST pin the tab's
  // ``workDir`` to the task's recorded directory regardless of any
  // settings change (this path was already correct; the test guards
  // against future regressions of the same invariant).
  const {win, posted} = makeWebview();
  const tabId = initialTabId(posted);

  send(win, {
    type: 'configData',
    config: {work_dir: '/path/initial'},
    apiKeys: {},
  });

  send(win, {
    type: 'task_events',
    tabId: tabId,
    chat_id: 'chat-real-task',
    task_id: 42,
    task: 'Real persisted task',
    events: [],
    extra: JSON.stringify({work_dir: '/path/task-recorded'}),
  });

  send(win, {
    type: 'configData',
    config: {work_dir: '/path/new'},
    apiKeys: {},
  });

  send(win, {
    type: 'autocommit_prompt',
    tabId: tabId,
    changedFiles: ['file.txt'],
  });

  win.document.querySelector('.wt-merge').click();
  const msg = lastMsg(posted, 'autocommitAction');
  assert.strictEqual(
    msg.workDir,
    '/path/task-recorded',
    'extra.work_dir must pin the tab even when configWorkDir later changes',
  );
  win.close();
  console.log('  ok - extra.work_dir pin survives settings change');
}

function main() {
  testInvariantHoldsAfterSettingsChange_ClearBind();
  testInvariantHoldsAfterSettingsChange_TaskEventsBindNoExtraWorkdir();
  testTaskEventsExtraWorkDirStillWinsOverConfig();
  console.log('tabWorkDirSettingsInvariant.test.js: all assertions passed.');
}

main();
