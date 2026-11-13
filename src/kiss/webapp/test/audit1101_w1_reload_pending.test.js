// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/* global require, console, process */

// Audit 2026-11-01 (W1 F1-F3): a content tab's "Reload from disk" is
// one tokened openFile, and only its own reply settles it.
//
//  F1  edits typed while the reload's reply was on its way were never
//      consented away: the reply keeps the edited editor (as any open
//      of a dirty file does) instead of replacing it with the disk's
//      text and dropping the edits;
//  F2  an error reply for the same path and chat on ANOTHER machine
//      (the chat's open of the file's twin on gpu-box) does not cancel
//      the local reload;
//  F3  the reply of a reload whose tab was closed meanwhile is dropped
//      instead of reopening the closed file in a new tab (its error
//      toasts nothing either); a later plain open still opens the file.
//
// Driven through the real chat.html + api.js + main.js in jsdom with the
// stub Monaco of the ui_antipattern harness.  The fixtures answer a
// reload the way the daemon does: web_server._handle_open_file echoes
// the request's token on every reply, success or error (a forwarded
// request is echoed by the machine's daemon, and a refused one by
// sidebar_panels._machine_reply).

const assert = require('assert');
const H = require('./ui_antipattern_harness');
const {contentTabs} = require('./subagent_tab_helpers');

const {makePage, send, sleep, click, toast, toastButton} = H;

const FILE = '/shared/notes.txt';

function setUp(ctx) {
  const {win} = ctx;
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {work_dir: '/ws/a'}});
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 'a1', workDir: '/ws/a', chatId: 'chat-1'}],
  });
}

/** A fileContent reply for FILE to chat a1's request, *fields* on top. */
function reply(win, fields) {
  send(
    win,
    Object.assign(
      {
        type: 'fileContent',
        tabId: 'a1',
        path: FILE,
        name: 'notes.txt',
        content: 'disk v1',
        version: 'v1',
      },
      fields,
    ),
  );
}

function lastOpenFile(posted) {
  return posted.filter(m => m.type === 'openFile').pop();
}

/**
 * Open FILE in an editor, edit it, have the save refused as a conflict
 * and click the toast's "Reload from disk".  Returns the editor and the
 * openFile the reload sent.
 */
async function armReload(ctx) {
  const {win, posted, created} = ctx;
  setUp(ctx);
  reply(win);
  // The stubbed editor is created on the next microtask.
  await sleep(0);
  assert.strictEqual(created.length, 1, 'the file got an editor');
  const editor = created[0].editor;
  editor._type('unsaved pre-reload');
  click(win, win.document.querySelector('.content-save-btn'));
  const save = posted.filter(m => m.type === 'saveFile').pop();
  assert.ok(save, 'Save sends saveFile');
  send(win, {
    type: 'fileSaved',
    token: save.token,
    ok: false,
    conflict: true,
    error: 'notes.txt changed on disk',
  });
  const conflict = win.document.querySelector(
    '[data-notification-id^="file-save-conflict-"]',
  );
  assert.ok(conflict, 'the conflict toast');
  click(win, toastButton(conflict, 'Reload from disk'));
  const request = lastOpenFile(posted);
  assert.ok(request, 'the reload asks the daemon for the file');
  assert.strictEqual(request.path, FILE);
  assert.strictEqual(request.tabId, 'a1', 'asked as the owning chat');
  return {editor, request};
}

async function testReloadIsATokenedOpenFile() {
  const ctx = makePage();
  const {win, posted, created} = ctx;
  try {
    const {request} = await armReload(ctx);
    assert.strictEqual(
      typeof request.token,
      'string',
      'BUG - the reload must carry its own token',
    );
    assert.ok(request.token, 'the token is not empty');
    reply(win, {token: request.token, content: 'disk v2', version: 'v2'});
    await sleep(0);
    assert.strictEqual(created.length, 2, 'the reply replaces the editor');
    assert.strictEqual(created[1].editor.getModel().getValue(), 'disk v2');
    // Asked again from the Save bar: a reload of its own.
    click(win, win.document.querySelector('.content-refresh-btn'));
    const again = lastOpenFile(posted);
    assert.ok(again.token, 'the second reload carries a token');
    assert.notStrictEqual(
      again.token,
      request.token,
      'every reload has its own token',
    );
    // The daemon's machine-bound request: the token travels with it.
    reply(win, {token: again.token, content: 'disk v3', version: 'v3'});
    await sleep(0);
    assert.strictEqual(created.length, 3, 'the second reply lands too');
    assert.strictEqual(created[2].editor.getModel().getValue(), 'disk v3');
  } finally {
    win.close();
  }
}

async function testEditsTypedWhileTheReloadIsPendingSurvive() {
  const ctx = makePage();
  const {win, posted, created} = ctx;
  try {
    const {editor, request} = await armReload(ctx);
    // A slow read: the user types again before the reply lands.
    editor._type('typed after the reload request');
    reply(win, {token: request.token, content: 'disk v2', version: 'v2'});
    await sleep(0);
    assert.strictEqual(
      created.length,
      1,
      'BUG - the reply must not replace an editor edited after the request',
    );
    assert.strictEqual(
      editor.getModel().getValue(),
      'typed after the reload request',
      'the edits typed meanwhile survive',
    );
    assert.ok(
      win.document.querySelector('.chat-tab-dirty'),
      'the tab strip still shows the dirty marker',
    );
    // That reload is spent: a plain open of the file by the same chat
    // keeps the edits like any open of a dirty file.
    reply(win, {content: 'disk v2', version: 'v2'});
    await sleep(0);
    assert.strictEqual(created.length, 1, 'a plain open keeps the editor');
    // Asked again without typing, the reload lands.
    click(win, win.document.querySelector('.content-refresh-btn'));
    const again = lastOpenFile(posted);
    reply(win, {token: again.token, content: 'disk v3', version: 'v3'});
    await sleep(0);
    assert.strictEqual(created.length, 2, 'the new reload replaces the editor');
    assert.strictEqual(created[1].editor.getModel().getValue(), 'disk v3');
    assert.ok(
      !win.document.querySelector('.chat-tab-dirty'),
      'the reloaded tab is clean',
    );
  } finally {
    win.close();
  }
}

async function testAnotherMachinesErrorKeepsTheLocalReload() {
  const ctx = makePage();
  const {win, created} = ctx;
  try {
    const {request} = await armReload(ctx);
    // The chat's own open of the same path on gpu-box fails meanwhile:
    // another request (no token), another machine, same path and chat.
    reply(win, {machine: 'gpu-box', error: 'remote file not found'});
    reply(win, {token: request.token, content: 'disk v2', version: 'v2'});
    await sleep(0);
    assert.strictEqual(
      created.length,
      2,
      "BUG - gpu-box's error must not cancel the local reload",
    );
    assert.strictEqual(created[1].editor.getModel().getValue(), 'disk v2');
    assert.ok(
      !win.document.querySelector('.chat-tab-dirty'),
      'the reloaded tab is clean',
    );
  } finally {
    win.close();
  }
}

async function testTheReplyOfAClosedTabsReloadIsDropped() {
  const ctx = makePage();
  const {win, created} = ctx;
  try {
    const {request} = await armReload(ctx);
    const content = contentTabs(win)[0];
    const row = Array.from(
      win.document.querySelectorAll('.chat-tab.content-tab'),
    ).find(el => el.dataset.tabId === content.id);
    assert.ok(row, 'the file is on the tab strip');
    click(win, row.querySelector('.chat-tab-close'));
    click(
      win,
      toastButton(toast(win, 'close-dirty-' + content.id), "Don't save"),
    );
    assert.strictEqual(contentTabs(win).length, 0, 'the file is closed');
    // The reload's reply lands after the close.
    reply(win, {token: request.token, content: 'disk v2', version: 'v2'});
    await sleep(0);
    assert.strictEqual(
      contentTabs(win).length,
      0,
      'BUG - the late reload reply must not reopen the closed file',
    );
    assert.strictEqual(created.length, 1, 'no new editor');
    // Nor does its failure toast for a tab that is gone.
    reply(win, {token: request.token, error: 'notes.txt was deleted'});
    assert.ok(
      !win.document.querySelector('[data-notification-id="file-open-error"]'),
      "a closed tab's reload error toasts nothing",
    );
    // A plain open of the file afterwards (a link click) still opens it.
    reply(win, {content: 'disk v2', version: 'v2'});
    await sleep(0);
    assert.strictEqual(contentTabs(win).length, 1, 'an open still opens it');
    assert.strictEqual(created.length, 2);
    assert.strictEqual(created[1].editor.getModel().getValue(), 'disk v2');
  } finally {
    win.close();
  }
}

async function testTheLateReplyOfAClosedPlainOpenIsDropped() {
  // A transcript link clicked repeatedly while the daemon is slow
  // sends one plain open per click.  The first reply opens the
  // editor, the user closes it, and a later reply of an open sent
  // before the close must not resurrect it; a click after the close
  // is a new open and opens it again -- and the replies of the opens
  // sent before the close stay rejected after that reopen too (their
  // old contents or their error must not land on the new editor).
  const ctx = makePage();
  const {win, posted, created} = ctx;
  try {
    setUp(ctx);
    const link = win.document.createElement('span');
    link.className = 'kiss-filelink';
    link.dataset.path = FILE;
    win.document.getElementById('output').appendChild(link);
    for (let i = 0; i < 4; i++) click(win, link);
    const opens = posted.filter(m => m.type === 'openFile');
    assert.strictEqual(opens.length, 4, 'four plain opens');
    assert.strictEqual(
      new Set(opens.map(o => o.token)).size,
      4,
      'each open has its own token',
    );
    reply(win, {token: opens[0].token});
    await sleep(0);
    assert.strictEqual(contentTabs(win).length, 1, 'the first reply opens');
    const content = contentTabs(win)[0];
    const row = Array.from(
      win.document.querySelectorAll('.chat-tab.content-tab'),
    ).find(el => el.dataset.tabId === content.id);
    click(win, row.querySelector('.chat-tab-close'));
    assert.strictEqual(contentTabs(win).length, 0, 'the file is closed');
    reply(win, {token: opens[1].token});
    await sleep(0);
    assert.strictEqual(
      contentTabs(win).length,
      0,
      'BUG - the late reply of an open sent before the close reopens it',
    );
    assert.strictEqual(created.length, 1, 'no new editor');
    click(win, link);
    const again = lastOpenFile(posted);
    assert.notStrictEqual(again.token, opens[1].token);
    reply(win, {token: again.token, content: 'disk v2', version: 'v2'});
    await sleep(0);
    assert.strictEqual(contentTabs(win).length, 1, 'a new open opens it');
    assert.strictEqual(created[1].editor.getModel().getValue(), 'disk v2');
    // The replies of the opens sent before the close arrive after the
    // reopen: still rejected.
    reply(win, {token: opens[2].token, content: 'old delayed data'});
    await sleep(0);
    assert.strictEqual(
      created[created.length - 1].editor.getModel().getValue(),
      'disk v2',
      'BUG - an old reply arriving after the reopen replaced the file',
    );
    assert.strictEqual(created.length, 2, 'no editor rebuilt');
    assert.strictEqual(contentTabs(win).length, 1);
    reply(win, {token: opens[3].token, error: 'notes.txt: gone'});
    await sleep(0);
    assert.strictEqual(
      toast(win, 'file-open-error'),
      null,
      'an old error reply arriving after the reopen toasts nothing',
    );
    assert.strictEqual(created[1].editor.getModel().getValue(), 'disk v2');
  } finally {
    win.close();
  }
}

/** Open a file with a fileContent reply of *fields*, then close its tab. */
async function openAndClose(win, fields) {
  reply(win, fields);
  await sleep(0);
  assert.strictEqual(contentTabs(win).length, 1, 'the reply opens the file');
  click(
    win,
    win.document.querySelector('.chat-tab.content-tab .chat-tab-close'),
  );
  assert.strictEqual(contentTabs(win).length, 0, 'the file is closed');
}

// W1 audit 2026-11-12: the page remembers every file it ever closed,
// one record each, for its whole life, on purpose.  A reply may be
// outstanding across any number of closes: forgetting the oldest closes
// reopened a closed file, and declaring every older request late
// instead dropped the reply of a file never closed (both tried and
// reviewed).  This pins the two behaviours a bound would trade.
async function testCloseRecordsOutliveAnyNumberOfCloses() {
  const ctx = makePage();
  const {win, posted} = ctx;
  try {
    setUp(ctx);
    const link = win.document.createElement('span');
    link.className = 'kiss-filelink';
    link.dataset.path = FILE;
    win.document.getElementById('output').appendChild(link);
    click(win, link);
    click(win, link);
    const other = win.document.createElement('span');
    other.className = 'kiss-filelink';
    other.dataset.path = '/shared/never-closed.txt';
    win.document.getElementById('output').appendChild(other);
    click(win, other);
    const opens = posted.filter(m => m.type === 'openFile');
    await openAndClose(win, {token: opens[0].token});
    for (let i = 0; i < 300; i++) {
      await openAndClose(win, {path: '/shared/f' + i, name: 'f' + i});
    }
    reply(win, {token: opens[1].token});
    await sleep(0);
    assert.strictEqual(
      contentTabs(win).length,
      0,
      'BUG - the late reply reopened a file closed 300 closes ago',
    );
    reply(win, {
      token: opens[2].token,
      path: '/shared/never-closed.txt',
      name: 'never-closed.txt',
    });
    await sleep(0);
    assert.strictEqual(
      contentTabs(win).length,
      1,
      'BUG - the late reply of a file never closed was dropped',
    );
  } finally {
    win.close();
  }
}

(async () => {
  await testReloadIsATokenedOpenFile();
  console.log('  ok - a reload is a tokened openFile, each with its own token');
  await testEditsTypedWhileTheReloadIsPendingSurvive();
  console.log(
    '  ok - edits typed while the reload is pending survive its reply',
  );
  await testAnotherMachinesErrorKeepsTheLocalReload();
  console.log("  ok - another machine's error keeps the local reload armed");
  await testTheReplyOfAClosedTabsReloadIsDropped();
  console.log(
    "  ok - a closed tab's reload reply is dropped, a later open opens",
  );
  await testTheLateReplyOfAClosedPlainOpenIsDropped();
  console.log("  ok - a closed editor's late plain-open reply is dropped");
  await testCloseRecordsOutliveAnyNumberOfCloses();
  console.log('  ok - close records outlive any number of closes');
})().catch(e => {
  console.error(e);
  process.exit(1);
});
