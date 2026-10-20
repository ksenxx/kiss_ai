// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Renaming a chat from the Chats panel (audit 2026-10-16 mainjs-3 F4,
// regression-diff F1): a double-click on the title opens the rename box
// and opens the chat ONCE (the two clicks of the gesture are not two
// header clicks); a chat whose rows all live on another machine offers
// no rename, as the summary could only be stored on this daemon.
// Runs with: node test/audit1016b_history_rename.test.js

const assert = require('assert');
const {makePage: buildPage, send} = require('./simplify2_harness');

function makePage() {
  const {win, posted} = buildPage({bodyClass: 'remote-chat'});
  win.document.getElementById('sidebar').classList.add('open');
  send(win, {type: 'configData', config: {work_dir: ''}, apiKeys: {}});
  return {win, posted};
}

function makeRow(overrides) {
  return Object.assign(
    {
      id: 'chat-' + overrides.task_id,
      task_id: overrides.task_id,
      title: 'task ' + overrides.task_id,
      timestamp: 1_700_000_000,
      preview: 'task ' + overrides.task_id,
      has_events: true,
      failed: false,
      is_running: false,
      awaiting_answer: false,
      tokens: 1,
      cost: 0,
      steps: 1,
      is_favorite: false,
      work_dir: '',
      startTs: 1_700_000_000_000,
      endTs: 0,
    },
    overrides,
  );
}

/** Reply to the latest getHistory request (or the first load). */
function sendHistory(win, posted, sessions) {
  const req = posted.filter(m => m && m.type === 'getHistory').pop();
  send(win, {
    type: 'history',
    offset: 0,
    generation: req ? req.generation : undefined,
    sessions,
  });
}

function header(win, chatId) {
  const g = win.document.querySelector(
    '#history-list .history-chat-group[data-chat-id="' + chatId + '"]',
  );
  assert.ok(g, 'chat panel for ' + chatId);
  return g.querySelector(':scope > .history-chat-header');
}

function title(win, chatId) {
  return header(win, chatId).querySelector(':scope > .history-chat-title');
}

function pencil(win, chatId) {
  return header(win, chatId).querySelector('.history-chat-rename');
}

function pencilShown(win, chatId) {
  const p = pencil(win, chatId);
  return !!p && win.getComputedStyle(p).display !== 'none';
}

function byType(posted, type) {
  return posted.filter(m => m && m.type === type);
}

function chatTabs(win) {
  return win._testApi
    .openTabs()
    .filter(t => !t.isSubagentTab && !t.isContentTab);
}

function mouse(win, el, type, detail) {
  el.dispatchEvent(
    new win.MouseEvent(type, {bubbles: true, cancelable: true, detail}),
  );
}

/**
 * A double-click is click (detail 1), click (detail 2), dblclick.  The
 * first click opens the chat as a single click does; the second must
 * not open it AGAIN (a second new tab, a second newChat+resumeSession
 * and a closeTab of the first), and the dblclick opens the rename box.
 */
function testDoubleClickOpensTheChatOnce() {
  const {win, posted} = makePage();
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 1, chat_summary: 'Fix the login bug'}),
  ]);
  const welcomeTab = win._testApi.getActiveTabId();
  const newChatsBefore = byType(posted, 'newChat').length;
  const t = title(win, 'A');
  mouse(win, t, 'click', 1);
  mouse(win, t, 'click', 2);
  mouse(win, t, 'dblclick', 2);
  assert.strictEqual(
    byType(posted, 'newChat').length - newChatsBefore,
    1,
    'one newChat frame for the gesture',
  );
  assert.strictEqual(
    byType(posted, 'resumeSession').length,
    1,
    'the chat is resumed once',
  );
  // Leaving the idle welcome chat retires it (by design); the tab the
  // first click opened must not be retired by a second opening.
  assert.deepStrictEqual(
    byType(posted, 'closeTab').map(m => m.tabId),
    [welcomeTab],
    'only the idle welcome tab was closed',
  );
  const open = chatTabs(win);
  assert.strictEqual(open.length, 1, 'one chat tab is open');
  assert.notStrictEqual(open[0].id, welcomeTab);
  assert.ok(
    title(win, 'A').querySelector('input'),
    'the double-click opened the rename box',
  );
  win.close();
  console.log('  ok - a double-click on the title opens the chat once');
}

/**
 * A chat with rows only from a remote machine shows no pencil and
 * ignores a double-click on its title: setChatSummary stores the
 * summary on this daemon, where the remote chat's rows never read it.
 * A chat with a local row (alone or mixed with remote rows) keeps it.
 */
function testRemoteOnlyChatOffersNoRename() {
  const {win, posted} = makePage();
  sendHistory(win, posted, [
    makeRow({id: 'R', task_id: 1, machine: 'boxb', chat_summary: 'Remote'}),
    makeRow({id: 'R', task_id: 2, machine: 'boxb', chat_summary: 'Remote'}),
    makeRow({id: 'M', task_id: 3, machine: 'boxb', chat_summary: 'Mixed'}),
    makeRow({id: 'M', task_id: 4, chat_summary: 'Mixed'}),
    makeRow({id: 'L', task_id: 5, chat_summary: 'Local'}),
  ]);
  assert.strictEqual(pencilShown(win, 'R'), false, 'remote-only: no pencil');
  assert.strictEqual(pencilShown(win, 'M'), true, 'mixed chat keeps it');
  assert.strictEqual(pencilShown(win, 'L'), true, 'local chat keeps it');
  mouse(win, title(win, 'R'), 'dblclick', 2);
  assert.strictEqual(
    title(win, 'R').querySelector('input'),
    null,
    'remote-only: the double-click opens no rename box',
  );
  assert.strictEqual(title(win, 'R').textContent, 'Remote');
  assert.strictEqual(byType(posted, 'setChatSummary').length, 0);
  // The mixed chat renames as a local one does.
  mouse(win, title(win, 'M'), 'dblclick', 2);
  const input = title(win, 'M').querySelector('input');
  assert.ok(input, 'mixed chat: the double-click opens the rename box');
  input.value = 'Mixed, renamed';
  input.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
  );
  assert.deepStrictEqual(
    JSON.parse(JSON.stringify(byType(posted, 'setChatSummary'))),
    [{type: 'setChatSummary', chatId: 'M', summary: 'Mixed, renamed'}],
  );
  // A refresh that turns the chat remote-only takes the pencil away.
  sendHistory(win, posted, [
    makeRow({id: 'L', task_id: 5, machine: 'boxb', chat_summary: 'Local'}),
  ]);
  assert.strictEqual(pencilShown(win, 'L'), false, 'now remote-only');
  win.close();
  console.log('  ok - a remote-only chat offers no rename');
}

testDoubleClickOpensTheChatOnce();
testRemoteOnlyChatOffersNoRename();
console.log('audit1016b_history_rename.test.js: all tests passed');
