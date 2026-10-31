// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for the composer's sheets and pickers in
// media/main.js:
//
// * a promptlet clicked into the composer ends the prompt-history
//   cycling, so ArrowUp/ArrowDown no longer replace the composed text;
// * a double Add in the promptlet sheet posts one addTrick;
// * a `models` reply keeps the open model picker filtered by its search;
// * a later history page leaves a chat header's trash icon hidden while
//   its Delete / Cancel question is up;
// * the settings sheet that follows the chat header from a machine back
//   to this daemon shows this daemon's run toggles at once.

'use strict';

const assert = require('assert');
const {makePage, send, stubRaf, click, byId} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {
  makeSession,
  loadHistory,
  sendHistoryPage,
  makeConnectedRemotePage,
} = require('./history_helpers');
const {MACHINES} = require('./settings_helpers');

const {test, report} = makeRunner();

function keydown(win, el, key) {
  el.dispatchEvent(new win.KeyboardEvent('keydown', {key, bubbles: true}));
}

function typeInto(win, el, value) {
  el.value = value;
  el.dispatchEvent(new win.Event('input', {bubbles: true}));
}

async function main() {
  await test('a clicked promptlet ends the prompt-history cycling', () => {
    const {win} = makePage({beforeScripts: stubRaf});
    const doc = win.document;
    const inp = doc.getElementById('task-input');
    send(win, {type: 'inputHistory', tasks: ['old prompt', 'older prompt']});
    keydown(win, inp, 'ArrowUp');
    assert.strictEqual(inp.value, 'old prompt');
    win.__TRICKS__ = ['Promptlet A'];
    doc.getElementById('tricks-btn').click();
    doc.getElementById('tricks-list').children[0].click();
    assert.strictEqual(inp.value, 'old prompt Promptlet A');
    keydown(win, inp, 'ArrowUp');
    assert.strictEqual(inp.value, 'old prompt Promptlet A', 'ArrowUp kept it');
    keydown(win, inp, 'ArrowDown');
    assert.strictEqual(
      inp.value,
      'old prompt Promptlet A',
      'ArrowDown kept it',
    );
    win.close();
  });

  await test('a double Add posts one addTrick', () => {
    const {win, posted} = makePage({beforeScripts: stubRaf});
    const doc = win.document;
    win.__TRICKS__ = [];
    doc.getElementById('tricks-btn').click();
    typeInto(win, doc.getElementById('tricks-add-input'), 'New promptlet');
    const add = doc.getElementById('tricks-add-btn');
    add.click();
    add.click();
    const adds = posted.filter(m => m.type === 'addTrick');
    assert.strictEqual(adds.length, 1, 'one add in flight');
    assert.strictEqual(adds[0].text, 'New promptlet');
    win.close();
  });

  await test('a models reply keeps the open picker filtered', () => {
    const models = [
      {name: 'claude-opus-5', inp: 5, out: 25, uses: 3, vendor: 'anthropic'},
      {name: 'gpt-5.6-sol', inp: 2, out: 8, uses: 0, vendor: 'openai'},
      {name: 'gemini-3.5-flash', inp: 1, out: 4, uses: 0, vendor: 'google'},
    ];
    const {win} = makePage({
      modelName: 'claude-opus-5',
      beforeScripts: stubRaf,
    });
    const doc = win.document;
    send(win, {type: 'models', models, selected: 'claude-opus-5'});
    click(win, doc.getElementById('model-btn'));
    const search = doc.getElementById('model-search');
    typeInto(win, search, 'gemini');
    send(win, {type: 'models', models, selected: 'claude-opus-5'});
    const shown = Array.from(
      doc.querySelectorAll('#model-list .model-item'),
      el => el.textContent,
    );
    assert.ok(
      shown.length > 0 && shown.every(t => t.includes('gemini')),
      JSON.stringify(shown),
    );
    keydown(win, search, 'Enter');
    assert.strictEqual(
      doc.getElementById('model-name').textContent.replace(/\u200e/g, ''),
      'gemini-3.5-flash',
    );
    win.close();
  });

  await test("a later history page keeps a chat header's trash hidden under its question", () => {
    const {win, posted} = makeConnectedRemotePage();
    loadHistory(win, [
      makeSession(),
      makeSession({task_id: 'task-2', timestamp: 1700000100}),
    ]);
    const header = win.document.querySelector('.history-chat-header');
    const del = header.querySelector(
      '.history-chat-actions > .sidebar-item-delete',
    );
    const confirm = header.querySelector(
      '.history-chat-actions > .sidebar-item-confirm',
    );
    click(win, del);
    assert.strictEqual(confirm.style.display, '');
    // An older task of the same chat arrives on the next page (it
    // starts where the first one ended).
    sendHistoryPage(win, posted, 2, [
      makeSession({task_id: 'task-0', timestamp: 1699990000}),
    ]);
    assert.ok(header.isConnected, 'the page was appended, not rebuilt');
    assert.strictEqual(
      win.document.querySelectorAll('#history-list .running-item').length,
      3,
      'the older task is listed',
    );
    assert.strictEqual(confirm.style.display, '', 'the question is still up');
    assert.strictEqual(del.style.display, 'none', 'the trash stays hidden');
    click(win, confirm.querySelector('.sidebar-confirm-no'));
    assert.strictEqual(del.style.display, '', 'Cancel brings it back');
    win.close();
  });

  await test("the settings sheet coming home shows this daemon's toggles at once", () => {
    const {win} = makePage({beforeScripts: stubRaf});
    const panel = byId(win, 'settings-panel');
    const webTools = byId(win, 'cfg-use-web-tools');
    send(win, {
      type: 'configData',
      config: {use_web_browser: false, use_memory: true},
      apiKeys: {},
      machine: 'laptop',
    });
    send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
    // The chat ran a task on gpu-box (the daemon's binding).
    const boxTab = win._testApi.getActiveTabId();
    send(win, {type: 'tabMachine', tabId: boxTab, machine: 'gpu-box'});
    click(win, byId(win, 'settings-btn'));
    send(win, {
      type: 'configData',
      forwarded: true,
      machine: 'gpu-box',
      config: {use_web_browser: true, use_memory: false},
      apiKeys: {},
    });
    assert.ok(panel.classList.contains('for-machine'));
    assert.strictEqual(webTools.checked, true, "gpu-box's toggle");
    // A new chat (this machine) with the sheet open: this daemon's
    // toggles before its reply, so a submit now carries its values.
    win._testApi.createNewTab();
    assert.ok(!panel.classList.contains('for-machine'), 'the sheet came home');
    assert.strictEqual(webTools.checked, false, "this daemon's web tools");
    assert.strictEqual(byId(win, 'cfg-use-memory').checked, true);
    win.close();
  });

  report('composerSheets_sweep1030');
}

main();
