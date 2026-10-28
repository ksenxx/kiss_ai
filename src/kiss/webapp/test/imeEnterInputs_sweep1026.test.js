// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for Enter inside an IME composition
// (media/main.js): the Enter that commits a Japanese / Chinese / Korean
// composition arrives as a keydown with `isComposing` (keyCode 229) and
// must not act as a submit.  The composer and the chat search already
// honour that; so must the Tricks "add" box, the history chat-rename
// box and the model picker's search box.

'use strict';

const assert = require('assert');
const {
  makePage: buildPage,
  send,
  ofType,
  stubRaf,
  typeInto,
} = require('./simplify2_harness');
const {sendHistory, makeRow} = require('./history_helpers');

function imeEnter(win, el) {
  el.dispatchEvent(
    new win.KeyboardEvent('keydown', {
      key: 'Enter',
      isComposing: true,
      bubbles: true,
    }),
  );
}

function plainEnter(win, el) {
  el.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
  );
}

function testTricksAddBox() {
  const {win, posted} = buildPage();
  win.__TRICKS__ = ['Alpha promptlet'];
  win.document.getElementById('tricks-btn').click();
  const addInput = win.document.getElementById('tricks-add-input');
  typeInto(win, addInput, '日本語の促進');
  imeEnter(win, addInput);
  assert.strictEqual(
    ofType(posted, 'addTrick').length,
    0,
    'IME Enter does not add',
  );
  plainEnter(win, addInput);
  assert.strictEqual(
    ofType(posted, 'addTrick').length,
    1,
    'a plain Enter adds',
  );
  win.close();
  console.log('  ok - IME Enter in the Tricks add box does not submit');
}

function testHistoryRenameBox() {
  const {win, posted} = buildPage({bodyClass: 'remote-chat'});
  win.document.getElementById('sidebar').classList.add('open');
  send(win, {type: 'configData', config: {work_dir: ''}, apiKeys: {}});
  sendHistory(win, posted, [
    makeRow({id: 'A', task_id: 1, chat_summary: 'Fix the login bug'}),
  ]);
  win.document.querySelector('#history-list .history-chat-rename').click();
  const titleEl = win.document.querySelector(
    '#history-list .history-chat-group[data-chat-id="A"] > .history-chat-header > .history-chat-title',
  );
  const input = titleEl.querySelector('input');
  assert.ok(input, 'the pencil opened the rename box');
  input.value = 'ログイン修正';
  imeEnter(win, input);
  assert.ok(
    titleEl.querySelector('input'),
    'IME Enter keeps the rename box open',
  );
  assert.strictEqual(
    ofType(posted, 'setChatSummary').length,
    0,
    'nothing renamed yet',
  );
  // Escape inside the composition belongs to the IME too.
  input.dispatchEvent(
    new win.KeyboardEvent('keydown', {
      key: 'Escape',
      isComposing: true,
      bubbles: true,
    }),
  );
  assert.ok(
    titleEl.querySelector('input'),
    'IME Escape keeps the rename box open',
  );
  plainEnter(win, input);
  assert.deepStrictEqual(
    JSON.parse(JSON.stringify(ofType(posted, 'setChatSummary'))),
    [{type: 'setChatSummary', chatId: 'A', summary: 'ログイン修正'}],
    'a plain Enter renames',
  );
  win.close();
  console.log('  ok - IME Enter in the chat rename box does not commit');
}

function testModelSearchBox() {
  const {win, posted} = buildPage({
    modelName: 'claude-a',
    beforeScripts: stubRaf,
  });
  send(win, {
    type: 'models',
    models: [
      {name: 'claude-a', inp: 5, out: 25, uses: 3, vendor: 'anthropic'},
      {name: 'gpt-b', inp: 2, out: 8, uses: 0, vendor: 'openai'},
    ],
    selected: 'claude-a',
  });
  win.document
    .getElementById('model-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const search = win.document.getElementById('model-search');
  search.value = 'g';
  search.dispatchEvent(new win.Event('input', {bubbles: true}));
  const picked = () => ofType(posted, 'selectModel').map(m => m.model);
  imeEnter(win, search);
  assert.deepStrictEqual(picked(), [], 'IME Enter picks no model');
  assert.ok(
    win.document.getElementById('model-list').querySelector('.model-item'),
    'the picker is still open',
  );
  plainEnter(win, search);
  assert.deepStrictEqual(
    picked(),
    ['gpt-b'],
    'a plain Enter picks the first match',
  );
  win.close();
  console.log('  ok - IME Enter in the model search box does not pick');
}

function main() {
  testTricksAddBox();
  testHistoryRenameBox();
  testModelSearchBox();
  console.log('imeEnterInputs_sweep1026.test.js: all tests passed');
}

main();
