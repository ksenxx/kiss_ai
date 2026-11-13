// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for Enter inside an IME composition
// (media/main.js): the Enter that commits a Japanese / Chinese / Korean
// composition arrives as a keydown with `isComposing` or with keyCode
// 229 (some browsers send only one of them) and must not act as a
// submit; each case below sends both forms.  The composer and the chat
// search already honour that; so must the Tricks "add" box, the history
// chat-rename box and the model picker's search box.

'use strict';

const assert = require('assert');
const {
  makePage: buildPage,
  send,
  ofType,
  plain,
  stubRaf,
  typeInto,
} = require('./simplify2_harness');
const {sendHistory, makeRow} = require('./history_helpers');
const {key, IME_FORMS} = require('./ui_antipattern_harness');

function testTricksAddBox() {
  const {win, posted} = buildPage();
  win.__TRICKS__ = ['Alpha promptlet'];
  win.document.getElementById('tricks-btn').click();
  const addInput = win.document.getElementById('tricks-add-input');
  typeInto(win, addInput, '日本語の促進');
  for (const form of IME_FORMS) {
    key(win, addInput, 'Enter', form);
    assert.strictEqual(
      ofType(posted, 'addTrick').length,
      0,
      'IME Enter does not add: ' + JSON.stringify(form),
    );
  }
  key(win, addInput, 'Enter');
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
  for (const form of IME_FORMS) {
    const label = JSON.stringify(form);
    key(win, input, 'Enter', form);
    assert.ok(
      titleEl.querySelector('input'),
      'IME Enter keeps the rename box open: ' + label,
    );
    assert.strictEqual(
      ofType(posted, 'setChatSummary').length,
      0,
      'nothing renamed yet: ' + label,
    );
    // Escape inside the composition belongs to the IME too.
    key(win, input, 'Escape', form);
    assert.ok(
      titleEl.querySelector('input'),
      'IME Escape keeps the rename box open: ' + label,
    );
  }
  key(win, input, 'Enter');
  assert.deepStrictEqual(
    plain(ofType(posted, 'setChatSummary')),
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
  for (const form of IME_FORMS) {
    const label = JSON.stringify(form);
    key(win, search, 'Enter', form);
    assert.deepStrictEqual(picked(), [], 'IME Enter picks no model: ' + label);
    assert.ok(
      win.document.getElementById('model-list').querySelector('.model-item'),
      'the picker is still open: ' + label,
    );
  }
  key(win, search, 'Enter');
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
