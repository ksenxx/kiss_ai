// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');

const {makePage: makeChatPage, send, stubRaf} = harness;

function makePage(templateModel, seededState) {
  return makeChatPage({
    modelName: templateModel,
    initialState: seededState,
    beforeScripts: stubRaf,
  });
}

// Put chat *tabId* on screen and check it got there.
function clickTab(win, tabId) {
  harness.clickTab(win, tabId);
  assert.strictEqual(win._testApi.getActiveTabId(), tabId);
}

function pickerText(win) {
  // Strip the U+200E marks refreshModelLabel adds for the pill's
  // leading-truncation (RTL line) rendering.
  const el = win.document.getElementById('model-name');
  return ((el && el.textContent) || '').replace(/\u200e/g, '');
}

function testRestoredTabInheritsTemplateModelOnLaunch() {
  // Tabs are server-canonical now: only the selection is persisted,
  // and the daemon's `tabs_state` snapshot supplies the tab set.
  const seeded = {chatId: 'persisted-tab-a'};
  const {win} = makePage('claude-opus-4-7', seeded);
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 'persisted-tab-a', chatId: '', title: 'first', workDir: ''},
      {tabId: 'persisted-tab-b', chatId: '', title: 'second', workDir: ''},
    ],
  });

  assert.strictEqual(
    pickerText(win),
    'claude-opus-4-7',
    'launch-time picker must reflect the {{MODEL_NAME}} template',
  );

  clickTab(win, 'persisted-tab-b');
  assert.strictEqual(
    pickerText(win),
    'claude-opus-4-7',
    'BUG: model picker turned blank after switching to the second ' +
      'restored tab — the launch IIFE ran before the DOM-read of ' +
      '#model-name, so every restored tab inherited an empty ' +
      'selectedModel from the closure variable',
  );

  win.close();
  console.log('  ok - restored tabs inherit template model on launch');
}

function testModelsEventPropagatesIntoTabState() {
  // Tabs are server-canonical now: only the selection is persisted,
  // and the daemon's `tabs_state` snapshot supplies the tab set.
  const seeded = {chatId: 'persisted-tab-a'};
  const {win} = makePage('No model', seeded);
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 'persisted-tab-a', chatId: '', title: 'first', workDir: ''},
      {tabId: 'persisted-tab-b', chatId: '', title: 'second', workDir: ''},
    ],
  });

  assert.strictEqual(pickerText(win), 'No model');

  send(win, {
    type: 'models',
    models: [
      {name: 'claude-opus-4-7', inp: 15, out: 75, uses: 0, vendor: 'Anthropic'},
      {name: 'gpt-5.5', inp: 5, out: 25, uses: 0, vendor: 'OpenAI'},
    ],
    selected: 'claude-opus-4-7',
  });
  assert.strictEqual(
    pickerText(win),
    'claude-opus-4-7',
    'models event must update the live picker label',
  );

  clickTab(win, 'persisted-tab-b');
  assert.strictEqual(
    pickerText(win),
    'claude-opus-4-7',
    'BUG: tab state was not updated by the models event; ' +
      'switching tabs reverts the picker to the stale launch-time value',
  );

  win.close();
  console.log(
    '  ok - models event propagates new selection into existing tab state',
  );
}

function testModelsEventPreservesUserPickedTabs() {
  // Tabs are server-canonical now: only the selection is persisted,
  // and the daemon's `tabs_state` snapshot supplies the tab set.
  const seeded = {chatId: 'persisted-tab-a'};
  const {win, posted} = makePage('claude-opus-4-7', seeded);
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 'persisted-tab-a', chatId: '', title: 'first', workDir: ''},
      {tabId: 'persisted-tab-b', chatId: '', title: 'second', workDir: ''},
    ],
  });

  const modelList = [
    {name: 'claude-opus-4-7', inp: 15, out: 75, uses: 0, vendor: 'Anthropic'},
    {name: 'gpt-5.5', inp: 5, out: 25, uses: 0, vendor: 'OpenAI'},
  ];

  send(win, {type: 'models', models: modelList, selected: 'claude-opus-4-7'});

  const modelItems = win.document.querySelectorAll('#model-list .model-item');
  let picked = null;
  modelItems.forEach(it => {
    if ((it.textContent || '').includes('gpt-5.5')) picked = it;
  });
  assert.ok(picked, 'gpt-5.5 must be listed in the model dropdown');
  picked.click();
  assert.strictEqual(pickerText(win), 'gpt-5.5');

  clickTab(win, 'persisted-tab-b');
  assert.strictEqual(
    pickerText(win),
    'claude-opus-4-7',
    'sibling tab must keep its own model (not inherit the just-picked value)',
  );

  clickTab(win, 'persisted-tab-a');
  assert.strictEqual(
    pickerText(win),
    'gpt-5.5',
    'user-picked model must persist across tab switches',
  );

  void posted;

  win.close();
  console.log('  ok - user-picked tab is not overwritten by models event');
}

function runTests() {
  testModelsEventPropagatesIntoTabState();
  testRestoredTabInheritsTemplateModelOnLaunch();
  testModelsEventPreservesUserPickedTabs();
}

try {
  runTests();
  console.log('\n3 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
