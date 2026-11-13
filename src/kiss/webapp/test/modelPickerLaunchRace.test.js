// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  clickTab,
  lastMsg,
} = require('./simplify2_harness');
const {typeAndEnter} = require('./attachment_helpers');

function makePage(templateModel, seededState) {
  return makeChatPage({
    modelName: templateModel,
    initialState: seededState,
    beforeScripts: stubRaf,
  });
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

// A reconnect's `ready` is answered with the daemon's model list too.
// The page stamps it like a getModels (`generation`, which the daemon
// echoes), so the answer to a ready sent BEFORE the user's pick cannot
// hand the tab the old default back.
function testReconnectReplyKeepsTheModelPickedSince() {
  const {win, posted} = makePage('model-a');
  const models = ['model-a', 'model-b'].map(name => {
    return {name, inp: 1, out: 1, uses: 0, vendor: 'v'};
  });
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'models', models, selected: 'model-a'});
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  send(win, {type: 'daemonStatus', connected: true});
  const ready = lastMsg(posted, 'ready');
  assert.ok(ready.generation > 0, 'BUG: the reconnect ready is not stamped');
  win._testApi.endLaunch();
  win.document.getElementById('model-btn').click();
  Array.from(win.document.querySelectorAll('#model-list .model-item'))
    .find(el => el.textContent.includes('model-b'))
    .click();
  // The daemon read its default before the pick: its late answer.
  send(win, {
    type: 'models',
    models,
    selected: 'model-a',
    generation: ready.generation,
  });
  assert.strictEqual(
    pickerText(win),
    'model-b',
    "BUG: the reconnect's late model list replaced the user's pick",
  );
  typeAndEnter(win, 'run with my model');
  assert.strictEqual(lastMsg(posted, 'submit').model, 'model-b');
  win.close();
  console.log('  ok - a reconnect reply keeps the model picked since');
}

// A hand-picked model surviving a tab switch and a models refresh is
// pinned by modelPickerAgentOverride.test.js.

try {
  testModelsEventPropagatesIntoTabState();
  testRestoredTabInheritsTemplateModelOnLaunch();
  testReconnectReplyKeepsTheModelPickedSince();
  console.log('\n3 passed, 0 failed');
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
