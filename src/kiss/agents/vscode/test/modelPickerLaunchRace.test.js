// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(templateModel, seededState) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, templateModel);
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
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

  const posted = [];
  win.acquireVsCodeApi = function () {
    let state = seededState;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function clickTab(win, tabId) {
  const tabEl = win.document.querySelector(
    `.chat-tab[data-tab-id="${tabId}"]`,
  );
  assert.ok(tabEl, `tab ${tabId} must exist in the tab bar`);
  tabEl.click();
}

function pickerText(win) {
  const el = win.document.getElementById('model-name');
  return (el && el.textContent) || '';
}

function testRestoredTabInheritsTemplateModelOnLaunch() {
  // Tabs are server-canonical now: only the selection is persisted,
  // and the daemon's `tabs_state` snapshot supplies the tab set.
  const seeded = {chatId: 'persisted-tab-a'};
  const {win} = makeWebview('claude-opus-4-7', seeded);
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
  console.log(
    '  ok - restored tabs inherit template model on launch',
  );
}

function testModelsEventPropagatesIntoTabState() {
  // Tabs are server-canonical now: only the selection is persisted,
  // and the daemon's `tabs_state` snapshot supplies the tab set.
  const seeded = {chatId: 'persisted-tab-a'};
  const {win} = makeWebview('No model', seeded);
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
  const {win, posted} = makeWebview('claude-opus-4-7', seeded);
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
