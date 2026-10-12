// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Audit 2026-10-12 (mainjs-1): a chat tab closed on ANOTHER surface
// (it vanishes from the daemon's `tabs_state` snapshot, reconcileTabs
// drops it) must forget its machine pick exactly like a tab closed
// here does (closeTab deletes the tabMachines / tabMachinePicked
// entries).  Otherwise every remotely closed tab leaves a pick behind
// in the persisted webview state, forever: the maps are written back
// on every persistTabState and re-read on every reload.
'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(state) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace('{{BODY_CLASS_ATTR}}', ' class="remote-chat"');
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
  let saved = state || undefined;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => saved,
      setState: s => {
        saved = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted, state: () => saved};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

const MACHINES = [
  {name: 'laptop', local: true, status: 'reachable', address: 'https://a'},
  {name: 'gpu-box', status: 'reachable', address: 'ssh me@gpu'},
];

function pick(win, value) {
  const sel = win.document.getElementById('chat-machine');
  sel.value = value;
  sel.dispatchEvent(new win.Event('change', {bubbles: true}));
}

function persistedPicks(state) {
  return JSON.parse(JSON.stringify(state().tabMachines));
}

function persistedPicked(state) {
  return JSON.parse(JSON.stringify(state().tabMachinePicked)).sort();
}

function testRemotelyClosedTabForgetsItsPick() {
  const {win, state} = makeWebview();
  const api = win._testApi;
  send(win, {type: 'configData', config: {}, machine: 'laptop'});
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  const first = api.getActiveTabId();
  api.createNewTab();
  const second = api.getActiveTabId();
  pick(win, 'gpu-box');
  // The registry lists both tabs: both are now mirrored tabs.
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: first, title: 'one'},
      {tabId: second, title: 'two'},
    ],
  });
  assert.deepStrictEqual(persistedPicks(state), {[second]: 'gpu-box'});
  assert.deepStrictEqual(persistedPicked(state), [second]);
  // Another surface closes the second tab: the next snapshot omits it.
  send(win, {type: 'tabs_state', tabs: [{tabId: first, title: 'one'}]});
  assert.ok(!api.openTabs().some(t => t.id === second), 'the tab is gone');
  assert.deepStrictEqual(
    persistedPicks(state),
    {},
    'a tab closed on another surface forgets its machine pick',
  );
  assert.deepStrictEqual(
    persistedPicked(state),
    [],
    'and its "picked here" mark',
  );
  win.close();
  console.log('  ok - a tab closed on another surface forgets its pick');
}

try {
  testRemotelyClosedTabForgetsItsPick();
  console.log('\n1 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
