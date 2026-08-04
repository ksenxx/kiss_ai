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
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return win;
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function mergeToolbar(win) {
  return win.document.getElementById('merge-toolbar');
}

function testBackgroundMergeDoesNotSwitchTabs() {
  const win = makeWebview();
  const api = win._testApi;
  const mergeTab = api.getActiveTabId();

  api.createNewTab();
  const userTab = api.getActiveTabId();
  assert.ok(userTab !== mergeTab, 'a second tab must be active');

  send(win, {type: 'merge_started', tabId: mergeTab});

  assert.strictEqual(
    api.getActiveTabId(),
    userTab,
    'merge_started in a background tab must not switch tabs',
  );
  assert.strictEqual(
    mergeToolbar(win),
    null,
    'a background merge must not mount its toolbar over the active tab',
  );

  clickTab(win, mergeTab);
  assert.strictEqual(
    api.getActiveTabId(),
    mergeTab,
    'clicking the merging tab must switch to it',
  );
  assert.ok(
    mergeToolbar(win),
    'the merge toolbar must appear once the user visits the merging tab',
  );

  clickTab(win, userTab);
  assert.strictEqual(
    mergeToolbar(win),
    null,
    'leaving the merging tab must take its toolbar away again',
  );

  win.close();
  console.log('  ok - background merge_started never steals focus');
}

function testActiveTabMergeShowsToolbarImmediately() {
  const win = makeWebview();
  const api = win._testApi;
  const activeTab = api.getActiveTabId();

  send(win, {type: 'merge_started', tabId: activeTab});

  assert.strictEqual(
    api.getActiveTabId(),
    activeTab,
    'the active tab must stay active',
  );
  assert.ok(
    mergeToolbar(win),
    'merge_started for the active tab must mount the toolbar at once',
  );

  send(win, {type: 'merge_ended', tabId: activeTab});
  assert.strictEqual(
    mergeToolbar(win),
    null,
    'merge_ended must remove the toolbar',
  );

  win.close();
  console.log('  ok - active-tab merge_started still shows the toolbar');
}

function testBackgroundMergeEndedClearsPendingToolbar() {
  const win = makeWebview();
  const api = win._testApi;
  const mergeTab = api.getActiveTabId();
  api.createNewTab();
  const userTab = api.getActiveTabId();

  send(win, {type: 'merge_started', tabId: mergeTab});
  send(win, {type: 'merge_ended', tabId: mergeTab});

  assert.strictEqual(
    api.getActiveTabId(),
    userTab,
    'neither merge event may switch tabs',
  );

  clickTab(win, mergeTab);
  assert.strictEqual(
    mergeToolbar(win),
    null,
    'a merge finished in the background must not resurrect its toolbar',
  );

  win.close();
  console.log('  ok - background merge_ended leaves no stale toolbar');
}

function runTests() {
  testBackgroundMergeDoesNotSwitchTabs();
  testActiveTabMergeShowsToolbarImmediately();
  testBackgroundMergeEndedClearsPendingToolbar();
}

try {
  runTests();
  console.log('\n3 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
