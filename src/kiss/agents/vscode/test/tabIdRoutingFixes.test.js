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
  const posted = [];

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage(msg) {
        posted.push(msg);
      },
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

function initialTabId(posted) {
  const ready = posted.find(m => m && m.type === 'ready' && m.tabId);
  assert.ok(ready, 'webview posted a ready message with its active tabId');
  return ready.tabId;
}

function tabElement(win, tabId) {
  return win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
}

function notificationToasts(win) {
  return Array.from(win.document.querySelectorAll('[data-notification-id]'));
}

function testPhantomSubagentTabNotMaterialised() {
  const {win} = makeWebview();
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'phantom-sub-1',
    parent_tab_id: '',
    description: 'leaked sub-agent',
    task_id: 'sub-task-9',
    isSubagentTab: true,
    isDone: false,
  });
  assert.strictEqual(
    tabElement(win, 'phantom-sub-1'),
    null,
    'a webview that does not own the target tab must not materialise ' +
      'a phantom sub-agent tab from a blank-parent openSubagentTab',
  );
  send(win, {
    type: 'task_events',
    task: 'leaked sub-agent',
    task_id: 'sub-task-9',
    tabId: 'phantom-sub-1',
    events: [
      {type: 'prompt', text: 'SECRET-SUBAGENT-PROMPT', taskId: 'sub-task-9'},
    ],
  });
  assert.ok(
    !win.document.body.textContent.includes('SECRET-SUBAGENT-PROMPT'),
    'the phantom tab transcript leaked into a foreign webview',
  );
  console.log('  ok - blank-parent openSubagentTab does not create phantom tabs');
}

function testOwnedTabConversionStillWorks() {
  const {win, posted} = makeWebview();
  send(win, {type: 'new_tab', task_id: 'sub-task-9'});
  const resume = posted.find(m => m && m.type === 'resumeSession');
  assert.ok(resume && resume.tabId, 'created tab posted resumeSession');
  const ownedTabId = resume.tabId;
  send(win, {
    type: 'openSubagentTab',
    tab_id: ownedTabId,
    parent_tab_id: '',
    description: 'converted sub-agent',
    task_id: 'sub-task-9',
    isSubagentTab: true,
    isDone: true,
  });
  const el = tabElement(win, ownedTabId);
  assert.ok(el, 'the owned tab still exists');
  assert.ok(
    el.className.includes('subagent-tab'),
    'the owned tab was converted into a sub-agent tab',
  );
  console.log('  ok - blank-parent openSubagentTab still converts the owned tab');
}

function testUnknownParentStillDropped() {
  const {win} = makeWebview();
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-of-foreign-parent',
    parent_tab_id: 'foreign-parent-tab',
    description: 'other window fan-out',
    task_id: 'sub-task-10',
  });
  assert.strictEqual(
    tabElement(win, 'sub-of-foreign-parent'),
    null,
    'non-empty unknown parent_tab_id must still be dropped',
  );
  console.log('  ok - unknown-parent openSubagentTab still dropped');
}

function testForeignTabNotificationDropped() {
  const {win, posted} = makeWebview();
  const ownTab = initialTabId(posted);

  send(win, {
    type: 'notification',
    id: 'n-own',
    severity: 'info',
    message: 'own-tab toast',
    tabId: ownTab,
  });
  assert.strictEqual(
    notificationToasts(win).length,
    1,
    'a toast stamped with a LOCAL tab id must render',
  );

  send(win, {
    type: 'notification',
    id: 'n-foreign',
    severity: 'info',
    message: 'foreign-window toast',
    tabId: 'foreign-window-tab',
  });
  const ids = notificationToasts(win).map(el =>
    el.getAttribute('data-notification-id'),
  );
  assert.ok(
    !ids.includes('n-foreign'),
    "another window's tab-stamped notification must not render here",
  );

  send(win, {
    type: 'notification',
    id: 'n-global',
    severity: 'info',
    message: 'global toast',
  });
  assert.ok(
    notificationToasts(win)
      .map(el => el.getAttribute('data-notification-id'))
      .includes('n-global'),
    'a tabless (global) notification must still render',
  );
  console.log('  ok - notifications route by local tab ownership');
}

function testBackgroundTabErrorRetained() {
  const {win, posted} = makeWebview();
  const bgTabId = initialTabId(posted);
  win._testApi.createNewTab();
  assert.notStrictEqual(
    tabElement(win, bgTabId).className.includes('active'),
    true,
    'the initial tab is now a background tab',
  );

  send(win, {type: 'error', text: 'bg boom', tabId: bgTabId});
  assert.ok(
    !win.document.getElementById('output').textContent.includes('bg boom'),
    "a background tab's error must not render into the ACTIVE tab",
  );

  tabElement(win, bgTabId).dispatchEvent(
    new win.MouseEvent('click', {bubbles: true}),
  );
  assert.ok(
    win.document.getElementementById === undefined,
  );
  assert.ok(
    win.document.getElementById('output').textContent.includes('bg boom'),
    "the background tab's error banner must appear after switching to it",
  );
  console.log('  ok - background-tab error retained and shown on switch');
}

function testForeignTabErrorDropped() {
  const {win} = makeWebview();
  send(win, {type: 'error', text: 'foreign boom', tabId: 'foreign-tab-1'});
  assert.ok(
    !win.document.body.textContent.includes('foreign boom'),
    "another window's error must not render here",
  );
  console.log('  ok - foreign-tab error dropped');
}

function testActiveTabErrorRendered() {
  const {win, posted} = makeWebview();
  const ownTab = initialTabId(posted);
  send(win, {type: 'error', text: 'active boom', tabId: ownTab});
  assert.ok(
    win.document.getElementById('output').textContent.includes('active boom'),
    'an active-tab error must render immediately',
  );
  send(win, {type: 'error', text: 'tabless boom'});
  assert.ok(
    win.document.getElementById('output').textContent.includes('tabless boom'),
    'a tabless error must render immediately',
  );
  console.log('  ok - active-tab and tabless errors render immediately');
}

function main() {
  testPhantomSubagentTabNotMaterialised();
  testOwnedTabConversionStillWorks();
  testUnknownParentStillDropped();
  testForeignTabNotificationDropped();
  testBackgroundTabErrorRetained();
  testForeignTabErrorDropped();
  testActiveTabErrorRendered();
  console.log('tabIdRoutingFixes: all tests passed');
}

try {
  main();
  process.exit(0);
} catch (err) {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
}
