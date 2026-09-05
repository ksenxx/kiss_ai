// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for tab-bar scroll discipline: the tab bar
// must NOT auto-scroll the active tab into view on the many re-renders
// renderTabBar() runs for unrelated reasons (task status events, title
// updates from tabs_state snapshots), because that would yank the bar
// away from wherever the user manually scrolled it.  The bar scrolls
// the active tab into view ONLY when the active tab actually changes:
// a click on another tab, creating a new tab, or the successor switch
// after the active tab is closed.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// Loads the real chat webview (chat.html + panelCopy.js + api.js +
// main.js) into JSDOM and records every scrollIntoView call made on a
// tab strip (an element with class "chat-tab"), by tab id.  Calls on
// other elements (chat log autoscroll, dropdown items) are ignored:
// this suite is about the tab bar only.
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

  const tabScrolls = [];
  win.Element.prototype.scrollIntoView = function () {
    if (this.classList && this.classList.contains('chat-tab')) {
      tabScrolls.push(this.dataset.tabId || this.className);
    }
  };
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  let state;
  win.acquireVsCodeApi = function () {
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted, tabScrolls};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function entry(tabId, title, workDir) {
  return {
    tabId: tabId,
    chatId: 'chat-' + tabId,
    title: title || tabId,
    workDir: workDir || '',
  };
}

function tabEl(win, tabId) {
  return win.document.querySelector(
    '.chat-tab[data-tab-id="' + tabId + '"]',
  );
}

function activeTabId(win) {
  const el = win.document.querySelector('.chat-tab.active');
  return el ? el.dataset.tabId : null;
}

function clickEl(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

// Puts the webview in a known state: three shared tabs t1/t2/t3 with
// t1 active (reconcileTabs keeps the current active tab when it is in
// the snapshot; the boot placeholder is not, so the first snapshot tab
// becomes active).  Clears the scroll log afterwards so each test
// asserts only about its own actions.
function makeThreeTabs() {
  const ctx = makeWebview();
  send(ctx.win, {
    type: 'tabs_state',
    tabs: [entry('t1'), entry('t2'), entry('t3')],
  });
  assert.strictEqual(
    activeTabId(ctx.win),
    't1',
    'setup: t1 must be the active tab after the first snapshot',
  );
  ctx.tabScrolls.length = 0;
  return ctx;
}

function testStatusRerenderDoesNotScroll() {
  // Task status events (running started/stopped, for the active tab or
  // a background tab) re-render the tab bar; none of them is a tab
  // switch, so none may scroll the bar.
  const {win, tabScrolls} = makeThreeTabs();

  send(win, {type: 'status', running: true, tabId: 't2', startTs: 1000});
  send(win, {type: 'status', running: true, tabId: 't1', startTs: 1001});
  send(win, {type: 'status', running: false, tabId: 't2'});
  send(win, {type: 'status', running: false, tabId: 't1'});

  assert.deepStrictEqual(
    tabScrolls,
    [],
    'status re-renders must not scroll the tab bar, got scrolls: ' +
      JSON.stringify(tabScrolls),
  );
  assert.strictEqual(activeTabId(win), 't1', 'active tab must not change');
}

function testTitleUpdateDoesNotScroll() {
  // A tabs_state snapshot that only renames tabs (the agent updating a
  // task title) re-renders the bar without a switch: no scroll.
  const {win, tabScrolls} = makeThreeTabs();

  send(win, {
    type: 'tabs_state',
    tabs: [entry('t1', 'renamed one'), entry('t2', 'renamed two'), entry('t3')],
  });

  assert.strictEqual(
    tabEl(win, 't1').textContent.includes('renamed one'),
    true,
    'setup: the rename must actually reach the tab strip',
  );
  assert.deepStrictEqual(
    tabScrolls,
    [],
    'a title-only snapshot must not scroll the tab bar, got: ' +
      JSON.stringify(tabScrolls),
  );
}

function testSwitchingTabsScrollsExactlyTheNewActiveTab() {
  // Clicking another tab IS a switch: the newly active tab must be
  // scrolled into view (and only it).
  const {win, tabScrolls} = makeThreeTabs();

  clickEl(win, tabEl(win, 't3'));
  assert.strictEqual(activeTabId(win), 't3', 't3 must become active');
  assert.deepStrictEqual(
    tabScrolls,
    ['t3'],
    'switching to t3 must scroll exactly t3 into view, got: ' +
      JSON.stringify(tabScrolls),
  );

  // Re-renders after the switch must stay quiet again.
  tabScrolls.length = 0;
  send(win, {type: 'status', running: true, tabId: 't3', startTs: 2000});
  send(win, {type: 'status', running: false, tabId: 't3'});
  assert.deepStrictEqual(
    tabScrolls,
    [],
    'post-switch re-renders must not scroll, got: ' +
      JSON.stringify(tabScrolls),
  );
}

function testClickingActiveTabDoesNotScroll() {
  // Clicking the tab that is already active is not a switch; the user
  // may have scrolled the bar away on purpose.
  const {win, tabScrolls} = makeThreeTabs();

  clickEl(win, tabEl(win, 't1'));
  assert.strictEqual(activeTabId(win), 't1', 't1 must stay active');
  assert.deepStrictEqual(
    tabScrolls,
    [],
    'clicking the already-active tab must not scroll, got: ' +
      JSON.stringify(tabScrolls),
  );
}

function testNewTabScrollsIntoView() {
  // The "+" button creates AND activates a new tab: that is a switch,
  // so the new tab must be brought into view.
  const {win, tabScrolls} = makeThreeTabs();

  clickEl(win, win.document.querySelector('.chat-tab-add'));
  const newId = activeTabId(win);
  assert.ok(newId && newId !== 't1', 'a new tab must become active');
  assert.ok(
    tabScrolls.includes(newId),
    'the newly created tab must be scrolled into view, got: ' +
      JSON.stringify(tabScrolls),
  );
  assert.ok(
    tabScrolls.every(id => id === newId),
    'only the new tab may be scrolled, got: ' + JSON.stringify(tabScrolls),
  );
}

function testCloseSuccessorScrollsIntoView() {
  // Closing the active tab activates a successor: that is a switch, so
  // the successor must be brought into view.
  const {win, tabScrolls} = makeThreeTabs();

  const closeBtn = tabEl(win, 't1').querySelector('.chat-tab-close');
  assert.ok(closeBtn, 'setup: the active tab must have a close control');
  clickEl(win, closeBtn);

  const successor = activeTabId(win);
  assert.ok(
    successor && successor !== 't1',
    'closing the active tab must activate a successor',
  );
  assert.ok(
    tabScrolls.includes(successor),
    'the successor tab must be scrolled into view, got: ' +
      JSON.stringify(tabScrolls),
  );
  assert.ok(
    tabScrolls.every(id => id === successor),
    'only the successor may be scrolled, got: ' + JSON.stringify(tabScrolls),
  );
}

function testClosingBackgroundTabDoesNotScroll() {
  // Closing a background tab re-renders the bar but the active tab is
  // unchanged: no switch, no scroll.
  const {win, tabScrolls} = makeThreeTabs();

  const closeBtn = tabEl(win, 't2').querySelector('.chat-tab-close');
  assert.ok(closeBtn, 'setup: t2 must have a close control');
  clickEl(win, closeBtn);

  assert.strictEqual(activeTabId(win), 't1', 't1 must stay active');
  assert.strictEqual(tabEl(win, 't2'), null, 't2 must be gone from the bar');
  assert.deepStrictEqual(
    tabScrolls,
    [],
    'closing a background tab must not scroll the bar, got: ' +
      JSON.stringify(tabScrolls),
  );
}

function testSubagentRetagDoesNotScroll() {
  // The daemon may re-announce an already-open sub-agent under a new
  // tab id; the client then RENAMES the existing tab onto the new id
  // (retagSubagentTab) instead of opening a second one.  When that tab
  // is the active one, the rename moves activeTabId to the new id --
  // but the user did not switch tabs, so the bar must not scroll.
  const {win, tabScrolls} = makeThreeTabs();

  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-old',
    parent_tab_id: 't1',
    description: 'a sub-agent',
    task_id: 'sub-task-1',
    taskIndex: 0,
    isSubagentTab: true,
  });
  const subEl = tabEl(win, 'sub-old');
  assert.ok(subEl, 'setup: the sub-agent tab must open in the bar');

  // The user switches to the sub-agent tab: that scroll is expected.
  clickEl(win, subEl);
  assert.strictEqual(activeTabId(win), 'sub-old', 'sub tab must be active');
  tabScrolls.length = 0;

  // Re-announcement under a new id retags the SAME tab in place.
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-new',
    parent_tab_id: 't1',
    description: 'a sub-agent',
    task_id: 'sub-task-1',
    taskIndex: 0,
    isSubagentTab: true,
  });
  assert.strictEqual(
    activeTabId(win),
    'sub-new',
    'the retag must move the active tab onto the new id',
  );
  assert.strictEqual(
    tabEl(win, 'sub-old'),
    null,
    'the old id must be gone from the bar (rename, not a second tab)',
  );
  assert.deepStrictEqual(
    tabScrolls,
    [],
    'a retag is a rename, not a switch: no scroll, got: ' +
      JSON.stringify(tabScrolls),
  );
}

const tests = [
  testStatusRerenderDoesNotScroll,
  testTitleUpdateDoesNotScroll,
  testSwitchingTabsScrollsExactlyTheNewActiveTab,
  testClickingActiveTabDoesNotScroll,
  testNewTabScrollsIntoView,
  testCloseSuccessorScrollsIntoView,
  testClosingBackgroundTabDoesNotScroll,
  testSubagentRetagDoesNotScroll,
];

let failures = 0;
for (const t of tests) {
  try {
    t();
    console.log('PASS', t.name);
  } catch (err) {
    failures += 1;
    console.error('FAIL', t.name);
    console.error(err && err.stack ? err.stack : err);
  }
}
if (failures > 0) {
  console.error(`${failures} test(s) failed`);
  process.exit(1);
}
console.log(`All ${tests.length} tabBarNoAutoScroll tests passed`);
// The webview leaves reconnect timers behind in JSDOM; exit explicitly
// like the other suites so the process never hangs.
process.exit(0);
