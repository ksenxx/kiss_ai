// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the shared tab bar: every client of the
// daemon (VS Code webview or remote web app — both run this same
// main.js) shows the SAME registry tabs, whatever folder each tab runs
// in and whatever the client's own workspace directory (configWorkDir)
// is. The workspace still scopes the history filter and the Explorer
// views, never the tab bar: a task running anywhere stays visible on
// every surface until the user closes its tab.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(opts) {
  const remote = !!(opts && opts.remote);
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  if (remote) {
    html = html.replace('{{BODY_CLASS_ATTR}}', ' class="remote-chat"');
  }
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

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function setWorkspace(win, dir) {
  send(win, {type: 'configData', config: {work_dir: dir}, apiKeys: {}});
}

function tabBarIds(win) {
  // The chat whose group is on screen sits on the main row and on the
  // group strip under it; count each tab once.
  const ids = Array.from(win.document.querySelectorAll('.chat-tab'))
    .filter(el => !!el.dataset.tabId)
    .map(el => el.dataset.tabId);
  return ids.filter((id, i) => ids.indexOf(id) === i);
}

function activeTabId(win) {
  // The strip's active entry is the tab on screen (the main row only
  // highlights the group it belongs to).
  const el = win.document.querySelector('#tab-list .chat-tab.active');
  return el ? el.dataset.tabId : null;
}

function entry(tabId, workDir, chatId, scopeWorkDir) {
  return {
    tabId: tabId,
    chatId: chatId || '',
    title: tabId,
    workDir: workDir || '',
    scopeWorkDir: scopeWorkDir || '',
  };
}

function clickEl(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

const MIXED_SNAPSHOT = {
  type: 'tabs_state',
  tabs: [
    entry('a1', '/ws/a', 'chat-1'),
    entry('wt', '/ws/a/.kiss-worktrees/kiss_wt-7', 'chat-2'),
    entry('b1', '/ws/b', 'chat-3'),
    entry('un', '', 'chat-4'),
    // A run_agent sub-task: runs in a scratch dir, scoped elsewhere.
    entry('api', '/home/u/.kiss/channel_work', 'chat-5', '/ws/c'),
  ],
};

function testEveryRegistryTabIsShownWhateverTheWorkspace() {
  for (const remote of [false, true]) {
    const {win, posted} = makeWebview({remote});
    setWorkspace(win, '/ws/a');
    send(win, MIXED_SNAPSHOT);
    assert.deepStrictEqual(
      tabBarIds(win),
      ['a1', 'wt', 'b1', 'un', 'api'],
      (remote ? 'remote' : 'vscode') +
        ': every registry tab gets a strip, whatever folder it runs in',
    );
    assert.strictEqual(
      posted.filter(m => m && m.type === 'closeTab').length,
      0,
      'showing the shared tabs must not touch the registry',
    );
  }
}

function testWorkspaceChangeLeavesTheTabBarAlone() {
  // Switching the global working directory (configData from the
  // daemon, or its workDirChanged broadcast) re-scopes the history
  // filter only: the strips, the active tab and every draft stay put,
  // and no placeholder tab is spawned.
  const {win} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, MIXED_SNAPSHOT);
  clickEl(win, win.document.querySelector('.chat-tab[data-tab-id="b1"]'));
  assert.strictEqual(activeTabId(win), 'b1');
  const input = win.document.getElementById('task-input');
  input.value = 'draft on b1';
  input.dispatchEvent(new win.Event('input', {bubbles: true}));

  setWorkspace(win, '/ws/zzz');
  assert.deepStrictEqual(tabBarIds(win), ['a1', 'wt', 'b1', 'un', 'api']);
  assert.strictEqual(activeTabId(win), 'b1', 'the active tab survives');
  assert.strictEqual(input.value, 'draft on b1', 'the draft survives');

  send(win, {type: 'workDirChanged', workDir: '/ws/b'});
  assert.deepStrictEqual(tabBarIds(win), ['a1', 'wt', 'b1', 'un', 'api']);
  assert.strictEqual(activeTabId(win), 'b1');
}

function testConfigArrivingAfterSnapshotChangesNothing() {
  // A late configData (the daemon answers after the first snapshot)
  // must not reshuffle a tab bar that was rendered without a workspace.
  const {win} = makeWebview();
  send(win, MIXED_SNAPSHOT);
  const before = tabBarIds(win);
  const active = activeTabId(win);
  setWorkspace(win, '/ws/b');
  assert.deepStrictEqual(tabBarIds(win), before);
  assert.strictEqual(activeTabId(win), active);
}

function testHistoryClickOnAnotherWorkspaceChatActivatesItsTab() {
  // The chat of a history row is already open in a tab: the click
  // switches to that tab instead of resuming the chat in a new one.
  const {win, posted} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1'), entry('b1', '/ws/b', 'chat-2')],
  });
  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [
      {
        id: 'chat-2',
        task_id: 't-9',
        title: 'other workspace chat',
        preview: 'other workspace chat',
        timestamp: 1000,
        has_events: true,
        work_dir: '/ws/b',
      },
    ],
  });
  const row = win.document.querySelector('.sidebar-item');
  assert.ok(row, 'the history row must render');
  const opensBefore = posted.filter(m => m && m.type === 'openTab').length;
  clickEl(win, row);
  assert.strictEqual(activeTabId(win), 'b1', 'the existing tab is activated');
  assert.strictEqual(
    posted.filter(m => m && m.type === 'openTab').length,
    opensBefore,
    'no fresh tab is registered for a chat that already has one',
  );
}

function testFileFromAnotherWorkspaceTabOpensAsBackgroundContentTab() {
  // A file/report produced by a tab that runs in another folder opens
  // like any other background tab's file: its content tab joins that
  // tab's group (shown on the group strip once the user is on that
  // chat) and waits without pulling the user off the tab they read.
  const {win} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1'), entry('b1', '/ws/b', 'chat-2')],
  });
  send(win, {
    type: 'fileContent',
    tabId: 'b1',
    path: '/ws/b/report.html',
    name: 'report.html',
    content: '<p>report</p>',
  });
  assert.strictEqual(activeTabId(win), 'a1', 'a background file never steals focus');
  assert.deepStrictEqual(
    tabBarIds(win),
    ['a1', 'b1'],
    "the other chat's file stays in that chat's group, off the strip of this one",
  );
  clickEl(win, win.document.querySelector('.chat-tab[data-tab-id="b1"]'));
  const strip = Array.from(win.document.querySelectorAll('#tab-list .chat-tab'));
  assert.strictEqual(strip.length, 2, "the content tab gets a strip in its owner's group");
  assert.strictEqual(strip[0].dataset.tabId, 'b1');
  assert.ok(strip[1].classList.contains('content-tab'));
  assert.strictEqual(tabBarIds(win).length, 3, 'three tabs are open in all');
  clickEl(win, win.document.querySelectorAll('#tab-list .chat-tab')[1]);
  assert.strictEqual(
    win.document
      .querySelector('#tab-list .chat-tab.active')
      .textContent.includes('report.html'),
    true,
    'the content tab is a first-class tab the user can activate',
  );
}

function testOneContentTabPerPathAcrossWorkspaces() {
  // Two chats in different folders opening the same absolute path
  // share one content tab, like one editor per file in VS Code.
  const {win} = makeWebview();
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1'), entry('b1', '/ws/b', 'chat-2')],
  });
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/shared/notes.md',
    name: 'notes.md',
    content: 'v1',
  });
  send(win, {
    type: 'fileContent',
    tabId: 'b1',
    path: '/shared/notes.md',
    name: 'notes.md',
    content: 'v2',
  });
  assert.strictEqual(tabBarIds(win).length, 3, 'one content tab for the path');
}

function testCloseOthersClosesEveryOtherTab() {
  // "Close Others" from a1's context menu closes b1 too: there is no
  // hidden tab a bulk close could spare.
  const {win, posted} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1'), entry('b1', '/ws/b', 'chat-2')],
  });
  const strip = win.document.querySelector('.chat-tab[data-tab-id="a1"]');
  strip.dispatchEvent(
    new win.MouseEvent('contextmenu', {bubbles: true, clientX: 5, clientY: 5}),
  );
  const item = Array.from(
    win.document.querySelectorAll('.tab-context-menu-item, [role="menuitem"]'),
  ).find(el => (el.textContent || '').trim() === 'Close Others');
  assert.ok(item, 'the context menu offers Close Others');
  clickEl(win, item);
  const closed = posted
    .filter(m => m && m.type === 'closeTab')
    .map(m => m.tabId);
  assert.deepStrictEqual(closed, ['b1'], 'the other workspace tab is closed');
}

function testReadySendsCanonicalWorkDir() {
  // Restart recovery must serialize the registry's CANONICAL work dir
  // (including '' = unpinned), never the stale local pin, or an empty
  // registry would be re-seeded with wrong pins.
  const {win, posted} = makeWebview();
  send(win, {type: 'tabs_state', tabs: [entry('b1', '/ws/b', 'chat-1')]});
  send(win, {type: 'tabs_state', tabs: [entry('b1', '', 'chat-1')]});
  // A real outage (the daemon had been up), not a cold start: only
  // the former re-announces ready.
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  const readies = posted.filter(m => m && m.type === 'ready');
  const last = readies[readies.length - 1];
  const restored = Array.from(last.restoredTabs || []).find(
    t => t.tabId === 'b1',
  );
  assert.ok(restored, 'the tab must be offered for recovery');
  assert.strictEqual(restored.workDir, '', 'canonical (unpinned) work dir');
}

const tests = [
  testEveryRegistryTabIsShownWhateverTheWorkspace,
  testWorkspaceChangeLeavesTheTabBarAlone,
  testConfigArrivingAfterSnapshotChangesNothing,
  testHistoryClickOnAnotherWorkspaceChatActivatesItsTab,
  testFileFromAnotherWorkspaceTabOpensAsBackgroundContentTab,
  testOneContentTabPerPathAcrossWorkspaces,
  testCloseOthersClosesEveryOtherTab,
  testReadySendsCanonicalWorkDir,
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
  console.error(failures + ' test(s) failed');
  process.exit(1);
}
console.log('All ' + tests.length + ' sharedTabsAcrossWorkspaces tests passed');
// jsdom keeps timers alive; end the process like the other suites do.
process.exit(0);
