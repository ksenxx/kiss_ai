// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for shared-tab mirroring in the chat
// webview: the daemon owns the canonical tab registry and broadcasts
// full `tabs_state` snapshots; the webview reconciles its local tab
// bar against them.  Tabs are NEVER persisted client-side any more —
// only the active-tab selection and drawer preferences are.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(initialState) {
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

  const posted = [];
  let state = initialState;
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

  return {
    win,
    posted,
    getState: () => state,
  };
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

// Every open tab in tab order.  No row renders the chat tabs any more
// (the chat on screen is the one picked in the Chats panel; the strip
// under it only lists that chat's group), so the tab records are the
// mirror of the daemon registry that the tests compare against.
function tabBarIds(win) {
  return plain(win._testApi.openTabs().map(t => t.id));
}

function tabBarTitles(win) {
  return plain(win._testApi.openTabs().map(t => t.title));
}

// JSDOM objects come from another JS realm, so deepStrictEqual's
// prototype identity check rejects them; normalize through JSON.
function plain(value) {
  return JSON.parse(JSON.stringify(value));
}

function activeTabId(win) {
  return win._testApi.getActiveTabId();
}

function msgsOf(posted, type) {
  return posted.filter(m => m && m.type === type);
}

function snapshotEntry(tabId, title, chatId, workDir) {
  return {
    tabId: tabId,
    chatId: chatId || '',
    title: title || 'new chat',
    workDir: workDir || '',
  };
}

// --- ready / boot ---------------------------------------------------------

function testReadyIgnoresStaleLocalTabSet() {
  // Tabs come from the daemon's registry alone: a `tabs` key left in
  // local storage by a pre-registry client is neither restored on
  // screen nor carried into `ready`.
  const staleState = {
    tabs: [
      {
        title: 'Old task',
        chatId: 'legacy-tab-1',
        backendChatId: 'chat-1',
        workDir: '/w1',
      },
    ],
    chatId: 'legacy-tab-1',
  };
  const {win, posted} = makeWebview(staleState);
  const readies = msgsOf(posted, 'ready');
  assert.strictEqual(readies.length, 1, 'exactly one ready at boot');
  assert.deepStrictEqual(plain(readies[0].restoredTabs), []);
  // The placeholder tab is on screen until the daemon's snapshot arrives.
  assert.strictEqual(tabBarIds(win).length, 1);
}

function testReadyWithoutLegacyStateSendsEmptyRestoredTabs() {
  const {posted} = makeWebview(undefined);
  const readies = msgsOf(posted, 'ready');
  assert.strictEqual(readies.length, 1);
  assert.deepStrictEqual(plain(readies[0].restoredTabs), []);
}

function testTabsAreNoLongerPersistedLocally() {
  const {win, getState} = makeWebview(undefined);
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'one'), snapshotEntry('t2', 'two')],
  });
  const state = getState();
  assert.ok(state, 'client-local state is still persisted');
  assert.strictEqual(
    state.tabs,
    undefined,
    'the tab set must not be persisted client-side any more — it is ' +
      'server-canonical',
  );
  assert.ok('chatId' in state, 'active-tab selection stays client-local');
  assert.ok('drawersVersion' in state, 'drawer prefs stay client-local');
}

// --- tabs_state reconciliation --------------------------------------------

function testSnapshotAdoptsTabsAndDropsPlaceholder() {
  const {win} = makeWebview(undefined);
  const placeholder = activeTabId(win);
  send(win, {
    type: 'tabs_state',
    tabs: [
      snapshotEntry('t1', 'first tab', 'chat-1'),
      snapshotEntry('t2', 'second tab'),
    ],
  });
  assert.deepStrictEqual(tabBarIds(win), ['t1', 't2']);
  assert.deepStrictEqual(tabBarTitles(win), ['first tab', 'second tab']);
  assert.ok(
    !tabBarIds(win).includes(placeholder),
    'the boot placeholder is replaced by the canonical snapshot',
  );
  assert.ok(
    ['t1', 't2'].includes(activeTabId(win)),
    'a canonical tab becomes active when the placeholder goes away',
  );
}

function testSnapshotRestoresSavedSelection() {
  const {win} = makeWebview({chatId: 't2', drawersVersion: 3});
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'one'), snapshotEntry('t2', 'two')],
  });
  assert.strictEqual(
    activeTabId(win),
    't2',
    'this client\'s own saved selection is restored from the snapshot',
  );
}

function testSnapshotFollowsTitlesOrderAndRemovals() {
  const {win} = makeWebview(undefined);
  send(win, {
    type: 'tabs_state',
    tabs: [
      snapshotEntry('t1', 'one'),
      snapshotEntry('t2', 'two'),
      snapshotEntry('t3', 'three'),
    ],
  });
  send(win, {
    type: 'tabs_state',
    tabs: [
      snapshotEntry('t3', 'three renamed'),
      snapshotEntry('t1', 'one'),
    ],
  });
  assert.deepStrictEqual(tabBarIds(win), ['t3', 't1']);
  assert.deepStrictEqual(tabBarTitles(win), ['three renamed', 'one']);
}

function testSnapshotClipsLongTitles() {
  const {win} = makeWebview(undefined);
  const long = 'x'.repeat(64);
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', long)]});
  const label = tabBarTitles(win)[0];
  assert.strictEqual(label, 'x'.repeat(30) + '\u2026');
}

function testActiveTabRemovalActivatesSurvivor() {
  const {win} = makeWebview(undefined);
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'one'), snapshotEntry('t2', 'two')],
  });
  const active = activeTabId(win);
  const survivor = active === 't1' ? 't2' : 't1';
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry(survivor, 'survivor')],
  });
  assert.strictEqual(activeTabId(win), survivor);
}

function testEmptySnapshotKeepsUnregisteredPlaceholder() {
  const {win, posted} = makeWebview(undefined);
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'one')],
  });
  posted.length = 0;
  send(win, {type: 'tabs_state', tabs: []});
  assert.strictEqual(
    tabBarIds(win).length,
    1,
    'an empty registry leaves one local placeholder so the composer ' +
      'always exists',
  );
  assert.strictEqual(
    msgsOf(posted, 'openTab').length,
    0,
    'the placeholder is NOT registered — the daemon adopts it when a ' +
      'task first runs in it',
  );
}

// --- local tab creation ----------------------------------------------------

function testCreateNewTabRegistersWithDaemon() {
  const {win, posted} = makeWebview(undefined);
  posted.length = 0;
  win._testApi.createNewTab();
  const opens = msgsOf(posted, 'openTab');
  assert.strictEqual(opens.length, 1, 'createNewTab announces openTab');
  assert.strictEqual(opens[0].tabId, activeTabId(win));
  const news = msgsOf(posted, 'newChat');
  assert.strictEqual(news.length, 1, 'newChat still initializes the chat');
}

function testInFlightOpenSurvivesStaleSnapshot() {
  const {win} = makeWebview(undefined);
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  win._testApi.createNewTab();
  const fresh = activeTabId(win);
  // A snapshot broadcast before the daemon processed this client's
  // openTab must not remove the brand-new local tab.
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  assert.ok(
    tabBarIds(win).includes(fresh),
    'a snapshot that predates the in-flight openTab must keep the tab',
  );
  // Once the echo lists it, the registration is complete...
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'one'), snapshotEntry(fresh, 'new chat')],
  });
  assert.deepStrictEqual(tabBarIds(win), ['t1', fresh]);
  // ...and a LATER snapshot without it (another client closed it)
  // removes it like any other tab.
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  assert.deepStrictEqual(tabBarIds(win), ['t1']);
}

// --- client-local tabs and per-tab state -----------------------------------

function testSubagentAndContentTabsSurviveReconcile() {
  const {win} = makeWebview(undefined);
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'parent', 'chat-1')],
  });
  // A live sub-agent announcement creates a derived, client-local tab.
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-1',
    parent_tab_id: 't1',
    description: 'sub work',
    task_id: 'task-9',
    isSubagentTab: true,
    isDone: false,
  });
  assert.deepStrictEqual(tabBarIds(win), ['t1', 'sub-1']);
  // Reconcile with a snapshot that (of course) does not list the
  // sub-agent tab: it must survive, anchored after its parent.
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t2', 'other'), snapshotEntry('t1', 'parent')],
  });
  assert.deepStrictEqual(tabBarIds(win), ['t2', 't1', 'sub-1']);
  // Closing the parent on ANOTHER client cascades to the local
  // sub-agent tab.
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t2', 'other')]});
  assert.deepStrictEqual(tabBarIds(win), ['t2']);
}

function testPerTabComposerAndModelSurviveReconcile() {
  // Each tab keeps its own composer draft and model pick; a snapshot
  // (e.g. a rename from another client) must never clobber them.
  const {win} = makeWebview(undefined);
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'one'), snapshotEntry('t2', 'two')],
  });
  send(win, {
    type: 'models',
    models: [
      {name: 'model-a', inp: 1, out: 1, uses: 0, vendor: 'v'},
      {name: 'model-b', inp: 1, out: 1, uses: 0, vendor: 'v'},
    ],
    selected: 'model-a',
  });
  send(win, {type: 'modelPick', model: 'model-b', source: 'restore',
             tabId: 't2'});
  const inp = win.document.getElementById('task-input');
  // Type a draft into the ACTIVE tab's composer.
  const active = activeTabId(win);
  inp.value = 'my half-typed prompt';
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  // Another client renames both tabs.
  send(win, {
    type: 'tabs_state',
    tabs: [
      snapshotEntry('t1', 'one renamed'),
      snapshotEntry('t2', 'two renamed'),
    ],
  });
  assert.strictEqual(activeTabId(win), active, 'selection is untouched');
  assert.strictEqual(
    inp.value,
    'my half-typed prompt',
    'the composer draft survives a snapshot',
  );
  assert.deepStrictEqual(tabBarTitles(win), ['one renamed', 'two renamed']);
}

// --- daemon reconnect -------------------------------------------------------

function testReconnectResendsReadyWithCurrentTabs() {
  const {win, posted} = makeWebview(undefined);
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'bound tab', 'chat-7', '/w7')],
  });
  // The daemon was up (it answered the boot ready with that snapshot):
  // a later outage is a real reconnect.  A cold start -- connected:false
  // before any connected:true -- must NOT re-ready (the boot ready is
  // still queued in the host and goes out on auth).
  send(win, {type: 'daemonStatus', connected: true});
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  const readies = msgsOf(posted, 'ready');
  assert.strictEqual(
    readies.length,
    1,
    'a daemon reconnect re-announces ready exactly once',
  );
  assert.deepStrictEqual(plain(readies[0].restoredTabs), [
    {tabId: 't1', chatId: 'chat-7', title: 'bound tab', workDir: '/w7'},
  ]);
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: true});
  assert.strictEqual(
    msgsOf(posted, 'ready').length,
    0,
    'a repeated connected=true without a disconnect must not re-ready',
  );
}

function testCloseTabStillAnnouncedToDaemon() {
  const {win, posted} = makeWebview(undefined);
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'one'), snapshotEntry('t2', 'two')],
  });
  // Only the chat on screen is rendered (its own entry on the group
  // strip carries the close button), so bring t2 on screen first.
  win._testApi.switchToTab('t2');
  posted.length = 0;
  const closeBtn = win.document.querySelector(
    '#tab-list .chat-tab[data-tab-id="t2"] .chat-tab-close',
  );
  assert.ok(closeBtn, 'the chat on screen has a close button on the strip');
  closeBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const closes = msgsOf(posted, 'closeTab');
  assert.strictEqual(closes.length, 1);
  assert.strictEqual(closes[0].tabId, 't2');
}

const tests = [
  testReadyIgnoresStaleLocalTabSet,
  testReadyWithoutLegacyStateSendsEmptyRestoredTabs,
  testTabsAreNoLongerPersistedLocally,
  testSnapshotAdoptsTabsAndDropsPlaceholder,
  testSnapshotRestoresSavedSelection,
  testSnapshotFollowsTitlesOrderAndRemovals,
  testSnapshotClipsLongTitles,
  testActiveTabRemovalActivatesSurvivor,
  testEmptySnapshotKeepsUnregisteredPlaceholder,
  testCreateNewTabRegistersWithDaemon,
  testInFlightOpenSurvivesStaleSnapshot,
  testSubagentAndContentTabsSurviveReconcile,
  testPerTabComposerAndModelSurviveReconcile,
  testReconnectResendsReadyWithCurrentTabs,
  testCloseTabStillAnnouncedToDaemon,
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
console.log(`All ${tests.length} tabsMirror tests passed`);
