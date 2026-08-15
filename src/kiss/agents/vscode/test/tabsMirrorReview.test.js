// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the tab-mirroring review fixes:
//  [2] a sub-agent tab closed on one client is removed on every other
//      client via the daemon's `closeSubagentTab` broadcast, applied
//      locally WITHOUT echoing a `closeTab` back (no feedback loop);
//  [5] a cap-rejected `openTab` is answered with `openTabRejected`,
//      which drops the local pending tab, and pending ids that no
//      snapshot ever confirms expire instead of staying immune.

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

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function tabBarIds(win) {
  return Array.from(win.document.querySelectorAll('.chat-tab'))
    .filter(el => !!el.dataset.tabId)
    .map(el => el.dataset.tabId);
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

// Give a webview a mirrored parent tab plus one live sub-agent tab.
function seedSubagent(win) {
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('parent', 'Parent')]});
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'parent__sub_1',
    parent_tab_id: 'parent',
    description: 'Sub one',
    task_id: '42',
  });
  assert.ok(
    tabBarIds(win).includes('parent__sub_1'),
    'harness: the sub-agent tab never opened',
  );
}

// --- [2] sub-agent close mirroring ----------------------------------------

function testCloseSubagentTabBroadcastRemovesTabWithoutEcho() {
  // Client B shows the same sub-agent tab client A just closed; the
  // daemon's `closeSubagentTab` broadcast must remove it on B without
  // B echoing a `closeTab` back to the daemon (feedback loop).
  const {win, posted} = makeWebview(undefined);
  seedSubagent(win);
  const closesBefore = msgsOf(posted, 'closeTab').length;

  send(win, {type: 'closeSubagentTab', tab_id: 'parent__sub_1'});

  assert.ok(
    !tabBarIds(win).includes('parent__sub_1'),
    'the broadcast close never removed the sub-agent tab — the tab ' +
      'sets diverge between clients',
  );
  assert.strictEqual(
    msgsOf(posted, 'closeTab').length,
    closesBefore,
    'applying a broadcast close must NOT echo closeTab back to the ' +
      'daemon',
  );
  // The parent tab is untouched.
  assert.ok(tabBarIds(win).includes('parent'));
}

function testCloseSubagentTabForUnknownIdIsIgnored() {
  const {win, posted} = makeWebview(undefined);
  seedSubagent(win);
  const before = tabBarIds(win);
  const postedBefore = posted.length;
  send(win, {type: 'closeSubagentTab', tab_id: 'no-such-tab__sub_7'});
  assert.deepStrictEqual(tabBarIds(win), before);
  assert.strictEqual(posted.length, postedBefore);
}

function testManualSubagentCloseStillAnnouncedToDaemon() {
  // The origin side of the mirror: a hand-closed sub-agent tab must
  // still send `closeTab` so the daemon can broadcast the close.
  const {win, posted} = makeWebview(undefined);
  seedSubagent(win);
  const closeBtn = win.document.querySelector(
    '.chat-tab[data-tab-id="parent__sub_1"] .chat-tab-close',
  );
  closeBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const closes = msgsOf(posted, 'closeTab');
  assert.strictEqual(closes.length, 1);
  assert.strictEqual(closes[0].tabId, 'parent__sub_1');
  assert.ok(!tabBarIds(win).includes('parent__sub_1'));
}

// --- [5] rejected openTab + pending expiry ---------------------------------

function testOpenTabRejectedDropsLocalPendingTab() {
  const {win, posted} = makeWebview(undefined);
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  win._testApi.createNewTab();
  const opens = msgsOf(posted, 'openTab');
  assert.strictEqual(opens.length, 1);
  const newId = opens[0].tabId;
  assert.ok(tabBarIds(win).includes(newId));
  const closesBefore = msgsOf(posted, 'closeTab').length;

  send(win, {type: 'openTabRejected', tabId: newId});

  assert.ok(
    !tabBarIds(win).includes(newId),
    'a rejected openTab left a permanently local tab no other ' +
      'client will ever see',
  );
  assert.strictEqual(
    msgsOf(posted, 'closeTab').length,
    closesBefore,
    'dropping a rejected tab must not echo closeTab to the daemon',
  );
  assert.deepStrictEqual(tabBarIds(win), ['t1']);
}

function testOpenTabRejectedForLastTabKeepsLocalPlaceholder() {
  // Rejecting the ONLY tab must not close the composer (and must not
  // re-register in a loop): the tab stays as a local placeholder.
  const {win, posted} = makeWebview(undefined);
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  win._testApi.createNewTab();
  const newId = msgsOf(posted, 'openTab')[0].tabId;
  // The registry tab disappears (another client closed it) leaving
  // only the pending local tab...
  send(win, {type: 'tabs_state', tabs: []});
  assert.deepStrictEqual(tabBarIds(win), [newId]);
  const opensBefore = msgsOf(posted, 'openTab').length;

  send(win, {type: 'openTabRejected', tabId: newId});

  assert.strictEqual(
    tabBarIds(win).length,
    1,
    'rejecting the last tab must keep one local placeholder',
  );
  assert.strictEqual(
    msgsOf(posted, 'openTab').length,
    opensBefore,
    'a rejection must not trigger another openTab (registration loop)',
  );
}

function testUnconfirmedPendingOpenExpiresAfterSnapshots() {
  // A pending openTab the daemon never confirms must stop shielding
  // the local tab after a few snapshots, so the client converges back
  // to the canonical tab set instead of keeping a ghost tab forever.
  const {win, posted} = makeWebview(undefined);
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  win._testApi.createNewTab();
  const newId = msgsOf(posted, 'openTab')[0].tabId;

  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  assert.ok(
    tabBarIds(win).includes(newId),
    'an in-flight openTab must survive the first stale snapshot',
  );
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});

  assert.ok(
    !tabBarIds(win).includes(newId),
    'a pending openTab no snapshot ever confirmed must expire — the ' +
      'tab stayed snapshot-immune forever',
  );
  assert.deepStrictEqual(tabBarIds(win), ['t1']);
}

function testConfirmedPendingOpenIsAdoptedDespiteEarlierMisses() {
  // Two stale snapshots then the echo: the tab must be adopted, not
  // expired — expiry only applies to ids that never get confirmed.
  const {win, posted} = makeWebview(undefined);
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  win._testApi.createNewTab();
  const newId = msgsOf(posted, 'openTab')[0].tabId;

  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'one'), snapshotEntry(newId, 'new chat')],
  });
  assert.ok(
    tabBarIds(win).includes(newId),
    'the confirming snapshot must adopt the tab',
  );
  // And it stays through later snapshots that list it.
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'one'), snapshotEntry(newId, 'new chat')],
  });
  assert.ok(tabBarIds(win).includes(newId));
}

const tests = [
  testCloseSubagentTabBroadcastRemovesTabWithoutEcho,
  testCloseSubagentTabForUnknownIdIsIgnored,
  testManualSubagentCloseStillAnnouncedToDaemon,
  testOpenTabRejectedDropsLocalPendingTab,
  testOpenTabRejectedForLastTabKeepsLocalPlaceholder,
  testUnconfirmedPendingOpenExpiresAfterSnapshots,
  testConfirmedPendingOpenIsAdoptedDespiteEarlierMisses,
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
console.log(`All ${tests.length} tabsMirrorReview tests passed`);
