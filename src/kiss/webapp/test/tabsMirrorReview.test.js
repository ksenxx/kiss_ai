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
const harness = require('./simplify2_harness');
const {trackWindow, runTests} = require('./workdir_helpers');
const {clickClose} = require('./subagent_tab_helpers');
const {
  installFakeWebSocket,
  bootShim,
  frames,
} = require('./transport_page_helpers');
const {send, tabIds: openTabIds} = harness;

/** A chat page, closed by the runner once its test ends. */
function makePage() {
  const page = harness.makePage();
  trackWindow(page.win);
  return page;
}

/**
 * A chat page on the REAL transport (media/transport.js) over a fake
 * WebSocket born CONNECTING, so what the daemon would receive, and
 * when, is exactly what the shim sends: the commands it queues during
 * an outage, re-sends while a reconnect's batch is unconfirmed, or
 * holds back behind one.  `connect()` completes the handshake of the
 * socket the shim opened last; `reconnect()` makes the shim open the
 * next one at once (it reconnects without backoff on `online`) and
 * completes that handshake, which flushes the queued batch and
 * announces `daemonStatus` to main.js.
 */
function makeTransportPage() {
  const win = harness.newDom({runScripts: 'outside-only'});
  trackWindow(win);
  const sockets = [];
  installFakeWebSocket(win, sockets, {connecting: true});
  bootShim(win);
  harness.loadPageScripts(win);
  function connect() {
    const sock = sockets[sockets.length - 1];
    sock.fireOpen();
    sock.fireMessage({type: 'auth_ok'});
    return sock;
  }
  function reconnect() {
    const before = sockets.length;
    win.dispatchEvent(new win.Event('online'));
    assert.strictEqual(sockets.length, before + 1, 'the shim reconnected');
    return connect();
  }
  return {win, sockets, connect, reconnect};
}

/** Every `openTab` frame sent on any socket of a transport page. */
function openTabFrames(sockets) {
  return sockets.flatMap(frames).filter(f => f.type === 'openTab');
}

/** The tab ids on the strip (#tab-list), in order. */
function tabBarIds(win) {
  return Array.from(win.document.querySelectorAll('.chat-tab'))
    .filter(el => !!el.dataset.tabId)
    .map(el => el.dataset.tabId);
}

const {ofType: msgsOf, snapshotEntry} = harness;

// Give a page a mirrored parent tab plus one sub-agent tab: live by
// default, finished history with `done`.  The user may hand-close
// either: a running sub-agent's tab stays open only until the user
// closes it.
function seedSubagent(win, done) {
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('parent', 'Parent')]});
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'parent__sub_1',
    parent_tab_id: 'parent',
    description: 'Sub one',
    task_id: '42',
    isDone: !!done,
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
  const {win, posted} = makePage();
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
  const {win, posted} = makePage();
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
  const {win, posted} = makePage();
  seedSubagent(win, true);
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
  const {win, posted} = makePage();
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  win._testApi.createNewTab();
  const opens = msgsOf(posted, 'openTab');
  assert.strictEqual(opens.length, 1);
  const newId = opens[0].tabId;
  assert.ok(openTabIds(win).includes(newId));
  const closesBefore = msgsOf(posted, 'closeTab').length;

  send(win, {type: 'openTabRejected', tabId: newId});

  assert.ok(
    !openTabIds(win).includes(newId),
    'a rejected openTab left a permanently local tab no other ' +
      'client will ever see',
  );
  assert.strictEqual(
    msgsOf(posted, 'closeTab').length,
    closesBefore,
    'dropping a rejected tab must not echo closeTab to the daemon',
  );
  assert.deepStrictEqual(openTabIds(win), ['t1']);
}

function testOpenTabRejectedForLastTabKeepsLocalPlaceholder() {
  // Rejecting the ONLY tab must not close the composer (and must not
  // re-register in a loop): the tab stays as a local placeholder.
  const {win, posted} = makePage();
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  win._testApi.createNewTab();
  const newId = msgsOf(posted, 'openTab')[0].tabId;
  // The registry tab disappears (another client closed it) leaving
  // only the pending local tab...
  send(win, {type: 'tabs_state', tabs: []});
  assert.deepStrictEqual(openTabIds(win), [newId]);
  const opensBefore = msgsOf(posted, 'openTab').length;

  send(win, {type: 'openTabRejected', tabId: newId});

  assert.strictEqual(
    openTabIds(win).length,
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
  // Against a daemon that echoes no acknowledgments (its snapshots
  // carry no `acks`), a pending openTab the daemon never confirms must
  // stop shielding the local tab after a few snapshots, so the client
  // converges back to the canonical tab set instead of keeping a ghost
  // tab forever.  A daemon that echoes them is pinned below (A18-1).
  const {win, posted} = makePage();
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  win._testApi.createNewTab();
  const newId = msgsOf(posted, 'openTab')[0].tabId;

  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  assert.ok(
    openTabIds(win).includes(newId),
    'an in-flight openTab must survive the first stale snapshot',
  );
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});

  assert.ok(
    !openTabIds(win).includes(newId),
    'a pending openTab no snapshot ever confirmed must expire — the ' +
      'tab stayed snapshot-immune forever',
  );
  assert.deepStrictEqual(openTabIds(win), ['t1']);
}

// Two stale snapshots then the echo adopting the tab (expiry only
// applies to ids that never get confirmed) is pinned by
// tabsMirror.test.js::testInFlightOpenSurvivesStaleSnapshot.

// 2026-11-05 A18-1: the daemon echoes the `ack` token of an `openTab` in
// the `acks` of the first snapshot taken after it processed the command.
// Snapshots other activity publishes meanwhile (every one carrying an
// `acks` list that names no token of ours) are no acknowledgment,
// however many: the pending open keeps shielding the tab and its draft.
// The acknowledging snapshot settles it: the tab is listed (opened) or
// not (the open was rejected, dropped or the tab closed meanwhile).
function testPendingOpenWaitsForItsAcknowledgment() {
  const {win, posted} = makePage();
  const stale = () => ({
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'one')],
    acks: ['someone-else'],
  });
  send(win, stale());
  win._testApi.createNewTab();
  const open = msgsOf(posted, 'openTab')[0];
  assert.ok(open.ack, 'openTab carries an ack token for the daemon to echo');
  const newId = open.tabId;
  win.document.getElementById('task-input').value = 'typed while opening';

  for (let i = 0; i < 5; i++) send(win, stale());
  assert.ok(
    openTabIds(win).includes(newId),
    'REGRESSION: unrelated snapshots retired the pending open and ' +
      'dropped the brand-new tab',
  );
  assert.strictEqual(
    win.document.getElementById('task-input').value,
    'typed while opening',
    'the draft typed into the new tab survives the stale snapshots',
  );

  // The acknowledging snapshot lists the tab: registered for good; a
  // later snapshot without it is another client's close, as usual.
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'one'), snapshotEntry(newId, 'new chat')],
    acks: [open.ack],
  });
  assert.deepStrictEqual(openTabIds(win), ['t1', newId]);
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')], acks: []});
  assert.deepStrictEqual(openTabIds(win), ['t1']);

  // A second open whose acknowledging snapshot OMITS the tab: the
  // daemon processed the open and the tab is not there (dropped, or
  // closed meanwhile), so the shield lifts and the tab goes.
  win._testApi.createNewTab();
  const open2 = msgsOf(posted, 'openTab')[1];
  assert.notStrictEqual(open2.ack, open.ack, 'tokens are unique per command');
  send(win, stale());
  assert.ok(openTabIds(win).includes(open2.tabId), 'still shielded');
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'one')],
    acks: [open2.ack],
  });
  assert.deepStrictEqual(
    openTabIds(win),
    ['t1'],
    'the acknowledging snapshot that omits the tab retires the open',
  );
}

// A18-1: an open the daemon had not acknowledged when the connection
// dropped is not announced again on reconnect.  A lost acknowledgment
// is no lost open: the daemon may have processed it, and the user may
// have closed the tab on another surface meanwhile -- a second
// `openTab` of that id is a new open to the daemon (TabRegistry.open_tab
// registers any absent id), which would bring the closed tab back on
// every surface.  The reconnect's `ready` snapshot is authoritative
// for opens as for closes: a tab it lists is kept, draft and all; one
// it omits is gone, as a lost open is in any case.  An open the
// transport still owes is different (round 3): the `daemonStatus` of
// the reconnect names the frames the transport has not had delivered
// yet (`unconfirmed`: the batch it just flushed, re-sent on the next
// connection if this one dies before its `pong`, and the frames held
// back behind it), and the shield of an open among them holds until
// the daemon answers it, through the snapshots other surfaces publish
// meanwhile and through every reconnect it is still owed at.  Here the
// recording transport plays the shim: the `daemonStatus` it sends
// carries what the real one would.
function testUnacknowledgedOpenIsNotReannouncedAfterReconnect() {
  const a = makePage();
  const b = makePage();
  const root = snapshotEntry('t1', 'one');
  for (const p of [a, b]) {
    send(p.win, {type: 'tabs_state', tabs: [root], acks: []});
    send(p.win, {type: 'daemonStatus', connected: true});
  }
  a.win._testApi.createNewTab();
  const open = msgsOf(a.posted, 'openTab')[0];
  const fresh = snapshotEntry(open.tabId, open.title);
  // The daemon processed the open: its acknowledging snapshot reaches
  // B, but A's connection drops first.  The user then closes the
  // shared tab on B.
  send(b.win, {type: 'tabs_state', tabs: [root, fresh], acks: [open.ack]});
  assert.deepStrictEqual(openTabIds(b.win), ['t1', open.tabId]);
  send(a.win, {type: 'daemonStatus', connected: false});
  b.win._testApi.switchToTab(open.tabId);
  clickClose(b.win, open.tabId);
  const close = msgsOf(b.posted, 'closeTab').pop();
  send(b.win, {type: 'tabs_state', tabs: [root], acks: [close.ack]});
  assert.deepStrictEqual(openTabIds(b.win), ['t1']);
  // A reconnects: `ready` goes out, the open does not.
  a.posted.length = 0;
  send(a.win, {type: 'daemonStatus', connected: true});
  assert.strictEqual(
    msgsOf(a.posted, 'ready').length,
    1,
    'reconnect re-readies',
  );
  assert.deepStrictEqual(
    msgsOf(a.posted, 'openTab'),
    [],
    'REGRESSION: the reconnect announced the open again, which brings ' +
      'back on every surface the tab the user closed on another one',
  );
  // The `ready` reply omits the tab: it is gone here too.
  send(a.win, {type: 'tabs_state', tabs: [root], acks: []});
  assert.deepStrictEqual(openTabIds(a.win), ['t1'], 'the ready snapshot rules');

  // An open the daemon processed and nobody closed: the `ready` reply
  // lists the tab, which is kept with its draft, and no `openTab` is
  // sent for it either.
  a.win._testApi.createNewTab();
  const open2 = msgsOf(a.posted, 'openTab')[0];
  a.win.document.getElementById('task-input').value = 'draft';
  a.posted.length = 0;
  send(a.win, {type: 'daemonStatus', connected: false});
  send(a.win, {type: 'daemonStatus', connected: true});
  assert.deepStrictEqual(msgsOf(a.posted, 'openTab'), []);
  send(a.win, {
    type: 'tabs_state',
    tabs: [root, snapshotEntry(open2.tabId, open2.title)],
    acks: [],
  });
  assert.deepStrictEqual(openTabIds(a.win), ['t1', open2.tabId]);
  assert.strictEqual(
    a.win.document.getElementById('task-input').value,
    'draft',
  );

  // A tab opened while the daemon was down: its `openTab` is on its
  // way once the connection is back, so the snapshots that arrive
  // before the daemon answers it (other surfaces re-readying after
  // the same outage) must not drop the tab and what was typed into it.
  send(a.win, {type: 'daemonStatus', connected: false});
  a.win._testApi.createNewTab();
  const open3 = msgsOf(a.posted, 'openTab').pop();
  a.win.document.getElementById('task-input').value = 'typed offline';
  a.posted.length = 0;
  // (An entry that is no command is skipped.)
  send(a.win, {
    type: 'daemonStatus',
    connected: true,
    unconfirmed: [null, open3],
  });
  assert.deepStrictEqual(
    msgsOf(a.posted, 'openTab'),
    [],
    'not announced twice',
  );
  const listed = [root, snapshotEntry(open2.tabId, open2.title)];
  send(a.win, {type: 'tabs_state', tabs: listed, acks: ['someone-else']});
  send(a.win, {type: 'tabs_state', tabs: listed, acks: []});
  assert.deepStrictEqual(
    openTabIds(a.win),
    ['t1', open2.tabId, open3.tabId],
    'REGRESSION: the reconnect dropped the tab opened during the outage ' +
      'before the daemon could answer its open',
  );
  assert.strictEqual(
    a.win.document.getElementById('task-input').value,
    'typed offline',
  );
  send(a.win, {
    type: 'tabs_state',
    tabs: listed.concat(snapshotEntry(open3.tabId, open3.title)),
    acks: [open3.ack],
  });
  assert.deepStrictEqual(openTabIds(a.win), ['t1', open2.tabId, open3.tabId]);

  // Opened during an outage, still owed when the next connection is
  // lost too (the transport re-sends its batch on the one after): the
  // shield holds across that reconnect as well, until the daemon's
  // answer -- which, the tab having been closed on another surface
  // meanwhile, acknowledges the open without listing it.
  const listed3 = listed.concat(snapshotEntry(open3.tabId, open3.title));
  send(a.win, {type: 'daemonStatus', connected: false});
  a.win._testApi.createNewTab();
  const open4 = msgsOf(a.posted, 'openTab').pop();
  a.win.document.getElementById('task-input').value = 'typed offline too';
  send(a.win, {type: 'daemonStatus', connected: true, unconfirmed: [open4]});
  send(a.win, {type: 'tabs_state', tabs: listed3, acks: []});
  assert.ok(openTabIds(a.win).includes(open4.tabId), 'shielded once');
  a.posted.length = 0;
  send(a.win, {type: 'daemonStatus', connected: false});
  send(a.win, {type: 'daemonStatus', connected: true, unconfirmed: [open4]});
  assert.deepStrictEqual(msgsOf(a.posted, 'openTab'), [], 'never re-sent');
  send(a.win, {type: 'tabs_state', tabs: listed3, acks: ['someone-else']});
  assert.deepStrictEqual(
    openTabIds(a.win),
    ['t1', open2.tabId, open3.tabId, open4.tabId],
    'REGRESSION: the second reconnect dropped a tab whose open the ' +
      'transport still owes, and what was typed into it',
  );
  assert.strictEqual(
    a.win.document.getElementById('task-input').value,
    'typed offline too',
  );
  send(a.win, {type: 'tabs_state', tabs: listed3, acks: [open4.ack]});
  assert.deepStrictEqual(
    openTabIds(a.win),
    ['t1', open2.tabId, open3.tabId],
    'the acknowledging snapshot rules: the tab closed elsewhere is gone',
  );

  // Opened during an outage and delivered on the reconnect (the batch
  // was confirmed; only the answer was lost with the connection): the
  // next reconnect owes nothing for it, so the `ready` reply settles
  // it like any other open, and no ghost tab is shielded for ever.
  send(a.win, {type: 'daemonStatus', connected: false});
  a.win._testApi.createNewTab();
  const open5 = msgsOf(a.posted, 'openTab').pop();
  send(a.win, {type: 'daemonStatus', connected: true, unconfirmed: [open5]});
  send(a.win, {type: 'tabs_state', tabs: listed3, acks: []});
  assert.ok(openTabIds(a.win).includes(open5.tabId), 'shielded while owed');
  send(a.win, {type: 'daemonStatus', connected: false});
  send(a.win, {type: 'daemonStatus', connected: true, unconfirmed: []});
  send(a.win, {type: 'tabs_state', tabs: listed3, acks: []});
  assert.deepStrictEqual(
    openTabIds(a.win),
    ['t1', open2.tabId, open3.tabId],
    'REGRESSION: an open the transport delivered and the daemon never ' +
      'listed keeps shielding a tab no other surface has',
  );
}

// Audit 2026-11-11 (W1): the same verdict from an EMPTY registry.  An
// empty snapshot spares the local placeholder it never listed, but a
// tab this page announced is the registry's to keep or drop: opens the
// daemon answered without listing them (every tab was closed on another
// surface during the outage) must not survive as tabs no other surface
// has.  One unannounced placeholder takes their place, and a tab the
// daemon REJECTED as the page's only chat is that placeholder again.
function testAnsweredOpensDoNotSurviveAnEmptyRegistry() {
  const {win, posted} = makePage();
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')], acks: []});
  win._testApi.createNewTab();
  win._testApi.createNewTab();
  const opens = msgsOf(posted, 'openTab');
  assert.strictEqual(opens.length, 2, 'setup: two opens announced');
  // Both opens were delivered; their snapshots were lost with the
  // connection, and another surface closed every tab meanwhile.
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true, unconfirmed: []});
  send(win, {type: 'tabs_state', tabs: [], acks: opens.map(o => o.ack)});
  const ids = openTabIds(win);
  assert.strictEqual(
    ids.length,
    1,
    'BUG: tabs closed elsewhere survive an empty registry: ' + ids,
  );
  assert.ok(
    !opens.some(o => o.tabId === ids[0]),
    'the survivor is a fresh, unannounced placeholder',
  );
  assert.strictEqual(
    msgsOf(posted, 'openTab').length,
    2,
    'the placeholder is not announced',
  );

  // A rejected sole tab stays as the local placeholder: later empty
  // snapshots keep sparing it, as they spare any placeholder.
  const b = makePage();
  send(b.win, {type: 'tabs_state', tabs: [snapshotEntry('t1', 'one')]});
  b.win._testApi.createNewTab();
  const newId = msgsOf(b.posted, 'openTab')[0].tabId;
  send(b.win, {type: 'tabs_state', tabs: []});
  send(b.win, {type: 'openTabRejected', tabId: newId});
  send(b.win, {type: 'tabs_state', tabs: []});
  assert.deepStrictEqual(
    openTabIds(b.win),
    [newId],
    'the rejected sole tab is the placeholder an empty registry spares',
  );
}

// Round 3 (a): the shield follows the transport's real delivery state.
// A frame posted while a reconnect's batch awaits its `pong` is held
// back behind the batch, connected or not, and sent only after the
// `pong` -- through as many reconnects as it takes.  Until then the
// open was never sent: a snapshot omitting the tab is no verdict on
// it, and retiring the shield at a reconnect lost the tab and its
// draft to the next unrelated snapshot, after which the open, sent at
// last, recreated the tab empty.  The real shim over a fake socket:
// the open's delivery is whatever the shim does with it.
function heldOpenSurvivesReconnects(openOffline) {
  const {win, sockets, connect, reconnect} = makeTransportPage();
  const root = snapshotEntry('t1', 'one');
  const first = connect();
  first.fireMessage({type: 'pong'});
  first.fireMessage({type: 'tabs_state', tabs: [root], acks: []});
  first.fireClose();
  // A command queued during the outage makes the reconnect's batch,
  // which awaits its `pong`; the open is posted behind it.
  win.acquireTransport().postMessage({type: 'getSettings'});
  const second = reconnect();
  if (openOffline) second.fireClose();
  win._testApi.createNewTab();
  const newId = openTabIds(win).find(id => id !== 't1');
  const input = win.document.getElementById('task-input');
  input.value = 'unsent draft';
  input.dispatchEvent(new win.Event('input', {bubbles: true}));
  assert.deepStrictEqual(openTabFrames(sockets), [], 'the open is held back');
  if (!openOffline) second.fireClose();
  const unrelated = {type: 'tabs_state', tabs: [root], acks: ['other-surface']};
  let sock = reconnect();
  sock.fireMessage(unrelated);
  assert.deepStrictEqual(
    openTabIds(win),
    ['t1', newId],
    'REGRESSION: an unrelated snapshot dropped a tab whose open the ' +
      'transport has not sent yet',
  );
  if (openOffline) {
    sock.fireClose();
    sock = reconnect();
    sock.fireMessage(unrelated);
    assert.deepStrictEqual(
      openTabIds(win),
      ['t1', newId],
      'REGRESSION: the second reconnect dropped the still-unsent open',
    );
  }
  assert.deepStrictEqual(openTabFrames(sockets), [], 'still not sent');
  sock.fireMessage({type: 'pong'});
  const sent = openTabFrames(sockets);
  assert.strictEqual(sent.length, 1, 'the open goes out once, after the pong');
  assert.strictEqual(sent[0].tabId, newId);
  sock.fireMessage({
    type: 'tabs_state',
    tabs: [root, snapshotEntry(newId, sent[0].title)],
    acks: [sent[0].ack],
  });
  assert.deepStrictEqual(openTabIds(win), ['t1', newId]);
  win._testApi.switchToTab(newId);
  assert.strictEqual(input.value, 'unsent draft', 'the draft survived');
}

function testHeldOpenMadeOnlineSurvivesReconnects() {
  heldOpenSurvivesReconnects(false);
}

function testHeldOpenMadeOfflineSurvivesReconnects() {
  heldOpenSurvivesReconnects(true);
}

// Round 3 (b), the page's side: an open made during an outage goes out
// in the reconnect's batch; when that connection dies before the
// daemon's `pong`, the shim re-sends the batch on the next one, token
// and all (at-least-once), and main.js announces nothing on top.  The
// daemon decides a token once (OpenTabOutcome.REPLAYED, see
// test_web_server_tab_mirroring.py): its answer echoes the token and
// lists the tab, or omits it when another surface closed it in
// between.  The shield, kept while the open is owed, settles on that
// answer either way -- never on an unrelated snapshot before it.
function testReplayedOpenIsSettledByTheDaemonsAnswer() {
  const {win, sockets, connect, reconnect} = makeTransportPage();
  const root = snapshotEntry('t1', 'one');
  const first = connect();
  first.fireMessage({type: 'pong'});
  first.fireMessage({type: 'tabs_state', tabs: [root], acks: []});
  first.fireClose();
  win._testApi.createNewTab();
  const newId = openTabIds(win).find(id => id !== 't1');
  const second = reconnect();
  const [open] = openTabFrames(sockets);
  assert.ok(open && open.tabId === newId, 'the queued open went out');
  // The daemon processed it, but this connection died before its
  // answer and the batch's pong arrived; the user closed the tab on
  // another surface meanwhile.
  second.fireClose();
  const third = reconnect();
  assert.deepStrictEqual(
    openTabFrames(sockets),
    [open, open],
    'the shim re-sends the open with its token; main.js adds no open',
  );
  third.fireMessage({
    type: 'tabs_state',
    tabs: [root],
    acks: ['other-surface'],
  });
  assert.deepStrictEqual(
    openTabIds(win),
    ['t1', newId],
    'REGRESSION: an unrelated snapshot dropped the tab whose open the ' +
      'transport is still delivering',
  );
  third.fireMessage({type: 'tabs_state', tabs: [root], acks: [open.ack]});
  assert.deepStrictEqual(
    openTabIds(win),
    ['t1'],
    'the acknowledging answer rules: the tab closed elsewhere is gone',
  );
  third.fireMessage({type: 'pong'});
  assert.strictEqual(openTabFrames(sockets).length, 2, 'nothing more is sent');
}

// The `ready` of a reconnect whose `pong` is lost was never sent: the
// shim holds it back behind the batch awaiting that `pong`, and sends
// it on the connection that follows.  main.js announces no second one
// there, which would have the daemon resync this page twice.
function testReadyGoesOutOncePerConnectionAfterALostPong() {
  const {win, connect, reconnect} = makeTransportPage();
  const first = connect();
  first.fireMessage({type: 'pong'});
  first.fireClose();
  // A command queued during the outage makes the reconnect's batch.
  win.acquireTransport().postMessage({type: 'getSettings'});
  reconnect().fireClose(); // died before its pong
  const third = reconnect();
  third.fireMessage({type: 'pong'});
  assert.strictEqual(
    frames(third).filter(f => f.type === 'ready').length,
    1,
    'BUG: `ready` went out twice on one connection',
  );
}

const tests = [
  testCloseSubagentTabBroadcastRemovesTabWithoutEcho,
  testCloseSubagentTabForUnknownIdIsIgnored,
  testManualSubagentCloseStillAnnouncedToDaemon,
  testOpenTabRejectedDropsLocalPendingTab,
  testOpenTabRejectedForLastTabKeepsLocalPlaceholder,
  testUnconfirmedPendingOpenExpiresAfterSnapshots,
  testPendingOpenWaitsForItsAcknowledgment,
  testUnacknowledgedOpenIsNotReannouncedAfterReconnect,
  testAnsweredOpensDoNotSurviveAnEmptyRegistry,
  testHeldOpenMadeOnlineSurvivesReconnects,
  testHeldOpenMadeOfflineSurvivesReconnects,
  testReplayedOpenIsSettledByTheDaemonsAnswer,
  testReadyGoesOutOncePerConnectionAfterALostPong,
];

runTests(tests.map(fn => [fn.name, fn]));
