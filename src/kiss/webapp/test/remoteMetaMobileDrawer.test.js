// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the mobile remote webapp's task-info drawer
// (media/chat.html + media/main.js, remote mode below 900px):
//
// * the status-bar mirror still fills the drawer's #meta-list rows on
//   mobile (the hidden #tab-status-bar keeps updating underneath);
// * switching to the desktop layout (matchMedia flip) closes the
//   drawer state so the docked panel never sits over a live backdrop;
// * a desktop-sized remote page polls getTaskUpdate with no drawer.
//
// The drawer open/close flows, the drawer-gated 5s poll and the refresh
// button are covered by sidebarChatModeDrawerAndComposer.test.js.

'use strict';

const assert = require('assert');
const {
  makePage: makeHarnessPage,
  send,
  stubRaf,
  fakeIntervals,
  waitFor,
  fakeMatchMedia,
} = require('./simplify2_harness');

const POLL_MS = 5000;

const {makeRunner} = require('./ui_antipattern_harness');
const {
  click,
  drawerOpen,
  overlayOpen,
  pollCount,
  lastPoll,
} = require('./remote_drawer_helpers');

const {test, report} = makeRunner();

function makePage(opts) {
  const {desktopMatches = false} = opts || {};
  let setDesktop;
  let timers;
  const {win, posted} = makeHarnessPage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js'],
    beforeScripts(win) {
      stubRaf(win);
      setDesktop = fakeMatchMedia(win, desktopMatches);
      timers = fakeIntervals(win);
    },
  });
  return {win, posted, setDesktop, tick: timers.tick};
}

async function main() {
  await test('the drawer mirrors the (hidden) status-bar values on mobile', async () => {
    const wv = makePage();
    const win = wv.win;
    send(win, {
      type: 'configData',
      config: {work_dir: '/cfg/dir', max_budget: 42},
      apiKeys: {},
      machine: 'phone-host',
    });
    send(win, {type: 'usage_info', total_tokens: 2000, cost: 0.25});
    const text = id => win.document.getElementById(id).textContent;
    // main.js mirrors the status-bar values into the panel through a
    // MutationObserver, which is delivered as a microtask.
    await waitFor(
      () => text('meta-tokens') === '2.00K' && text('meta-cost') === '$0.25',
      'the drawer never mirrored the status-bar tokens and cost',
    );
    assert.strictEqual(text('meta-tokens'), '2.00K');
    assert.strictEqual(text('meta-cost'), '$0.25');
    // The machine name sits in the status bar AND in the Machine row.
    assert.strictEqual(text('status-machine'), 'phone-host');
    assert.strictEqual(text('meta-machine'), 'phone-host');
    assert.strictEqual(text('meta-workdir'), '/cfg/dir');
    assert.strictEqual(text('meta-max-budget'), '$42.00');
  });

  await test('switching to the desktop layout closes the drawer', () => {
    const wv = makePage();
    const win = wv.win;
    click(win, 'meta-drawer-btn');
    assert.ok(drawerOpen(win) && overlayOpen(win));
    wv.setDesktop(true);
    assert.ok(
      win.document.body.classList.contains('remote-desktop'),
      'the desktop class follows the media query',
    );
    assert.ok(!drawerOpen(win), 'desktop docking drops the drawer state');
    assert.ok(!overlayOpen(win), 'the backdrop goes with it');
  });

  await test('desktop remote keeps polling without the drawer (docked panel)', () => {
    const wv = makePage({desktopMatches: true});
    const win = wv.win;
    send(win, {
      type: 'configData',
      config: {work_dir: '/cfg/dir', max_budget: 42},
      apiKeys: {},
    });
    send(win, {type: 'status', running: true});
    assert.ok(
      pollCount(wv) >= 1,
      'the docked desktop panel polls with no drawer involved',
    );
    const count = pollCount(wv);
    wv.tick(POLL_MS);
    assert.strictEqual(pollCount(wv), count + 1, 'and keeps polling');
    assert.strictEqual(lastPoll(wv).type, 'getTaskUpdate');
    assert.ok(!('workDir' in lastPoll(wv)));
  });

  report('remoteMetaMobileDrawer');
  process.exit(0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
