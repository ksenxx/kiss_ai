// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// Review 2026-10-18: on the real transport (media/transport.js) a
// password-protected daemon announces `daemonStatus connected` twice
// per connection, on `auth_required` (to reveal the page) and on
// `auth_ok`.  main.js requested every subpanel on each announcement, so
// a password-required startup sent two of each.  The subpanels are now
// asked once per connection; a reconnect still asks again.

const assert = require('assert');
const {
  newDom,
  loadPageScripts,
  readShimJs,
  installFakeWebSocket,
  waitForSocket,
} = require('./transport_page_helpers');

const PANEL_COMMANDS = [
  'getCronJobs',
  'getMachines',
  'getAppsStatus',
  'getSpendReport',
  'getRecommendations',
];

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

/** How many frames of each subpanel command *socket* carried. */
function panelCounts(socket) {
  const frames = socket.sent.map(s => JSON.parse(s));
  const counts = {};
  for (const type of PANEL_COMMANDS)
    counts[type] = frames.filter(m => m.type === type).length;
  return counts;
}

async function main() {
  const win = newDom();
  const sockets = [];
  installFakeWebSocket(win, sockets);
  win.prompt = () => 'test-password';
  win.eval(readShimJs());
  loadPageScripts(win);
  await waitForSocket(sockets, 1);
  const first = sockets[0];
  first.onopen();
  first.onmessage({data: JSON.stringify({type: 'auth_required'})});
  await sleep(30);
  first.onmessage({data: JSON.stringify({type: 'auth_ok'})});
  await sleep(30);
  const once = {};
  for (const type of PANEL_COMMANDS) once[type] = 1;
  assert.deepStrictEqual(
    panelCounts(first),
    once,
    'REGRESSION: a password-required startup asked for the subpanels twice',
  );

  // The server confirms the batch flushed on `auth_ok` (otherwise the
  // transport re-sends it on the next socket, at-least-once).  Then the
  // socket drops; the reconnect re-authenticates and asks once more.
  first.onmessage({data: JSON.stringify({type: 'pong'})});
  first.close();
  await waitForSocket(sockets, 2);
  const second = sockets[1];
  second.onopen();
  second.onmessage({data: JSON.stringify({type: 'auth_ok'})});
  await sleep(30);
  assert.deepStrictEqual(panelCounts(second), once, 'the reconnect asks once');
  win.close();
  console.log('  ok - a password-required startup asks for each subpanel once');
}

// The shim's reconnect and stale-check timers outlive the window: exit
// explicitly, like the other real-transport suites.
main().then(
  () => process.exit(0),
  error => {
    console.error(error);
    process.exit(1);
  },
);
