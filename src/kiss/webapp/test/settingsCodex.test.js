/**
 * JSDOM tests for the settings panel's Codex subpanel.
 *
 * The subpanel starts collapsed, holds the "Bill my ChatGPT
 * subscription" switch (on by default), the "Sign in with ChatGPT"
 * device-code row and the masked CODEX_ACCESS_TOKEN box; opening the
 * panel asks the daemon for `getCodexStatus`; the `codexStatus` reply
 * paints the status line; `configData` loads the switch and token and
 * closing the panel saves them (all shared with the Claude Code
 * subpanel: settings_agent_subpanel.js).  Codex-only: a `codexLogin`
 * reply paints the device-code link and code and polls `getCodexStatus`
 * every 5 s until the login shows up (or the panel closes, or 15 minutes
 * pass).  The subpanel order is pinned in
 * appearanceMenuAndMachineSettings.test.js.
 */

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  fakeIntervals,
  lastMsg,
  click,
  byId,
} = require('./simplify2_harness');
const {openSettings, closeSettings} = require('./settings_helpers');
const {runAgentSubpanelTests} = require('./settings_agent_subpanel');

const BASE = {installed: true, loggedIn: false, authMethod: 'none', useSubscription: true,
  hasToken: false};

const CODEX = {
  subpanel: 'codex',
  cli: 'codex',
  statusCmd: 'getCodexStatus',
  statusReply: 'codexStatus',
  tokenKey: 'CODEX_ACCESS_TOKEN',
  configKey: 'codex_use_subscription',
  loggedInAuth: 'chatgpt',
  statusCases: [
    [
      {...BASE, installed: false, error: "Codex CLI ('codex') not found on PATH."},
      /not installed on this machine/,
      'warn',
    ],
    [
      {...BASE, error: '`codex login status` (exit 1) answered: boom'},
      /Could not read the codex login: .*boom/,
      'warn',
    ],
    [BASE, /Not logged in: click "Sign in with ChatGPT"/, 'warn'],
    [
      {...BASE, useSubscription: false},
      /Not logged in; codex\/ models will use OPENAI_API_KEY/,
      'warn',
    ],
    [
      {...BASE, loggedIn: true, authMethod: 'chatgpt'},
      /^Logged in with ChatGPT: codex\/ models bill your ChatGPT subscription\.$/,
      'ok',
    ],
    [
      {...BASE, loggedIn: true, authMethod: 'access_token', hasToken: true},
      /^Logged in with a Codex access token: codex\/ models bill your ChatGPT workspace\.$/,
      'ok',
    ],
    [
      {...BASE, loggedIn: true, authMethod: 'api_key'},
      /logged in with an API key .*billed per token.*codex logout/,
      'warn',
    ],
    [
      {...BASE, loggedIn: true, authMethod: 'workload_identity', useSubscription: false},
      /authenticates with workload_identity: codex\/ models are billed per token/,
      'warn',
    ],
    [
      {...BASE, loggedIn: true, authMethod: ''},
      /authenticates with an unknown method/,
      'warn',
    ],
  ],
};

function statusLine(win) {
  return byId(win, 'cfg-codex-status');
}

function testDeviceCodeLogin() {
  const {win, posted} = makePage();
  openSettings(win);
  const btn = byId(win, 'cfg-codex-login-btn');
  const note = byId(win, 'cfg-codex-login-note');
  assert.ok(byId(win, 'codex-body').contains(btn), 'the sign-in button is in the subpanel');
  assert.strictEqual(btn.disabled, false);
  assert.ok(/device-code login/.test(note.textContent));

  click(win, btn);
  assert.ok(lastMsg(posted, 'startCodexLogin'), 'the button asks the daemon to start the login');
  assert.strictEqual(btn.disabled, true, 'the button is disabled while the CLI starts');
  assert.ok(/Starting/.test(note.textContent));

  send(win, {type: 'codexLogin', url: 'https://auth.openai.com/codex/device', code: '2YCJ-GJHWN'});
  assert.strictEqual(btn.disabled, false, 'the button is enabled again');
  const link = note.querySelector('a');
  assert.ok(link, 'the verification URL is a link');
  assert.strictEqual(link.href, 'https://auth.openai.com/codex/device');
  assert.strictEqual(link.target, '_blank');
  assert.strictEqual(note.querySelector('code').textContent, '2YCJ-GJHWN');
  assert.ok(/enter the code/.test(note.textContent));

  // The CLI finished the login: the status line reports it.
  send(win, {type: 'codexStatus', installed: true, loggedIn: true, authMethod: 'chatgpt',
    useSubscription: true, hasToken: false});
  assert.ok(/Logged in with ChatGPT/.test(statusLine(win).textContent));

  // A failure to start the login is reported in the row.
  click(win, btn);
  send(win, {type: 'codexLogin', error: '`codex login --device-auth` printed no login code: x'});
  assert.strictEqual(btn.disabled, false);
  assert.ok(/Sign-in failed: .*printed no login code/.test(note.textContent));
  assert.strictEqual(note.querySelector('a'), null, 'no stale link');

  // A reply for another machine's sheet is ignored, its link error
  // included: it is a notification that leaves this sheet's sign-in
  // waiting (audit 2026-10-18: it used to reset the row).
  click(win, btn);
  send(win, {type: 'codexLogin', forwarded: true, machine: 'other', url: 'https://x', code: 'AAAA-BBBB'});
  assert.strictEqual(note.querySelector('a'), null, "another machine's prompt is not painted");
  send(win, {type: 'codexLogin', forwarded: true, machine: 'other', error: 'machine unreachable'});
  assert.strictEqual(btn.disabled, true, "another machine's link error leaves the sign-in waiting");
  assert.ok(/Starting/.test(note.textContent));
  assert.ok(
    win.document.querySelector('.kiss-notification[data-notification-id="settings-machine-error"]'),
    'the link error is a notification',
  );
  send(win, {type: 'codexLogin', url: 'https://auth.openai.com/codex/device', code: 'ZZZZ-1111'});
  assert.strictEqual(note.querySelector('code').textContent, 'ZZZZ-1111', 'the prompt lands');
  win.close();
  console.log('  ok - device-code sign-in row');
}

/**
 * The sheet of a remote machine: that machine's own link error ends
 * its sign-in (the button works again); an outage ends it too and
 * stops the status poll; the poll posts nothing while the page is
 * hidden or the daemon is away; opening or closing the sheet stops it.
 */
function testRemoteSheetLinkErrorOutageAndPollGates() {
  const {win, posted} = makePage();
  const clock = installClock(win);
  const statusRequests = () => posted.filter(m => m.type === 'getCodexStatus').length;
  send(win, {type: 'configData', config: {}, machine: 'laptop'});
  send(win, {
    type: 'machines',
    machines: [
      {name: 'laptop', status: 'reachable', local: true},
      {name: 'gpu-box', status: 'reachable', address: 'ssh me@gpu'},
    ],
    checkedAt: 1,
  });
  win._testApi.createNewTab();
  const sel = byId(win, 'chat-machine');
  sel.value = 'gpu-box';
  sel.dispatchEvent(new win.Event('change', {bubbles: true}));
  openSettings(win);
  const btn = byId(win, 'cfg-codex-login-btn');
  const note = byId(win, 'cfg-codex-login-note');

  // gpu-box's own link error: the sign-in is over, the button works.
  click(win, btn);
  assert.strictEqual(lastMsg(posted, 'startCodexLogin').machine, 'gpu-box');
  assert.strictEqual(btn.disabled, true);
  send(win, {type: 'codexLogin', forwarded: true, machine: 'gpu-box', error: 'machine unreachable'});
  assert.strictEqual(btn.disabled, false, "the sheet's machine's link error re-enables the button");
  assert.ok(/device-code login/.test(note.textContent), 'the idle hint is back');

  // The prompt starts the poll; a hidden page or an outage posts nothing.
  click(win, btn);
  send(win, {...PROMPT, forwarded: true, machine: 'gpu-box'});
  assert.strictEqual(clock.pending(POLL_MS).length, 1);
  let before = statusRequests();
  clock.tick(POLL_MS);
  assert.strictEqual(statusRequests(), before + 1);
  assert.strictEqual(lastMsg(posted, 'getCodexStatus').machine, 'gpu-box');
  Object.defineProperty(win.document, 'hidden', {value: true, configurable: true});
  before = statusRequests();
  clock.tick(POLL_MS);
  assert.strictEqual(statusRequests(), before, 'REGRESSION: a hidden page polled');
  assert.strictEqual(clock.pending(POLL_MS).length, 1, 'the poll is still armed');
  Object.defineProperty(win.document, 'hidden', {value: false, configurable: true});

  // An outage: the sign-in awaiting its code is over, the poll stops.
  click(win, btn);
  assert.strictEqual(btn.disabled, true);
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  assert.strictEqual(btn.disabled, false, 'REGRESSION: the outage left the button disabled');
  assert.ok(/device-code login/.test(note.textContent));
  assert.strictEqual(clock.pending(POLL_MS).length, 0, 'REGRESSION: the poll outlived the daemon');
  send(win, {type: 'daemonStatus', connected: true});
  // (The reconnect arms the page's own 5 s pollers: count from here.)
  const others = clock.pending(POLL_MS).length;

  // Closing the sheet stops the poll at once; a prompt answered after
  // a reopen is the reopened sheet's, so a reopen stops it too.
  click(win, btn);
  send(win, {...PROMPT, forwarded: true, machine: 'gpu-box'});
  assert.strictEqual(clock.pending(POLL_MS).length, others + 1);
  closeSettings(win);
  assert.strictEqual(clock.pending(POLL_MS).length, others, 'closing stops the poll');
  // The next open is for the machine the header shows (gpu-box still).
  openSettings(win);
  click(win, btn);
  send(win, {...PROMPT, forwarded: true, machine: 'gpu-box'});
  assert.strictEqual(clock.pending(POLL_MS).length, others + 1);
  openSettings(win);
  assert.strictEqual(clock.pending(POLL_MS).length, others, 'reopening stops the poll');
  assert.strictEqual(btn.disabled, false);
  win.close();
  console.log('  ok - a remote sheet: link error, outage and the poll gates');
}

/**
 * Replace the booted page's interval timers with the harness's manual
 * clock (installed after boot so main.js's own timers are left alone);
 * `pending(ms)` lists the armed intervals of that period.
 */
function installClock(win) {
  const clock = fakeIntervals(win);
  clock.pending = ms => clock.intervals.filter(iv => iv.ms === ms);
  return clock;
}

const POLL_MS = 5000;
const PROMPT = {type: 'codexLogin', url: 'https://auth.openai.com/codex/device', code: '2YCJ-GJHWN'};
const NOT_LOGGED_IN = {type: 'codexStatus', installed: true, loggedIn: false, authMethod: 'none',
  useSubscription: true, hasToken: false};
const LOGGED_IN = {...NOT_LOGGED_IN, loggedIn: true, authMethod: 'chatgpt'};

function testLoginPollsTheStatusUntilLoggedIn() {
  const {win, posted} = makePage();
  const clock = installClock(win);
  const statusRequests = () => posted.filter(m => m.type === 'getCodexStatus').length;
  openSettings(win);
  const btn = byId(win, 'cfg-codex-login-btn');

  click(win, btn);
  assert.strictEqual(clock.pending(POLL_MS).length, 0, 'no poll before the daemon answers');
  send(win, PROMPT);
  assert.strictEqual(clock.pending(POLL_MS).length, 1, 'the device-code prompt starts the poll');
  let before = statusRequests();
  clock.tick(POLL_MS);
  assert.strictEqual(statusRequests(), before + 1, 'each tick asks the daemon for the login status');
  send(win, NOT_LOGGED_IN);
  assert.strictEqual(clock.pending(POLL_MS).length, 1, 'a not-logged-in status keeps polling');
  send(win, {...LOGGED_IN, authMethod: 'api_key'});
  assert.strictEqual(clock.pending(POLL_MS).length, 1, 'a saved API-key login is not the sign-in');
  send(win, LOGGED_IN);
  assert.strictEqual(clock.pending(POLL_MS).length, 0, 'a logged-in status stops the poll');
  assert.ok(/Logged in with ChatGPT/.test(statusLine(win).textContent));

  // A second prompt restarts a single poll; closing the panel ends it.
  click(win, btn);
  send(win, PROMPT);
  click(win, btn);
  send(win, PROMPT);
  assert.strictEqual(clock.pending(POLL_MS).length, 1, 'one poll at a time');
  closeSettings(win);
  before = statusRequests();
  clock.tick(POLL_MS);
  assert.strictEqual(clock.pending(POLL_MS).length, 0, 'a closed panel stops the poll');
  assert.strictEqual(statusRequests(), before, 'and asks nothing more');

  // The CLI gives up after 15 minutes; so does the poll.
  openSettings(win);
  click(win, btn);
  send(win, PROMPT);
  const realNow = win.Date.now;
  win.Date.now = () => realNow() + 16 * 60 * 1000;
  before = statusRequests();
  clock.tick(POLL_MS);
  assert.strictEqual(clock.pending(POLL_MS).length, 0, 'the poll stops after the deadline');
  assert.strictEqual(statusRequests(), before, 'without a final request');
  win.Date.now = realNow;
  win.close();
  console.log('  ok - device-code sign-in polls the status until the login shows up');
}

/**
 * Review 2026-10-18: a device code answered after the sheet closed (a
 * late local reply) started the 5 s status poll, which then ran with
 * no sheet to update until the login or the 15 min limit.  It starts
 * no poll now; a sign-in from the reopened sheet polls as before.
 */
function testLateDeviceCodeAfterCloseStartsNoPoll() {
  const {win, posted} = makePage();
  const clock = installClock(win);
  const statusRequests = () => posted.filter(m => m.type === 'getCodexStatus').length;
  openSettings(win);
  const btn = byId(win, 'cfg-codex-login-btn');
  click(win, btn);
  closeSettings(win);
  assert.strictEqual(byId(win, 'settings-panel').classList.contains('open'), false);
  send(win, PROMPT);
  assert.strictEqual(clock.pending(POLL_MS).length, 0, 'REGRESSION: a late device code armed the poll');
  const before = statusRequests();
  clock.tick(POLL_MS);
  assert.strictEqual(statusRequests(), before, 'nothing is asked while the sheet is closed');
  // The reopened sheet's own sign-in polls.
  openSettings(win);
  click(win, btn);
  send(win, PROMPT);
  assert.strictEqual(clock.pending(POLL_MS).length, 1, 'the open sheet polls');
  const opened = statusRequests();
  clock.tick(POLL_MS);
  assert.strictEqual(statusRequests(), opened + 1);
  win.close();
  console.log('  ok - a device code answered after the sheet closed starts no poll');
}

function main() {
  runAgentSubpanelTests(CODEX);
  testDeviceCodeLogin();
  testLoginPollsTheStatusUntilLoggedIn();
  testRemoteSheetLinkErrorOutageAndPollGates();
  testLateDeviceCodeAfterCloseStartsNoPoll();
  console.log('settingsCodex: all tests passed');
}

main();
