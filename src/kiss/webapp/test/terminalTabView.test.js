// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E tests for the remote webapp's terminal tab view (terminalTab.js):
// opening the shell on first show, refitting on resize and re-show,
// restarting an ended shell with Enter, the lost-session note after a
// reconnect, and dispose.  jsdom has no xterm.js, no ResizeObserver and
// no animation frames, so the page is given a minimal Terminal /
// FitAddon pair and synchronous stand-ins for the two browser APIs;
// the real terminalTab.js drives them.  setTimeout is recorded too, so
// the 100 ms resize debounce is fired by hand instead of slept past.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {
  settle,
  installXtermStub,
  newDom,
  loadPageScripts,
  makePage: makeAppPage,
  plain,
} = require('./simplify2_harness');
const {
  readShimJs,
  installFakeWebSocket,
  frames,
} = require('./transport_page_helpers');

const MEDIA = path.join(__dirname, '..', 'media');

function makePage() {
  const dom = new JSDOM('<!DOCTYPE html><body></body>', {
    runScripts: 'outside-only',
  });
  const win = dom.window;
  const {terminals, fits, observers} = installXtermStub(win);
  const frames = [];
  const timers = new Map();
  let nextTimer = 0;
  win.requestAnimationFrame = cb => {
    frames.push(cb);
    return frames.length;
  };
  win.setTimeout = (fn, ms) => {
    nextTimer++;
    timers.set(nextTimer, {fn, ms});
    return nextTimer;
  };
  win.clearTimeout = id => timers.delete(id);
  win.eval(fs.readFileSync(path.join(MEDIA, 'terminalTab.js'), 'utf8'));
  return {win, fits, terminals, observers, frames, timers};
}

/** Fire every timer still armed, as the clock would once they elapse. */
function runTimers(page) {
  const due = Array.from(page.timers.values());
  page.timers.clear();
  due.forEach(t => t.fn());
}

/**
 * The page's first terminal exists one event-loop turn after the show:
 * with the stub's Terminal in place ensureXterm() is already resolved.
 */
async function xtermLoaded(page) {
  await settle();
  assert.ok(page.terminals.length >= 1, 'xterm did not load');
}

/** Create a view, show it and let xterm load. */
async function openView(page) {
  const sent = [];
  const view = page.win.TerminalTabView.create('tab-1', {
    send: msg => sent.push(msg),
    workDir: '/work',
  });
  page.win.document.body.appendChild(view.el);
  view.setVisible(true);
  await xtermLoaded(page);
  return {view, sent};
}

async function testOpenFitAndResize() {
  const page = makePage();
  const {view, sent} = await openView(page);
  const term = page.terminals[0];
  assert.ok(term, 'xterm created on first show');
  assert.deepStrictEqual(plain(sent), [
    {
      type: 'terminalOpen',
      tab_id: 'tab-1',
      cols: 80,
      rows: 24,
      workDir: '/work',
    },
  ]);
  assert.strictEqual(page.fits.length, 1, 'fitted once on open');
  assert.strictEqual(term.focused, 1, 'focused when shown');

  // A surface resize refits after the debounce, only while visible.
  page.observers[0].cb();
  assert.deepStrictEqual(
    Array.from(page.timers.values(), t => t.ms),
    [100],
    'a resize arms the 100 ms debounce',
  );
  assert.strictEqual(page.fits.length, 1, 'no refit before the debounce');
  runTimers(page);
  assert.strictEqual(page.fits.length, 2, 'refit after a resize');
  view.setVisible(false);
  page.observers[0].cb();
  runTimers(page);
  assert.strictEqual(page.fits.length, 2, 'no refit while hidden');

  // Re-showing fits on the next frame.
  view.setVisible(true);
  assert.strictEqual(term.focused, 2);
  page.frames.forEach(cb => cb());
  assert.strictEqual(page.fits.length, 3, 'refit on re-show');

  // A changed grid is forwarded to the pty; an unchanged one is not.
  term.resizeCb({cols: 80, rows: 24});
  term.resizeCb({cols: 100, rows: 30});
  assert.deepStrictEqual(plain(sent[sent.length - 1]), {
    type: 'terminalResize',
    tab_id: 'tab-1',
    cols: 100,
    rows: 30,
  });
  assert.strictEqual(sent.length, 2);

  // Keystrokes go to the shell once the daemon confirmed it runs; typed
  // before that they wait (the connection does not own the shell yet).
  term.dataCb('l');
  assert.strictEqual(sent.length, 2, 'input waits for the terminalOpened');
  view.opened({attached: false});
  term.dataCb('s\r');
  assert.deepStrictEqual(plain(sent.slice(2)), [
    {type: 'terminalInput', tab_id: 'tab-1', data: 'l'},
    {type: 'terminalInput', tab_id: 'tab-1', data: 's\r'},
  ]);
  // Output arriving before xterm existed is replayed, later output
  // written directly.
  view.data('$ ');
  assert.ok(term.written.endsWith('$ '));
}

async function testExitRestartAndReconnectNote() {
  const page = makePage();
  const {view, sent} = await openView(page);
  const term = page.terminals[0];
  view.opened({attached: false});
  view.exit(2);
  assert.ok(view.exited);
  assert.ok(term.written.includes('The shell exited with code 2'));
  term.dataCb('x');
  assert.strictEqual(sent.length, 1, 'no input to an ended shell');

  // Enter on the ended shell asks for a new one in the same tab.
  const swallowed = term.keyHandler({type: 'keydown', key: 'Enter'});
  assert.strictEqual(swallowed, false);
  assert.strictEqual(sent.length, 2);
  assert.strictEqual(sent[1].type, 'terminalOpen');
  assert.strictEqual(view.exited, false);
  const before = term.written;
  view.opened({attached: false});
  assert.strictEqual(term.written, before, 'a restart is not a lost shell');

  // A reconnect re-sends the open; the shell that answers with
  // attached:false replaced one that ended meanwhile.
  view.reconnect();
  assert.strictEqual(sent.length, 3);
  view.opened({attached: true});
  assert.ok(!term.written.includes('while this page was disconnected'));
  view.opened({attached: false});
  assert.ok(term.written.includes('while this page was disconnected'));

  // Copy / paste chords stay with the browser; other keys reach xterm.
  assert.strictEqual(term.keyHandler({type: 'keyup', key: 'a'}), true);
  assert.strictEqual(term.keyHandler({type: 'keydown', key: 'a'}), true);
  assert.strictEqual(
    term.keyHandler({type: 'keydown', key: 'C', ctrlKey: true, shiftKey: true}),
    false,
  );
  assert.strictEqual(
    term.keyHandler({type: 'keydown', key: 'V', ctrlKey: true, shiftKey: true}),
    false,
  );
  assert.strictEqual(
    term.keyHandler({type: 'keydown', key: 'X', ctrlKey: true, shiftKey: true}),
    true,
  );

  // terminalError ends the shell too, and a reconnect then stays quiet.
  view.error('The shell could not be started: boom');
  assert.ok(view.exited);
  assert.ok(term.written.includes('could not be started'));
  view.reconnect();
  assert.strictEqual(sent.length, 3);

  view.retheme();
  assert.ok(term.options.theme);
}

// The daemon could not start the FIRST shell of the tab (a pty fork
// failure, a remote machine that refused): the tab is not dead.  Enter
// asks for a new shell exactly as it does after an exit, and the note
// says so.
async function testErrorOnFirstOpenCanBeRetried() {
  const page = makePage();
  const {view, sent} = await openView(page);
  const term = page.terminals[0];
  view.error('The shell could not be started: boom');
  assert.ok(view.exited);
  assert.ok(term.written.includes('could not be started: boom'));
  assert.ok(
    term.written.includes('Press Enter to try again'),
    'the note tells the user how to retry: ' + term.written,
  );
  term.dataCb('x');
  assert.strictEqual(sent.length, 1, 'no input to a shell that never ran');
  assert.strictEqual(term.keyHandler({type: 'keydown', key: 'Enter'}), false);
  assert.strictEqual(sent.length, 2, 'Enter asks for a shell again');
  assert.strictEqual(sent[1].type, 'terminalOpen');
  assert.strictEqual(view.exited, false);
  const before = term.written;
  view.opened({attached: false});
  assert.strictEqual(term.written, before, 'the retry is not a lost shell');
  term.dataCb('ls\r');
  assert.deepStrictEqual(plain(sent[2]), {
    type: 'terminalInput',
    tab_id: 'tab-1',
    data: 'ls\r',
  });
}

// An ended shell went through an outage: `disconnected` left the view
// detached and `reconnect` leaves an exited shell alone.  Enter then
// asks for a new shell, and a command typed before the daemon's
// terminalOpened belongs to that new shell: it used to be dropped as
// input for the old one, because the fresh shell "replaced" the ended
// one.  What was typed for the old shell while the socket was down is
// still not run in the shell the user explicitly asked for.
async function testExplicitRestartAfterAnOutageKeepsTheTypedCommand() {
  const page = makePage();
  const {view, sent} = await openView(page);
  const term = page.terminals[0];
  view.opened({attached: false});
  view.exit(0);
  view.disconnected();
  view.reconnect();
  assert.strictEqual(sent.length, 1, 'a reconnect leaves an ended shell alone');
  assert.strictEqual(term.keyHandler({type: 'keydown', key: 'Enter'}), false);
  assert.strictEqual(sent[1].type, 'terminalOpen');
  term.dataCb('echo typed-for-new-shell\r'); // before the open reply
  assert.strictEqual(sent.length, 2, 'input waits for the terminalOpened');
  view.opened({attached: false});
  assert.deepStrictEqual(
    plain(sent.slice(2)),
    [
      {
        type: 'terminalInput',
        tab_id: 'tab-1',
        data: 'echo typed-for-new-shell\r',
      },
    ],
    'the command typed for the new shell runs in it, once',
  );
  assert.ok(!term.written.includes('while this page was disconnected'));

  // Typed for the running shell while the socket was down, then the
  // daemon could not re-attach it: Enter asks for a new shell, and the
  // old shell's input does not run in it.
  view.disconnected();
  term.dataCb('echo stale\r');
  view.reconnect();
  assert.strictEqual(sent.length, 4);
  assert.strictEqual(sent[3].type, 'terminalOpen');
  view.error('The shell could not be started: the machine is unreachable.');
  assert.ok(view.exited);
  assert.strictEqual(term.keyHandler({type: 'keydown', key: 'Enter'}), false);
  term.dataCb('echo fresh\r');
  view.opened({attached: false});
  assert.deepStrictEqual(
    plain(sent.slice(4)),
    [
      {
        type: 'terminalOpen',
        tab_id: 'tab-1',
        cols: 80,
        rows: 24,
        workDir: '/work',
      },
      {type: 'terminalInput', tab_id: 'tab-1', data: 'echo fresh\r'},
    ],
    'only what was typed after the explicit restart reaches the new shell',
  );
}

async function testDispose() {
  const page = makePage();
  const {view} = await openView(page);
  const term = page.terminals[0];
  page.observers[0].cb(); // a resize timer in flight
  view.dispose();
  view.dispose();
  assert.ok(term.disposed);
  assert.ok(page.observers[0].disconnected);
  assert.strictEqual(page.timers.size, 0, 'dispose cancels the resize timer');
  runTimers(page);
  assert.strictEqual(page.fits.length, 1, 'no refit after dispose');
  page.frames.forEach(cb => cb());
  view.setVisible(true);
  await settle();
  assert.strictEqual(page.terminals.length, 1, 'no terminal after dispose');
}

async function testOutputBeforeXterm() {
  const page = makePage();
  const view = page.win.TerminalTabView.create('tab-2', {send: () => {}});
  view.data('early');
  view.exit();
  view.setVisible(true);
  await xtermLoaded(page);
  assert.ok(page.terminals[0].written.startsWith('early'));
  assert.ok(page.terminals[0].written.includes('The shell exited.'));
}

/** The daemon's side of a fake socket's handshake: auth_ok, then the pong. */
function handshake(sock) {
  sock.onopen();
  sock.fireMessage({type: 'auth_ok'});
  const last = frames(sock)[sock.sent.length - 1];
  if (last.type === 'ping') sock.fireMessage({type: 'pong'});
}

/** The terminalInput texts a fake socket carried. */
function inputs(sock) {
  return frames(sock)
    .filter(f => f.type === 'terminalInput')
    .map(f => f.data);
}

/**
 * The real chat page (transport.js, api.js, main.js, terminalTab.js)
 * over a fake WebSocket, its first socket past the handshake.
 */
function makeTransportPage() {
  const win = newDom();
  const sockets = [];
  installFakeWebSocket(win, sockets);
  const {terminals} = installXtermStub(win);
  win.eval(readShimJs());
  loadPageScripts(win);
  win.eval(fs.readFileSync(path.join(MEDIA, 'terminalTab.js'), 'utf8'));
  handshake(sockets[0]);
  return {win, sockets, terminals};
}

// Keystrokes typed while the socket is down used to join the
// transport's reconnect batch, which goes out before main.js re-sends
// terminalOpen: the daemon got them from a connection that did not own
// the shell yet and dropped them.  The real chat page (transport.js,
// api.js, main.js, terminalTab.js) over the fake WebSocket: the input
// waits until the daemon's terminalOpened confirms the re-attach, then
// goes out once, in order, after the terminalOpen.
async function testInputTypedWhileDisconnectedWaitsForTheReattach() {
  const {win, sockets, terminals} = makeTransportPage();
  win.document.getElementById('terminal-btn').click();
  await xtermLoaded({terminals});
  const term = terminals[0];
  const open = frames(sockets[0]).find(f => f.type === 'terminalOpen');
  assert.ok(open, 'the terminal tab asked for its shell');
  const tabId = open.tab_id;
  const opened = attached => ({
    type: 'terminalOpened',
    tab_id: tabId,
    attached,
  });
  sockets[0].fireMessage(opened(false));
  term.dataCb('echo live\r');
  assert.deepStrictEqual(inputs(sockets[0]), ['echo live\r']);

  // The socket dies; the user keeps typing into the still-open terminal.
  // The `online` event makes the shim reconnect at once, backoff or not.
  sockets[0].fireClose();
  term.dataCb('printf OFFLINE\r');
  win.dispatchEvent(new win.Event('online'));
  assert.strictEqual(sockets.length, 2);
  const sock = sockets[1];
  handshake(sock);
  await settle();
  const types = frames(sock).map(f => f.type);
  assert.ok(types.includes('terminalOpen'), 'the reconnect re-opens the shell');
  assert.deepStrictEqual(
    inputs(sock),
    [],
    'nothing typed offline goes out before the daemon re-attached the shell',
  );
  sock.fireMessage(opened(true));
  term.dataCb('echo after\r');
  const after = frames(sock).map(f => f.type);
  assert.ok(
    after.indexOf('terminalOpen') < after.indexOf('terminalInput'),
    'the held input follows the terminalOpen: ' + JSON.stringify(after),
  );
  assert.deepStrictEqual(inputs(sock), ['printf OFFLINE\r', 'echo after\r']);
  assert.strictEqual(
    sockets.flatMap(inputs).filter(d => d === 'printf OFFLINE\r').length,
    1,
    'the offline input is sent exactly once',
  );

  // A second outage, after which the daemon reports the shell gone: what
  // was typed for it is not run in the fresh shell.
  sock.fireClose();
  term.dataCb('rm -rf stale\r');
  win.dispatchEvent(new win.Event('online'));
  assert.strictEqual(sockets.length, 3);
  handshake(sockets[2]);
  await settle();
  sockets[2].fireMessage(opened(false));
  term.dataCb('echo fresh\r');
  assert.deepStrictEqual(inputs(sockets[2]), ['echo fresh\r']);
  assert.ok(term.written.includes('while this page was disconnected'));
  win.close();
}

// A terminal opened while the socket is down: its terminalOpen waits in
// the transport's offline batch, which the reconnect sends and, when
// that connection dies before the daemon's pong confirms it, sends
// AGAIN on the next one (at-least-once).  Keystrokes typed before the
// daemon confirmed the shell must not join that batch: a shell command
// run twice is not a settings save applied twice.  They wait in the
// view until terminalOpened, then go out once.
async function testTerminalOpenedDuringAnOutageHoldsInputUntilTheReattach() {
  const {win, sockets, terminals} = makeTransportPage();
  sockets[0].fireClose();
  win.document.getElementById('terminal-btn').click();
  await xtermLoaded({terminals});
  const term = terminals[0];
  term.dataCb('echo example\r');
  win.dispatchEvent(new win.Event('online'));
  assert.strictEqual(sockets.length, 2);
  const sock = sockets[1];
  sock.onopen();
  sock.fireMessage({type: 'auth_ok'});
  await settle();
  const open = frames(sock).find(f => f.type === 'terminalOpen');
  assert.ok(open, 'the reconnect batch carries the terminalOpen');
  assert.deepStrictEqual(
    inputs(sock),
    [],
    'nothing typed before the daemon confirmed the shell joins the batch',
  );
  sock.fireMessage({
    type: 'terminalOpened',
    tab_id: open.tab_id,
    attached: false,
  });
  // The connection dies before the pong: the next one re-sends the
  // batch; the input, posted after terminalOpened, goes out once.
  sock.fireClose();
  win.dispatchEvent(new win.Event('online'));
  assert.strictEqual(sockets.length, 3);
  handshake(sockets[2]);
  await settle();
  assert.deepStrictEqual(
    sockets.flatMap(inputs),
    ['echo example\r'],
    'the command typed during the outage runs exactly once',
  );
  win.close();
}

async function testOpenTerminalsCarryDistinctTitles() {
  // The "Terminal" menu item numbers a new tab one above the highest
  // number among the terminals open: counting them would hand the
  // number of a still-open terminal out again once an earlier one is
  // closed.
  const {win} = makeAppPage({
    extraScripts: ['terminalTab.js'],
    beforeScripts: installXtermStub,
  });
  const terminals = () =>
    plain(win._testApi.openTabs().filter(t => /^Terminal/.test(t.title)));
  const titles = () => terminals().map(t => t.title);
  const open = () => win.document.getElementById('terminal-btn').click();
  const close = id =>
    win.document
      .querySelector('.chat-tab[data-tab-id="' + id + '"] .chat-tab-close')
      .click();
  open();
  open();
  assert.deepStrictEqual(titles(), ['Terminal', 'Terminal 2']);
  close(terminals()[0].id);
  open();
  assert.deepStrictEqual(titles(), ['Terminal 2', 'Terminal 3']);
  // With none left the numbering starts over.
  terminals().forEach(t => close(t.id));
  open();
  assert.deepStrictEqual(titles(), ['Terminal']);
  // The views start their shells on a later tick.
  await settle();
  win.close();
}

(async () => {
  await testOpenTerminalsCarryDistinctTitles();
  await testOpenFitAndResize();
  await testExitRestartAndReconnectNote();
  await testErrorOnFirstOpenCanBeRetried();
  await testExplicitRestartAfterAnOutageKeepsTheTypedCommand();
  await testDispose();
  await testOutputBeforeXterm();
  await testInputTypedWhileDisconnectedWaitsForTheReattach();
  await testTerminalOpenedDuringAnOutageHoldsInputUntilTheReattach();
  console.log('terminalTabView.test.js passed');
})().catch(err => {
  console.error(err);
  process.exit(1);
});
