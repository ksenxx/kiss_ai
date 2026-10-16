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
// the real terminalTab.js drives them.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makePage() {
  const dom = new JSDOM('<!DOCTYPE html><body></body>', {
    runScripts: 'outside-only',
  });
  const win = dom.window;
  const fits = [];
  const terminals = [];
  const observers = [];
  const frames = [];

  class Terminal {
    constructor() {
      this.cols = 80;
      this.rows = 24;
      this.options = {};
      this.written = '';
      this.focused = 0;
      this.disposed = false;
      terminals.push(this);
    }
    open() {}
    loadAddon() {}
    attachCustomKeyEventHandler(handler) {
      this.keyHandler = handler;
    }
    write(text) {
      this.written += text;
    }
    onData(cb) {
      this.dataCb = cb;
    }
    onResize(cb) {
      this.resizeCb = cb;
    }
    focus() {
      this.focused++;
    }
    getSelection() {
      return '';
    }
    dispose() {
      this.disposed = true;
    }
  }
  class FitAddon {
    fit() {
      fits.push(1);
    }
  }
  win.Terminal = Terminal;
  win.FitAddon = {FitAddon};
  win.ResizeObserver = class {
    constructor(cb) {
      this.cb = cb;
      this.disconnected = false;
      observers.push(this);
    }
    observe() {}
    disconnect() {
      this.disconnected = true;
    }
  };
  win.requestAnimationFrame = cb => {
    frames.push(cb);
    return frames.length;
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'terminalTab.js'), 'utf8'));
  return {win, fits, terminals, observers, frames};
}

/** Create a view, show it and let xterm load (one microtask turn). */
async function openView(page) {
  const sent = [];
  const view = page.win.TerminalTabView.create('tab-1', {
    send: msg => sent.push(msg),
    workDir: '/work',
  });
  page.win.document.body.appendChild(view.el);
  view.setVisible(true);
  await Promise.resolve();
  await Promise.resolve();
  return {view, sent};
}

/** Objects from the page realm have another Object prototype. */
function plain(value) {
  return JSON.parse(JSON.stringify(value));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
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
  await sleep(150);
  assert.strictEqual(page.fits.length, 2, 'refit after a resize');
  view.setVisible(false);
  page.observers[0].cb();
  await sleep(150);
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

  // Keystrokes go to the shell while it runs.
  term.dataCb('ls\r');
  assert.deepStrictEqual(plain(sent[2]), {
    type: 'terminalInput',
    tab_id: 'tab-1',
    data: 'ls\r',
  });
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

async function testDispose() {
  const page = makePage();
  const {view} = await openView(page);
  const term = page.terminals[0];
  page.observers[0].cb(); // a resize timer in flight
  view.dispose();
  view.dispose();
  assert.ok(term.disposed);
  assert.ok(page.observers[0].disconnected);
  await sleep(150);
  assert.strictEqual(page.fits.length, 1, 'no refit after dispose');
  page.frames.forEach(cb => cb());
  view.setVisible(true);
  await Promise.resolve();
  assert.strictEqual(page.terminals.length, 1, 'no terminal after dispose');
}

async function testOutputBeforeXterm() {
  const page = makePage();
  const view = page.win.TerminalTabView.create('tab-2', {send: () => {}});
  view.data('early');
  view.exit();
  view.setVisible(true);
  await Promise.resolve();
  await Promise.resolve();
  assert.ok(page.terminals[0].written.startsWith('early'));
  assert.ok(page.terminals[0].written.includes('The shell exited.'));
}

(async () => {
  await testOpenFitAndResize();
  await testExitRestartAndReconnectNote();
  await testDispose();
  await testOutputBeforeXterm();
  console.log('terminalTabView.test.js passed');
})().catch(err => {
  console.error(err);
  process.exit(1);
});
