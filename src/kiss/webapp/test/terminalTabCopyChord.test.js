// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// terminalTab.js: Ctrl/Cmd+Shift+C copies the xterm selection through
// PanelCopy.copyText, like every other copy control of the page, so a
// clipboard API that rejects (document not focused, permission denied)
// falls back to the execCommand('copy') textarea instead of leaving an
// unhandled rejection and no copy.  jsdom has no xterm.js, no
// ResizeObserver and no animation frames, so the page is given the
// same minimal stand-ins as terminalTabView.test.js; the real
// terminalTab.js and panelCopy.js drive them.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {queueRaf} = require('./simplify2_harness');

const MEDIA = path.join(__dirname, '..', 'media');

const unhandled = [];
process.on('unhandledRejection', err => unhandled.push(err));

/** A page whose terminal view is shown with *selection* selected. */
async function openView(selection) {
  const dom = new JSDOM('<!DOCTYPE html><body></body>', {
    runScripts: 'outside-only',
    url: 'https://localhost/',
  });
  const win = dom.window;
  const log = {rejectedWrites: 0, execCopies: []};
  const terminals = [];
  class Terminal {
    constructor() {
      this.cols = 80;
      this.rows = 24;
      this.options = {};
      terminals.push(this);
    }
    open() {}
    loadAddon() {}
    attachCustomKeyEventHandler(handler) {
      this.keyHandler = handler;
    }
    write() {}
    onData() {}
    onResize() {}
    focus() {}
    getSelection() {
      return selection;
    }
    dispose() {}
  }
  win.Terminal = Terminal;
  win.FitAddon = {
    FitAddon: class {
      fit() {}
    },
  };
  win.ResizeObserver = class {
    observe() {}
    disconnect() {}
  };
  queueRaf(win); // no frame ever runs
  // The async clipboard refuses every write, as it does when the
  // document is not focused; execCommand('copy') records the value of
  // the textarea the fallback appended and selected (jsdom's select()
  // does not move focus, so it is found in the body, not activeElement).
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: {
      writeText: () => {
        log.rejectedWrites++;
        const err = new Error('Document is not focused.');
        err.name = 'NotAllowedError';
        return Promise.reject(err);
      },
    },
  });
  win.document.execCommand = () => {
    const ta = win.document.body.querySelector('textarea');
    log.execCopies.push(ta ? ta.value : null);
    return true;
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'terminalTab.js'), 'utf8'));
  const view = win.TerminalTabView.create('tab-1', {send: () => {}});
  win.document.body.appendChild(view.el);
  view.setVisible(true);
  await Promise.resolve();
  await Promise.resolve();
  assert.strictEqual(terminals.length, 1, 'xterm created on first show');
  return {keyHandler: terminals[0].keyHandler, log};
}

/**
 * Let the clipboard rejection and the execCommand fallback settle:
 * `PanelCopy.copyText` is promise-only, so one macrotask turn drains it.
 */
function settle() {
  return new Promise(resolve => setImmediate(resolve));
}

async function main() {
  {
    const {keyHandler, log} = await openView('ls -la\n');
    const chord = {type: 'keydown', key: 'C', ctrlKey: true, shiftKey: true};
    assert.strictEqual(keyHandler(chord), false, 'the chord is swallowed');
    await settle();
    assert.strictEqual(log.rejectedWrites, 1, 'the clipboard API was tried');
    assert.deepStrictEqual(
      log.execCopies,
      ['ls -la\n'],
      'the refused write falls back to execCommand with the selection',
    );
  }
  {
    // Nothing selected: nothing is copied and nothing rejects.
    const {keyHandler, log} = await openView('');
    const chord = {type: 'keydown', key: 'c', metaKey: true, shiftKey: true};
    assert.strictEqual(keyHandler(chord), false);
    await settle();
    assert.strictEqual(log.rejectedWrites, 0);
    assert.deepStrictEqual(log.execCopies, []);
  }
  assert.deepStrictEqual(unhandled, [], 'no unhandled rejection');
  console.log('terminalTabCopyChord.test.js passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
