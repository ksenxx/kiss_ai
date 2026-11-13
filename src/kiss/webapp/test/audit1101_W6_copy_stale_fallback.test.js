// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E regression for audit 2026-11-01 W6 F3: a copy whose async
// clipboard write rejects falls back to the hidden-textarea
// execCommand('copy').  When a newer copy had already succeeded in the
// meantime, that late fallback replaced the newer clipboard contents
// with the older text, while the button (whose feedback is generation
// guarded) kept showing the newer copy as the one that landed.  Both
// helpers had the pattern: PanelCopy.copyText (every same-page copy
// control) and the context menu's own copyText (also embedded into the
// sandboxed preview iframe).
//
// The real modules run in jsdom; only the clipboard API and execCommand
// are stand-ins that record every write and let the test settle the
// async writes in the order it chooses.  The "stale" tests failed
// against the pre-fix modules; the guards keep the lone-rejection
// fallback working.
//
// Round 2 (F4): the parent document loads both helpers (main.js installs
// the context menu over the content tab area), so a menu copy and a
// panel copy of the same page must retire each other's older pending
// writes too: in the parent the menu's Copy goes through PanelCopy,
// which owns the page's copy generation; the menu's own guarded copy
// serves the sandboxed preview iframe, where PanelCopy is not loaded.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {makeRunner, rightClick} = require('./ui_antipattern_harness');
const {MEDIA, flushMicrotasks} = require('./simplify2_harness');

const {test, report} = makeRunner();

/**
 * A page whose clipboard writeText is settled by hand per text, with
 * execCommand('copy') recording what the hidden textarea held.
 * `script` names the media module(s) to load, in order.
 */
function makeWindow(html, script) {
  const win = new JSDOM('<body>' + html + '</body>', {
    runScripts: 'outside-only',
    url: 'https://localhost/',
  }).window;
  const writes = [];
  const pending = new Map();
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: {
      writeText: text =>
        new Promise((resolve, reject) => {
          pending.set(text, {resolve, reject});
        }),
    },
  });
  win.document.execCommand = cmd => {
    assert.strictEqual(cmd, 'copy');
    writes.push('fallback:' + win.document.querySelector('textarea').value);
    return true;
  };
  for (const name of [].concat(script)) {
    win.eval(fs.readFileSync(path.join(MEDIA, name), 'utf8'));
  }
  /** Settle the async write of `text`: `ok` true resolves it. */
  async function settle(text, ok) {
    const p = pending.get(text);
    assert.ok(p, 'a write of ' + JSON.stringify(text) + ' is pending');
    pending.delete(text);
    if (ok) {
      writes.push('native:' + text);
      p.resolve();
    } else {
      p.reject(new Error('refused'));
    }
    await flushMicrotasks();
  }
  return {win, writes, settle};
}

function menuCopy(win, target) {
  rightClick(win, target);
  const item = win.document.querySelector(
    '#sorcar-content-context-menu .sorcar-ctx-item[data-action="copy"]',
  );
  assert.ok(item && !item.classList.contains('disabled'), 'Copy is offered');
  item.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true, detail: 1}),
  );
}

function selectText(win, node) {
  const range = win.document.createRange();
  range.selectNodeContents(node);
  const sel = win.getSelection();
  sel.removeAllRanges();
  sel.addRange(range);
}

async function main() {
  await test(
    'W6 F3: a copy button whose first write rejects after a second copy ' +
      'succeeded does not fall back over the newer clipboard contents',
    async () => {
      const {win, writes, settle} = makeWindow(
        '<button id="copy">Copy</button>',
        'panelCopy.js',
      );
      const btn = win.document.getElementById('copy');
      let text = 'OLD';
      win.PanelCopy.wireCopyButton(btn, () => text);
      btn.click();
      text = 'NEW';
      btn.click();
      await settle('NEW', true);
      assert.deepStrictEqual(writes, ['native:NEW']);
      assert.strictEqual(btn.title, 'Copied');
      await settle('OLD', false);
      assert.deepStrictEqual(
        writes,
        ['native:NEW'],
        'the stale rejection must not execCommand the old text',
      );
      assert.strictEqual(btn.title, 'Copied', 'the newer outcome stands');
    },
  );

  await test(
    'W6 F3: the guard spans copy controls — a later PanelCopy.copyText ' +
      'from elsewhere also retires an older pending write',
    async () => {
      const {win, writes, settle} = makeWindow(
        '<button id="copy">Copy</button>',
        'panelCopy.js',
      );
      const btn = win.document.getElementById('copy');
      win.PanelCopy.wireCopyButton(btn, 'OLD');
      btn.click();
      const later = win.PanelCopy.copyText('NEW');
      await settle('NEW', true);
      assert.strictEqual(await later, true);
      await settle('OLD', false);
      assert.deepStrictEqual(writes, ['native:NEW']);
    },
  );

  await test(
    'W6 F3 guard: a rejection with no newer copy still falls back, and ' +
      'so do a missing or throwing clipboard API',
    async () => {
      const {win, writes, settle} = makeWindow(
        '<button id="copy">Copy</button>',
        'panelCopy.js',
      );
      const btn = win.document.getElementById('copy');
      win.PanelCopy.wireCopyButton(btn, 'ONLY');
      btn.click();
      await settle('ONLY', false);
      assert.deepStrictEqual(writes, ['fallback:ONLY']);
      assert.strictEqual(btn.title, 'Copied');
      win.navigator.clipboard.writeText = () => {
        throw new Error('sync refusal');
      };
      assert.strictEqual(await win.PanelCopy.copyText('SYNC'), true);
      delete win.navigator.clipboard.writeText;
      assert.strictEqual(await win.PanelCopy.copyText('NONE'), true);
      assert.deepStrictEqual(writes, [
        'fallback:ONLY',
        'fallback:SYNC',
        'fallback:NONE',
      ]);
    },
  );

  await test(
    "W6 F3: the context menu's Copy has the same guard — a stale " +
      'rejection after a newer menu copy succeeded writes nothing',
    async () => {
      const {win, writes, settle} = makeWindow(
        '<p id="a">OLD</p><p id="b">NEW</p>',
        'contentContextMenu.js',
      );
      win.ContentContextMenu.installContentContextMenu(win.document);
      const doc = win.document;
      selectText(win, doc.getElementById('a'));
      menuCopy(win, doc.getElementById('a'));
      selectText(win, doc.getElementById('b'));
      menuCopy(win, doc.getElementById('b'));
      await settle('NEW', true);
      await settle('OLD', false);
      assert.deepStrictEqual(
        writes,
        ['native:NEW'],
        'the stale rejection must not execCommand the old text',
      );
      // Alone, a rejection still falls back.
      selectText(win, doc.getElementById('a'));
      menuCopy(win, doc.getElementById('a'));
      await settle('OLD', false);
      assert.deepStrictEqual(writes, ['native:NEW', 'fallback:OLD']);
    },
  );

  for (const first of ['panel', 'menu']) {
    const later = first === 'panel' ? 'menu' : 'panel';
    await test(
      `W6 F4: in one document a ${later} copy retires an older ${first} ` +
        'copy — the stale rejection falls back over nothing',
      async () => {
        const {win, writes, settle} = makeWindow('<p id="p">MENU</p>', [
          'panelCopy.js',
          'contentContextMenu.js',
        ]);
        win.ContentContextMenu.installContentContextMenu(win.document);
        const p = win.document.getElementById('p');
        const panelCopy = () => win.PanelCopy.copyText('PANEL');
        const menuCopyP = () => {
          selectText(win, p);
          menuCopy(win, p);
        };
        if (first === 'panel') {
          panelCopy();
          menuCopyP();
        } else {
          menuCopyP();
          panelCopy();
        }
        const old = first === 'panel' ? 'PANEL' : 'MENU';
        const latest = first === 'panel' ? 'MENU' : 'PANEL';
        await settle(latest, true);
        await settle(old, false);
        assert.deepStrictEqual(
          writes,
          ['native:' + latest],
          'the older ' + first + ' copy must not execCommand over the newer one',
        );
        // A lone rejection of either control still falls back.
        menuCopyP();
        await settle('MENU', false);
        assert.deepStrictEqual(writes, ['native:' + latest, 'fallback:MENU']);
      },
    );
  }

  report('audit1101_W6_copy_stale_fallback');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  },
);
