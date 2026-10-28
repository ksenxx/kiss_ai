// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const projectRoot = path.resolve(__dirname, '..');

const {JSDOM} = require('jsdom');
const {makeRunner} = require('./ui_antipattern_harness');
const {stubClipboard} = require('./simplify2_harness');
const {test, report} = makeRunner();

// A page with tips.js (and its PanelCopy dependency) loaded over
// `window.__TIPS__ = cfg`.  `withStorage` gives the page an http origin
// so localStorage works; `seenVersion` seeds its kissTipsSeenVersion.
//
// The style tests below read the stylesheet text the component embeds
// in its shadow root: jsdom neither cascades a shadow root's <style>
// into getComputedStyle (`.tips-panel` reads `position: static`) nor
// exposes it as a CSSStyleSheet (`style.sheet` is null).
function loadTipsDom(
  cfg,
  {withMarked = true, withConfig = true, withStorage = false, seenVersion} = {},
) {
  const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {
    runScripts: 'outside-only',
    url: withStorage ? 'http://localhost/' : undefined,
  });
  if (seenVersion !== undefined) {
    dom.window.localStorage.setItem('kissTipsSeenVersion', seenVersion);
  }
  const run = file =>
    dom.window.eval(
      fs.readFileSync(path.join(projectRoot, 'media', file), 'utf-8'),
    );
  if (withMarked) run('marked.min.js');
  if (withConfig) dom.window.eval(`window.__TIPS__ = ${JSON.stringify(cfg)};`);
  // tips.js wires its Copy buttons through PanelCopy, which every real
  // host page loads before it (chat.html and the kiss-web server render
  // panelCopy.js first).
  run('panelCopy.js');
  run('tips.js');
  return dom.window;
}

function panelParts(win) {
  const host = win.document.querySelector('kiss-tips-panel');
  assert.ok(host, 'kiss-tips-panel must be mounted on document.body');
  const root = host.shadowRoot;
  assert.ok(root, 'kiss-tips-panel must use shadow DOM');
  return {
    host,
    root,
    body: root.querySelector('.tips-body'),
    counter: root.querySelector('.tips-counter'),
    prev: root.querySelector('.tips-prev'),
    next: root.querySelector('.tips-next'),
    close: root.querySelector('.tips-close'),
  };
}

const THREE_TIPS = [
  '## First tip\n\n- alpha\n- beta',
  'Second tip with **bold** text.',
  '### Third tip\n\n`code` here.',
];

test('tips panel auto-mounts centered and renders markdown as HTML', () => {
  const win = loadTipsDom({tips: THREE_TIPS, show: true});
  const {root, body, counter, prev, next} = panelParts(win);
  const css = root.querySelector('style').textContent;
  assert.ok(css.includes('position: fixed'), 'overlay must be fixed');
  assert.ok(css.includes('justify-content: center'), 'centered horizontally');
  assert.ok(css.includes('align-items: center'), 'centered vertically');
  assert.ok(
    body.innerHTML.includes('<h2') && body.innerHTML.includes('<li>'),
    `tip markdown must render to HTML, got: ${body.innerHTML}`,
  );
  assert.strictEqual(counter.textContent, '1 / 3');
  assert.strictEqual(prev.disabled, true, 'Previous disabled on first tip');
  assert.strictEqual(next.disabled, false, 'Next enabled on first tip');
});

test('next/previous navigate tips with the usual boundary semantics', () => {
  const win = loadTipsDom({tips: THREE_TIPS, show: true});
  const {body, counter, prev, next} = panelParts(win);

  next.click();
  assert.strictEqual(counter.textContent, '2 / 3');
  assert.ok(body.innerHTML.includes('<strong>bold</strong>'));
  assert.strictEqual(prev.disabled, false);
  assert.strictEqual(next.disabled, false);

  next.click();
  assert.strictEqual(counter.textContent, '3 / 3');
  assert.ok(body.innerHTML.includes('<code>code</code>'));
  assert.strictEqual(next.disabled, true, 'Next disabled on last tip');

  next.click();
  assert.strictEqual(counter.textContent, '3 / 3');

  prev.click();
  assert.strictEqual(counter.textContent, '2 / 3');
  prev.click();
  assert.strictEqual(counter.textContent, '1 / 3');
  assert.strictEqual(prev.disabled, true);
  prev.click();
  assert.strictEqual(counter.textContent, '1 / 3');
});

test('close removes the tips panel from the DOM', () => {
  const win = loadTipsDom({tips: THREE_TIPS, show: true});
  const {close} = panelParts(win);
  close.click();
  assert.strictEqual(win.document.querySelector('kiss-tips-panel'), null);
});

test('a single tip disables both Previous and Next', () => {
  const win = loadTipsDom({tips: ['Only one.'], show: true});
  const {counter, prev, next} = panelParts(win);
  assert.strictEqual(counter.textContent, '1 / 1');
  assert.strictEqual(prev.disabled, true);
  assert.strictEqual(next.disabled, true);
});

test('tips panel does not mount when show is false', () => {
  const win = loadTipsDom({tips: THREE_TIPS, show: false});
  assert.strictEqual(win.document.querySelector('kiss-tips-panel'), null);
});

test('tips panel does not mount when there are no tips', () => {
  const win = loadTipsDom({tips: [], show: true});
  assert.strictEqual(win.document.querySelector('kiss-tips-panel'), null);
});

test('tips panel does not mount when window.__TIPS__ is missing', () => {
  const win = loadTipsDom(null, {withConfig: false});
  assert.strictEqual(win.document.querySelector('kiss-tips-panel'), null);
});

test('tips fall back to plain text when marked is unavailable', () => {
  const win = loadTipsDom(
    {tips: ['**not rendered**'], show: true},
    {withMarked: false},
  );
  const {body} = panelParts(win);
  assert.ok(
    body.textContent.includes('**not rendered**'),
    'raw markdown text must be shown when marked is missing',
  );
  assert.ok(
    !body.innerHTML.includes('<strong>'),
    'no HTML rendering without marked',
  );
});

const CODE_TIP =
  'Run this:\n\n```\nnpm install\nnpm test\n```\n\nAnd `inline` code.';
const TWO_CODE_TIP =
  'First:\n\n```\necho one\n```\n\nSecond:\n\n```\necho two\n```\n';

function copyButtons(body) {
  return Array.from(body.querySelectorAll('button.tips-copy'));
}

test('tips panel has a fixed, viewport-capped width and height', () => {
  const win = loadTipsDom({tips: THREE_TIPS, show: true});
  const {root} = panelParts(win);
  const css = root.querySelector('style').textContent;
  const panelDecls = css.split('.tips-panel {')[1].split('}')[0];
  // The size is a fixed pixel size capped by the viewport (`min(Npx,
  // Mvw)` / `min(Npx, Mvh)`), never a content-driven or unbounded one;
  // the exact pixels are the designer's, but both terms must be positive
  // and the viewport term at most 100% so the cap really is the viewport.
  for (const [axis, unit] of [
    ['width', 'vw'],
    ['height', 'vh'],
  ]) {
    const m = new RegExp(
      String.raw`(?:^|[^-])${axis}:\s*min\((\d+)px,\s*(\d+)${unit}\)`,
    ).exec(panelDecls);
    assert.ok(m, `fixed, viewport-capped panel ${axis}`);
    const [px, viewport] = [Number(m[1]), Number(m[2])];
    assert.ok(px > 0, `positive pixel ${axis}: ${m[0].trim()}`);
    assert.ok(
      viewport > 0 && viewport <= 100,
      `${axis} viewport cap must be in (0, 100]${unit}: ${m[0].trim()}`,
    );
  }
  assert.doesNotMatch(
    panelDecls,
    /(max|min)-(width|height)/,
    'no min/max bounds: the size is fixed',
  );
  assert.ok(
    panelDecls.includes('box-sizing: border-box'),
    'fixed size must include the panel border',
  );
});

test('tips body scrolls when the content overflows the fixed panel', () => {
  const win = loadTipsDom({tips: THREE_TIPS, show: true});
  const {root, body} = panelParts(win);
  const css = root.querySelector('style').textContent;
  const bodyRule = css.split('.tips-body {')[1];
  assert.ok(bodyRule, '.tips-body rule must exist');
  const decls = bodyRule.split('}')[0];
  assert.ok(decls.includes('overflow: auto'), 'body must scroll on overflow');
  assert.ok(decls.includes('flex: 1 1 auto'), 'body fills the fixed panel');
  assert.ok(
    decls.includes('min-height: 0'),
    'body must be allowed to shrink so overflow: auto can engage',
  );
  assert.strictEqual(body.className, 'tips-body');
});

test('tips panel is styled from the shared design tokens, never a fixed palette', () => {
  const win = loadTipsDom({tips: THREE_TIPS, show: true});
  const {root} = panelParts(win);
  const css = root.querySelector('style').textContent;
  // The tokens main.css defines on :root (and remote-codex.css
  // re-derives for the remote light theme) inherit into the shadow
  // root, so the panel follows both remote themes.  Each of these
  // must be read through var().
  for (const token of [
    '--bg2',
    '--fg',
    '--dim',
    '--border',
    '--panel-line',
    '--accent',
    '--scrim',
    '--radius-xl',
    '--shadow-lg',
    '--z-dialog',
    '--dur-slow',
    '--theme-button-background',
    '--theme-button-foreground',
    '--theme-button-secondaryBackground',
    '--theme-textCodeBlock-background',
    '--theme-focusBorder',
    '--font-mono',
  ]) {
    assert.ok(css.includes('var(' + token), `token ${token} must be read`);
  }
  // Every literal colour is a fallback inside var(...).  A colour that
  // stands alone would bypass the theme.
  const stripped = css.replace(/var\([^()]*(?:\([^()]*\)[^()]*)*\)/g, '');
  const literalColors = stripped.match(/#[0-9a-fA-F]{3,8}\b|rgba?\(/g) || [];
  assert.deepStrictEqual(
    literalColors,
    [],
    `colours outside var() fallbacks: ${literalColors.join(', ')}`,
  );
  // The dialog layer sits above the settings drawer (z-drawer) that
  // opens it and below toasts.
  assert.ok(css.includes('z-index: var(--z-dialog, 300)'));
});

test('header shows the lightbulb icon and the reading progress grows per tip', () => {
  const win = loadTipsDom({tips: THREE_TIPS, show: true});
  const {root, next} = panelParts(win);
  assert.ok(root.querySelector('.tips-icon svg'), 'header icon is inline SVG');
  const bar = root.querySelector('.tips-progress-bar');
  assert.ok(bar, 'a progress bar under the header');
  const pct = () => Math.round(parseFloat(bar.style.width));
  assert.strictEqual(pct(), 33);
  next.click();
  assert.strictEqual(pct(), 67);
  next.click();
  assert.strictEqual(pct(), 100);
});

test('Left/Right arrow keys page through the tips', () => {
  const win = loadTipsDom({tips: THREE_TIPS, show: true});
  const {counter} = panelParts(win);
  const key = k =>
    win.document.dispatchEvent(
      new win.KeyboardEvent('keydown', {
        key: k,
        bubbles: true,
        cancelable: true,
      }),
    );
  key('ArrowRight');
  assert.strictEqual(counter.textContent, '2 / 3');
  key('ArrowRight');
  key('ArrowRight');
  assert.strictEqual(counter.textContent, '3 / 3', 'clamped at the last tip');
  key('ArrowLeft');
  assert.strictEqual(counter.textContent, '2 / 3');
  key('ArrowLeft');
  key('ArrowLeft');
  assert.strictEqual(counter.textContent, '1 / 3', 'clamped at the first tip');
});

/** loadTipsDom on an http origin so localStorage works. */
function loadTipsDomWithStorage(cfg, seenVersion) {
  return loadTipsDom(cfg, {withStorage: true, seenVersion});
}

test('a tipsData bootstrap after load opens the tips like an embedded config', () => {
  // The page starts with an empty config; the daemon's
  // `tipsData` answer to `ready` reaches tips.js through
  // window.__kissApplyTipsConfig (main.js) and opens the window once.
  const win = loadTipsDom({tips: [], show: false, version: ''});
  assert.strictEqual(win.document.querySelector('kiss-tips-panel'), null);
  win.__kissApplyTipsConfig({tips: THREE_TIPS, show: true, version: '9.9.9'});
  const {counter} = panelParts(win);
  assert.strictEqual(counter.textContent.trim(), '1 / 3');
  assert.deepStrictEqual(
    win.__TIPS__.tips,
    THREE_TIPS,
    'the button reuses the tips',
  );
  win.__kissApplyTipsConfig({tips: THREE_TIPS, show: true, version: '9.9.9'});
  assert.strictEqual(
    win.document.querySelectorAll('kiss-tips-panel').length,
    1,
    'a second bootstrap (reconnect) never stacks a second window',
  );
});

test('with a version, the tips auto-open once per version per browser', () => {
  const cfg = {tips: THREE_TIPS, show: true, version: '2026.10.1'};
  // Fresh browser: opens and remembers the version.
  const first = loadTipsDomWithStorage(cfg);
  assert.ok(first.document.querySelector('kiss-tips-panel'), 'opens once');
  assert.strictEqual(
    first.localStorage.getItem('kissTipsSeenVersion'),
    '2026.10.1',
  );
  // Same version already seen here: stays closed.
  const again = loadTipsDomWithStorage(cfg, '2026.10.1');
  assert.strictEqual(again.document.querySelector('kiss-tips-panel'), null);
  // An update: the new version opens the tips again.
  const updated = loadTipsDomWithStorage(
    {tips: THREE_TIPS, show: true, version: '2026.11.1'},
    '2026.10.1',
  );
  assert.ok(updated.document.querySelector('kiss-tips-panel'), 'reopens');
  assert.strictEqual(
    updated.localStorage.getItem('kissTipsSeenVersion'),
    '2026.11.1',
  );
  // The host's verdict and the opt-out still win.
  const closed = loadTipsDomWithStorage(
    {tips: THREE_TIPS, show: false, version: '2026.12.1'},
    '2026.10.1',
  );
  assert.strictEqual(closed.document.querySelector('kiss-tips-panel'), null);
  assert.strictEqual(
    closed.localStorage.getItem('kissTipsSeenVersion'),
    '2026.10.1',
    'a closed window does not mark the version as seen',
  );
});

test('without localStorage the version gate degrades to the host verdict', () => {
  // loadTipsDom runs on an opaque origin where localStorage throws.
  const win = loadTipsDom({tips: THREE_TIPS, show: true, version: '1.2.3'});
  assert.throws(() => win.localStorage, 'precondition: no storage here');
  assert.ok(win.document.querySelector('kiss-tips-panel'), 'still opens');
});

test('each fenced code block gets a copy button', () => {
  const win = loadTipsDom({tips: [CODE_TIP], show: true});
  const {body} = panelParts(win);
  const buttons = copyButtons(body);
  assert.strictEqual(buttons.length, 1, 'one fenced block → one copy button');
  const btn = buttons[0];
  assert.strictEqual(btn.textContent, 'Copy');
  assert.strictEqual(btn.type, 'button');
  assert.strictEqual(btn.getAttribute('aria-label'), 'Copy code to clipboard');
  const wrapper = btn.parentElement;
  assert.ok(
    wrapper.classList.contains('tips-code'),
    'button must live in a .tips-code wrapper',
  );
  assert.ok(
    wrapper.querySelector('pre > code'),
    'wrapper must contain the fenced code block',
  );
});

test('a tip with two fenced blocks gets two copy buttons', () => {
  const win = loadTipsDom({tips: [TWO_CODE_TIP], show: true});
  const {body} = panelParts(win);
  assert.strictEqual(copyButtons(body).length, 2);
});

test('inline code does not get a copy button', () => {
  const win = loadTipsDom({tips: ['Only `inline` code here.'], show: true});
  const {body} = panelParts(win);
  assert.strictEqual(copyButtons(body).length, 0);
});

test('plain-text fallback (no marked) has no copy buttons', () => {
  const win = loadTipsDom({tips: [CODE_TIP], show: true}, {withMarked: false});
  const {body} = panelParts(win);
  assert.strictEqual(copyButtons(body).length, 0);
});

test('navigation re-creates copy buttons for each tip', () => {
  const win = loadTipsDom({
    tips: [CODE_TIP, 'No code at all.', TWO_CODE_TIP],
    show: true,
  });
  const {body, next, prev} = panelParts(win);
  assert.strictEqual(copyButtons(body).length, 1);
  next.click();
  assert.strictEqual(copyButtons(body).length, 0);
  next.click();
  assert.strictEqual(copyButtons(body).length, 2);
  prev.click();
  assert.strictEqual(copyButtons(body).length, 0);
  prev.click();
  assert.strictEqual(copyButtons(body).length, 1);
});

function tickAsync() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

/**
 * Replace the window's setTimeout with a recorder that never fires, so a
 * test can run the "Copied!" reset by hand (unlike the chat harness's
 * captureTimers, which keeps real timers).
 */
function recordTimers(win) {
  const timers = [];
  win.setTimeout = fn => {
    timers.push(fn);
    return 0;
  };
  return timers;
}

// The copy tests await clipboard promises, so they run one after another
// inside this async block; `test` returns the promise of an async body.
(async () => {
  await test('copy button uses navigator.clipboard and shows Copied!', async () => {
    const win = loadTipsDom({tips: [CODE_TIP], show: true});
    const {body} = panelParts(win);
    const {writes: copied} = stubClipboard(win);
    const timers = recordTimers(win);
    const btn = copyButtons(body)[0];
    btn.click();
    await tickAsync();
    assert.strictEqual(copied.length, 1, 'clipboard.writeText called once');
    assert.strictEqual(copied[0], body.querySelector('pre code').textContent);
    assert.ok(
      copied[0].includes('npm install') && copied[0].includes('npm test'),
      'the full fenced block text must be copied',
    );
    assert.strictEqual(btn.textContent, 'Copied!');
    assert.strictEqual(timers.length, 1, 'a label-reset timer is scheduled');
    timers[0]();
    assert.strictEqual(btn.textContent, 'Copy', 'label resets after the timer');
  });

  // The clipboard -> execCommand fallback, the helper-textarea cleanup
  // and the both-paths-fail outcome belong to PanelCopy.copyText and are
  // pinned by ui_antipattern_panel_copy_fallback.test.js and
  // panelCopy.test.js (missing / throwing writeText).

  await test('copy shows Failed when execCommand returns false', async () => {
    const win = loadTipsDom({tips: [CODE_TIP], show: true});
    const {body} = panelParts(win);
    Object.defineProperty(win.navigator, 'clipboard', {
      configurable: true,
      value: undefined,
    });
    win.document.execCommand = () => false;
    const timers = recordTimers(win);
    const btn = copyButtons(body)[0];
    btn.click();
    await tickAsync();
    assert.strictEqual(btn.textContent, 'Failed');
    assert.strictEqual(timers.length, 1, 'a label-reset timer is scheduled');
    timers[0]();
    assert.strictEqual(btn.textContent, 'Copy', 'label resets after Failed');
  });

  report('tipsWindow');
})();
