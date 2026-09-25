// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the task-info panel's SECTIONS (the
// metasections block of media/main.js): the Task Info and Task update
// panels are collapsible, their bodies scroll, and the separator
// between them drags the boundary.  Covered here, on every surface
// that shows the panel (remote webapp, sidebar-chat drawer, Task Info
// view):
//
// * every section starts EXPANDED (aria-expanded="true", no
//   `collapsed` class) when nothing is stored;
// * the chevron button collapses / expands its section, updates
//   aria-expanded and persists the choice in localStorage, which a
//   fresh page restores;
// * the layout rule: the last expanded section's body fills
//   (flex-grow 1, basis 0), a dragged body has a fixed pixel basis,
//   every other body keeps its natural height; the resizer after a
//   section is hidden while nothing shown follows it, `static` unless
//   both sides are expanded, and a real handle otherwise;
// * the Task update section joins the stack only once it has content
//   (the `visible` class), and leaves it again when emptied;
// * the resizer's keyboard, double-click and pointer paths, and a
//   window whose localStorage is unavailable (opaque origin), where
//   every choice lasts for the page only.
//
// Geometry (pixel heights, scrolling) needs a layout engine and is
// covered by tests/agents/vscode/test_meta_panel_sections.py.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

let failures = 0;

async function test(name, fn) {
  try {
    await fn();
    console.log(`  ok - ${name}`);
  } catch (e) {
    failures++;
    console.log(`  FAIL - ${name}`);
    console.log(`      ${e.stack || e.message}`);
  }
}

/**
 * Boot a chat webview with the given body attributes.  *storage*
 * pre-seeds localStorage before main.js runs (so a stored collapse
 * state or height is restored at boot); `url: null` opens an OPAQUE
 * origin, where jsdom's localStorage throws SecurityError exactly as a
 * browser's does.
 */
// A third panel, added the way chat.html documents it: one more
// section (header with a toggle, a body) plus its resizer.
const EXTRA_SECTION =
  '<section class="meta-section" id="meta-section-extra">' +
  '<div class="sidebar-hdr meta-section-hdr">' +
  '<button type="button" class="meta-section-toggle" aria-expanded="true" ' +
  'aria-controls="meta-extra-body"><span>Extra</span></button></div>' +
  '<div id="meta-extra-body" class="meta-section-body"><p>extra</p></div>' +
  '</section>' +
  '<div class="meta-section-resizer" role="separator" aria-orientation="horizontal" ' +
  'aria-label="Resize the Extra section" tabindex="0"></div>';

function makeWebview(bodyAttrs, opts) {
  const {
    storage = {},
    url = 'https://localhost/',
    extraSection = false,
  } = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  html = html.replace('<body', '<body' + bodyAttrs);
  if (extraSection)
    html = html.replace(
      '<div id="meta-resizer"',
      EXTRA_SECTION + '<div id="meta-resizer"',
    );
  const domOpts = {runScripts: 'dangerously', pretendToBeVisual: true};
  if (url) domOpts.url = url;
  const dom = new JSDOM(html, domOpts);
  const win = dom.window;
  if (url) {
    for (const [k, v] of Object.entries(storage))
      win.localStorage.setItem(k, v);
  }
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  win.cancelAnimationFrame = function () {};
  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.matchMedia = function (query) {
    return {
      matches: query === '(min-width: 900px)',
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=metasections-main.js',
  );
  return {win, posted};
}

const REMOTE = ' class="remote-chat"';
const SIDEBAR = '';
const META_VIEW =
  ' class="editor-tab-mode meta-panel-mode" data-kiss-tab-id="meta-panel"';

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function el(win, id) {
  return win.document.getElementById(id);
}

function toggleOf(section) {
  return section.querySelector('.meta-section-toggle');
}

/** The .meta-section-resizer right after *section*. */
function resizerAfter(section) {
  const r = section.nextElementSibling;
  assert.ok(r && r.classList.contains('meta-section-resizer'));
  return r;
}

function sections(win) {
  return Array.from(
    win.document.querySelectorAll('#meta-panel > .meta-section'),
  );
}

function assertExpanded(section, expanded) {
  assert.strictEqual(
    section.classList.contains('collapsed'),
    !expanded,
    section.id + ' collapsed class',
  );
  assert.strictEqual(
    toggleOf(section).getAttribute('aria-expanded'),
    expanded ? 'true' : 'false',
    section.id + ' aria-expanded',
  );
}

/** 'fill' | 'natural' | '<px>' from a body's inline flex. */
function bodyLayout(body) {
  const grow = body.style.flexGrow;
  const basis = body.style.flexBasis;
  if (grow === '1') return 'fill';
  if (basis === '' || basis === 'auto') return 'natural';
  return basis;
}

function assertResizer(r, state) {
  const actual = r.hidden
    ? 'hidden'
    : r.classList.contains('static')
      ? 'static'
      : 'handle';
  assert.strictEqual(actual, state, r.getAttribute('aria-label') + ' state');
  assert.strictEqual(
    r.getAttribute('aria-disabled'),
    state === 'handle' ? 'false' : 'true',
  );
  assert.strictEqual(r.tabIndex, state === 'handle' ? 0 : -1);
}

/** Make the remote webview's Task update section visible with *html*. */
function showTaskUpdate(wv, html) {
  const win = wv.win;
  send(win, {type: 'configData', config: {}, apiKeys: {}});
  send(win, {type: 'status', running: true});
  const poll = wv.posted.filter(m => m.type === 'getTaskUpdate').pop();
  assert.ok(poll, 'a getTaskUpdate poll must have fired');
  send(win, {
    type: 'taskUpdate',
    tabId: poll.tabId,
    token: poll.token,
    taskId: 'task-1',
    exists: true,
    sig: 'sig-1',
    content: html,
    error: '',
    running: false,
    cost: 0,
    updatedAt: 0,
  });
}

function pointer(win, target, type, fields) {
  const ev = new win.Event(type, {bubbles: true, cancelable: true});
  Object.assign(ev, {button: 0, clientY: 0, pointerId: 1}, fields);
  target.dispatchEvent(ev);
}

function key(win, target, k) {
  target.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: k, bubbles: true, cancelable: true}),
  );
}

async function main() {
  for (const [label, attrs] of [
    ['remote webapp', REMOTE],
    ['sidebar-chat drawer', SIDEBAR],
    ['Task Info view', META_VIEW],
  ]) {
    await test(`${label}: both sections start expanded, the Task update one hidden`, () => {
      const {win} = makeWebview(attrs);
      const [info, update] = sections(win);
      assert.strictEqual(info.id, 'meta-section-info');
      assert.strictEqual(update.id, 'meta-info');
      assert.strictEqual(toggleOf(info).textContent, 'Task Info');
      assert.strictEqual(toggleOf(update).textContent, 'Task update');
      assertExpanded(info, true);
      assertExpanded(update, true);
      assert.ok(
        !update.classList.contains('visible'),
        'no task: Task update section hidden',
      );
      // Only Task Info is shown, so it is the last expanded section and
      // fills; no separator follows it.
      assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'fill');
      assertResizer(resizerAfter(info), 'hidden');
      assertResizer(resizerAfter(update), 'hidden');
      // Bodies are the scrolling parts.
      assert.ok(el(win, 'meta-list').classList.contains('meta-section-body'));
      assert.ok(
        el(win, 'meta-info-content').classList.contains('meta-section-body'),
      );
      assert.strictEqual(
        toggleOf(info).getAttribute('aria-controls'),
        'meta-list',
      );
      assert.strictEqual(
        toggleOf(update).getAttribute('aria-controls'),
        'meta-info-content',
      );
    });

    await test(`${label}: the chevron collapses and re-expands, persisting the choice`, () => {
      const {win} = makeWebview(attrs);
      const [info] = sections(win);
      click(win, toggleOf(info));
      assertExpanded(info, false);
      assert.strictEqual(
        win.localStorage.getItem(
          'kiss-meta-section-collapsed:meta-section-info',
        ),
        '1',
      );
      // Collapsed: the body no longer fills (nothing is expanded).
      assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'natural');
      click(win, toggleOf(info));
      assertExpanded(info, true);
      assert.strictEqual(
        win.localStorage.getItem(
          'kiss-meta-section-collapsed:meta-section-info',
        ),
        null,
      );
      assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'fill');
    });

    await test(`${label}: a stored collapse state and height are restored at boot`, () => {
      const {win} = makeWebview(attrs, {
        storage: {
          'kiss-meta-section-collapsed:meta-info': '1',
          'kiss-meta-section-h:meta-section-info': '123',
          'kiss-meta-section-h:meta-info': 'junk',
        },
      });
      const [info, update] = sections(win);
      assertExpanded(info, true);
      assertExpanded(update, false);
      // Task Info is still the last expanded section, so it fills
      // regardless of its stored height...
      assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'fill');
      // ...until the collapsed Task update section is shown and
      // expanded: then the stored height applies.
      update.classList.add('visible');
      click(win, toggleOf(update));
      assertExpanded(update, true);
      assert.strictEqual(bodyLayout(el(win, 'meta-list')), '123px');
      assert.strictEqual(bodyLayout(el(win, 'meta-info-content')), 'fill');
      assertResizer(resizerAfter(info), 'handle');
    });
  }

  await test('remote webapp: the Task update section joins the stack with content and leaves when emptied', () => {
    const wv = makeWebview(REMOTE);
    const win = wv.win;
    const [info, update] = sections(win);
    showTaskUpdate(wv, '<p>progress</p>');
    assert.ok(update.classList.contains('visible'));
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'natural');
    assert.strictEqual(bodyLayout(el(win, 'meta-info-content')), 'fill');
    assertResizer(resizerAfter(info), 'handle');
    assertResizer(resizerAfter(update), 'hidden');

    // Collapse the Task update section: Task Info fills again and the
    // separator stays but is no longer a handle.
    click(win, toggleOf(update));
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'fill');
    assertResizer(resizerAfter(info), 'static');
    click(win, toggleOf(update));

    // Collapse Task Info instead: the separator under it is static
    // too (nothing above to resize), Task update fills.
    click(win, toggleOf(info));
    assertResizer(resizerAfter(info), 'static');
    assert.strictEqual(bodyLayout(el(win, 'meta-info-content')), 'fill');
    click(win, toggleOf(info));

    // The task ends: the update is emptied and its section leaves.
    send(win, {type: 'status', running: false});
    assert.ok(
      !update.classList.contains('visible'),
      'emptied update hides its section',
    );
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'fill');
    assertResizer(resizerAfter(info), 'hidden');
  });

  await test('remote webapp: arrow keys, double-click and pointer drag on the separator', () => {
    const wv = makeWebview(REMOTE);
    const win = wv.win;
    const [info] = sections(win);
    const r = resizerAfter(info);
    const list = el(win, 'meta-list');
    const HK = 'kiss-meta-section-h:meta-section-info';

    // Hidden / static separators ignore every input.
    key(win, r, 'ArrowDown');
    pointer(win, r, 'pointerdown');
    pointer(win, r, 'pointermove', {clientY: 40});
    pointer(win, r, 'pointerup');
    r.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
    assert.strictEqual(win.localStorage.getItem(HK), null);

    showTaskUpdate(wv, '<p>progress</p>');
    assertResizer(r, 'handle');

    // Arrow keys fix the height (jsdom has no layout: every rect is 0,
    // so the clamp lands on 0) and persist it; other keys are ignored.
    key(win, r, 'Tab');
    assert.strictEqual(win.localStorage.getItem(HK), null);
    key(win, r, 'ArrowDown');
    assert.strictEqual(win.localStorage.getItem(HK), '0');
    assert.strictEqual(bodyLayout(list), '0px');
    assert.strictEqual(r.getAttribute('aria-valuenow'), '0');
    assert.strictEqual(r.getAttribute('aria-valuemax'), '0');
    key(win, r, 'ArrowUp');
    assert.strictEqual(bodyLayout(list), '0px');

    // Double-click restores the natural height.
    r.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
    assert.strictEqual(win.localStorage.getItem(HK), null);
    assert.strictEqual(bodyLayout(list), 'natural');

    // A pointer drag: the body class marks the drag, moves resize,
    // release ends it; a move without a press does nothing.
    pointer(win, r, 'pointermove', {clientY: 40});
    assert.strictEqual(bodyLayout(list), 'natural');
    pointer(win, r, 'pointerdown', {button: 2});
    assert.ok(
      !win.document.body.classList.contains('meta-section-resizing'),
      'right button ignored',
    );
    pointer(win, r, 'pointerdown');
    assert.ok(win.document.body.classList.contains('meta-section-resizing'));
    pointer(win, r, 'pointermove', {clientY: 40});
    assert.strictEqual(bodyLayout(list), '0px');
    pointer(win, r, 'pointerup');
    assert.ok(!win.document.body.classList.contains('meta-section-resizing'));
    pointer(win, r, 'pointerup');
    assert.strictEqual(win.localStorage.getItem(HK), '0');
    // pointercancel ends a drag too.
    pointer(win, r, 'pointerdown');
    pointer(win, r, 'pointercancel');
    assert.ok(!win.document.body.classList.contains('meta-section-resizing'));
  });

  await test('a third section added per the chat.html recipe joins the stack; `hidden` removes it', () => {
    const wv = makeWebview(REMOTE, {extraSection: true});
    const win = wv.win;
    const [info, update, extra] = sections(win);
    assert.strictEqual(extra.id, 'meta-section-extra');
    assertExpanded(extra, true);
    // No task: Task Info and Extra are shown; Extra, being last, fills.
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'natural');
    assert.strictEqual(bodyLayout(el(win, 'meta-extra-body')), 'fill');
    assertResizer(resizerAfter(info), 'handle');
    assertResizer(resizerAfter(update), 'hidden');
    assertResizer(resizerAfter(extra), 'hidden');

    showTaskUpdate(wv, '<p>progress</p>');
    assert.strictEqual(bodyLayout(el(win, 'meta-info-content')), 'natural');
    assert.strictEqual(bodyLayout(el(win, 'meta-extra-body')), 'fill');
    assertResizer(resizerAfter(info), 'handle');
    assertResizer(resizerAfter(update), 'handle');
    assertResizer(resizerAfter(extra), 'hidden');

    // Collapse the middle one: the separator above it is still a
    // handle (Extra below is expanded), the one below it is static.
    click(win, toggleOf(update));
    assertResizer(resizerAfter(info), 'handle');
    assertResizer(resizerAfter(update), 'static');
    click(win, toggleOf(update));

    // A section hidden with the `hidden` attribute is out of the stack
    // (its owner re-applies the layout; a toggle round trip does here).
    extra.hidden = true;
    click(win, toggleOf(info));
    click(win, toggleOf(info));
    assert.strictEqual(bodyLayout(el(win, 'meta-info-content')), 'fill');
    assertResizer(resizerAfter(update), 'hidden');
    assertResizer(resizerAfter(extra), 'hidden');
    extra.hidden = false;
    click(win, toggleOf(info));
    click(win, toggleOf(info));
    assert.strictEqual(bodyLayout(el(win, 'meta-extra-body')), 'fill');
    assertResizer(resizerAfter(update), 'handle');
  });

  await test('opaque origin (no localStorage): sections still work, choices last for the page', () => {
    const wv = makeWebview(REMOTE, {url: null});
    const win = wv.win;
    assert.throws(
      () => win.localStorage,
      'jsdom must refuse localStorage on an opaque origin',
    );
    const [info, update] = sections(win);
    assertExpanded(info, true);
    assertExpanded(update, true);
    click(win, toggleOf(update));
    assertExpanded(update, false);
    click(win, toggleOf(update));
    assertExpanded(update, true);
    update.classList.add('visible');
    win.eval("document.getElementById('meta-info').classList.add('visible')");
    click(win, toggleOf(info));
    click(win, toggleOf(info));
    const r = resizerAfter(info);
    assertResizer(r, 'handle');
    key(win, r, 'ArrowDown');
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), '0px');
    r.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'natural');
  });

  if (failures) {
    console.log(`\n${failures} test(s) failed`);
    process.exit(1);
  }
  console.log('\nall metaPanelSections tests passed');
  // main.js starts timers (task-update polling, debounces) in every
  // window booted above; exit explicitly instead of waiting for them.
  process.exit(0);
}

main();
