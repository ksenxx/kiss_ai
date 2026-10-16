// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the task-info panel's SECTIONS (the
// metasections block of media/main.js): the Task Info and Task update
// panels are collapsible, their bodies scroll, and the separator
// between them drags the boundary.  Covered here:
//
// * every section starts EXPANDED (aria-expanded="true", no
//   `collapsed` class) when nothing is stored;
// * the chevron button collapses / expands its section, updates
//   aria-expanded and persists the choice in localStorage, which a
//   fresh page restores;
// * the layout rule: by default every expanded body takes an equal
//   share (flex-grow 1, basis 0); a dragged body has a fixed pixel
//   basis, except the last expanded one, which always fills (every
//   body keeps the minimum height main.css sets); a drag pins the
//   expanded bodies above the boundary it moves; the resizer after a
//   section is hidden while nothing shown follows it, `static` unless
//   both sides are expanded, and a real handle otherwise;
// * the Task update section joins the stack only once it has content
//   (the `visible` class), and leaves it again when emptied;
// * the resizer's keyboard, double-click and pointer paths, and a
//   window whose localStorage is unavailable (opaque origin), where
//   every choice lasts for the page only.
//
// Geometry (pixel heights, scrolling) needs a layout engine and is
// covered by tests/webapp/test_meta_panel_sections.py.

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
 * Boot a chat page with the given body attributes.  *storage*
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

const pages = [];

function makePage(opts) {
  const {
    storage = {},
    url = 'https://localhost/',
    extraSection = false,
    tips = undefined,
  } = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  html = html.replace('<body', '<body class="remote-chat"');
  if (extraSection)
    html = html.replace(
      '<div id="meta-resizer"',
      EXTRA_SECTION + '<div id="meta-resizer"',
    );
  const domOpts = {runScripts: 'dangerously', pretendToBeVisual: true};
  if (url) domOpts.url = url;
  const dom = new JSDOM(html, domOpts);
  const win = dom.window;
  pages.push(win);
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
  // jsdom has no ResizeObserver: record the observers so a test can
  // fire a panel resize by hand.
  const observers = [];
  win.ResizeObserver = class {
    constructor(fn) {
      this.fn = fn;
      observers.push(this);
    }
    observe(target) {
      this.target = target;
    }
  };
  const posted = [];
  win.acquireTransport = function () {
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
  // The page embeds the daemon's tips bootstrap (with the running
  // version) before main.js runs; *tips* stands in for it.
  if (tips !== undefined) win.__TIPS__ = tips;
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=metasections-main.js',
  );
  return {win, posted, observers};
}

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

/**
 * A body's layout from its inline flex: 'equal' (an equal share,
 * flex-grow 1, basis 0) or '<px>' (a dragged height).
 */
function bodyLayout(body) {
  if (body.style.flexGrow === '1') {
    assert.strictEqual(body.style.flexBasis, '0px');
    return 'equal';
  }
  assert.strictEqual(body.style.flexGrow, '0');
  return body.style.flexBasis;
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

/** Make the page's Task update section visible with *html*. */
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

/**
 * Register chat tab *tabId* (its status known, so the next running
 * status counts as a start) and start a task in it, the way the daemon
 * does for a prompt the user submits.
 */
function startOnChat(win, tabId) {
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId, chatId: 'chat-' + tabId, title: tabId, workDir: '/ws'}],
  });
  send(win, {type: 'task_events', tabId, task: 'the task', events: []});
  send(win, {type: 'status', running: true, tabId, startTs: 1});
}

/**
 * Take the workspace (Explorer, Source Control) and global (Schedule,
 * Machines, Apps, Spend, Recommendations) sections out of the stack (the `hidden`
 * attribute, then the panel's resize observer re-applies the layout),
 * leaving only the two per-task sections.
 */
function hideGlobalSections(wv) {
  const win = wv.win;
  for (const id of [
    'meta-explorer',
    'meta-scm',
    'meta-schedule',
    'meta-machines',
    'meta-apps',
    'meta-spend',
    'meta-recommendations',
  ])
    el(win, id).hidden = true;
  const panel = el(win, 'meta-panel');
  wv.observers.find(o => o.target === panel).fn([]);
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
  await test('remote webapp: every section starts expanded, the Task update one hidden', () => {
    const {win} = makePage();
    const [
      info,
      update,
      explorer,
      scm,
      schedule,
      machines,
      apps,
      spend,
      recommendations,
    ] = sections(win);
    assert.strictEqual(info.id, 'meta-section-info');
    assert.strictEqual(update.id, 'meta-info');
    assert.strictEqual(explorer.id, 'meta-explorer');
    assert.strictEqual(scm.id, 'meta-scm');
    assert.strictEqual(schedule.id, 'meta-schedule');
    assert.strictEqual(machines.id, 'meta-machines');
    assert.strictEqual(apps.id, 'meta-apps');
    assert.strictEqual(spend.id, 'meta-spend');
    assert.strictEqual(recommendations.id, 'meta-recommendations');
    assert.strictEqual(toggleOf(info).textContent, 'Task Info');
    assert.strictEqual(toggleOf(update).textContent, 'Task update');
    assert.strictEqual(toggleOf(explorer).textContent, 'Explorer');
    // The header count (uncommitted files) is empty until a git status.
    assert.strictEqual(toggleOf(scm).textContent, 'Source Control');
    assert.strictEqual(toggleOf(schedule).textContent, 'Schedule');
    assert.strictEqual(toggleOf(machines).textContent, 'Machines');
    assert.strictEqual(toggleOf(apps).textContent, 'Apps');
    assert.strictEqual(toggleOf(spend).textContent, 'Spend');
    assert.strictEqual(
      toggleOf(recommendations).textContent,
      'Recommendations',
    );
    for (const section of sections(win)) assertExpanded(section, true);
    assert.ok(
      !update.classList.contains('visible'),
      'no task: Task update section hidden',
    );
    // The workspace sections (Explorer, Source Control) are shown.
    assert.strictEqual(explorer.hidden, false, 'Explorer shown');
    assert.strictEqual(scm.hidden, false, 'Source Control shown');
    // Nothing is dragged: every expanded body takes an equal share
    // and no separator follows Recommendations, the last shown section.
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'equal');
    assert.strictEqual(bodyLayout(el(win, 'meta-schedule-list')), 'equal');
    assert.strictEqual(bodyLayout(el(win, 'meta-apps-list')), 'equal');
    assert.strictEqual(bodyLayout(el(win, 'meta-spend-body')), 'equal');
    assert.strictEqual(
      bodyLayout(el(win, 'meta-recommendations-list')),
      'equal',
    );
    assertResizer(resizerAfter(info), 'handle');
    assertResizer(resizerAfter(update), 'hidden');
    assertResizer(resizerAfter(schedule), 'handle');
    assertResizer(resizerAfter(machines), 'handle');
    assertResizer(resizerAfter(apps), 'handle');
    assertResizer(resizerAfter(spend), 'handle');
    assertResizer(resizerAfter(recommendations), 'hidden');
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

  await test('remote webapp: the chevron collapses; re-expanding folds every other section, persisting the choices', () => {
    const {win} = makePage();
    const stored = id =>
      win.localStorage.getItem('kiss-meta-section-collapsed:' + id);
    const [info, ...others] = sections(win);
    click(win, toggleOf(info));
    assertExpanded(info, false);
    assert.strictEqual(stored('meta-section-info'), '1');
    for (const section of others) assertExpanded(section, true);
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'equal');
    // The separator under a collapsed section is no longer a handle.
    assertResizer(resizerAfter(info), 'static');
    // Opening a collapsed section folds the rest: the panel shows one
    // section at a time, as much of it as fits.
    click(win, toggleOf(info));
    assertExpanded(info, true);
    assert.strictEqual(stored('meta-section-info'), null);
    for (const section of others) {
      assertExpanded(section, false);
      assert.strictEqual(stored(section.id), '1', section.id + ' folded');
    }
    // Nothing expanded follows Task Info: its separator is static.
    assertResizer(resizerAfter(info), 'static');
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'equal');
    // Opening Apps folds Task Info in turn; a click on the open Apps
    // just folds it.
    const apps = el(win, 'meta-apps');
    click(win, toggleOf(apps));
    assertExpanded(apps, true);
    assertExpanded(info, false);
    assert.strictEqual(stored('meta-apps'), null);
    assert.strictEqual(stored('meta-section-info'), '1');
    assert.strictEqual(bodyLayout(el(win, 'meta-apps-list')), 'equal');
    click(win, toggleOf(apps));
    assertExpanded(apps, false);
    assert.strictEqual(stored('meta-apps'), '1');
    for (const section of sections(win)) assertExpanded(section, false);
  });

  await test('remote webapp: a stored collapse state and height are restored at boot', () => {
    const {win} = makePage({
      storage: {
        'kiss-meta-section-collapsed:meta-info': '1',
        'kiss-meta-section-h:meta-section-info': '123',
        'kiss-meta-section-h:meta-info': 'junk',
      },
    });
    const [info, update] = sections(win);
    assertExpanded(info, true);
    assertExpanded(update, false);
    // Task Info is not the last expanded section (Schedule, Apps and
    // Spend follow), so its stored height applies.
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), '123px');
    assert.strictEqual(bodyLayout(el(win, 'meta-apps-list')), 'equal');
    assert.strictEqual(bodyLayout(el(win, 'meta-spend-body')), 'equal');
    // The collapsed Task update section, shown and opened by a click,
    // folds Task Info (one section at a time) and keeps an equal
    // share ('junk' is no stored height).
    update.classList.add('visible');
    click(win, toggleOf(update));
    assertExpanded(update, true);
    assertExpanded(info, false);
    assert.strictEqual(bodyLayout(el(win, 'meta-info-content')), 'equal');
    assertResizer(resizerAfter(info), 'static');
  });

  await test('remote webapp: a task start expands Task Info and Task update and collapses the rest', () => {
    // The user's arrangement before the start: the two per-task
    // sections folded, every other one open.
    const {win} = makePage({
      storage: {
        'kiss-meta-section-collapsed:meta-section-info': '1',
        'kiss-meta-section-collapsed:meta-info': '1',
      },
    });
    const [info, update, ...others] = sections(win);
    assertExpanded(info, false);
    assertExpanded(update, false);
    for (const section of others) assertExpanded(section, true);
    // The page arranges the panel when the daemon's running status
    // starts a task in the chat tab on screen.
    startOnChat(win, 'a1');
    assertExpanded(info, true);
    assertExpanded(update, true);
    for (const section of others) {
      assertExpanded(section, false);
      assert.strictEqual(
        win.localStorage.getItem('kiss-meta-section-collapsed:' + section.id),
        '1',
        section.id + ' collapse stored',
      );
    }
    for (const section of [info, update])
      assert.strictEqual(
        win.localStorage.getItem('kiss-meta-section-collapsed:' + section.id),
        null,
        section.id + ' expansion stored',
      );
    // The arrangement is a starting point: the user unfolds Apps
    // again (which folds the two per-task sections), and a repeated
    // running status (not a start) keeps it.
    const apps = el(win, 'meta-apps');
    click(win, toggleOf(apps));
    assertExpanded(apps, true);
    assertExpanded(info, false);
    assertExpanded(update, false);
    send(win, {type: 'status', running: true, tabId: 'a1', startTs: 1});
    assertExpanded(apps, true);
  });

  await test('remote webapp: the Task update section joins the stack with content and leaves when emptied', () => {
    const wv = makePage();
    const win = wv.win;
    const [info, update] = sections(win);
    hideGlobalSections(wv);
    showTaskUpdate(wv, '<p>progress</p>');
    assert.ok(update.classList.contains('visible'));
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'equal');
    assert.strictEqual(bodyLayout(el(win, 'meta-info-content')), 'equal');
    assertResizer(resizerAfter(info), 'handle');
    assertResizer(resizerAfter(update), 'hidden');

    // Collapse the Task update section: Task Info takes it all and the
    // separator stays but is no longer a handle.
    click(win, toggleOf(update));
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'equal');
    assertResizer(resizerAfter(info), 'static');

    // Open Task update again: that folds Task Info (one section at a
    // time), so the separator under it is static too (nothing above
    // to resize) and Task update takes it all.
    click(win, toggleOf(update));
    assertExpanded(info, false);
    assertResizer(resizerAfter(info), 'static');
    assert.strictEqual(bodyLayout(el(win, 'meta-info-content')), 'equal');
    // Task Info back (folding Task update); then the task ends: the
    // update is emptied and its section leaves.
    click(win, toggleOf(info));
    assertExpanded(info, true);
    assertExpanded(update, false);
    send(win, {type: 'status', running: false});
    assert.ok(
      !update.classList.contains('visible'),
      'emptied update hides its section',
    );
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'equal');
    assertResizer(resizerAfter(info), 'hidden');

    // With the Schedule, Machines, Apps and Spend sections
    // back, the Task update section (still folded) sits between Task
    // Info and Schedule; opening it folds the rest.
    el(win, 'meta-schedule').hidden = false;
    el(win, 'meta-machines').hidden = false;
    el(win, 'meta-apps').hidden = false;
    el(win, 'meta-spend').hidden = false;
    el(win, 'meta-recommendations').hidden = false;
    showTaskUpdate(wv, '<p>progress</p>');
    assert.ok(update.classList.contains('visible'));
    assertExpanded(update, false);
    // The Task Info click folded them too (hidden or not): nothing
    // expanded follows Task Info until a click opens one.
    assertExpanded(el(win, 'meta-schedule'), false);
    assertResizer(resizerAfter(info), 'static');
    assertResizer(resizerAfter(update), 'static');
    click(win, toggleOf(update));
    assertExpanded(update, true);
    assertExpanded(info, false);
    assertExpanded(el(win, 'meta-schedule'), false);
    assert.strictEqual(bodyLayout(el(win, 'meta-info-content')), 'equal');
    assertResizer(resizerAfter(update), 'static');
  });

  await test('remote webapp: arrow keys, double-click and pointer drag on the separator', () => {
    const wv = makePage();
    const win = wv.win;
    const [info] = sections(win);
    hideGlobalSections(wv);
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
    // The last expanded body below a dragged one still fills.
    assert.strictEqual(bodyLayout(el(win, 'meta-info-content')), 'equal');
    assert.strictEqual(r.getAttribute('aria-valuenow'), '0');
    assert.strictEqual(r.getAttribute('aria-valuemax'), '0');
    key(win, r, 'ArrowUp');
    assert.strictEqual(bodyLayout(list), '0px');

    // Double-click restores the equal share.
    r.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
    assert.strictEqual(win.localStorage.getItem(HK), null);
    assert.strictEqual(bodyLayout(list), 'equal');

    // A pointer drag: the body class marks the drag, moves resize,
    // release ends it; a move without a press does nothing.
    pointer(win, r, 'pointermove', {clientY: 40});
    assert.strictEqual(bodyLayout(list), 'equal');
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
    const wv = makePage({extraSection: true});
    const win = wv.win;
    hideGlobalSections(wv);
    const [info, update, , , , , , , , extra] = sections(win);
    assert.strictEqual(extra.id, 'meta-section-extra');
    assertExpanded(extra, true);
    // No task: Task Info and Extra are shown, sharing equally.
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'equal');
    assert.strictEqual(bodyLayout(el(win, 'meta-extra-body')), 'equal');
    assertResizer(resizerAfter(info), 'handle');
    assertResizer(resizerAfter(update), 'hidden');
    assertResizer(resizerAfter(extra), 'hidden');

    showTaskUpdate(wv, '<p>progress</p>');
    assert.strictEqual(bodyLayout(el(win, 'meta-info-content')), 'equal');
    assert.strictEqual(bodyLayout(el(win, 'meta-extra-body')), 'equal');
    assertResizer(resizerAfter(info), 'handle');
    assertResizer(resizerAfter(update), 'handle');
    assertResizer(resizerAfter(extra), 'hidden');

    // Collapse the middle one: the separator above it is still a
    // handle (Extra below is expanded), the one below it is static.
    click(win, toggleOf(update));
    assertResizer(resizerAfter(info), 'handle');
    assertResizer(resizerAfter(update), 'static');
    // Opening it again folds the other two: both separators static.
    click(win, toggleOf(update));
    assertExpanded(extra, false);
    assertResizer(resizerAfter(info), 'static');
    assertResizer(resizerAfter(update), 'static');
    click(win, toggleOf(extra));
    assertExpanded(extra, true);
    assertExpanded(update, false);

    // A section hidden with the `hidden` attribute is out of the stack
    // (its owner re-applies the layout; the panel's resize observer
    // does here).
    const relayout = () =>
      wv.observers.find(o => o.target === el(win, 'meta-panel')).fn([]);
    extra.hidden = true;
    relayout();
    assertResizer(resizerAfter(update), 'hidden');
    assertResizer(resizerAfter(extra), 'hidden');
    extra.hidden = false;
    relayout();
    assert.strictEqual(bodyLayout(el(win, 'meta-extra-body')), 'equal');
    assertResizer(resizerAfter(update), 'static');
  });

  await test('a drag pins the expanded bodies above the moved boundary, and only those', () => {
    const HK = 'kiss-meta-section-h:';
    // No task: Task Info, Schedule, Apps and Spend share the panel equally.
    let win = makePage().win;
    let schedule = el(win, 'meta-schedule');
    const r = resizerAfter(schedule);
    assertResizer(r, 'handle');
    // Dragging the Schedule / Apps boundary pins Task Info (jsdom has no
    // layout, so at 0px) and fixes Schedule; Apps keeps its share and
    // Spend, last, fills.  The hidden Task update section above is not
    // pinned.
    pointer(win, r, 'pointerdown');
    pointer(win, r, 'pointermove', {clientY: 40});
    pointer(win, r, 'pointerup');
    assert.strictEqual(win.localStorage.getItem(HK + 'meta-section-info'), '0');
    assert.strictEqual(win.localStorage.getItem(HK + 'meta-schedule'), '0');
    assert.strictEqual(win.localStorage.getItem(HK + 'meta-info'), null);
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), '0px');
    assert.strictEqual(bodyLayout(el(win, 'meta-schedule-list')), '0px');
    assert.strictEqual(bodyLayout(el(win, 'meta-apps-list')), 'equal');
    assert.strictEqual(bodyLayout(el(win, 'meta-spend-body')), 'equal');
    // Double-click returns Schedule to an equal share; the pinned Task
    // Info keeps its height.
    r.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
    assert.strictEqual(bodyLayout(el(win, 'meta-schedule-list')), 'equal');
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), '0px');

    // A body above that already has a dragged height keeps it.
    win = makePage({
      storage: {[HK + 'meta-section-info']: '123'},
    }).win;
    schedule = el(win, 'meta-schedule');
    key(win, resizerAfter(schedule), 'ArrowDown');
    assert.strictEqual(
      win.localStorage.getItem(HK + 'meta-section-info'),
      '123',
    );
    assert.strictEqual(win.localStorage.getItem(HK + 'meta-schedule'), '0');

    // A collapsed body above is not pinned either.
    win = makePage().win;
    schedule = el(win, 'meta-schedule');
    click(win, toggleOf(el(win, 'meta-section-info')));
    key(win, resizerAfter(schedule), 'ArrowDown');
    assert.strictEqual(
      win.localStorage.getItem(HK + 'meta-section-info'),
      null,
    );
    assert.strictEqual(win.localStorage.getItem(HK + 'meta-schedule'), '0');
    assert.strictEqual(bodyLayout(el(win, 'meta-spend-body')), 'equal');
    // Collapsing the dragged body leaves nothing dragged above the last:
    // it returns to a plain share.
    click(win, toggleOf(schedule));
    assert.strictEqual(bodyLayout(el(win, 'meta-spend-body')), 'equal');
  });

  await test('opaque origin (no localStorage): sections still work, choices last for the page', () => {
    const wv = makePage({url: null});
    const win = wv.win;
    assert.throws(
      () => win.localStorage,
      'jsdom must refuse localStorage on an opaque origin',
    );
    const [info, update] = sections(win);
    assertExpanded(info, true);
    assertExpanded(update, true);
    // Every section starts expanded, so the separator under Task Info
    // is a handle: a height is set and reset without storage.
    const r = resizerAfter(info);
    assertResizer(r, 'handle');
    key(win, r, 'ArrowDown');
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), '0px');
    r.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'equal');
    update.classList.add('visible');
    click(win, toggleOf(update));
    assertExpanded(update, false);
    assertExpanded(info, true);
    click(win, toggleOf(update));
    assertExpanded(update, true);
    assertExpanded(info, false, 'opening Task update folded Task Info');
    click(win, toggleOf(info));
    assertExpanded(info, true);
    assertExpanded(update, false);
  });

  // ---- First load after an install or an update ----
  const CK = 'kiss-meta-section-collapsed:';
  const VK = 'kiss-meta-sections-version';
  const RECS = 'meta-recommendations';

  /** Only Recommendations is expanded; every other section is folded. */
  function assertFreshLayout(win) {
    for (const section of sections(win))
      assertExpanded(section, section.id === RECS);
  }

  await test('remote webapp: the first load of a version expands only Recommendations, whatever was stored', () => {
    // A profile that had Recommendations folded and Task Info open.
    let win = makePage({
      storage: {[CK + RECS]: '1', [CK + 'meta-info']: '1'},
      tips: {tips: [], show: false, version: '2026.10.1'},
    }).win;
    assertFreshLayout(win);
    assert.strictEqual(win.localStorage.getItem(VK), '2026.10.1');
    // The reset persists like clicks on the chevrons.
    assert.strictEqual(win.localStorage.getItem(CK + RECS), null);
    assert.strictEqual(win.localStorage.getItem(CK + 'meta-section-info'), '1');
    assert.strictEqual(win.localStorage.getItem(CK + 'meta-spend'), '1');
    // The user opens Task Info (which folds Recommendations) ...
    click(win, toggleOf(el(win, 'meta-section-info')));
    assertExpanded(el(win, 'meta-section-info'), true);
    assertExpanded(el(win, RECS), false);
    const kept = {};
    for (let i = 0; i < win.localStorage.length; i++) {
      const k = win.localStorage.key(i);
      kept[k] = win.localStorage.getItem(k);
    }
    // ... and the next load of the SAME version keeps that choice.
    win = makePage({
      storage: kept,
      tips: {tips: [], show: false, version: '2026.10.1'},
    }).win;
    assertExpanded(el(win, 'meta-section-info'), true);
    assertExpanded(el(win, RECS), false);
    // An update: the fresh layout again, once.
    win = makePage({
      storage: kept,
      tips: {tips: [], show: false, version: '2026.10.2'},
    }).win;
    assertFreshLayout(win);
    assert.strictEqual(win.localStorage.getItem(VK), '2026.10.2');
  });

  await test('no version (an empty bootstrap version, or none at all): nothing is reset', () => {
    let win = makePage({
      storage: {[CK + RECS]: '1'},
      tips: {tips: [], show: false, version: ''},
    }).win;
    assertExpanded(el(win, RECS), false);
    assertExpanded(el(win, 'meta-section-info'), true);
    assert.strictEqual(win.localStorage.getItem(VK), null);
    win = makePage({storage: {[CK + RECS]: '1'}}).win;
    assertExpanded(el(win, RECS), false);
    assert.strictEqual(win.localStorage.getItem(VK), null);
  });

  await test('a page booted without a version learns it from tipsData and resets once', () => {
    {
      const win = makePage({
        storage: {[CK + RECS]: '1'},
        tips: {tips: [], show: false, version: ''},
      }).win;
      assertExpanded(el(win, RECS), false);
      send(win, {
        type: 'tipsData',
        tips: [],
        show: false,
        version: '2026.10.1',
      });
      assertFreshLayout(win);
      assert.strictEqual(win.localStorage.getItem(VK), '2026.10.1');
      // The user's later choice survives the tipsData a reconnect
      // re-sends.
      click(win, toggleOf(el(win, 'meta-spend')));
      assertExpanded(el(win, 'meta-spend'), true);
      assertExpanded(el(win, RECS), false);
      send(win, {
        type: 'tipsData',
        tips: [],
        show: false,
        version: '2026.10.1',
      });
      assertExpanded(el(win, 'meta-spend'), true);
      assertExpanded(el(win, RECS), false);
      // A tipsData without a version string changes nothing.
      send(win, {type: 'tipsData', tips: [], show: false});
      assertExpanded(el(win, 'meta-spend'), true);
    }
  });

  await test('opaque origin (no localStorage): a version makes every load a first one', () => {
    const win = makePage({
      url: null,
      tips: {tips: [], show: false, version: '2026.10.1'},
    }).win;
    assertFreshLayout(win);
    click(win, toggleOf(el(win, 'meta-spend')));
    assertExpanded(el(win, 'meta-spend'), true);
    // The page itself remembers the version: the tipsData of the
    // connect (and of every reconnect) leaves its choices alone.
    send(win, {type: 'tipsData', tips: [], show: false, version: '2026.10.1'});
    assertExpanded(el(win, 'meta-spend'), true);
    assertExpanded(el(win, RECS), false);
    // A new version (the daemon updated under an open page) resets.
    send(win, {type: 'tipsData', tips: [], show: false, version: '2026.10.2'});
    assertFreshLayout(win);
  });

  await test('the equal share is published as --meta-body-share and re-measured when the panel resizes', () => {
    const wv = makePage();
    const win = wv.win;
    const panel = el(win, 'meta-panel');
    // jsdom has no layout: every body measures 0, so the share is 0px
    // (and every minimum with it); a real browser measures the free
    // height (test_meta_panel_sections.py checks the numbers).
    assert.strictEqual(
      panel.style.getPropertyValue('--meta-body-share'),
      '0.00px',
    );
    const observer = wv.observers.find(o => o.target === panel);
    assert.ok(observer, 'setupMetaSections observes #meta-panel');
    panel.style.setProperty('--meta-body-share', '99px');
    observer.fn([]);
    assert.strictEqual(
      panel.style.getPropertyValue('--meta-body-share'),
      '0.00px',
    );
    // Layout changes re-measure too.
    panel.style.setProperty('--meta-body-share', '99px');
    click(win, toggleOf(sections(win)[0]));
    assert.strictEqual(
      panel.style.getPropertyValue('--meta-body-share'),
      '0.00px',
    );
  });

  for (const win of pages) win.close();
  if (failures) {
    console.log(`\n${failures} test(s) failed`);
    process.exitCode = 1;
    return;
  }
  console.log('\nall metaPanelSections tests passed');
}

main();
