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
// * a .meta-section-pinned section (Recommendations, hidden while it
//   has no task) sits on the bottom strip: its body takes no share, no
//   separator beside it is a handle, and only its own chevron folds or
//   opens it, whatever the other sections do (the first load of a
//   version opens it);
// * the Task update section joins the stack only once it has content
//   (the `visible` class), and leaves it again when emptied;
// * the resizer's keyboard, double-click and pointer paths, and a
//   window whose localStorage is unavailable (opaque origin), where
//   every choice lasts for the page only.
//
// Geometry (pixel heights, scrolling) needs a layout engine and is
// covered by tests/webapp/test_meta_panel_sections.py; here the resize
// arithmetic runs on bodies given a height (layOutSectionBodies).

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  click,
  stubRaf,
  fakeMatchMedia,
  flushMicrotasks,
} = require('./simplify2_harness');
const {makeRunner, key} = require('./ui_antipattern_harness');
const {pointer} = require('./panel_resize_helpers');

const {test, report} = makeRunner();

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

/**
 * Lay out the section bodies, which jsdom does not: a body that shares
 * the panel is *height* px tall and a dragged one as tall as its fixed
 * flex-basis, as in a panel with room for both.  Every other element
 * keeps jsdom's empty geometry.
 */
function layOutSectionBodies(win, height) {
  const proto = win.Element.prototype;
  const jsdomRect = proto.getBoundingClientRect;
  proto.getBoundingClientRect = function () {
    if (!this.classList.contains('meta-section-body'))
      return jsdomRect.call(this);
    const h =
      this.style.flexGrow === '0' ? parseFloat(this.style.flexBasis) : height;
    return {top: 0, bottom: h, height: h, left: 0, right: 300, width: 300};
  };
}

/**
 * Boot a chat page.  *storage* pre-seeds localStorage before main.js
 * runs (so a stored collapse state or height is restored at boot);
 * `url: null` opens an OPAQUE origin, where jsdom's localStorage throws
 * SecurityError exactly as a browser's does; *extraSection* adds a
 * third section the way chat.html documents it; *bodyHeight* lays the
 * section bodies out that tall (layOutSectionBodies) instead of
 * leaving them jsdom's 0px.
 */
function makePage(opts) {
  const {
    storage = {},
    url = 'https://localhost/',
    extraSection = false,
    tips = undefined,
    bodyHeight = 0,
  } = opts || {};
  // jsdom has no ResizeObserver: record the observers so a test can
  // fire a panel resize by hand.
  const observers = [];
  function beforeScripts(win) {
    pages.push(win);
    if (extraSection)
      win.document
        .getElementById('meta-resizer')
        .insertAdjacentHTML('beforebegin', EXTRA_SECTION);
    if (url) {
      for (const [k, v] of Object.entries(storage))
        win.localStorage.setItem(k, v);
    }
    stubRaf(win);
    win.ResizeObserver = class {
      constructor(fn) {
        this.fn = fn;
        observers.push(this);
      }
      observe(target) {
        this.target = target;
      }
    };
    fakeMatchMedia(win, true);
    if (bodyHeight) layOutSectionBodies(win, bodyHeight);
    // The page embeds the daemon's tips bootstrap (with the running
    // version) before main.js runs; *tips* stands in for it.
    if (tips !== undefined) win.__TIPS__ = tips;
  }
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    url,
    beforeScripts,
    extraScripts: ['marked.min.js'],
    sourceUrl: 'metasections-main.js',
  });
  return {win, posted, observers};
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

/** The sections that fold together: every one but the pinned Recommendations. */
function foldable(win) {
  return sections(win).filter(
    s => !s.classList.contains('meta-section-pinned'),
  );
}

/** A section's state: its class and, where it has a toggle, what that says. */
function assertExpanded(section, expanded) {
  assert.strictEqual(
    section.classList.contains('collapsed'),
    !expanded,
    section.id + ' collapsed class',
  );
  const toggle = toggleOf(section);
  if (toggle)
    assert.strictEqual(
      toggle.getAttribute('aria-expanded'),
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
 * Machines, Skills, Connectors, Spend) sections out of the stack (the
 * `hidden` attribute, which the page watches to re-apply the layout),
 * leaving only the two per-task sections: the page itself keeps
 * Recommendations hidden while it has no task.
 */
async function hideGlobalSections(win) {
  for (const id of [
    'meta-explorer',
    'meta-scm',
    'meta-schedule',
    'meta-machines',
    'meta-skills',
    'meta-apps',
    'meta-spend',
  ])
    el(win, id).hidden = true;
  await flushMicrotasks();
}

async function main() {
  await test('remote webapp: every section starts expanded, the Task update and Recommendations ones hidden', () => {
    const {win} = makePage();
    const [
      info,
      update,
      explorer,
      scm,
      schedule,
      machines,
      skills,
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
    assert.strictEqual(skills.id, 'meta-skills');
    assert.strictEqual(apps.id, 'meta-apps');
    assert.strictEqual(spend.id, 'meta-spend');
    assert.strictEqual(recommendations.id, 'meta-recommendations');
    assert.strictEqual(toggleOf(info).textContent, 'Task Info');
    assert.strictEqual(toggleOf(update).textContent, 'Task Progress');
    assert.strictEqual(toggleOf(explorer).textContent, 'Explorer');
    // The header count (uncommitted files) is empty until a git status.
    assert.strictEqual(toggleOf(scm).textContent, 'Source Control');
    assert.strictEqual(toggleOf(schedule).textContent, 'Schedule');
    assert.strictEqual(toggleOf(machines).textContent, 'Machines');
    assert.strictEqual(toggleOf(skills).textContent, 'Skills');
    // The section keeps the id it had when it was headed "Apps".
    assert.strictEqual(toggleOf(apps).textContent, 'Connectors');
    assert.strictEqual(apps.getAttribute('aria-label'), 'Connectors');
    assert.strictEqual(
      resizerAfter(apps).getAttribute('aria-label'),
      'Resize the Connectors section',
    );
    assert.strictEqual(toggleOf(spend).textContent, 'Spend');
    // Recommendations is pinned to the strip: its chevron is the first
    // control of the section (its top left) and names the body it
    // folds.  The section is hidden until it has a task to show.
    assert.strictEqual(toggleOf(recommendations).textContent, 'Recommendations');
    assert.strictEqual(
      recommendations.firstElementChild.firstElementChild,
      toggleOf(recommendations),
    );
    for (const control of [
      toggleOf(recommendations),
      resizerAfter(recommendations),
    ])
      assert.strictEqual(
        control.getAttribute('aria-controls'),
        'meta-recommendations-body',
      );
    assert.strictEqual(recommendations.hidden, true, 'no task: hidden');
    assert.deepStrictEqual(foldable(win), sections(win).slice(0, -1));
    for (const section of sections(win)) assertExpanded(section, true);
    assert.ok(
      !update.classList.contains('visible'),
      'no task: Task update section hidden',
    );
    // The workspace sections (Explorer, Source Control) are shown.
    assert.strictEqual(explorer.hidden, false, 'Explorer shown');
    assert.strictEqual(scm.hidden, false, 'Source Control shown');
    // Nothing is dragged: every expanded body that folds takes an
    // equal share.  The pinned body is sized by main.css alone.
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'equal');
    // The Schedule and Machines bodies scroll an example line with the
    // list under it: the toggle and the separator name the body.
    for (const section of [schedule, machines]) {
      const body = el(win, section.id + '-body');
      assert.strictEqual(body.lastElementChild.id, section.id + '-list');
      for (const control of [toggleOf(section), resizerAfter(section)])
        assert.strictEqual(control.getAttribute('aria-controls'), body.id);
      assert.strictEqual(bodyLayout(body), 'equal');
    }
    // The Skills body scrolls under its search row, which keeps its
    // own height.
    assert.strictEqual(
      toggleOf(skills).getAttribute('aria-controls'),
      'meta-skills-body',
    );
    assert.strictEqual(bodyLayout(el(win, 'meta-skills-body')), 'equal');
    assert.strictEqual(bodyLayout(el(win, 'meta-apps-list')), 'equal');
    assert.strictEqual(bodyLayout(el(win, 'meta-spend-body')), 'equal');
    assert.strictEqual(el(win, 'meta-recommendations-body').style.flex, '');
    assertResizer(resizerAfter(info), 'handle');
    assertResizer(resizerAfter(update), 'hidden');
    assertResizer(resizerAfter(schedule), 'handle');
    assertResizer(resizerAfter(machines), 'handle');
    assertResizer(resizerAfter(skills), 'handle');
    assertResizer(resizerAfter(apps), 'handle');
    // Nothing shown follows Spend while Recommendations has no task.
    assertResizer(resizerAfter(spend), 'hidden');
    assertResizer(resizerAfter(recommendations), 'hidden');
    // A task brings the pinned section on show: Spend, the last that
    // shares, has no handle under it, and no separator follows
    // Recommendations, the last shown section.
    send(win, {
      type: 'recommendations',
      generatedAt: 1,
      tasks: ['Fix the parser'],
    });
    assert.strictEqual(recommendations.hidden, false);
    assert.strictEqual(el(win, 'meta-recommendations-body').style.flex, '');
    assertResizer(resizerAfter(spend), 'static');
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
    const [info, ...others] = foldable(win);
    const pinned = el(win, 'meta-recommendations');
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
    // ... but for the pinned section, which only its own chevron folds.
    assertExpanded(pinned, true);
    assert.strictEqual(stored(pinned.id), null);
    // Nothing that shares follows Task Info: its separator is static.
    assertResizer(resizerAfter(info), 'static');
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), 'equal');
    // Opening Connectors folds Task Info in turn; a click on the open
    // Connectors just folds it.
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
    for (const section of foldable(win)) assertExpanded(section, false);
    assertExpanded(pinned, true);
    // The pinned section's chevron folds it and opens it again, alone:
    // the open Task Info stays open through both.
    click(win, toggleOf(info));
    click(win, toggleOf(pinned));
    assertExpanded(pinned, false);
    assert.strictEqual(stored(pinned.id), '1');
    assertExpanded(info, true);
    // Another section opening, or a task starting, leaves it folded.
    click(win, toggleOf(apps));
    startOnChat(win, 'a1');
    assertExpanded(pinned, false);
    assertExpanded(info, true);
    click(win, toggleOf(pinned));
    assertExpanded(pinned, true);
    assert.strictEqual(stored(pinned.id), null);
    assertExpanded(info, true);
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
    // Task Info is not the last expanded section (Schedule, Connectors
    // and Spend follow), so its stored height applies.
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
    // A static separator ignores a double-click: the stored height stays.
    resizerAfter(info).dispatchEvent(
      new win.MouseEvent('dblclick', {bubbles: true}),
    );
    assert.strictEqual(
      win.localStorage.getItem('kiss-meta-section-h:meta-section-info'),
      '123',
    );
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
    const [info, update, ...others] = foldable(win);
    assertExpanded(info, false);
    assertExpanded(update, false);
    for (const section of others) assertExpanded(section, true);
    const schedule = el(win, 'meta-schedule');
    assertResizer(resizerAfter(schedule), 'handle');
    // The page arranges the panel when the daemon's running status
    // starts a task in the chat tab on screen.
    startOnChat(win, 'a1');
    assertExpanded(info, true);
    assertExpanded(update, true);
    // The separators follow: nothing is expanded below Schedule now.
    assertResizer(resizerAfter(schedule), 'static');
    for (const section of others) {
      assertExpanded(section, false);
      assert.strictEqual(
        win.localStorage.getItem('kiss-meta-section-collapsed:' + section.id),
        '1',
        section.id + ' collapse stored',
      );
    }
    // The pinned Recommendations stays open under the run's sections.
    for (const section of [info, update, el(win, 'meta-recommendations')]) {
      assertExpanded(section, true);
      assert.strictEqual(
        win.localStorage.getItem('kiss-meta-section-collapsed:' + section.id),
        null,
        section.id + ' expansion stored',
      );
    }
    // The arrangement is a starting point: the user unfolds Connectors
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

  await test('remote webapp: the Task update section joins the stack with content and leaves when emptied', async () => {
    const wv = makePage();
    const win = wv.win;
    const [info, update] = sections(win);
    await hideGlobalSections(win);
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

    // With the Schedule, Machines, Connectors and Spend sections
    // back, and Recommendations with a task to show, the Task update
    // section (still folded) sits between Task Info and Schedule;
    // opening it folds the rest.
    el(win, 'meta-schedule').hidden = false;
    el(win, 'meta-machines').hidden = false;
    el(win, 'meta-apps').hidden = false;
    el(win, 'meta-spend').hidden = false;
    send(win, {
      type: 'recommendations',
      generatedAt: 1,
      tasks: ['Fix the parser'],
    });
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

  await test('remote webapp: arrow keys, double-click and pointer drag on the separator', async () => {
    // Each sharing body is laid out 200px tall, so a move shows.
    const wv = makePage({bodyHeight: 200});
    const win = wv.win;
    const [info] = sections(win);
    await hideGlobalSections(win);
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

    // Arrow keys move the boundary by 16px, fixing the height, and
    // persist it (it can take the 200px of the body below at most);
    // other keys are ignored.  (key() returns false when the page
    // consumed the key.)
    assert.strictEqual(key(win, r, 'Tab'), true, 'Tab keeps its default');
    assert.strictEqual(win.localStorage.getItem(HK), null);
    assert.strictEqual(
      key(win, r, 'ArrowDown'),
      false,
      'the arrow is consumed',
    );
    assert.strictEqual(win.localStorage.getItem(HK), '216');
    assert.strictEqual(bodyLayout(list), '216px');
    // The last expanded body below a dragged one still fills.
    assert.strictEqual(bodyLayout(el(win, 'meta-info-content')), 'equal');
    assert.strictEqual(r.getAttribute('aria-valuenow'), '216');
    assert.strictEqual(r.getAttribute('aria-valuemax'), '400');
    assert.strictEqual(key(win, r, 'ArrowUp'), false, 'the arrow is consumed');
    assert.strictEqual(win.localStorage.getItem(HK), '200');
    assert.strictEqual(bodyLayout(list), '200px');
    assert.strictEqual(r.getAttribute('aria-valuenow'), '200');

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
    // jsdom has no pointer capture: record what the page asks for.
    const captured = [];
    r.setPointerCapture = id => captured.push(id);
    const press = pointer(win, r, 'pointerdown', {pointerId: 7, clientY: 100});
    assert.ok(press.defaultPrevented, 'the press is consumed');
    assert.deepStrictEqual(captured, [7], 'the drag captures its pointer');
    assert.ok(win.document.body.classList.contains('meta-section-resizing'));
    // 40px down from the press: the body is 40px taller.
    pointer(win, r, 'pointermove', {clientY: 140});
    assert.strictEqual(bodyLayout(list), '240px');
    assert.strictEqual(r.getAttribute('aria-valuenow'), '240');
    // The drag is clamped: going down at the 400px the two bodies have
    // between them, going up at 0px.
    pointer(win, r, 'pointermove', {clientY: 1000});
    assert.strictEqual(bodyLayout(list), '400px');
    pointer(win, r, 'pointermove', {clientY: -1000});
    assert.strictEqual(bodyLayout(list), '0px');
    // Released 40px below the press: that height is the one kept.
    pointer(win, r, 'pointermove', {clientY: 140});
    pointer(win, r, 'pointerup');
    assert.ok(!win.document.body.classList.contains('meta-section-resizing'));
    pointer(win, r, 'pointerup');
    assert.strictEqual(win.localStorage.getItem(HK), '240');
    // pointercancel ends a drag too.
    pointer(win, r, 'pointerdown');
    pointer(win, r, 'pointercancel');
    assert.ok(!win.document.body.classList.contains('meta-section-resizing'));
    // A move after the drag ended resizes nothing.
    r.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
    pointer(win, r, 'pointermove', {clientY: 40});
    assert.strictEqual(bodyLayout(list), 'equal');
  });

  await test('a third section added per the chat.html recipe joins the stack; `hidden` removes it', async () => {
    const wv = makePage({extraSection: true});
    const win = wv.win;
    await hideGlobalSections(win);
    const [info, update, , , , , , , , , extra] = sections(win);
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

    // A section hidden with the `hidden` attribute is out of the stack:
    // the page watches the attribute and lays the panel out again.
    extra.hidden = true;
    await flushMicrotasks();
    assertResizer(resizerAfter(update), 'hidden');
    assertResizer(resizerAfter(extra), 'hidden');
    extra.hidden = false;
    await flushMicrotasks();
    assert.strictEqual(bodyLayout(el(win, 'meta-extra-body')), 'equal');
    assertResizer(resizerAfter(update), 'static');
  });

  await test('a drag pins the expanded bodies above the moved boundary, and only those', () => {
    const HK = 'kiss-meta-section-h:';
    // No task: Task Info, Schedule, Connectors and Spend share the panel
    // equally, each body laid out 200px tall.
    let win = makePage({bodyHeight: 200}).win;
    let schedule = el(win, 'meta-schedule');
    const r = resizerAfter(schedule);
    assertResizer(r, 'handle');
    // Dragging the boundary under Schedule 40px down pins Task Info at
    // the height it has and fixes Schedule 40px taller; Connectors keeps
    // its share and Spend, last, fills.  The hidden Task update section
    // above is not pinned.
    pointer(win, r, 'pointerdown');
    pointer(win, r, 'pointermove', {clientY: 40});
    pointer(win, r, 'pointerup');
    assert.strictEqual(
      win.localStorage.getItem(HK + 'meta-section-info'),
      '200',
    );
    assert.strictEqual(win.localStorage.getItem(HK + 'meta-schedule'), '240');
    assert.strictEqual(win.localStorage.getItem(HK + 'meta-info'), null);
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), '200px');
    assert.strictEqual(bodyLayout(el(win, 'meta-schedule-body')), '240px');
    assert.strictEqual(bodyLayout(el(win, 'meta-apps-list')), 'equal');
    assert.strictEqual(bodyLayout(el(win, 'meta-spend-body')), 'equal');
    // Double-click returns Schedule to an equal share; the pinned Task
    // Info keeps its height.
    r.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
    assert.strictEqual(bodyLayout(el(win, 'meta-schedule-body')), 'equal');
    assert.strictEqual(bodyLayout(el(win, 'meta-list')), '200px');

    // A body above that already has a dragged height keeps it (jsdom
    // has no layout: every rect is 0, so Schedule's clamp lands on 0).
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
    assertExpanded(info, false);
    click(win, toggleOf(info));
    assertExpanded(info, true);
    assertExpanded(update, false);
  });

  // ---- First load after an install or an update ----
  const CK = 'kiss-meta-section-collapsed:';
  const VK = 'kiss-meta-sections-version';
  const RECS = 'meta-recommendations';

  /**
   * Only the pinned Recommendations is open; every section that folds
   * is folded, so no separator is a handle.
   */
  function assertFreshLayout(win) {
    for (const section of sections(win))
      assertExpanded(section, section.id === RECS);
    assertResizer(resizerAfter(el(win, 'meta-schedule')), 'static');
  }

  await test('remote webapp: the first load of a version folds every section but the pinned Recommendations, whatever was stored', () => {
    // A profile in which Recommendations was folded and Task Info
    // open.
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
    // The user opens Task Info (Recommendations stays under it) ...
    click(win, toggleOf(el(win, 'meta-section-info')));
    assertExpanded(el(win, 'meta-section-info'), true);
    assertExpanded(el(win, RECS), true);
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
    assertExpanded(el(win, 'meta-spend'), false);
    // An update: the fresh layout again, once, a Recommendations the
    // user folded meanwhile open again.
    kept[CK + RECS] = '1';
    win = makePage({
      storage: kept,
      tips: {tips: [], show: false, version: '2026.10.2'},
    }).win;
    assertFreshLayout(win);
    assert.strictEqual(win.localStorage.getItem(VK), '2026.10.2');
  });

  await test('no version (an empty bootstrap version, or none at all): nothing is reset', () => {
    // A folded Spend and a folded Recommendations: both are the
    // user's choices, and both are kept.
    const storage = {[CK + 'meta-spend']: '1', [CK + RECS]: '1'};
    let win = makePage({
      storage,
      tips: {tips: [], show: false, version: ''},
    }).win;
    assertExpanded(el(win, 'meta-spend'), false);
    assertExpanded(el(win, 'meta-section-info'), true);
    assertExpanded(el(win, RECS), false);
    assert.strictEqual(win.localStorage.getItem(CK + RECS), '1');
    assert.strictEqual(win.localStorage.getItem(VK), null);
    win = makePage({storage}).win;
    assertExpanded(el(win, 'meta-spend'), false);
    assert.strictEqual(win.localStorage.getItem(VK), null);
  });

  await test('a page booted without a version learns it from tipsData and resets once', () => {
    {
      const win = makePage({
        storage: {[CK + 'meta-spend']: '1'},
        tips: {tips: [], show: false, version: ''},
      }).win;
      assertExpanded(el(win, 'meta-section-info'), true);
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
      send(win, {
        type: 'tipsData',
        tips: [],
        show: false,
        version: '2026.10.1',
      });
      assertExpanded(el(win, 'meta-spend'), true);
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
  report('metaPanelSections');
}

main();
