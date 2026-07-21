// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests: DRAWER-style widgets for the pinned task panel
// (#task-panel) and the input textbox + buttons panel (#input-area) in
// the chat webview — both the VS Code extension webview and the remote
// web app (body.remote-chat).
//
// Feature under test:
//
//   1. #task-panel carries a drawer toggle (#task-panel-drawer-btn).
//      Collapsing tucks the panel into a slim one-line drawer: the
//      task text is clamped to a single ellipsized line.  Expanding
//      restores the full panel showing the entire task text.
//   2. #input-area carries a drawer toggle (#input-drawer-btn).
//      Collapsing hides EVERY child of #input-area except the handle
//      itself — the composer (#input-container), the #autocomplete
//      popover and any merge/worktree action bar — even when those
//      children carry inline display styles.  Expanding restores them.
//   3. The freed space goes to the chat events area: #output is the
//      flex:1 child of the #app column, so it absorbs whatever height
//      the collapsed drawers give up.
//   4. Both toggles update aria-expanded + aria-label, and the
//      collapsed/expanded state is persisted via vscode.setState and
//      restored when the webview is disposed and re-opened.
//   5. Everything behaves identically in the remote web app, where
//      remote-codex.css is layered over main.css.
//
// This drives the production chat.html + panelCopy.js + main.js in
// jsdom with the REAL main.css (and remote-codex.css for remote mode)
// attached — jsdom 29 resolves the stylesheet cascade for
// getComputedStyle, so the display/clamping assertions below exercise
// the real CSS, not a re-implementation.  jsdom performs no layout,
// so the "space goes to the events area" invariant is asserted
// through the mechanism that produces it: #output keeps flex-grow 1
// while the collapsed drawers hide their contents.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/drawerPanels.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// Persisted webview state shared across webview instances — the
// vscode.getState()/setState() blob VS Code keeps alive while a view
// is closed and hands back on reopen (the remote web app's WS shim
// exposes the same API).
let persistedState;

/**
 * Build a jsdom window running the production chat webview with the
 * real stylesheets attached.
 *
 * @param {object} [opts]
 * @param {boolean} [opts.remote=false] add class="remote-chat" to body
 *     and layer remote-codex.css over main.css (the web app cascade).
 * @param {boolean} [opts.stripDrawerButtons=false] remove the drawer
 *     toggle buttons from the HTML before boot (an embedder serving a
 *     stale cached chat.html) — main.js must still boot cleanly.
 * @param {string} [opts.userAgent] navigator.userAgent for the window
 *     (jsdom's userAgent option) — simulates a phone/tablet browser.
 * @param {object} [opts.userAgentData] value installed as
 *     navigator.userAgentData before main.js boots (UA-CH hint, e.g.
 *     {mobile: true} as Chrome for Android exposes it).
 * @param {number} [opts.maxTouchPoints] value installed as
 *     navigator.maxTouchPoints before main.js boots (iPadOS Safari
 *     masquerades as "Macintosh" but reports a multi-touch screen).
 * @returns {{win: object, posted: Array}}
 */
function makeWebview(opts) {
  const {
    remote = false,
    stripDrawerButtons = false,
    userAgent,
    userAgentData,
    maxTouchPoints,
  } = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  if (remote) html = html.replace('<body', '<body class="remote-chat"');
  if (stripDrawerButtons) {
    html = html.replace(
      /<button id="task-panel-drawer-btn"[\s\S]*?<\/button>/,
      '',
    );
    html = html.replace(/<button id="input-drawer-btn"[\s\S]*?<\/button>/, '');
  }

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  // Overridden directly on the navigator (rather than through jsdom's
  // ``resources: {userAgent}`` setting) so the harness never enables
  // automatic subresource fetching.
  if (userAgent) {
    Object.defineProperty(win.navigator, 'userAgent', {
      value: userAgent,
      configurable: true,
    });
  }
  if (userAgentData !== undefined) {
    Object.defineProperty(win.navigator, 'userAgentData', {
      value: userAgentData,
      configurable: true,
    });
  }
  if (maxTouchPoints !== undefined) {
    Object.defineProperty(win.navigator, 'maxTouchPoints', {
      value: maxTouchPoints,
      configurable: true,
    });
  }

  // Attach the REAL stylesheet cascade so getComputedStyle resolves
  // the drawer rules exactly as a browser would.
  const style = win.document.createElement('style');
  style.textContent = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  win.document.head.appendChild(style);
  if (remote) {
    const remoteStyle = win.document.createElement('style');
    remoteStyle.textContent = fs.readFileSync(
      path.join(MEDIA, 'remote-codex.css'),
      'utf8',
    );
    win.document.head.appendChild(remoteStyle);
  }

  const posted = [];
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => persistedState,
      setState: s => {
        persistedState = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  // The sourceURL pragma names this eval instance in V8 coverage
  // output so drawerPanels.coverage.js can locate main.js and enforce
  // 100% line coverage of the drawer-coverage regions.
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=drawer-main.js',
  );
  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Click element *id* like a user would. */
function click(win, id) {
  const el = win.document.getElementById(id);
  assert.ok(el, `element #${id} must exist`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/** getComputedStyle shorthand. */
function cs(win, id) {
  return win.getComputedStyle(win.document.getElementById(id));
}

/** Make the task panel visible by replaying a task, as the daemon does. */
function showTaskPanel(win, posted) {
  const ready = posted.find(m => m.type === 'ready');
  send(win, {
    type: 'task_events',
    events: [],
    task: 'refactor the parser and keep the CLI flags backward compatible',
    tabId: ready.tabId,
    chat_id: 'chat-drawer',
  });
  assert.ok(
    win.document.getElementById('task-panel').classList.contains('visible'),
    'task panel must be visible after a task replay',
  );
}

/** Assert drawer button state: aria-expanded + aria-label action word. */
function assertBtnState(win, id, expanded) {
  const btn = win.document.getElementById(id);
  assert.strictEqual(
    btn.getAttribute('aria-expanded'),
    expanded ? 'true' : 'false',
    `#${id} aria-expanded must be ${expanded}`,
  );
  const label = btn.getAttribute('aria-label') || '';
  const want = expanded ? /^Collapse / : /^Expand /;
  assert.ok(
    want.test(label),
    `#${id} aria-label must start with "${expanded ? 'Collapse' : 'Expand'}" (got "${label}")`,
  );
}

// ── 1. Defaults: both drawers open, events area is the flex child ───
function testDefaultsExpanded() {
  persistedState = undefined;
  const {win, posted} = makeWebview();
  showTaskPanel(win, posted);
  const d = win.document;

  const taskBtn = d.getElementById('task-panel-drawer-btn');
  const inputBtn = d.getElementById('input-drawer-btn');
  assert.ok(taskBtn, '#task-panel-drawer-btn must exist');
  assert.ok(inputBtn, '#input-drawer-btn must exist');
  assert.ok(
    d.getElementById('task-panel').contains(taskBtn),
    'task drawer toggle must live inside #task-panel',
  );
  assert.ok(
    d.getElementById('input-area').contains(inputBtn),
    'input drawer toggle must live inside #input-area',
  );
  assert.strictEqual(
    taskBtn.getAttribute('aria-controls'),
    'task-panel-text',
    'task drawer toggle must declare its controlled region',
  );
  assert.strictEqual(
    inputBtn.getAttribute('aria-controls'),
    'input-container',
    'input drawer toggle must declare its controlled region',
  );
  assertBtnState(win, 'task-panel-drawer-btn', true);
  assertBtnState(win, 'input-drawer-btn', true);

  assert.ok(
    !d.getElementById('task-panel').classList.contains('drawer-collapsed'),
    'task drawer must start expanded',
  );
  assert.ok(
    !d.getElementById('input-area').classList.contains('drawer-collapsed'),
    'input drawer must start expanded',
  );
  assert.notStrictEqual(
    cs(win, 'input-container').display,
    'none',
    'composer must be visible while the input drawer is expanded',
  );
  assert.strictEqual(
    cs(win, 'task-panel-text').whiteSpace,
    'pre-wrap',
    'task text must wrap normally while the task drawer is expanded',
  );
  assert.strictEqual(
    cs(win, 'output').flexGrow,
    '1',
    '#output must be the flex:1 child that absorbs freed drawer space',
  );
  win.close();
}

// ── 2. Input drawer: collapse hides EVERY child except the handle ───
function testInputDrawerToggle() {
  persistedState = undefined;
  const {win} = makeWebview();
  const d = win.document;
  const area = d.getElementById('input-area');

  // The autocomplete popover is open (inline style, as main.js sets it)
  // and a worktree/merge action bar sits inside #input-area — both must
  // be tucked away by the drawer despite the inline styles.
  d.getElementById('autocomplete').style.display = 'block';
  const bar = d.createElement('div');
  bar.id = 'fake-merge-bar';
  bar.style.display = 'flex';
  area.insertBefore(bar, area.firstChild);

  click(win, 'input-drawer-btn');
  assert.ok(
    area.classList.contains('drawer-collapsed'),
    'clicking the handle must collapse the input drawer',
  );
  assertBtnState(win, 'input-drawer-btn', false);
  assert.strictEqual(
    cs(win, 'input-container').display,
    'none',
    'collapsed input drawer must hide the composer',
  );
  assert.strictEqual(
    cs(win, 'autocomplete').display,
    'none',
    'collapsed input drawer must hide the autocomplete popover ' +
      'even with an inline display:block',
  );
  assert.strictEqual(
    cs(win, 'fake-merge-bar').display,
    'none',
    'collapsed input drawer must hide action bars inserted into ' +
      '#input-area even with inline display styles',
  );
  assert.notStrictEqual(
    cs(win, 'input-drawer-btn').display,
    'none',
    'the drawer handle itself must stay visible to re-open the drawer',
  );
  assert.strictEqual(
    cs(win, 'output').flexGrow,
    '1',
    '#output must keep flex:1 so it absorbs the freed space',
  );

  click(win, 'input-drawer-btn');
  assert.ok(
    !area.classList.contains('drawer-collapsed'),
    'clicking the handle again must expand the input drawer',
  );
  assertBtnState(win, 'input-drawer-btn', true);
  assert.notStrictEqual(
    cs(win, 'input-container').display,
    'none',
    'expanded input drawer must show the composer again',
  );
  assert.strictEqual(
    cs(win, 'fake-merge-bar').display,
    'flex',
    'expanding must restore inline-styled action bars',
  );
  win.close();
}

// ── 3. Task drawer: collapse clamps to a slim one-line drawer ───────
function testTaskDrawerToggle() {
  persistedState = undefined;
  const {win, posted} = makeWebview();
  showTaskPanel(win, posted);
  const d = win.document;
  const panel = d.getElementById('task-panel');

  click(win, 'task-panel-drawer-btn');
  assert.ok(
    panel.classList.contains('drawer-collapsed'),
    'clicking the toggle must collapse the task drawer',
  );
  assertBtnState(win, 'task-panel-drawer-btn', false);
  const textCs = cs(win, 'task-panel-text');
  assert.strictEqual(
    textCs.whiteSpace,
    'nowrap',
    'collapsed task drawer must clamp the task text to one line',
  );
  assert.strictEqual(
    textCs.overflow,
    'hidden',
    'collapsed task drawer must hide the clamped overflow',
  );
  assert.strictEqual(
    textCs.textOverflow,
    'ellipsis',
    'collapsed task drawer must ellipsize the clamped text',
  );
  assert.strictEqual(
    cs(win, 'task-panel').display,
    'block',
    'the slim task drawer itself must stay visible',
  );
  assert.strictEqual(
    d.getElementById('task-panel-text').textContent,
    'refactor the parser and keep the CLI flags backward compatible',
    'the task text must stay readable in the slim drawer',
  );

  click(win, 'task-panel-drawer-btn');
  assert.ok(
    !panel.classList.contains('drawer-collapsed'),
    'clicking the toggle again must expand the task drawer',
  );
  assertBtnState(win, 'task-panel-drawer-btn', true);
  assert.strictEqual(
    cs(win, 'task-panel-text').whiteSpace,
    'pre-wrap',
    'expanded task drawer must restore the wrapped task text',
  );
  win.close();
}

// ── 4. Persistence: both drawer states survive a dispose/reopen ─────
function testPersistenceAcrossReopen() {
  persistedState = undefined;
  const wv1 = makeWebview();
  showTaskPanel(wv1.win, wv1.posted);
  click(wv1.win, 'task-panel-drawer-btn');
  click(wv1.win, 'input-drawer-btn');
  assert.ok(persistedState, 'toggling a drawer must persist state');
  wv1.win.close();

  const wv2 = makeWebview();
  showTaskPanel(wv2.win, wv2.posted);
  const d = wv2.win.document;
  assert.ok(
    d.getElementById('task-panel').classList.contains('drawer-collapsed'),
    'a re-opened webview must restore the collapsed task drawer',
  );
  assert.ok(
    d.getElementById('input-area').classList.contains('drawer-collapsed'),
    'a re-opened webview must restore the collapsed input drawer',
  );
  assertBtnState(wv2.win, 'task-panel-drawer-btn', false);
  assertBtnState(wv2.win, 'input-drawer-btn', false);
  assert.strictEqual(
    cs(wv2.win, 'input-container').display,
    'none',
    'the restored collapsed input drawer must hide the composer',
  );

  // Expanding in the restored webview persists back.
  click(wv2.win, 'task-panel-drawer-btn');
  click(wv2.win, 'input-drawer-btn');
  wv2.win.close();

  const wv3 = makeWebview();
  const d3 = wv3.win.document;
  assert.ok(
    !d3.getElementById('task-panel').classList.contains('drawer-collapsed'),
    'a re-opened webview must restore the re-expanded task drawer',
  );
  assert.ok(
    !d3.getElementById('input-area').classList.contains('drawer-collapsed'),
    'a re-opened webview must restore the re-expanded input drawer',
  );
  wv3.win.close();
}

// ── 5. Persistence is per-drawer: one collapsed, one expanded ───────
function testPersistenceSingleDrawer() {
  persistedState = undefined;
  const wv1 = makeWebview();
  showTaskPanel(wv1.win, wv1.posted);
  click(wv1.win, 'task-panel-drawer-btn');
  wv1.win.close();

  const wv2 = makeWebview();
  showTaskPanel(wv2.win, wv2.posted);
  const d = wv2.win.document;
  assert.ok(
    d.getElementById('task-panel').classList.contains('drawer-collapsed'),
    'the collapsed task drawer must be restored',
  );
  assert.ok(
    !d.getElementById('input-area').classList.contains('drawer-collapsed'),
    'the untouched input drawer must stay expanded',
  );
  assertBtnState(wv2.win, 'input-drawer-btn', true);
  wv2.win.close();
}

// ── 6. Remote web app: same drawers under the remote-codex skin ─────
function testRemoteWebApp() {
  persistedState = undefined;
  const {win, posted} = makeWebview({remote: true});
  showTaskPanel(win, posted);
  const d = win.document;

  click(win, 'task-panel-drawer-btn');
  click(win, 'input-drawer-btn');
  assert.ok(
    d.getElementById('task-panel').classList.contains('drawer-collapsed'),
    'remote: task drawer must collapse',
  );
  assert.ok(
    d.getElementById('input-area').classList.contains('drawer-collapsed'),
    'remote: input drawer must collapse',
  );
  assert.strictEqual(
    cs(win, 'input-container').display,
    'none',
    'remote: the remote-codex.css cascade must not resurrect the ' +
      'collapsed composer',
  );
  assert.strictEqual(
    cs(win, 'task-panel-text').whiteSpace,
    'nowrap',
    'remote: collapsed task drawer must clamp the task text',
  );
  assert.strictEqual(
    cs(win, 'output').flexGrow,
    '1',
    'remote: #output must keep flex:1 to absorb the freed space',
  );

  click(win, 'task-panel-drawer-btn');
  click(win, 'input-drawer-btn');
  assert.notStrictEqual(
    cs(win, 'input-container').display,
    'none',
    'remote: expanding must restore the composer',
  );
  assert.strictEqual(
    cs(win, 'task-panel-text').whiteSpace,
    'pre-wrap',
    'remote: expanding must restore the wrapped task text',
  );
  win.close();
}

// ── 7. Drawer state survives status + task replay churn ─────────────
function testDrawerStateSurvivesTaskChurn() {
  persistedState = undefined;
  const {win, posted} = makeWebview();
  showTaskPanel(win, posted);
  const ready = posted.find(m => m.type === 'ready');
  const d = win.document;

  click(win, 'task-panel-drawer-btn');
  click(win, 'input-drawer-btn');
  send(win, {
    type: 'status',
    running: true,
    tabId: ready.tabId,
    startTs: Date.now(),
  });
  send(win, {
    type: 'task_events',
    events: [],
    task: 'a brand new task text arriving mid-flight',
    tabId: ready.tabId,
    chat_id: 'chat-drawer',
  });
  send(win, {type: 'status', running: false, tabId: ready.tabId});

  assert.ok(
    d.getElementById('task-panel').classList.contains('drawer-collapsed'),
    'status/task churn must not re-open the task drawer',
  );
  assert.ok(
    d.getElementById('input-area').classList.contains('drawer-collapsed'),
    'status/task churn must not re-open the input drawer',
  );
  assert.strictEqual(
    d.getElementById('task-panel-text').textContent,
    'a brand new task text arriving mid-flight',
    'the slim task drawer must keep tracking the latest task text',
  );
  win.close();
}

// ── 8. The removed Collapse Chats button never comes back ───────────
function testChatsCollapseButtonRemoved() {
  persistedState = undefined;
  const {win, posted} = makeWebview();
  showTaskPanel(win, posted);
  const d = win.document;
  assert.strictEqual(
    d.getElementById('task-panel-collapse-btn'),
    null,
    'the Collapse/Uncollapse Chats button must not exist',
  );
  assert.strictEqual(
    d.getElementById('task-panel-collapse-label'),
    null,
    'the Collapse/Uncollapse Chats label must not exist',
  );
  win.close();
}

// ── 9. Both toggles offer at-least-24px hit targets (WCAG 2.2) ──────
function testDrawerButtonsBigEnough() {
  persistedState = undefined;
  for (const remote of [false, true]) {
    const {win} = makeWebview({remote});
    const taskBtn = cs(win, 'task-panel-drawer-btn');
    assert.ok(
      parseFloat(taskBtn.width) >= 24 && parseFloat(taskBtn.height) >= 24,
      `task drawer toggle must be at least 24x24px (remote=${remote}, ` +
        `got ${taskBtn.width} x ${taskBtn.height})`,
    );
    const inputBtn = cs(win, 'input-drawer-btn');
    assert.ok(
      parseFloat(inputBtn.width) >= 24 && parseFloat(inputBtn.height) >= 24,
      `input drawer handle must be at least 24x24px (remote=${remote}, ` +
        `got ${inputBtn.width} x ${inputBtn.height})`,
    );
    win.close();
  }
}

// ── 10. Toggles do not keep keyboard focus after a click ────────────
function testDrawerButtonsLoseFocusAfterClick() {
  persistedState = undefined;
  const {win} = makeWebview();
  for (const id of ['task-panel-drawer-btn', 'input-drawer-btn']) {
    const el = win.document.getElementById(id);
    el.focus();
    assert.strictEqual(
      win.document.activeElement,
      el,
      `#${id} must be focusable (test setup)`,
    );
    el.dispatchEvent(
      new win.MouseEvent('click', {bubbles: true, cancelable: true}),
    );
    assert.notStrictEqual(
      win.document.activeElement,
      el,
      `#${id} must not keep focus after being clicked (same ` +
        'blur-after-click contract as every other chat control)',
    );
  }
  win.close();
}

// ── 11. Remote skin restates the slim collapsed paddings ────────────
function testRemoteCollapsedPadding() {
  persistedState = undefined;
  const {win, posted} = makeWebview({remote: true});
  showTaskPanel(win, posted);
  click(win, 'task-panel-drawer-btn');
  click(win, 'input-drawer-btn');
  // The remote base rules set shorthand ``padding`` with higher
  // specificity than main.css's collapsed longhands, so the slim
  // paddings must be restated remote-side — with remote-specific
  // values distinct from main.css's (6px/10px vs 4px/12px), proving
  // the remote rules win the cascade here.
  assert.strictEqual(
    cs(win, 'task-panel').paddingTop,
    '6px',
    'remote: collapsed task drawer must get the slim remote padding',
  );
  assert.strictEqual(
    cs(win, 'input-area').paddingTop,
    '10px',
    'remote: collapsed input drawer must get the slim remote padding',
  );
  win.close();
}

// ── 12. Graceful boot when an embedder serves stale HTML ────────────
function testMissingDrawerButtonsGracefulBoot() {
  persistedState = undefined;
  const {win, posted} = makeWebview({stripDrawerButtons: true});
  assert.strictEqual(
    win.document.getElementById('input-drawer-btn'),
    null,
    'harness sanity: the drawer buttons were stripped',
  );
  assert.ok(
    posted.some(m => m.type === 'ready'),
    'main.js must boot (post ready) even without the drawer buttons',
  );
  win.close();
}

// ── Mobile device simulation: real browser UA strings ───────────────
const UA_IPHONE =
  'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) ' +
  'AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 ' +
  'Mobile/15E148 Safari/604.1';
const UA_ANDROID =
  'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 ' +
  '(KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36';
const UA_IPAD_MASQUERADE =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ' +
  'AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15';
const UA_DESKTOP =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' +
  '(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36';

/** Assert both drawers are (not) collapsed, with matching toggles. */
function assertDrawers(win, collapsed, why) {
  const d = win.document;
  assert.strictEqual(
    d.getElementById('task-panel').classList.contains('drawer-collapsed'),
    collapsed,
    `task drawer must ${collapsed ? '' : 'NOT '}be collapsed: ${why}`,
  );
  assert.strictEqual(
    d.getElementById('input-area').classList.contains('drawer-collapsed'),
    collapsed,
    `input drawer must ${collapsed ? '' : 'NOT '}be collapsed: ${why}`,
  );
  assertBtnState(win, 'task-panel-drawer-btn', !collapsed);
  assertBtnState(win, 'input-drawer-btn', !collapsed);
}

// ── 13. Mobile remote web app: drawers OPEN COLLAPSED ───────────────
// Opening the remote web app on a phone/tablet must start with the
// pinned task panel AND the composer (input textbox + buttons panel)
// tucked into their slim drawers so the small screen is spent on the
// chat events area.
function testMobileRemoteOpensCollapsed() {
  for (const [name, ua] of [
    ['iPhone Safari', UA_IPHONE],
    ['Android Chrome', UA_ANDROID],
  ]) {
    persistedState = undefined;
    const {win, posted} = makeWebview({remote: true, userAgent: ua});
    showTaskPanel(win, posted);
    const d = win.document;
    assertDrawers(win, true, `${name} remote web app must open collapsed`);
    // The input textbox and the buttons panel both live inside the
    // hidden #input-container, so the collapsed drawer tucks them away.
    const container = d.getElementById('input-container');
    assert.ok(
      container.contains(d.getElementById('task-input')),
      'the input textbox must live inside the collapsed composer',
    );
    assert.ok(
      container.contains(d.getElementById('input-footer')),
      'the buttons panel must live inside the collapsed composer',
    );
    assert.strictEqual(
      cs(win, 'input-container').display,
      'none',
      `${name}: the composer must be hidden on open`,
    );
    assert.strictEqual(
      cs(win, 'task-panel-text').whiteSpace,
      'nowrap',
      `${name}: the task text must be clamped to the slim drawer`,
    );
    assert.strictEqual(
      cs(win, 'output').flexGrow,
      '1',
      `${name}: #output must absorb the freed space`,
    );
    win.close();
  }
}

// ── 14. Desktop remote web app: drawers still open EXPANDED ─────────
function testDesktopRemoteOpensExpanded() {
  persistedState = undefined;
  const {win, posted} = makeWebview({remote: true, userAgent: UA_DESKTOP});
  showTaskPanel(win, posted);
  assertDrawers(win, false, 'desktop remote web app keeps the old default');
  assert.notStrictEqual(
    cs(win, 'input-container').display,
    'none',
    'desktop remote: the composer must stay visible on open',
  );
  win.close();
}

// ── 15. UA-Client-Hints: navigator.userAgentData.mobile drives it ───
function testUserAgentDataMobileRemote() {
  persistedState = undefined;
  const wvMobile = makeWebview({
    remote: true,
    userAgent: UA_DESKTOP,
    userAgentData: {mobile: true},
  });
  assertDrawers(
    wvMobile.win,
    true,
    'userAgentData.mobile=true must open the remote drawers collapsed',
  );
  wvMobile.win.close();

  persistedState = undefined;
  const wvDesktop = makeWebview({
    remote: true,
    userAgent: UA_DESKTOP,
    userAgentData: {mobile: false},
  });
  assertDrawers(
    wvDesktop.win,
    false,
    'userAgentData.mobile=false must keep the remote drawers expanded',
  );
  wvDesktop.win.close();
}

// ── 16. iPadOS Safari masquerading as desktop "Macintosh" ───────────
function testIpadMasqueradeRemote() {
  persistedState = undefined;
  const wvIpad = makeWebview({
    remote: true,
    userAgent: UA_IPAD_MASQUERADE,
    maxTouchPoints: 5,
  });
  assertDrawers(
    wvIpad.win,
    true,
    'Macintosh UA with a multi-touch screen is an iPad — collapse',
  );
  wvIpad.win.close();

  persistedState = undefined;
  const wvMac = makeWebview({
    remote: true,
    userAgent: UA_IPAD_MASQUERADE,
    maxTouchPoints: 0,
  });
  assertDrawers(
    wvMac.win,
    false,
    'Macintosh UA without touch is a real Mac — stay expanded',
  );
  wvMac.win.close();
}

// ── 17. VS Code webview is never treated as mobile ──────────────────
function testMobileUaVscodeWebviewUnaffected() {
  persistedState = undefined;
  const {win, posted} = makeWebview({remote: false, userAgent: UA_IPHONE});
  showTaskPanel(win, posted);
  assertDrawers(
    win,
    false,
    'the extension webview (no body.remote-chat) keeps expanded defaults',
  );
  win.close();
}

// ── 18. Mobile default yields to the user's persisted choice ────────
function testMobileUserChoicePersists() {
  // First mobile visit: opens collapsed, the user expands both drawers.
  persistedState = undefined;
  const wv1 = makeWebview({remote: true, userAgent: UA_IPHONE});
  showTaskPanel(wv1.win, wv1.posted);
  assertDrawers(wv1.win, true, 'first mobile visit must open collapsed');
  click(wv1.win, 'task-panel-drawer-btn');
  click(wv1.win, 'input-drawer-btn');
  assertDrawers(wv1.win, false, 'the user expanded both drawers');
  wv1.win.close();

  // Reload on the same device: the expanded choice must win over the
  // mobile collapsed default.
  const wv2 = makeWebview({remote: true, userAgent: UA_IPHONE});
  showTaskPanel(wv2.win, wv2.posted);
  assertDrawers(
    wv2.win,
    false,
    'a reload must restore the user-expanded drawers on mobile',
  );
  wv2.win.close();

  // And the mobile collapsed default itself persists across reloads.
  persistedState = undefined;
  const wv3 = makeWebview({remote: true, userAgent: UA_ANDROID});
  wv3.win.close();
  const wv4 = makeWebview({remote: true, userAgent: UA_ANDROID});
  assertDrawers(
    wv4.win,
    true,
    'an untouched mobile session must stay collapsed after a reload',
  );
  wv4.win.close();
}

// ── 19. Legacy persisted blobs migrate to the mobile default ────────
// Builds without the mobile default auto-persisted
// ``taskDrawerCollapsed: false`` on every boot — never a user choice.
// A mobile session upgraded in place (sessionStorage survives the
// reload) must therefore IGNORE legacy drawer booleans and open
// collapsed; only blobs written by this build (drawersVersion) may
// override the mobile default.
function testLegacyStateMigratesOnMobile() {
  // Legacy blob: tabs + auto-persisted expanded drawers, no version.
  persistedState = {
    tabs: [{title: 'old chat', chatId: 'tab-1'}],
    activeTabIndex: 0,
    chatId: 'tab-1',
    taskDrawerCollapsed: false,
    inputDrawerCollapsed: false,
  };
  const wv = makeWebview({remote: true, userAgent: UA_IPHONE});
  assertDrawers(
    wv.win,
    true,
    'a legacy (unversioned) blob must not resurrect expanded drawers ' +
      'on mobile',
  );
  wv.win.close();

  // The same legacy blob on desktop keeps restoring as before.
  persistedState = {
    tabs: [{title: 'old chat', chatId: 'tab-1'}],
    activeTabIndex: 0,
    chatId: 'tab-1',
    taskDrawerCollapsed: true,
    inputDrawerCollapsed: true,
  };
  const wvDesk = makeWebview({remote: true, userAgent: UA_DESKTOP});
  assertDrawers(
    wvDesk.win,
    true,
    'desktop must keep restoring legacy drawer values (no regression)',
  );
  wvDesk.win.close();

  // A current-version blob with explicit expanded drawers restores
  // expanded even on mobile (the user's choice).
  persistedState = {
    tabs: [{title: 'old chat', chatId: 'tab-1'}],
    activeTabIndex: 0,
    chatId: 'tab-1',
    taskDrawerCollapsed: false,
    inputDrawerCollapsed: false,
    drawersVersion: 2,
  };
  const wvNew = makeWebview({remote: true, userAgent: UA_IPHONE});
  assertDrawers(
    wvNew.win,
    false,
    'a versioned blob with expanded drawers must restore expanded ' +
      'on mobile (user choice wins)',
  );
  wvNew.win.close();
}

// ── 20. Malformed persisted state never breaks the boot ─────────────
function testMalformedStateGracefulBoot() {
  for (const bad of ['garbage', 42, true]) {
    persistedState = bad;
    const {win, posted} = makeWebview({remote: true, userAgent: UA_IPHONE});
    assert.ok(
      posted.some(m => m.type === 'ready'),
      `main.js must boot with a ${typeof bad} persisted state`,
    );
    assertDrawers(
      win,
      true,
      `a malformed (${typeof bad}) blob must fall back to the mobile ` +
        'collapsed default',
    );
    win.close();
  }
}

function runTests() {
  const tests = [
    testDefaultsExpanded,
    testInputDrawerToggle,
    testTaskDrawerToggle,
    testPersistenceAcrossReopen,
    testPersistenceSingleDrawer,
    testRemoteWebApp,
    testDrawerStateSurvivesTaskChurn,
    testChatsCollapseButtonRemoved,
    testDrawerButtonsBigEnough,
    testDrawerButtonsLoseFocusAfterClick,
    testRemoteCollapsedPadding,
    testMissingDrawerButtonsGracefulBoot,
    testMobileRemoteOpensCollapsed,
    testDesktopRemoteOpensExpanded,
    testUserAgentDataMobileRemote,
    testIpadMasqueradeRemote,
    testMobileUaVscodeWebviewUnaffected,
    testMobileUserChoicePersists,
    testLegacyStateMigratesOnMobile,
    testMalformedStateGracefulBoot,
  ];
  for (const t of tests) {
    t();
    console.log('PASS', t.name);
  }
}

try {
  runTests();
  console.log('\nAll tests passed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
