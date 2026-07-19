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
//      task text is clamped to a single ellipsized line and the
//      "Collapse/Uncollapse Chats" button row is hidden.  Expanding
//      restores the full panel.
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
 * @returns {{win: object, posted: Array}}
 */
function makeWebview(opts) {
  const {remote = false, stripDrawerButtons = false} = opts || {};
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
    cs(win, 'task-panel-collapse-btn').display,
    'none',
    'collapsed task drawer must hide the Collapse Chats button row',
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
  assert.notStrictEqual(
    cs(win, 'task-panel-collapse-btn').display,
    'none',
    'expanded task drawer must restore the Collapse Chats button',
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

// ── 8. Existing Collapse Chats button is untouched by the drawers ───
function testChatsCollapseButtonUnaffected() {
  persistedState = undefined;
  const {win, posted} = makeWebview();
  showTaskPanel(win, posted);
  const d = win.document;
  const chatsBtn = d.getElementById('task-panel-collapse-btn');
  const before = chatsBtn.getAttribute('aria-expanded');

  click(win, 'task-panel-drawer-btn');
  click(win, 'task-panel-drawer-btn');
  assert.strictEqual(
    chatsBtn.getAttribute('aria-expanded'),
    before,
    'the drawer toggle must not drive the Collapse Chats button state',
  );

  // And vice versa: Collapse Chats must not collapse the drawer.
  chatsBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    !d.getElementById('task-panel').classList.contains('drawer-collapsed'),
    'the Collapse Chats button must not drive the drawer state',
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

function runTests() {
  const tests = [
    testDefaultsExpanded,
    testInputDrawerToggle,
    testTaskDrawerToggle,
    testPersistenceAcrossReopen,
    testPersistenceSingleDrawer,
    testRemoteWebApp,
    testDrawerStateSurvivesTaskChurn,
    testChatsCollapseButtonUnaffected,
    testDrawerButtonsBigEnough,
    testDrawerButtonsLoseFocusAfterClick,
    testRemoteCollapsedPadding,
    testMissingDrawerButtonsGracefulBoot,
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
