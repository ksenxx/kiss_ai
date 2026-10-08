// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for keyboard and screen-reader access to
// the tab strip: the add-tab, theme, and settings controls, the tabs
// on the group strip (#tab-list: the chat on screen and its sub-agent
// tabs -- there is no row of chat tabs, the Chats panel lists those),
// and each tab's close control must be reachable from the keyboard,
// expose an accessible role and name, and activate on Enter/Space
// exactly like a click -- exactly once, with the default Space scroll
// suppressed.  The tabs implement the WAI-ARIA tabs pattern with a
// roving tabindex: only the active tab is a Tab stop,
// ArrowLeft/ArrowRight/Home/End move focus between tabs, and each tab
// points at the shared chat surface via aria-controls.  Without this,
// keyboard-only and screen-reader users cannot switch tabs, close
// tabs, open settings, or create a new chat in the remote web app, and
// UI automation via the accessibility tree cannot reach them.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(initialState, opts) {
  opts = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  // The remote web server injects class="remote-chat" into the body
  // before any script runs (BODY_CLASS_ATTR in web_server.py), so
  // adding it before main.js is evaluated reproduces remote mode.
  if (opts.remote) win.document.body.classList.add('remote-chat');

  const posted = [];
  let state = initialState;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function snapshotEntry(tabId, title, chatId) {
  return {
    tabId: tabId,
    chatId: chatId || '',
    title: title || 'new chat',
    workDir: '',
  };
}

// The ids on the group strip, in order: the chat on screen and its
// sub-agent tabs.
function tabBarIds(win) {
  return Array.from(win.document.querySelectorAll('#tab-list .chat-tab'))
    .filter(el => !!el.dataset.tabId)
    .map(el => el.dataset.tabId);
}

// Every open chat tab (the Chats panel's subject), in tab order.
function openChatIds(win) {
  return Array.from(win._testApi.openTabs())
    .filter(t => !t.isSubagentTab && !t.isContentTab)
    .map(t => t.id);
}

function activeTabId(win) {
  // The strip's active entry is the tab on screen.
  const el = win.document.querySelector('#tab-list .chat-tab.active');
  return el ? el.dataset.tabId : null;
}

// A tab's entry on the group strip.
function tabEl(win, tabId) {
  return win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id="${tabId}"]`,
  );
}

// A chat "parent" on screen with one sub-agent tab per *subIds* entry,
// which is what fills the group strip with several keyboard-reachable
// tabs.  The chat comes from the registry and its sub-agents from the
// daemon's replay announcements; none of them is activated.
function openGroup(win, subIds) {
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('parent', 'parent chat', 'chat-A')],
  });
  win._testApi.switchToTab('parent');
  subIds.forEach((id, i) => {
    send(win, {
      type: 'openSubagentTab',
      tab_id: id,
      parent_tab_id: 'parent',
      description: `sub-agent ${id}`,
      task_id: `task-${id}`,
      taskIndex: i,
      isSubagentTab: true,
    });
  });
  assert.strictEqual(activeTabId(win), 'parent', 'the chat stays on screen');
  assert.deepStrictEqual(
    tabBarIds(win),
    ['parent'].concat(subIds),
    'the strip lists the chat and its sub-agents in order',
  );
}

// Focus the control first (a real keyboard user can only send keys to
// the focused element), then dispatch a cancelable keydown.  Returns
// the event so callers can assert defaultPrevented -- Space must be
// preventDefault-ed or the page scrolls on activation.
function pressKey(win, el, key) {
  el.focus();
  assert.strictEqual(
    win.document.activeElement,
    el,
    `control must accept focus before receiving "${key}" ` +
      `(tag=${el.tagName} tabindex=${el.getAttribute('tabindex')})`,
  );
  const ev = new win.KeyboardEvent('keydown', {
    key: key,
    bubbles: true,
    cancelable: true,
  });
  el.dispatchEvent(ev);
  return ev;
}

// A control is keyboard-reachable iff it is a native button or carries
// tabindex >= 0; it is screen-reader-usable iff it also exposes a role
// and an accessible name (aria-label, or text content for tabs).
function assertFocusable(el, what) {
  const nativeButton = el.tagName === 'BUTTON';
  assert.ok(
    nativeButton || el.tabIndex >= 0,
    `${what} must be keyboard-focusable (native <button> or tabindex>=0), ` +
      `got tag=${el.tagName} tabindex=${el.getAttribute('tabindex')}`,
  );
  const role = nativeButton ? 'button' : el.getAttribute('role');
  assert.ok(role, `${what} must expose an accessible role`);
  return role;
}

// The tabs on the group strip, the row the keyboard tests drive.
function tabEls(win) {
  return Array.from(win.document.querySelectorAll('#tab-list [role="tab"]'));
}

function testAddTabControlIsAccessibleButton() {
  // The "+" (new chat) control now lives in the input footer.  It must
  // be a NATIVE <button> (native buttons activate on Enter/Space in a
  // real browser -- JSDOM does not synthesize that click, so keyboard
  // activation is guaranteed by the element type rather than emulated
  // here) named "New chat", and a click must create exactly one tab.
  const {win} = makeWebview(undefined);
  const addBtn = win.document.getElementById('new-chat-btn');
  assert.ok(addBtn, 'new-chat control missing from the input footer');

  assert.strictEqual(
    addBtn.tagName,
    'BUTTON',
    'the new-chat control must be a native <button> so Enter/Space ' +
      'activate it without bespoke keydown handlers',
  );
  const role = assertFocusable(addBtn, 'the new-chat control');
  assert.strictEqual(role, 'button', 'new-chat control must be a button');
  assert.strictEqual(
    addBtn.getAttribute('aria-label'),
    'New chat',
    'new-chat control must be named "New chat" for screen readers',
  );

  // "+" opens one fresh chat and retires the idle one it leaves (no
  // task, no question, no draft), so the count does not grow: the set
  // of open chats gains exactly one id.
  const left = win._testApi.getActiveTabId();
  const before = openChatIds(win);
  addBtn.click();
  const after = openChatIds(win);
  const added = after.filter(id => before.indexOf(id) === -1);
  assert.strictEqual(
    added.length,
    1,
    'a click on + must create exactly one tab (no double-fire)',
  );
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    added[0],
    'the new chat is the one on screen',
  );
  assert.ok(
    after.indexOf(left) === -1,
    'the idle chat left behind is retired',
  );
}

function testSettingsControlIsAccessibleButton() {
  // The Settings control now lives in the "..." overflow menu.  It must
  // be a real accessible NATIVE <button> (so Enter/Space activate it in
  // a real browser without bespoke keydown handlers) named "Settings",
  // reachable through the "..." button, and a click must open the
  // settings panel exactly once and close the overflow menu.
  const {win, posted} = makeWebview(undefined);
  const panel = win.document.getElementById('settings-panel');
  assert.ok(panel, 'settings panel missing from chat.html');
  const getConfigCount = () =>
    posted.filter(m => m && m.type === 'getConfig').length;

  const moreBtn = win.document.getElementById('more-btn');
  assert.ok(moreBtn, 'the "..." button missing from the input footer');
  assert.strictEqual(
    moreBtn.tagName,
    'BUTTON',
    'the "..." control must be a native <button>',
  );
  assert.strictEqual(
    moreBtn.getAttribute('aria-haspopup'),
    'true',
    'the "..." control must announce its popup menu',
  );

  const moreMenu = win.document.getElementById('more-menu');
  assert.ok(moreMenu, 'the "..." menu missing');
  moreBtn.click();
  assert.ok(
    moreMenu.classList.contains('open'),
    'clicking "..." must open the overflow menu',
  );
  assert.strictEqual(
    moreBtn.getAttribute('aria-expanded'),
    'true',
    'the "..." control must report the open menu via aria-expanded',
  );

  const settingsBtn = win.document.getElementById('settings-btn');
  assert.ok(settingsBtn, 'settings control missing from the "..." menu');
  assert.strictEqual(
    settingsBtn.tagName,
    'BUTTON',
    'the settings control must be a native <button> so Enter/Space ' +
      'activate it without bespoke keydown handlers',
  );
  const role = assertFocusable(settingsBtn, 'the settings control');
  assert.strictEqual(role, 'button', 'settings control must be a button');
  assert.ok(
    (settingsBtn.textContent || '').indexOf('Settings') !== -1,
    'settings control must be named "Settings"',
  );

  // Click: opens the panel, requests the config exactly once
  // (openSettingsPanel calls api.getConfig(), so the posted getConfig
  // count is the activation count) and closes the overflow menu.
  const base = getConfigCount();
  settingsBtn.click();
  assert.ok(
    panel.classList.contains('open'),
    'a click on the settings control must open the settings panel',
  );
  assert.strictEqual(
    getConfigCount(),
    base + 1,
    'a click must activate the settings handler exactly once',
  );
  assert.ok(
    !moreMenu.classList.contains('open'),
    'activating a menu item must close the "..." menu',
  );
}

function testChatTabIsKeyboardActivatable() {
  // Each tab on the strip must expose its title as accessible name and
  // switch on Enter/Space exactly like a click.  Under the
  // roving-tabindex pattern only the ACTIVE tab is a Tab stop; other
  // tabs are reached with the arrow keys (covered separately below),
  // so activation is exercised from the roving focus position.
  const {win} = makeWebview(undefined);
  openGroup(win, ['sub-1']);

  const parent = tabEl(win, 'parent');
  const sub = tabEl(win, 'sub-1');
  assert.ok(parent && sub, 'both tabs must render');

  assert.strictEqual(parent.getAttribute('role'), 'tab');
  const name =
    parent.getAttribute('aria-label') ||
    (parent.querySelector('.chat-tab-label') || parent).textContent;
  assert.ok(
    name && name.indexOf('parent chat') !== -1,
    `a chat tab must expose its title as accessible name, got "${name}"`,
  );

  // Click baseline: activates the tab.
  sub.click();
  assert.strictEqual(
    activeTabId(win),
    'sub-1',
    'click must activate (baseline)',
  );

  // Arrow to the other tab, then Enter activates like the click did.
  let active = win.document.querySelector('#tab-list .chat-tab.active');
  pressKey(win, active, 'ArrowLeft');
  const focused = win.document.activeElement;
  assert.strictEqual(
    focused.dataset.tabId,
    'parent',
    'ArrowLeft must move focus to the previous tab',
  );
  pressKey(win, focused, 'Enter');
  assert.strictEqual(
    activeTabId(win),
    'parent',
    'Enter on a focused tab must activate it like a click',
  );

  // Space activates too, and must suppress the default page scroll.
  active = win.document.querySelector('#tab-list .chat-tab.active');
  pressKey(win, active, 'ArrowRight');
  const spaceEv = pressKey(win, win.document.activeElement, ' ');
  assert.strictEqual(
    activeTabId(win),
    'sub-1',
    'Space on a focused tab must activate it like a click',
  );
  assert.ok(
    spaceEv.defaultPrevented,
    'Space on a tab must preventDefault (or the page scrolls)',
  );
}

function testRovingTabindexFollowsActiveTab() {
  // FINDING 2a: the tablist must use a roving tabindex -- exactly the
  // active tab is a Tab stop (tabindex=0), every other tab has
  // tabindex=-1, and the stop follows activation.  Close controls keep
  // tabindex=0 so they stay directly Tab-reachable.
  const {win} = makeWebview(undefined);
  openGroup(win, ['sub-1', 'sub-2']);

  const stops = () =>
    tabEls(win).map(
      el => `${el.dataset.tabId}:${el.getAttribute('tabindex')}`,
    );
  assert.deepStrictEqual(
    stops(),
    ['parent:0', 'sub-1:-1', 'sub-2:-1'],
    'only the active tab may be a Tab stop (roving tabindex)',
  );

  tabEl(win, 'sub-2').click();
  assert.deepStrictEqual(
    stops(),
    ['parent:-1', 'sub-1:-1', 'sub-2:0'],
    'the Tab stop must follow the active tab',
  );

  const closers = win.document.querySelectorAll('#tab-list .chat-tab-close');
  assert.strictEqual(closers.length, 3, 'every strip entry has a close');
  for (const closeEl of closers) {
    assert.strictEqual(
      closeEl.getAttribute('tabindex'),
      '0',
      'close controls must stay directly Tab-reachable (tabindex=0)',
    );
  }
}

function testArrowKeysMoveFocusBetweenTabs() {
  // FINDING 2b: ArrowLeft/ArrowRight move focus between tabs (with
  // wrap-around), Home/End jump to the first/last tab, the moves are
  // preventDefault-ed (no page scroll), and focus movement alone does
  // NOT activate (manual-activation tabs pattern).
  const {win} = makeWebview(undefined);
  openGroup(win, ['sub-1', 'sub-2']);
  tabEl(win, 'sub-1').click();

  const focusedTab = () => {
    const el = win.document.activeElement;
    return el && el.dataset ? el.dataset.tabId || null : null;
  };

  let el = tabEl(win, 'sub-1');
  let ev = pressKey(win, el, 'ArrowRight');
  assert.strictEqual(
    focusedTab(),
    'sub-2',
    'ArrowRight must focus the next tab',
  );
  assert.ok(ev.defaultPrevented, 'ArrowRight must preventDefault');

  ev = pressKey(win, win.document.activeElement, 'ArrowRight');
  assert.strictEqual(
    focusedTab(),
    'parent',
    'ArrowRight on the last tab must wrap to the first',
  );

  pressKey(win, win.document.activeElement, 'ArrowLeft');
  assert.strictEqual(
    focusedTab(),
    'sub-2',
    'ArrowLeft on the first tab must wrap to the last',
  );

  ev = pressKey(win, win.document.activeElement, 'Home');
  assert.strictEqual(focusedTab(), 'parent', 'Home must focus the first tab');
  assert.ok(ev.defaultPrevented, 'Home must preventDefault');

  ev = pressKey(win, win.document.activeElement, 'End');
  assert.strictEqual(focusedTab(), 'sub-2', 'End must focus the last tab');
  assert.ok(ev.defaultPrevented, 'End must preventDefault');

  // Moving focus must not activate: sub-1 is still the active tab.
  assert.strictEqual(
    activeTabId(win),
    'sub-1',
    'arrow navigation must move focus without activating (manual activation)',
  );

  // The roving Tab stop follows the focus so the user can Tab away and
  // come back to where they were.
  assert.strictEqual(
    win.document.activeElement.getAttribute('tabindex'),
    '0',
    'the focused tab must become the Tab stop',
  );
}

function testCloseControlIsAccessibleAndDoesNotSwitchTabs() {
  // The per-tab close control must be focusable with an accessible
  // name, close exactly one tab on Enter/Space exactly like a click,
  // and must NOT also activate a tab switch.
  const {win} = makeWebview(undefined);
  openGroup(win, ['sub-1', 'sub-2', 'sub-3']);

  tabEl(win, 'sub-3').click();
  assert.strictEqual(activeTabId(win), 'sub-3');

  const close1 = tabEl(win, 'sub-1').querySelector('.chat-tab-close');
  assert.ok(close1, 'close control missing');
  const role = assertFocusable(close1, 'the tab close control');
  assert.strictEqual(role, 'button', 'close control must be a button');
  assert.strictEqual(
    close1.getAttribute('aria-label'),
    'Close tab',
    'close control must be named "Close tab" for screen readers',
  );

  // Enter on a background tab's close: closes exactly that tab, does
  // NOT switch to it first (active tab must stay sub-3).
  pressKey(win, close1, 'Enter');
  assert.deepStrictEqual(
    tabBarIds(win),
    ['parent', 'sub-2', 'sub-3'],
    'Enter on the close control must close exactly one tab like a click',
  );
  assert.strictEqual(
    activeTabId(win),
    'sub-3',
    'closing a background tab via keyboard must not switch to it',
  );

  // Space works too, and must suppress the default page scroll.
  const close2 = tabEl(win, 'sub-2').querySelector('.chat-tab-close');
  const spaceEv = pressKey(win, close2, ' ');
  assert.deepStrictEqual(
    tabBarIds(win),
    ['parent', 'sub-3'],
    'Space on the close control must close exactly one tab like a click',
  );
  assert.strictEqual(activeTabId(win), 'sub-3');
  assert.ok(
    spaceEv.defaultPrevented,
    'Space on the close control must preventDefault (or the page scrolls)',
  );
}

function testTabListExposesTabSemantics() {
  // The container must be a tablist, each tab must carry aria-selected
  // reflecting the active tab, and each tab must reference the shared
  // chat surface via aria-controls -> role=tabpanel so assistive tech
  // can announce "tab 2 of 3, selected" and jump to the panel.
  const {win} = makeWebview(undefined);
  openGroup(win, ['sub-1']);

  const tabList = win.document.getElementById('tab-list');
  assert.strictEqual(
    tabList.getAttribute('role'),
    'tablist',
    'the tab container must be a tablist',
  );
  const parent = tabEl(win, 'parent');
  const sub = tabEl(win, 'sub-1');
  assert.strictEqual(parent.getAttribute('role'), 'tab');
  assert.strictEqual(sub.getAttribute('role'), 'tab');
  assert.strictEqual(parent.getAttribute('aria-selected'), 'true');
  assert.strictEqual(sub.getAttribute('aria-selected'), 'false');

  sub.click();
  assert.strictEqual(
    tabEl(win, 'sub-1').getAttribute('aria-selected'),
    'true',
    'aria-selected must follow the active tab',
  );

  // FINDING 2c: tab -> tabpanel association.  All tabs swap the one
  // shared chat surface, so a single shared panel is correct.
  for (const el of tabEls(win)) {
    const controls = el.getAttribute('aria-controls');
    assert.ok(
      controls,
      'every tab must declare aria-controls pointing at its panel',
    );
    const panelEl = win.document.getElementById(controls);
    assert.ok(panelEl, `aria-controls="${controls}" must resolve to an element`);
    assert.strictEqual(
      panelEl.getAttribute('role'),
      'tabpanel',
      'the surface referenced by aria-controls must be a tabpanel',
    );
  }
}

function testRenamedTabKeepsAccessibleNameInSync() {
  // When the daemon renames a tab mid-run (a tabs_state snapshot with
  // a new title), the visible label AND the accessible name must both
  // update -- otherwise screen readers keep announcing the stale name.
  const {win} = makeWebview(undefined);
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'new chat', 'chat-A')],
  });
  let t1 = tabEl(win, 't1');
  assert.strictEqual(t1.getAttribute('aria-label'), 'new chat');

  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('t1', 'fix the login bug', 'chat-A')],
  });
  t1 = tabEl(win, 't1');
  assert.strictEqual(
    t1.querySelector('.chat-tab-label').textContent,
    'fix the login bug',
    'the visible label must show the new title',
  );
  assert.strictEqual(
    t1.getAttribute('aria-label'),
    'fix the login bug',
    'the accessible name must stay in sync with the renamed title',
  );
}

function testSubagentTabsGetTabSemantics() {
  // Sub-agent tabs live in the same tablist and must carry the same
  // tab semantics: role=tab, accessible name, aria-selected, and
  // participation in the roving tabindex.
  const {win} = makeWebview(undefined);
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('parent', 'parent chat', 'chat-A')],
  });
  tabEl(win, 'parent').click();
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-1',
    parent_tab_id: 'parent',
    description: 'index the codebase',
    task_id: 'task-sub-1',
    taskIndex: 0,
    isSubagentTab: true,
  });

  const sub = tabEl(win, 'sub-1');
  assert.ok(sub, 'sub-agent tab must render in the tab bar');
  assert.ok(
    sub.classList.contains('subagent-tab'),
    'sanity: the rendered tab is the sub-agent tab',
  );
  assert.strictEqual(
    sub.getAttribute('role'),
    'tab',
    'a sub-agent tab must be a tab for assistive tech',
  );
  const name = sub.getAttribute('aria-label') || '';
  assert.ok(
    name.indexOf('index the codebase') !== -1,
    `a sub-agent tab must expose its title as accessible name, got "${name}"`,
  );
  assert.strictEqual(
    sub.getAttribute('aria-selected'),
    'false',
    'a background sub-agent tab must announce as not selected',
  );
  assert.strictEqual(
    sub.getAttribute('tabindex'),
    '-1',
    'a background sub-agent tab must participate in the roving tabindex',
  );

  // Arrow navigation along the group strip reaches it from the parent
  // chat's strip entry, Enter activates it.
  pressKey(
    win,
    win.document.querySelector('#tab-list .chat-tab[data-tab-id="parent"]'),
    'ArrowRight',
  );
  assert.strictEqual(
    win.document.activeElement.dataset.tabId,
    'sub-1',
    'ArrowRight must move focus onto the sub-agent tab',
  );
  pressKey(win, win.document.activeElement, 'Enter');
  assert.strictEqual(
    activeTabId(win),
    'sub-1',
    'Enter must activate the sub-agent tab like a click',
  );
}

function testRemoteControlOrderAndThemeAccessibility() {
  // The footer must read, in order: burger menu, Task info drawer
  // toggle, New chat (+), inject, then the "..." overflow control on
  // the left; model picker, spinner, send, stop on the right.  The
  // theme toggle lives in the "..." menu and must be a named,
  // keyboard-activatable native button.
  const {win} = makeWebview(undefined, {remote: true});

  const tools = Array.from(
    win.document.getElementById('footer-tools').children,
  ).map(el => el.id);
  assert.deepStrictEqual(
    tools,
    [
      'menu-btn',
      'meta-drawer-btn',
      'new-chat-btn',
      'tricks-btn',
      'more-menu-wrap',
    ],
    'footer tools must keep the order burger, Task info, +, inject, "..."',
  );

  const actions = Array.from(
    win.document.getElementById('input-actions').children,
  ).map(el => el.id);
  assert.deepStrictEqual(
    actions,
    ['model-picker', 'wait-spinner', 'send-btn', 'stop-btn'],
    'the model picker must sit right of the footer, before the ' +
      'spinner, send and stop controls',
  );

  const themeBtn = win.document.getElementById('theme-btn');
  assert.ok(themeBtn, 'theme toggle missing from the "..." menu');
  assert.strictEqual(
    themeBtn.tagName,
    'BUTTON',
    'theme toggle must be a native <button> so Enter/Space activate it',
  );
  const role = assertFocusable(themeBtn, 'the theme toggle');
  assert.strictEqual(role, 'button', 'theme toggle must be a button');
  assert.ok(
    (themeBtn.getAttribute('aria-label') || '').indexOf('Switch to') === 0,
    'theme toggle must be named for screen readers',
  );

  const wasLight = win.document.body.classList.contains('light-theme');
  themeBtn.click();
  assert.notStrictEqual(
    win.document.body.classList.contains('light-theme'),
    wasLight,
    'a click on the theme toggle must switch the theme',
  );
}

const tests = [
  testAddTabControlIsAccessibleButton,
  testSettingsControlIsAccessibleButton,
  testChatTabIsKeyboardActivatable,
  testRovingTabindexFollowsActiveTab,
  testArrowKeysMoveFocusBetweenTabs,
  testCloseControlIsAccessibleAndDoesNotSwitchTabs,
  testTabListExposesTabSemantics,
  testRenamedTabKeepsAccessibleNameInSync,
  testSubagentTabsGetTabSemantics,
  testRemoteControlOrderAndThemeAccessibility,
];

let failures = 0;
for (const t of tests) {
  try {
    t();
    console.log('PASS', t.name);
  } catch (err) {
    failures += 1;
    console.error('FAIL', t.name);
    console.error(err && err.stack ? err.stack : err);
  }
}
if (failures > 0) {
  console.error(`${failures} test(s) failed`);
  process.exit(1);
}
console.log(`All ${tests.length} tabBarAccessibility tests passed`);
// The remote-mode webview leaves reconnect timers behind in JSDOM;
// exit explicitly like the other suites so the process never hangs.
process.exit(0);
