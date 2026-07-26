// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

let persistedState;

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

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=drawer-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function click(win, id) {
  const el = win.document.getElementById(id);
  assert.ok(el, `element #${id} must exist`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function cs(win, id) {
  return win.getComputedStyle(win.document.getElementById(id));
}

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

function testInputDrawerToggle() {
  persistedState = undefined;
  const {win} = makeWebview();
  const d = win.document;
  const area = d.getElementById('input-area');

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

function testRemoteCollapsedPadding() {
  persistedState = undefined;
  const {win, posted} = makeWebview({remote: true});
  showTaskPanel(win, posted);
  click(win, 'task-panel-drawer-btn');
  click(win, 'input-drawer-btn');
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

function testMobileUserChoicePersists() {
  persistedState = undefined;
  const wv1 = makeWebview({remote: true, userAgent: UA_IPHONE});
  showTaskPanel(wv1.win, wv1.posted);
  assertDrawers(wv1.win, true, 'first mobile visit must open collapsed');
  click(wv1.win, 'task-panel-drawer-btn');
  click(wv1.win, 'input-drawer-btn');
  assertDrawers(wv1.win, false, 'the user expanded both drawers');
  wv1.win.close();

  const wv2 = makeWebview({remote: true, userAgent: UA_IPHONE});
  showTaskPanel(wv2.win, wv2.posted);
  assertDrawers(
    wv2.win,
    false,
    'a reload must restore the user-expanded drawers on mobile',
  );
  wv2.win.close();

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

function testLegacyStateMigratesOnMobile() {
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
