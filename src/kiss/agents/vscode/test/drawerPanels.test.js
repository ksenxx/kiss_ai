// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the two chat drawers, in the extension webview and in
// the remote web app.
//
// The static task panel is a header, not content: it opens collapsed
// everywhere and only a click on its own chevron may expand it. The composer
// is the opposite -- it opens reachable, and folds away only on a phone while
// a task is running, where the transcript needs the whole screen.

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

function readyTabId(posted) {
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready, 'main.js must post a ready message');
  return ready.tabId;
}

function showTaskPanel(win, posted) {
  send(win, {
    type: 'task_events',
    events: [],
    task: 'refactor the parser and keep the CLI flags backward compatible',
    tabId: readyTabId(posted),
    chat_id: 'chat-drawer',
  });
  assert.ok(
    win.document.getElementById('task-panel').classList.contains('visible'),
    'task panel must be visible after a task replay',
  );
}

function setRunning(win, posted, running) {
  send(win, {
    type: 'status',
    running: running,
    tabId: readyTabId(posted),
    startTs: running ? Date.now() : 0,
  });
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

function assertTaskDrawer(win, collapsed, why) {
  assert.strictEqual(
    win.document
      .getElementById('task-panel')
      .classList.contains('drawer-collapsed'),
    collapsed,
    `task drawer must ${collapsed ? '' : 'NOT '}be collapsed: ${why}`,
  );
  assertBtnState(win, 'task-panel-drawer-btn', !collapsed);
}

function assertInputDrawer(win, collapsed, why) {
  assert.strictEqual(
    win.document
      .getElementById('input-area')
      .classList.contains('drawer-collapsed'),
    collapsed,
    `input drawer must ${collapsed ? '' : 'NOT '}be collapsed: ${why}`,
  );
  assertBtnState(win, 'input-drawer-btn', !collapsed);
  const display = cs(win, 'input-container').display;
  if (collapsed) {
    assert.strictEqual(display, 'none', `the composer must be hidden: ${why}`);
  } else {
    assert.notStrictEqual(
      display,
      'none',
      `the composer must be visible: ${why}`,
    );
  }
}

// The shipped defaults: task panel folded into its slim header, composer
// reachable.
function testDefaults() {
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

  assertTaskDrawer(win, true, 'the task panel opens collapsed');
  assertInputDrawer(win, false, 'the composer opens reachable');
  assert.strictEqual(
    cs(win, 'task-panel-text').whiteSpace,
    'nowrap',
    'the collapsed task drawer clamps the task text to one line',
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
  assertInputDrawer(win, true, 'clicking the handle collapses the composer');
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
  assertInputDrawer(win, false, 'clicking the handle again expands it');
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
  const task = 'refactor the parser and keep the CLI flags backward compatible';

  assertTaskDrawer(win, true, 'the task panel opens collapsed');
  const collapsedCs = cs(win, 'task-panel-text');
  assert.strictEqual(
    collapsedCs.overflow,
    'hidden',
    'collapsed task drawer must hide the clamped overflow',
  );
  assert.strictEqual(
    collapsedCs.textOverflow,
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
    task,
    'the task text must stay readable in the slim drawer',
  );

  click(win, 'task-panel-drawer-btn');
  assertTaskDrawer(win, false, 'clicking the toggle expands the task drawer');
  assert.strictEqual(
    cs(win, 'task-panel-text').whiteSpace,
    'pre-wrap',
    'expanded task drawer must wrap the task text',
  );

  click(win, 'task-panel-drawer-btn');
  assertTaskDrawer(win, true, 'clicking the toggle again collapses it');
  assert.strictEqual(
    cs(win, 'task-panel-text').whiteSpace,
    'nowrap',
    'the re-collapsed task drawer clamps the text again',
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
  assertTaskDrawer(
    wv2.win,
    false,
    'a re-opened webview restores the task drawer the user expanded',
  );
  assertInputDrawer(
    wv2.win,
    true,
    'a re-opened webview restores the composer the user collapsed',
  );

  click(wv2.win, 'task-panel-drawer-btn');
  click(wv2.win, 'input-drawer-btn');
  wv2.win.close();

  const wv3 = makeWebview();
  assertTaskDrawer(
    wv3.win,
    true,
    'a re-opened webview restores the re-collapsed task drawer',
  );
  assertInputDrawer(
    wv3.win,
    false,
    'a re-opened webview restores the re-expanded composer',
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
  assertTaskDrawer(wv2.win, false, 'the expanded task drawer is restored');
  assertInputDrawer(wv2.win, false, 'the untouched composer stays reachable');
  wv2.win.close();
}

function testRemoteWebApp() {
  persistedState = undefined;
  const {win, posted} = makeWebview({remote: true});
  showTaskPanel(win, posted);

  assertTaskDrawer(win, true, 'remote: the task panel opens collapsed');
  assert.strictEqual(
    cs(win, 'task-panel-text').whiteSpace,
    'nowrap',
    'remote: the collapsed task drawer clamps the task text',
  );

  click(win, 'task-panel-drawer-btn');
  click(win, 'input-drawer-btn');
  assertTaskDrawer(win, false, 'remote: the task drawer expands on click');
  assertInputDrawer(win, true, 'remote: the composer collapses on click');
  assert.strictEqual(
    cs(win, 'task-panel-text').whiteSpace,
    'pre-wrap',
    'remote: expanding must wrap the task text',
  );
  assert.strictEqual(
    cs(win, 'output').flexGrow,
    '1',
    'remote: #output must keep flex:1 to absorb the freed space',
  );

  click(win, 'task-panel-drawer-btn');
  click(win, 'input-drawer-btn');
  assertTaskDrawer(win, true, 'remote: the task drawer collapses again');
  assertInputDrawer(win, false, 'remote: the composer comes back');
  win.close();
}

// Nothing the backend says may open the task panel: only the chevron can.
function testTaskPanelNeverAutoExpands() {
  for (const remote of [false, true]) {
    persistedState = undefined;
    const {win, posted} = makeWebview({remote});
    const tabId = readyTabId(posted);
    const why = remote ? 'remote' : 'extension';

    showTaskPanel(win, posted);
    assertTaskDrawer(win, true, `${why}: a task replay must not expand it`);

    setRunning(win, posted, true);
    assertTaskDrawer(win, true, `${why}: a task starting must not expand it`);

    send(win, {
      type: 'setTaskText',
      text: 'a much longer task text that would love the extra room',
      tabId: tabId,
    });
    assertTaskDrawer(win, true, `${why}: new task text must not expand it`);

    send(win, {
      type: 'task_events',
      events: [],
      task: 'a brand new task text arriving mid-flight',
      tabId: tabId,
      chat_id: 'chat-drawer',
    });
    assertTaskDrawer(win, true, `${why}: a fresh replay must not expand it`);

    setRunning(win, posted, false);
    assertTaskDrawer(win, true, `${why}: a task ending must not expand it`);
    assert.strictEqual(
      win.document.getElementById('task-panel-text').textContent,
      'a brand new task text arriving mid-flight',
      `${why}: the slim drawer still tracks the latest task text`,
    );
    win.close();
  }
}

function testDrawerStateSurvivesTaskChurn() {
  persistedState = undefined;
  const {win, posted} = makeWebview();
  showTaskPanel(win, posted);

  click(win, 'task-panel-drawer-btn');
  click(win, 'input-drawer-btn');
  setRunning(win, posted, true);
  send(win, {
    type: 'task_events',
    events: [],
    task: 'a brand new task text arriving mid-flight',
    tabId: readyTabId(posted),
    chat_id: 'chat-drawer',
  });
  setRunning(win, posted, false);

  assertTaskDrawer(
    win,
    false,
    'status/task churn must not re-collapse the task drawer the user opened',
  );
  assertInputDrawer(
    win,
    true,
    'status/task churn must not re-open the composer the user closed',
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
  click(win, 'input-drawer-btn');
  assert.strictEqual(
    cs(win, 'task-panel').paddingTop,
    '4px',
    'remote: collapsed task drawer must keep the extension padding ' +
      '(remote-codex.css no longer restyles #task-panel)',
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

// The whole point of requirement 3: a phone with nothing running shows the
// textbox and the buttons.
function testMobileRemoteIdleShowsComposer() {
  for (const [name, ua] of [
    ['iPhone Safari', UA_IPHONE],
    ['Android Chrome', UA_ANDROID],
  ]) {
    persistedState = undefined;
    const {win, posted} = makeWebview({remote: true, userAgent: ua});
    showTaskPanel(win, posted);
    const d = win.document;
    assertTaskDrawer(win, true, `${name}: the task panel opens collapsed`);
    assertInputDrawer(win, false, `${name}: nothing is running, so type away`);
    const container = d.getElementById('input-container');
    assert.ok(
      container.contains(d.getElementById('task-input')),
      'the input textbox must live inside the composer',
    );
    assert.ok(
      container.contains(d.getElementById('input-footer')),
      'the buttons panel must live inside the composer',
    );
    assert.notStrictEqual(
      cs(win, 'task-input').display,
      'none',
      `${name}: the input textbox itself must be visible`,
    );
    assert.notStrictEqual(
      cs(win, 'input-footer').display,
      'none',
      `${name}: the buttons must be visible`,
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

// While a task runs the phone gives the transcript the screen, and hands the
// composer back the moment the task ends.
function testMobileRemoteFollowsRunningState() {
  persistedState = undefined;
  const {win, posted} = makeWebview({remote: true, userAgent: UA_IPHONE});
  showTaskPanel(win, posted);
  assertInputDrawer(win, false, 'idle phone shows the composer');

  setRunning(win, posted, true);
  assertInputDrawer(win, true, 'a running task folds the phone composer');
  assertTaskDrawer(win, true, 'the task panel stays collapsed throughout');

  setRunning(win, posted, false);
  assertInputDrawer(win, false, 'the finished task hands the composer back');
  win.close();
}

// A running task restored by the launch replay counts too: the registry
// snapshot opens the tab and the replayed `status` reports it running.
function testMobileRemoteRunningSnapshotFoldsComposer() {
  persistedState = undefined;
  const {win} = makeWebview({remote: true, userAgent: UA_ANDROID});
  assertInputDrawer(win, false, 'nothing running yet');
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 'tab-live', chatId: 'chat-live', title: 'live', workDir: ''},
    ],
  });
  send(win, {type: 'status', running: true, tabId: 'tab-live', startTs: 1000});
  assertInputDrawer(win, true, 'the launch replay says a task is running');
  win.close();
}

function testDesktopRemoteDefaults() {
  persistedState = undefined;
  const {win, posted} = makeWebview({remote: true, userAgent: UA_DESKTOP});
  showTaskPanel(win, posted);
  assertTaskDrawer(win, true, 'desktop remote: the task panel opens collapsed');
  assertInputDrawer(win, false, 'desktop remote: the composer is reachable');
  setRunning(win, posted, true);
  assertInputDrawer(
    win,
    false,
    'a desktop screen fits both, so a running task keeps the composer',
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
  setRunning(wvMobile.win, wvMobile.posted, true);
  assertInputDrawer(
    wvMobile.win,
    true,
    'userAgentData.mobile=true is a phone: fold the running composer',
  );
  wvMobile.win.close();

  persistedState = undefined;
  const wvDesktop = makeWebview({
    remote: true,
    userAgent: UA_DESKTOP,
    userAgentData: {mobile: false},
  });
  setRunning(wvDesktop.win, wvDesktop.posted, true);
  assertInputDrawer(
    wvDesktop.win,
    false,
    'userAgentData.mobile=false keeps the composer while running',
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
  setRunning(wvIpad.win, wvIpad.posted, true);
  assertInputDrawer(
    wvIpad.win,
    true,
    'Macintosh UA with a multi-touch screen is an iPad',
  );
  wvIpad.win.close();

  persistedState = undefined;
  const wvMac = makeWebview({
    remote: true,
    userAgent: UA_IPAD_MASQUERADE,
    maxTouchPoints: 0,
  });
  setRunning(wvMac.win, wvMac.posted, true);
  assertInputDrawer(
    wvMac.win,
    false,
    'Macintosh UA without touch is a real Mac',
  );
  wvMac.win.close();
}

function testMobileUaVscodeWebviewUnaffected() {
  persistedState = undefined;
  const {win, posted} = makeWebview({remote: false, userAgent: UA_IPHONE});
  showTaskPanel(win, posted);
  assertTaskDrawer(win, true, 'the extension webview also opens collapsed');
  setRunning(win, posted, true);
  assertInputDrawer(
    win,
    false,
    'the extension webview (no body.remote-chat) never folds the composer',
  );
  win.close();
}

function testMobileUserChoicePersists() {
  persistedState = undefined;
  const wv1 = makeWebview({remote: true, userAgent: UA_IPHONE});
  showTaskPanel(wv1.win, wv1.posted);
  click(wv1.win, 'input-drawer-btn');
  assertInputDrawer(wv1.win, true, 'the user folded the composer away');
  setRunning(wv1.win, wv1.posted, false);
  assertInputDrawer(
    wv1.win,
    true,
    'an idle phone must not undo the fold the user asked for',
  );
  wv1.win.close();

  const wv2 = makeWebview({remote: true, userAgent: UA_IPHONE});
  showTaskPanel(wv2.win, wv2.posted);
  assertInputDrawer(wv2.win, true, 'a reload restores the user-folded drawer');
  click(wv2.win, 'input-drawer-btn');
  setRunning(wv2.win, wv2.posted, true);
  assertInputDrawer(
    wv2.win,
    false,
    'a running task must not fold a composer the user opened by hand',
  );
  wv2.win.close();
}

// A blob from an older build has no `*UserSet` flags, so it cannot prove the
// user ever clicked anything: the defaults win.
function testLegacyStateDoesNotResurrectExpandedPanel() {
  for (const ua of [UA_IPHONE, UA_DESKTOP]) {
    persistedState = {
      tabs: [{title: 'old chat', chatId: 'tab-1'}],
      activeTabIndex: 0,
      chatId: 'tab-1',
      taskDrawerCollapsed: false,
      inputDrawerCollapsed: true,
    };
    const wv = makeWebview({remote: true, userAgent: ua});
    assertTaskDrawer(
      wv.win,
      true,
      'a legacy blob must not resurrect an expanded task panel',
    );
    assertInputDrawer(
      wv.win,
      false,
      'a legacy blob must not resurrect a folded composer',
    );
    wv.win.close();

    persistedState = {
      tabs: [{title: 'old chat', chatId: 'tab-1'}],
      activeTabIndex: 0,
      chatId: 'tab-1',
      taskDrawerCollapsed: false,
      inputDrawerCollapsed: true,
      drawersVersion: 2,
    };
    const wvV2 = makeWebview({remote: true, userAgent: ua});
    assertTaskDrawer(
      wvV2.win,
      true,
      'a v2 blob predates the *UserSet flags, so it proves nothing',
    );
    assertInputDrawer(wvV2.win, false, 'a v2 blob cannot fold the composer');
    wvV2.win.close();
  }

  persistedState = {
    tabs: [{title: 'old chat', chatId: 'tab-1'}],
    activeTabIndex: 0,
    chatId: 'tab-1',
    taskDrawerCollapsed: false,
    taskDrawerUserSet: true,
    inputDrawerCollapsed: true,
    inputDrawerUserSet: true,
    drawersVersion: 3,
  };
  const wvNew = makeWebview({remote: true, userAgent: UA_IPHONE});
  assertTaskDrawer(
    wvNew.win,
    false,
    'a current blob that records the click restores the expanded panel',
  );
  assertInputDrawer(
    wvNew.win,
    true,
    'a current blob that records the click restores the folded composer',
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
    assertTaskDrawer(
      win,
      true,
      `a malformed (${typeof bad}) blob falls back to the collapsed default`,
    );
    assertInputDrawer(
      win,
      false,
      `a malformed (${typeof bad}) blob falls back to a reachable composer`,
    );
    win.close();
  }
}

function runTests() {
  const tests = [
    testDefaults,
    testInputDrawerToggle,
    testTaskDrawerToggle,
    testPersistenceAcrossReopen,
    testPersistenceSingleDrawer,
    testRemoteWebApp,
    testTaskPanelNeverAutoExpands,
    testDrawerStateSurvivesTaskChurn,
    testChatsCollapseButtonRemoved,
    testDrawerButtonsBigEnough,
    testDrawerButtonsLoseFocusAfterClick,
    testRemoteCollapsedPadding,
    testMissingDrawerButtonsGracefulBoot,
    testMobileRemoteIdleShowsComposer,
    testMobileRemoteFollowsRunningState,
    testMobileRemoteRunningSnapshotFoldsComposer,
    testDesktopRemoteDefaults,
    testUserAgentDataMobileRemote,
    testIpadMasqueradeRemote,
    testMobileUaVscodeWebviewUnaffected,
    testMobileUserChoicePersists,
    testLegacyStateDoesNotResurrectExpandedPanel,
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
