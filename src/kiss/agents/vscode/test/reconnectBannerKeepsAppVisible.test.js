// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// main.js on a lost connection (remote webapp):
//
//   * `daemonStatus {connected:false, reconnecting:true}` — the shim's
//     post when the socket dropped AFTER this page was authenticated —
//     keeps #app visible and shows the overlay as the slim banner
//     (`kiss-server-loading--banner`), so a flaky link does not blank
//     the app on every blip;
//   * a plain `connected:false` (cold start, auth lockout, dismissed
//     password prompt) still hides #app behind the full overlay;
//   * `connected:true` clears both the overlay and the banner class;
//   * while the daemon is down the user's posts are held back — the
//     prompt stays in the composer (also when the drop happens while a
//     photo attachment is still converting), the answer stays in the
//     composer too and a dirty file tab stays dirty — instead of
//     being queued and then dropped by the reload that follows the
//     reconnect;
//   * the composer draft is persisted on `pagehide` and shown again by
//     the reloaded page, in the tab that gets the screen back;
//   * a prompt or ask-user answer the daemon never acknowledged (sent
//     into a connection that died silently) is persisted the same way,
//     so it comes back as a draft instead of vanishing.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(initialState) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  html = html.replace('<body', '<body class="remote-chat"');
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
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
  win.matchMedia = function (query) {
    return {
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    };
  };
  const created = [];
  win.monaco = {
    editor: {
      create: (holder, opts) => {
        let value = String(opts.value === undefined ? '' : opts.value);
        let version = 1;
        const listeners = [];
        const model = {
          getAlternativeVersionId: () => version,
          getValue: () => value,
          getLineCount: () => value.split('\n').length,
        };
        const editor = {
          getModel: () => model,
          onDidChangeModelContent: cb => listeners.push(cb),
          setPosition: () => {},
          revealLineInCenter: () => {},
          getDomNode: () => holder,
          dispose: () => {},
          layout: () => {},
          focus: () => {},
          _type: text => {
            value = String(text);
            version += 1;
            listeners.slice().forEach(cb => cb());
          },
        };
        created.push(editor);
        return editor;
      },
    },
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted, created, getState: () => state};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function uiState(win) {
  const overlay = win.document.getElementById('kiss-server-loading');
  const app = win.document.getElementById('app');
  return {
    overlayShown: overlay.style.display !== 'none',
    banner: overlay.classList.contains('kiss-server-loading--banner'),
    appShown: app.style.display !== 'none',
  };
}

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function clickSend(win) {
  click(win, win.document.getElementById('send-btn'));
}

function activateTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} is in the tab bar`);
  click(win, el);
  assert.strictEqual(win._testApi.getActiveTabId(), tabId);
}

// Objects persisted by the page live in the jsdom realm: compare by value.
function draftsOf(getState) {
  const d = getState().inputDrafts || {};
  return Object.keys(d)
    .sort()
    .map(id => [id, d[id]]);
}

function askDraftsOf(getState) {
  const d = getState().askDrafts || {};
  return Object.keys(d)
    .sort()
    .map(id => [id, d[id].question, d[id].answer]);
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function waitFor(predicate, message, timeoutMs = 2000) {
  const start = Date.now();
  for (;;) {
    const value = predicate();
    if (value) return value;
    if (Date.now() - start >= timeoutMs) throw new Error(message);
    await sleep(10);
  }
}

function testReconnectingKeepsAppVisibleUnderBanner() {
  const {win} = makeWebview();
  send(win, {type: 'daemonStatus', connected: true});
  assert.deepStrictEqual(
    uiState(win),
    {overlayShown: false, banner: false, appShown: true},
    'connected: overlay hidden, app shown',
  );
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  assert.deepStrictEqual(
    uiState(win),
    {overlayShown: true, banner: true, appShown: true},
    'reconnecting: overlay up AS A BANNER, app still on screen',
  );
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  assert.deepStrictEqual(
    uiState(win),
    {overlayShown: true, banner: true, appShown: true},
    'repeated reconnecting posts (each failed retry) keep the banner',
  );
  send(win, {type: 'daemonStatus', connected: true});
  assert.deepStrictEqual(
    uiState(win),
    {overlayShown: false, banner: false, appShown: true},
    'reconnected: overlay gone and the banner class cleared',
  );
  win.close();
  console.log('PASS reconnecting keeps #app visible under the banner');
}

function testPlainDisconnectStillHidesApp() {
  const {win} = makeWebview();
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  // Auth lockout / dismissed password prompt: re-gate the whole app.
  send(win, {type: 'daemonStatus', connected: false});
  assert.deepStrictEqual(
    uiState(win),
    {overlayShown: true, banner: false, appShown: false},
    'a plain connected:false after a banner hides #app behind the ' +
      'full overlay again',
  );
  win.close();
  console.log('PASS plain disconnect hides #app behind the full overlay');
}

function testColdStartIgnoresNonBooleanReconnecting() {
  const {win} = makeWebview();
  send(win, {type: 'daemonStatus', connected: false, reconnecting: 'yes'});
  assert.deepStrictEqual(
    uiState(win),
    {overlayShown: true, banner: false, appShown: false},
    'only reconnecting === true selects the banner',
  );
  win.close();
  console.log('PASS non-boolean reconnecting is not a banner');
}

function testSendHeldBackWhileDaemonDown() {
  const {win, posted} = makeWebview();
  send(win, {type: 'daemonStatus', connected: true});
  const inp = win.document.getElementById('task-input');
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  posted.length = 0;
  inp.value = 'typed during the outage';
  clickSend(win);
  assert.strictEqual(
    inp.value,
    'typed during the outage',
    'the prompt must stay in the composer while the daemon is down',
  );
  assert.ok(
    !posted.some(m => m.type === 'submit' || m.type === 'appendUserMessage'),
    'no submit may be posted while the daemon is down: ' +
      JSON.stringify(posted),
  );

  send(win, {type: 'daemonStatus', connected: true});
  posted.length = 0;
  clickSend(win);
  assert.ok(
    posted.some(
      m => m.type === 'submit' && m.prompt === 'typed during the outage',
    ),
    'once reconnected the same prompt goes out: ' + JSON.stringify(posted),
  );
  assert.strictEqual(inp.value, '', 'a sent prompt clears the composer');
  win.close();
  console.log('PASS send is held back while the daemon is down');
}

// A photo is still converting when Enter is pressed; the connection
// drops during that wait.  The send must notice and keep the prompt.
async function testSendReChecksConnectionAfterAttachmentWait() {
  const {win, posted} = makeWebview();
  // The HEIC decoder the page asks the browser for, parked forever.
  win.createImageBitmap = () => new Promise(() => {});
  win._testApi.endLaunch();
  send(win, {type: 'daemonStatus', connected: true});
  const inp = win.document.getElementById('task-input');
  const heic = new win.File([new Uint8Array([1, 2, 3])], 'IMG_0001.HEIC', {
    type: 'image/heic',
  });
  const paste = new win.Event('paste', {bubbles: true, cancelable: true});
  Object.defineProperty(paste, 'clipboardData', {
    value: {items: [{kind: 'file', getAsFile: () => heic}]},
  });
  inp.dispatchEvent(paste);
  await sleep(50);
  inp.value = 'prompt with a photo';
  inp.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
  );
  await sleep(50);
  assert.ok(
    !posted.some(m => m.type === 'submit'),
    'the send waits for the photo',
  );
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  // Removing the stuck chip releases the parked send while the daemon
  // is down.
  click(win, win.document.querySelector('.file-chip .fc-rm'));
  await sleep(50);
  assert.ok(
    !posted.some(m => m.type === 'submit' || m.type === 'appendUserMessage'),
    'a send released while the daemon is down must not be posted: ' +
      JSON.stringify(posted),
  );
  assert.strictEqual(
    inp.value,
    'prompt with a photo',
    'the prompt stays in the composer',
  );
  send(win, {type: 'daemonStatus', connected: true});
  clickSend(win);
  assert.ok(
    posted.some(m => m.type === 'submit' && m.prompt === 'prompt with a photo'),
    'once reconnected the prompt goes out',
  );
  win.close();
  console.log('PASS send re-checks the connection after the attachment wait');
}

function testAskAnswerHeldBackWhileDaemonDown() {
  const {win, posted} = makeWebview();
  send(win, {type: 'daemonStatus', connected: true});
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'askUser', tabId, question: 'Proceed?'});
  const input = win.document.getElementById('task-input');
  assert.ok(
    win.document.body.classList.contains('ask-answering'),
    'the composer is in answer mode',
  );
  input.value = 'yes, go ahead';
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  posted.length = 0;
  clickSend(win);
  assert.ok(
    !posted.some(m => m.type === 'userAnswer'),
    'no userAnswer may be posted while the daemon is down',
  );
  assert.strictEqual(
    input.value,
    'yes, go ahead',
    'the answer stays in the composer',
  );
  send(win, {type: 'daemonStatus', connected: true});
  clickSend(win);
  const answers = posted.filter(m => m.type === 'userAnswer');
  assert.deepStrictEqual(
    answers.map(a => [a.answer, a.tabId]),
    [['yes, go ahead', tabId]],
    'once reconnected the same answer goes out',
  );
  win.close();
  console.log('PASS ask-user answer is held back while the daemon is down');
}

async function testSaveHeldBackWhileDaemonDown() {
  const {win, posted, created} = makeWebview();
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {work_dir: '/ws'}, apiKeys: {}});
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 'a1', chatId: 'chat-1', title: 'a1', workDir: '/ws'}],
  });
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/notes.txt',
    name: 'notes.txt',
    content: 'v1 on disk',
    version: 'ver-1',
  });
  await waitFor(() => created.length >= 1, 'the file tab creates its editor');
  created[0]._type('v1 edited');
  const saveBtn = win.document.querySelector('.content-save-btn');
  assert.strictEqual(saveBtn.disabled, false, 'dirty tab: Save enabled');

  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  posted.length = 0;
  click(win, saveBtn);
  assert.ok(
    !posted.some(m => m.type === 'saveFile'),
    'no saveFile may be posted while the daemon is down',
  );
  const status = win.document.querySelector('.content-save-status');
  assert.strictEqual(
    status.textContent,
    'Not connected: try again when reconnected',
  );
  assert.ok(status.classList.contains('error'));
  assert.strictEqual(
    saveBtn.disabled,
    false,
    'the tab stays dirty: Save enabled',
  );

  send(win, {type: 'daemonStatus', connected: true});
  click(win, saveBtn);
  const saves = posted.filter(m => m.type === 'saveFile');
  assert.strictEqual(saves.length, 1, 'once reconnected the save goes out');
  assert.strictEqual(saves[0].content, 'v1 edited');
  win.close();
  console.log('PASS file save is held back while the daemon is down');
}

// The draft typed before the reload comes back in the reloaded page:
// shown at once in the boot tab and carried to the tab the registry
// snapshot puts on screen (the restored active tab when it still
// exists, the first visible tab otherwise).
function testComposerDraftSurvivesReload() {
  const first = makeWebview();
  send(first.win, {type: 'daemonStatus', connected: true});
  send(first.win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 't1', chatId: 'c1', title: 't1', workDir: ''},
      {tabId: 't2', chatId: 'c2', title: 't2', workDir: ''},
    ],
  });
  // A draft left in a background tab and one in the selected tab.
  first.win.document.getElementById('task-input').value = 'left in t1';
  activateTab(first.win, 't2');
  first.win.document.getElementById('task-input').value = 'half-written prompt';
  send(first.win, {type: 'daemonStatus', connected: false, reconnecting: true});
  // The shim reloads on reconnect: the page goes down.
  first.win.dispatchEvent(new first.win.Event('pagehide'));
  const persisted = first.getState();
  assert.deepStrictEqual(draftsOf(first.getState), [
    ['t1', 'left in t1'],
    ['t2', 'half-written prompt'],
  ]);
  assert.strictEqual(persisted.chatId, 't2');
  first.win.close();

  // Restored active tab still in the registry.
  const second = makeWebview(persisted);
  const inp2 = second.win.document.getElementById('task-input');
  assert.strictEqual(
    inp2.value,
    'half-written prompt',
    'the boot tab shows the draft before the first snapshot',
  );
  send(second.win, {type: 'daemonStatus', connected: true});
  send(second.win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 't1', chatId: 'c1', title: 't1', workDir: ''},
      {tabId: 't2', chatId: 'c2', title: 't2', workDir: ''},
    ],
  });
  assert.strictEqual(second.win._testApi.getActiveTabId(), 't2');
  assert.strictEqual(
    inp2.value,
    'half-written prompt',
    'the restored active tab shows the draft',
  );
  activateTab(second.win, 't1');
  assert.strictEqual(
    inp2.value,
    'left in t1',
    'the background tab has its own draft back',
  );
  activateTab(second.win, 't2');
  assert.strictEqual(inp2.value, 'half-written prompt');
  second.win.close();

  // Restored active tab gone (closed elsewhere during the outage): the
  // tab taking the screen keeps its own draft ...
  const third0 = makeWebview(persisted);
  send(third0.win, {type: 'daemonStatus', connected: true});
  send(third0.win, {
    type: 'tabs_state',
    tabs: [{tabId: 't1', chatId: 'c1', title: 't1', workDir: ''}],
  });
  assert.strictEqual(third0.win._testApi.getActiveTabId(), 't1');
  assert.strictEqual(
    third0.win.document.getElementById('task-input').value,
    'left in t1',
    'a tab with its own draft keeps it',
  );
  third0.win.close();
  // ... and inherits the orphaned draft when it has none.
  const third = makeWebview({
    chatId: 't2',
    inputDrafts: {t2: 'half-written prompt'},
  });
  const inp3 = third.win.document.getElementById('task-input');
  send(third.win, {type: 'daemonStatus', connected: true});
  send(third.win, {
    type: 'tabs_state',
    tabs: [{tabId: 't1', chatId: 'c1', title: 't1', workDir: ''}],
  });
  assert.strictEqual(third.win._testApi.getActiveTabId(), 't1');
  assert.strictEqual(
    inp3.value,
    'half-written prompt',
    'the tab that takes the screen inherits the draft',
  );
  // A second snapshot does not hand the draft out again.
  inp3.value = '';
  activateTab(third.win, 't1');
  send(third.win, {
    type: 'tabs_state',
    tabs: [{tabId: 't3', chatId: 'c3', title: 't3', workDir: ''}],
  });
  assert.strictEqual(third.win._testApi.getActiveTabId(), 't3');
  assert.strictEqual(inp3.value, '', 'the draft is handed out once');
  third.win.close();

  // No draft persisted (older state shape): nothing is seeded.
  const fourth = makeWebview({chatId: 'x', inputDraft: 'legacy'});
  assert.strictEqual(
    fourth.win.document.getElementById('task-input').value,
    '',
  );
  fourth.win.close();
  console.log('PASS the composer draft survives the reconnect reload');
}

// A prompt sent into a connection that died silently: the composer is
// already cleared and nothing tells the page.  While the daemon has not
// named the task (the tab still holds its pendingTaskId claim), the
// prompt is what the reloaded page shows as the draft; once any event
// names the task the prompt was taken and nothing comes back.
function testUnacknowledgedPromptComesBackAsDraft() {
  const {win, posted, getState} = makeWebview();
  win._testApi.endLaunch();
  send(win, {type: 'daemonStatus', connected: true});
  const tabId = win._testApi.getActiveTabId();
  const inp = win.document.getElementById('task-input');
  inp.value = 'sent into a dead socket';
  clickSend(win);
  assert.ok(
    posted.some(m => m.type === 'submit'),
    'the prompt was posted',
  );
  assert.strictEqual(inp.value, '', 'the composer is cleared on send');
  win.dispatchEvent(new win.Event('pagehide'));
  assert.deepStrictEqual(
    draftsOf(getState),
    [[tabId, 'sent into a dead socket']],
    'an unacknowledged prompt is persisted as the draft',
  );

  // The daemon echoes the prompt (setTaskText): it arrived.
  send(win, {type: 'setTaskText', text: 'sent into a dead socket', tabId});
  win.dispatchEvent(new win.Event('pagehide'));
  assert.deepStrictEqual(
    draftsOf(getState),
    [],
    'an acknowledged prompt is not',
  );

  // The task runs (the tab now has a task id); a follow-up typed while
  // it runs is a different command (appendUserMessage) with its own
  // echo (prompt).
  send(win, {type: 'system_output', text: 'started', tabId, taskId: 'task-1'});
  send(win, {type: 'status', running: true, tabId});
  inp.value = 'follow-up into a dead socket';
  clickSend(win);
  assert.ok(
    posted.some(
      m =>
        m.type === 'appendUserMessage' &&
        m.prompt === 'follow-up into a dead socket',
    ),
    'the follow-up was posted',
  );
  win.dispatchEvent(new win.Event('pagehide'));
  assert.deepStrictEqual(
    draftsOf(getState),
    [[tabId, 'follow-up into a dead socket']],
    'an unacknowledged follow-up is persisted',
  );
  send(win, {type: 'prompt', text: 'follow-up into a dead socket', tabId});
  win.dispatchEvent(new win.Event('pagehide'));
  assert.deepStrictEqual(draftsOf(getState), [], 'the echo acknowledges it');

  // The next task after this one ended: the tab keeps its old task id,
  // so this is the common second-prompt case.
  send(win, {type: 'status', running: false, tabId});
  inp.value = 'next task into a dead socket';
  clickSend(win);
  assert.ok(
    posted.some(
      m => m.type === 'submit' && m.prompt === 'next task into a dead socket',
    ),
  );
  win.dispatchEvent(new win.Event('pagehide'));
  assert.deepStrictEqual(
    draftsOf(getState),
    [[tabId, 'next task into a dead socket']],
    'an unacknowledged second task is persisted',
  );
  send(win, {type: 'setTaskText', text: 'next task into a dead socket', tabId});

  // An echo without a tabId (the VS Code extension's run path) is for
  // the tab on screen.
  inp.value = 'unaddressed echo';
  clickSend(win);
  send(win, {type: 'setTaskText', text: 'unaddressed echo'});
  win.dispatchEvent(new win.Event('pagehide'));
  assert.deepStrictEqual(
    draftsOf(getState),
    [],
    'an unaddressed echo acknowledges',
  );

  // Text typed since wins over the unacknowledged prompt.
  inp.value = 'a second prompt';
  clickSend(win);
  inp.value = 'typed after sending';
  win.dispatchEvent(new win.Event('pagehide'));
  assert.deepStrictEqual(draftsOf(getState), [[tabId, 'typed after sending']]);
  win.close();
  console.log('PASS an unacknowledged prompt comes back as the draft');
}

// Unacknowledged values of a tab the user has since switched away from
// are persisted too: the send happened on T1, the user moved to T2 before
// the dead socket was noticed.
function testBackgroundTabUnackedValuesArePersisted() {
  const {win, getState} = makeWebview();
  win._testApi.endLaunch();
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 't1', chatId: 'c1', title: 't1', workDir: ''},
      {tabId: 't2', chatId: 'c2', title: 't2', workDir: ''},
    ],
  });
  activateTab(win, 't1');
  win.document.getElementById('task-input').value = 'prompt from t1';
  clickSend(win);
  send(win, {type: 'askUser', tabId: 't1', question: 'Sure?'});
  win.document.getElementById('task-input').value = 'answer from t1';
  clickSend(win);
  activateTab(win, 't2');
  win.dispatchEvent(new win.Event('pagehide'));
  assert.deepStrictEqual(draftsOf(getState), [['t1', 'prompt from t1']]);
  assert.deepStrictEqual(askDraftsOf(getState), [
    ['t1', 'Sure?', 'answer from t1'],
  ]);
  win.close();
  console.log('PASS background-tab unacknowledged values are persisted');
}

// The same for the ask-user answer: typed or sent-but-unconfirmed, it is
// persisted with its question until the daemon's askUserDone confirms
// it, and the reloaded page puts it back into the composer only if the
// daemon re-asks that tab's question (an answered question is never
// re-asked).
function testUnacknowledgedAnswerComesBackInTheComposer() {
  const first = makeWebview();
  send(first.win, {type: 'daemonStatus', connected: true});
  send(first.win, {
    type: 'tabs_state',
    tabs: [{tabId: 't1', chatId: 'c1', title: 't1', workDir: ''}],
  });
  send(first.win, {type: 'askUser', tabId: 't1', question: 'Deploy?'});
  first.win.document.getElementById('task-input').value = 'yes, to staging';
  // Typed but not yet sent: persisted with its question, not as a
  // prompt draft (the composer is the answer box right now).
  first.win.dispatchEvent(new first.win.Event('pagehide'));
  assert.deepStrictEqual(draftsOf(first.getState), []);
  assert.deepStrictEqual(askDraftsOf(first.getState), [
    ['t1', 'Deploy?', 'yes, to staging'],
  ]);
  clickSend(first.win);
  assert.ok(
    first.posted.some(
      m => m.type === 'userAnswer' && m.answer === 'yes, to staging',
    ),
  );
  assert.ok(
    !first.win.document.body.classList.contains('ask-answering'),
    'answer mode ends on send as before',
  );
  first.win.dispatchEvent(new first.win.Event('pagehide'));
  const persisted = first.getState();
  assert.deepStrictEqual(draftsOf(first.getState), []);
  assert.deepStrictEqual(
    askDraftsOf(first.getState),
    [['t1', 'Deploy?', 'yes, to staging']],
    'an unconfirmed answer is persisted',
  );

  // Confirmed: nothing to carry over.
  send(first.win, {type: 'askUserDone', tabId: 't1'});
  first.win.dispatchEvent(new first.win.Event('pagehide'));
  assert.deepStrictEqual(askDraftsOf(first.getState), []);
  first.win.close();

  // Reloaded page: the daemon still has the question -> the answer is
  // back in the composer, once.
  const second = makeWebview(persisted);
  send(second.win, {type: 'daemonStatus', connected: true});
  send(second.win, {
    type: 'tabs_state',
    tabs: [{tabId: 't1', chatId: 'c1', title: 't1', workDir: ''}],
  });
  const inp2 = second.win.document.getElementById('task-input');
  send(second.win, {type: 'askUser', tabId: 't2', question: 'Other tab?'});
  send(second.win, {type: 'askUser', tabId: 't1', question: 'Deploy?'});
  assert.strictEqual(
    inp2.value,
    'yes, to staging',
    'the unconfirmed answer is shown again',
  );
  inp2.value = '';
  send(second.win, {type: 'askUserDone', tabId: 't1'});
  send(second.win, {type: 'askUser', tabId: 't1', question: 'Again?'});
  assert.strictEqual(inp2.value, '', 'the draft is handed out once');
  second.win.close();

  // Reloaded page where the daemon asks that tab a DIFFERENT question:
  // the old answer was taken; it must not be put into the new question.
  const other = makeWebview(persisted);
  send(other.win, {type: 'daemonStatus', connected: true});
  send(other.win, {
    type: 'tabs_state',
    tabs: [{tabId: 't1', chatId: 'c1', title: 't1', workDir: ''}],
  });
  const inpOther = other.win.document.getElementById('task-input');
  send(other.win, {type: 'askUser', tabId: 't1', question: 'Something new?'});
  assert.strictEqual(inpOther.value, '');
  send(other.win, {type: 'askUserDone', tabId: 't1'});
  send(other.win, {type: 'askUser', tabId: 't1', question: 'Deploy?'});
  assert.strictEqual(
    inpOther.value,
    '',
    'a different question drops the draft for good',
  );
  other.win.close();

  // Reloaded page for another tab: the draft stays out of its composer.
  const third = makeWebview(persisted);
  send(third.win, {type: 'daemonStatus', connected: true});
  send(third.win, {
    type: 'tabs_state',
    tabs: [{tabId: 't9', chatId: 'c9', title: 't9', workDir: ''}],
  });
  send(third.win, {type: 'askUser', tabId: 't9', question: 'Deploy?'});
  assert.strictEqual(third.win.document.getElementById('task-input').value, '');
  third.win.close();
  console.log('PASS an unacknowledged answer comes back in the composer');
}

async function main() {
  testReconnectingKeepsAppVisibleUnderBanner();
  testUnacknowledgedPromptComesBackAsDraft();
  testBackgroundTabUnackedValuesArePersisted();
  testUnacknowledgedAnswerComesBackInTheComposer();
  testPlainDisconnectStillHidesApp();
  testColdStartIgnoresNonBooleanReconnecting();
  testSendHeldBackWhileDaemonDown();
  await testSendReChecksConnectionAfterAttachmentWait();
  testAskAnswerHeldBackWhileDaemonDown();
  await testSaveHeldBackWhileDaemonDown();
  testComposerDraftSurvivesReload();
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
