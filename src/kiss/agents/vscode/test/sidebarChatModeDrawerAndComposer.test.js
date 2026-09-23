// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the extension's SIDEBAR chat view surface
// (media/chat.html + media/main.js loaded with no special body class):
//
// * main.js marks the surface with body.sidebar-chat-mode (never on
//   the remote page or an editor-tab webview);
// * main.css then shows the task-info drawer button in the tab bar,
//   hides the burger and the top status bar, and turns #meta-panel
//   into a right-hand drawer — the mobile remote webapp's behavior;
// * the drawer toggle/close/backdrop/Escape flows work, the closed
//   drawer is inert, and the 5s getTaskUpdate poll runs only while the
//   drawer is open and a task runs; the #meta-info refresh button posts
//   a getTaskUpdate with refresh:true and mirrors the agent's running
//   state;
// * the host's `openChatFromHistory` message (a primary-sidebar
//   history click in sidebar mode) mirrors the in-page history rows:
//   switch to the chat's tab, resume the chat in a fresh tab, or show
//   the task text read-only — and is ignored by editor-tab webviews;
// * body.content-tab-open (a file/webview tab in front) keeps the
//   composer's button row (+ and ...) while hiding the text box and
//   the chat-only controls (Inject promptlet, model picker, Send).

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const POLL_MS = 5000;

let passed = 0;
const failures = [];

async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.stack || e.message}`);
  }
}

/**
 * Replace the window's setInterval/clearInterval with a fake clock so
 * the 5s task-update poll can be advanced deterministically instead of
 * waited out. setTimeout stays real (message handlers debounce with it).
 *
 * @param {Window} win The jsdom window, before main.js is evaluated.
 * @returns {{tick: function(number), intervals: Array}} `tick(ms)`
 *   advances the clock and fires every due interval callback;
 *   `intervals` lists the live intervals as {cb, ms, next}.
 */
function installFakeIntervals(win) {
  let now = 0;
  let nextId = 1;
  const intervals = [];
  win.setInterval = function (cb, ms) {
    const iv = {id: nextId++, cb, ms: Number(ms) || 0, next: now + (Number(ms) || 0)};
    intervals.push(iv);
    return iv.id;
  };
  win.clearInterval = function (id) {
    const i = intervals.findIndex(iv => iv.id === id);
    if (i >= 0) intervals.splice(i, 1);
  };
  function tick(ms) {
    const target = now + ms;
    for (;;) {
      const due = intervals.filter(iv => iv.next <= target);
      if (due.length === 0) break;
      due.sort((a, b) => a.next - b.next);
      const iv = due[0];
      now = iv.next;
      iv.next += iv.ms;
      iv.cb();
    }
    now = target;
  }
  return {tick, intervals};
}

function makeWebview(bodyAttrs) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  if (bodyAttrs) html = html.replace('<body', '<body' + bodyAttrs);
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
  const timers = installFakeIntervals(win);
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=sidebarchat-main.js',
  );
  return {win, posted, tick: timers.tick, intervals: timers.intervals};
}

function injectMainCss(win) {
  const style = win.document.createElement('style');
  style.textContent = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  win.document.head.appendChild(style);
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function click(win, id) {
  win.document
    .getElementById(id)
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function drawerOpen(win) {
  return win.document.getElementById('meta-panel').classList.contains('open');
}

function polls(wv) {
  return wv.posted.filter(m => m.type === 'getTaskUpdate');
}

function pollCount(wv) {
  return polls(wv).length;
}

function lastPoll(wv) {
  const all = polls(wv);
  return all[all.length - 1];
}

/** A taskUpdate reply answering *poll*, with the given overrides. */
function reply(win, poll, fields) {
  send(
    win,
    Object.assign(
      {
        type: 'taskUpdate',
        tabId: poll.tabId,
        token: poll.token,
        taskId: 'task-1',
        exists: true,
        sig: '',
        content: '',
        error: '',
        running: false,
        cost: 0,
        updatedAt: 0,
      },
      fields,
    ),
  );
}

function chatTabs(win) {
  return Array.from(win.document.querySelectorAll('#tab-list .chat-tab'));
}

function activeTabLabel(win) {
  const tab = win.document.querySelector('#tab-list .chat-tab.active');
  const label = tab && tab.querySelector('.chat-tab-label');
  return label ? label.textContent : '';
}

function display(win, id) {
  return win.getComputedStyle(win.document.getElementById(id)).display;
}

async function main() {
  await test(
    'only the plain (sidebar) webview gets body.sidebar-chat-mode',
    () => {
      const plain = makeWebview('');
      assert.ok(
        plain.win.document.body.classList.contains('sidebar-chat-mode'),
        'the sidebar chat view marks itself',
      );
      const remote = makeWebview(' class="remote-chat"');
      assert.ok(
        !remote.win.document.body.classList.contains('sidebar-chat-mode'),
        'the remote page keeps its own drawer wiring',
      );
      const editor = makeWebview(
        ' class="editor-tab-mode" data-kiss-tab-id="root-1"',
      );
      assert.ok(
        !editor.win.document.body.classList.contains('sidebar-chat-mode'),
        'editor-tab webviews relay to the Task Info view instead',
      );
    },
  );

  await test(
    'sidebar mode shows the drawer button, hides the burger and the ' +
      'top status bar (main.css)',
    () => {
      const {win} = makeWebview('');
      injectMainCss(win);
      assert.strictEqual(
        display(win, 'meta-drawer-btn'),
        'inline-flex',
        'the task-info toggle rides the tab bar',
      );
      assert.strictEqual(
        display(win, 'meta-close'),
        'block',
        "the drawer's close button exists on this surface",
      );
      assert.strictEqual(display(win, 'menu-btn'), 'none', 'no burger');
      assert.strictEqual(
        display(win, 'tab-status-bar'),
        'none',
        'the drawer replaces the top status bar, like the mobile app',
      );
      assert.strictEqual(
        display(win, 'meta-resizer'),
        'none',
        'the drawer is not resizable',
      );
      assert.strictEqual(
        display(win, 'meta-panel'),
        'flex',
        'the panel renders (translated off-screen until opened)',
      );
    },
  );

  await test(
    'the drawer opens/closes from the button, close button, backdrop ' +
      'and Escape; closed it is inert',
    () => {
      const {win} = makeWebview('');
      const btn = win.document.getElementById('meta-drawer-btn');
      const panel = win.document.getElementById('meta-panel');
      assert.ok(!drawerOpen(win));
      assert.ok(
        panel.hasAttribute('inert'),
        'the drawer starts inert: its off-screen close button must not ' +
          'sit in the tab order before the drawer was ever opened',
      );
      click(win, 'meta-drawer-btn');
      assert.ok(drawerOpen(win), 'button opens the drawer');
      assert.strictEqual(btn.getAttribute('aria-expanded'), 'true');
      assert.ok(!panel.hasAttribute('inert'), 'an open drawer is usable');
      click(win, 'meta-close');
      assert.ok(!drawerOpen(win), 'the close button dismisses the drawer');
      assert.ok(
        panel.hasAttribute('inert'),
        'the closed drawer leaves the keyboard tab order',
      );
      click(win, 'meta-drawer-btn');
      click(win, 'meta-overlay');
      assert.ok(!drawerOpen(win), 'the backdrop dismisses the drawer');
      click(win, 'meta-drawer-btn');
      win.document.dispatchEvent(
        new win.KeyboardEvent('keydown', {key: 'Escape', bubbles: true}),
      );
      assert.ok(!drawerOpen(win), 'Escape dismisses the drawer');
    },
  );

  await test(
    'getTaskUpdate polls every 5s only while the drawer is open and a ' +
      'task runs, and the reply renders in the drawer',
    () => {
      const wv = makeWebview('');
      const win = wv.win;
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
      });
      wv.tick(POLL_MS * 2);
      assert.strictEqual(pollCount(wv), 0, 'no task, no poll');
      send(win, {type: 'status', running: true});
      wv.tick(POLL_MS * 2);
      assert.strictEqual(pollCount(wv), 0, 'a closed drawer must not poll');
      click(win, 'meta-drawer-btn');
      assert.strictEqual(pollCount(wv), 1, 'opening the drawer polls at once');
      assert.ok(
        wv.intervals.some(iv => iv.ms === POLL_MS),
        'the poll timer ticks every 5000 ms',
      );
      const poll = lastPoll(wv);
      assert.deepStrictEqual(Object.keys(poll).sort(), [
        'knownSig',
        'refresh',
        'tabId',
        'token',
        'type',
      ]);
      assert.ok(!('workDir' in poll), 'the poll names a tab, not a workdir');
      assert.ok(poll.tabId, 'the poll targets the visible chat tab');
      assert.strictEqual(poll.knownSig, '');
      assert.strictEqual(typeof poll.token, 'string');
      assert.strictEqual(poll.refresh, false);
      wv.tick(POLL_MS - 1);
      assert.strictEqual(pollCount(wv), 1, 'no poll before the interval');
      wv.tick(1);
      assert.strictEqual(pollCount(wv), 2, 'the timer polls at 5000 ms');

      reply(win, poll, {
        sig: '5:9',
        content: 'working on **it**\n',
        updatedAt: Date.now(),
        cost: 0.12,
      });
      const info = win.document.getElementById('meta-info');
      const content = win.document.getElementById('meta-info-content');
      assert.ok(info.classList.contains('visible'));
      assert.ok(
        content.innerHTML.includes('<strong>it</strong>'),
        'markdown content goes through marked',
      );
      const status = win.document.getElementById('meta-info-status').textContent;
      assert.ok(/^Updated \d/.test(status), status);
      assert.ok(status.endsWith(' \u00b7 $0.12'), status);
      wv.tick(POLL_MS);
      assert.strictEqual(lastPoll(wv).knownSig, '5:9');
      reply(win, lastPoll(wv), {unchanged: true});
      assert.ok(content.innerHTML.includes('<strong>it</strong>'));
      reply(win, lastPoll(wv), {
        sig: '6:0',
        content: '<p>Report <b>html</b></p><script>alert(1)</script>',
        updatedAt: Date.now(),
      });
      assert.ok(content.innerHTML.includes('<b>html</b>'), 'HTML as-is');
      assert.ok(!content.innerHTML.includes('<script'), 'scripts stripped');

      click(win, 'meta-close');
      const count = pollCount(wv);
      wv.tick(POLL_MS * 3);
      assert.strictEqual(
        pollCount(wv),
        count,
        'closing the drawer stops the poll',
      );
      assert.ok(!wv.intervals.some(iv => iv.ms === POLL_MS));
    },
  );

  await test(
    'the refresh button posts getTaskUpdate refresh:true and mirrors ' +
      'the agent\'s running state',
    () => {
      const wv = makeWebview('');
      const win = wv.win;
      send(win, {type: 'status', running: true});
      click(win, 'meta-drawer-btn');
      const btn = win.document.getElementById('meta-info-refresh');
      const status = win.document.getElementById('meta-info-status');
      const content = win.document.getElementById('meta-info-content');
      assert.strictEqual(
        btn.parentElement.parentElement.id,
        'meta-info',
        'the button belongs to the task-update subpanel',
      );
      const before = pollCount(wv);
      click(win, 'meta-info-refresh');
      assert.strictEqual(pollCount(wv), before + 1, 'the click polls at once');
      const refresh = lastPoll(wv);
      assert.strictEqual(refresh.type, 'getTaskUpdate');
      assert.strictEqual(refresh.refresh, true, 'the click asks for a run');
      assert.ok(!('workDir' in refresh));

      reply(win, refresh, {sig: 'r1', running: true});
      assert.strictEqual(btn.disabled, true, 'no second run while one runs');
      assert.ok(btn.classList.contains('spinning'));
      assert.strictEqual(status.textContent, 'Updating\u2026');
      assert.ok(
        win.document.getElementById('meta-info').classList.contains('visible'),
      );
      assert.ok(content.innerHTML.includes('Preparing the first update'));

      wv.tick(POLL_MS);
      assert.strictEqual(lastPoll(wv).refresh, false, 'timer polls are plain');
      reply(win, lastPoll(wv), {
        sig: 'r2',
        running: false,
        content: '<p>Half <em>done</em></p>',
        updatedAt: Date.now(),
        cost: 0.05,
      });
      assert.strictEqual(btn.disabled, false, 'the button is usable again');
      assert.ok(!btn.classList.contains('spinning'));
      assert.ok(/^Updated \d/.test(status.textContent), status.textContent);
      assert.ok(status.textContent.endsWith(' \u00b7 $0.05'));
      assert.ok(content.innerHTML.includes('<em>done</em>'));

      send(win, {type: 'status', running: false});
      assert.ok(
        !win.document.getElementById('meta-info').classList.contains('visible'),
        'the task ending empties the subpanel',
      );
      assert.strictEqual(status.textContent, '');
      assert.strictEqual(btn.disabled, false);
    },
  );

  await test(
    'openChatFromHistory switches to the tab already bound to the chat',
    () => {
      const {win, posted} = makeWebview('');
      const ready = posted.find(m => m && m.type === 'ready');
      send(win, {
        type: 'task_events',
        tabId: ready.tabId,
        chat_id: 'chat-existing',
        task_id: 101,
        task: 'Existing task opened already',
        events: [],
        extra: '{}',
      });
      win.document.querySelector('#new-chat-btn').click();
      assert.strictEqual(chatTabs(win).length, 2);
      assert.strictEqual(activeTabLabel(win), 'new chat');
      send(win, {
        type: 'openChatFromHistory',
        chatId: 'chat-existing',
        taskId: 101,
        title: 'Existing task opened already',
      });
      assert.strictEqual(
        activeTabLabel(win),
        'Existing task opened already',
        'the click lands on the existing tab',
      );
      assert.strictEqual(chatTabs(win).length, 2, 'no extra tab');
    },
  );

  await test(
    'openChatFromHistory resumes an unopened chat in a fresh tab',
    () => {
      const {win, posted} = makeWebview('');
      const tabsBefore = chatTabs(win).length;
      send(win, {
        type: 'openChatFromHistory',
        chatId: 'chat-far',
        taskId: 7,
        title: 'A chat from another day',
      });
      assert.strictEqual(chatTabs(win).length, tabsBefore + 1);
      const resume = posted.filter(m => m.type === 'resumeSession').pop();
      assert.ok(resume, 'the fresh tab resumes the chat');
      assert.strictEqual(resume.id, 'chat-far');
      assert.strictEqual(resume.taskId, 7);
      assert.strictEqual(
        win.document.getElementById('task-panel-text').textContent,
        'A chat from another day',
        'the task panel names the clicked task',
      );
    },
  );

  await test(
    'openChatFromHistory with nothing to resume shows the text read-only',
    () => {
      const {win, posted} = makeWebview('');
      const resumesBefore = posted.filter(
        m => m.type === 'resumeSession',
      ).length;
      send(win, {
        type: 'openChatFromHistory',
        chatId: '',
        taskId: null,
        title: 'An eventless task',
      });
      assert.strictEqual(
        posted.filter(m => m.type === 'resumeSession').length,
        resumesBefore,
        'nothing to resume',
      );
      assert.strictEqual(
        win.document.getElementById('task-panel-text').textContent,
        'An eventless task',
      );
    },
  );

  await test('editor-tab webviews ignore openChatFromHistory', () => {
    const {win, posted} = makeWebview(
      ' class="editor-tab-mode" data-kiss-tab-id="root-1"',
    );
    const resumesBefore = posted.filter(m => m.type === 'resumeSession').length;
    send(win, {
      type: 'openChatFromHistory',
      chatId: 'chat-far',
      taskId: 7,
      title: 'not for this surface',
    });
    assert.strictEqual(
      posted.filter(m => m.type === 'resumeSession').length,
      resumesBefore,
      'the host routes editor-tab opens through the panel manager instead',
    );
  });

  await test(
    'a content tab keeps the + and ... buttons but hides the text box, ' +
      'Inject promptlet, model picker and Send (main.css)',
    () => {
      const {win} = makeWebview('');
      injectMainCss(win);
      assert.notStrictEqual(display(win, 'input-wrap'), 'none');
      assert.notStrictEqual(display(win, 'tricks-btn'), 'none');
      win.document.body.classList.add('content-tab-open');
      assert.notStrictEqual(
        display(win, 'input-area'),
        'none',
        'the composer area stays',
      );
      assert.notStrictEqual(
        display(win, 'input-footer'),
        'none',
        'the button row stays',
      );
      assert.notStrictEqual(display(win, 'new-chat-btn'), 'none', '+ stays');
      assert.notStrictEqual(display(win, 'more-btn'), 'none', '... stays');
      assert.strictEqual(display(win, 'input-wrap'), 'none', 'no text box');
      assert.strictEqual(display(win, 'file-chips'), 'none');
      assert.strictEqual(display(win, 'input-drawer-btn'), 'none');
      assert.strictEqual(display(win, 'autocomplete'), 'none');
      assert.strictEqual(
        display(win, 'tricks-btn'),
        'none',
        'Inject promptlet hides',
      );
      assert.strictEqual(
        display(win, 'model-picker'),
        'none',
        'the model picker hides',
      );
      assert.strictEqual(display(win, 'send-btn'), 'none', 'Send hides');
      // Stop is NOT hidden by the content-tab CSS: while the owning
      // chat runs, its inline display keeps the button reachable to
      // stop the task from the file view.
      const stop = win.document.getElementById('stop-btn');
      stop.style.display = 'flex';
      assert.strictEqual(
        display(win, 'stop-btn'),
        'flex',
        'Stop stays reachable when the owner chat runs',
      );
      stop.style.display = 'none';
      // A composer hidden with an INLINE display:none (a finished
      // sub-agent tab, a worktree action bar) must still show the
      // button row on a content tab.
      const container = win.document.getElementById('input-container');
      container.style.display = 'none';
      assert.notStrictEqual(
        display(win, 'input-container'),
        'none',
        'the row survives an inline-hidden composer',
      );
      assert.notStrictEqual(display(win, 'new-chat-btn'), 'none');
      win.document.body.classList.remove('content-tab-open');
      assert.strictEqual(
        display(win, 'input-container'),
        'none',
        'the inline hide applies again once the content tab closes',
      );
      container.style.display = '';
      assert.notStrictEqual(display(win, 'input-wrap'), 'none');
    },
  );

  await test(
    'Stop clicked on a content tab stops the OWNING chat, and the ' +
      'button mirrors that chat\'s running state',
    () => {
      const {win, posted} = makeWebview('');
      const api = win._testApi;
      const chatId = api.getActiveTabId();
      send(win, {type: 'status', running: true, tabId: chatId});
      send(win, {
        type: 'fileContent',
        tabId: chatId,
        name: 'notes_stop.txt',
        path: '/tmp/notes_stop.txt',
        content: 'stop target body',
      });
      const contentId = api.getActiveTabId();
      assert.notStrictEqual(contentId, chatId, 'the file tab is in front');
      assert.strictEqual(
        win.document.getElementById('stop-btn').style.display,
        'flex',
        'Stop shows on the file tab while the owner chat runs',
      );
      posted.length = 0;
      click(win, 'stop-btn');
      const stop = posted.find(m => m.type === 'stop');
      assert.ok(stop, 'the click posts a stop');
      assert.strictEqual(
        stop.tabId,
        chatId,
        'the stop lands on the owning chat, not on the file view id',
      );
      // The owner idle: a later content-tab show hides Stop again.
      send(win, {type: 'status', running: false, tabId: chatId});
      send(win, {
        type: 'fileContent',
        tabId: chatId,
        name: 'notes_stop2.txt',
        path: '/tmp/notes_stop2.txt',
        content: 'second file',
      });
      assert.strictEqual(
        win.document.getElementById('stop-btn').style.display,
        'none',
        'no Stop on a file tab whose owner chat is idle',
      );
    },
  );

  await test(
    'host triggerStop, a file opened FROM a file view, and the pending ' +
      'pulse all follow the chat at the root of the owner chain',
    () => {
      const {win, posted} = makeWebview('');
      const api = win._testApi;
      const chatId = api.getActiveTabId();
      send(win, {type: 'status', running: true, tabId: chatId});
      send(win, {
        type: 'fileContent',
        tabId: chatId,
        name: 'nested_a.txt',
        path: '/tmp/nested_a.txt',
        content: 'file A',
      });
      const fileA = api.getActiveTabId();
      assert.notStrictEqual(fileA, chatId);
      // Opening a file FROM the file view chains the ownership:
      // B -> A -> chat. Every chat-scoped action lands on the chat.
      send(win, {
        type: 'fileContent',
        tabId: fileA,
        name: 'nested_b.txt',
        path: '/tmp/nested_b.txt',
        content: 'file B',
      });
      assert.notStrictEqual(api.getActiveTabId(), fileA);
      const stop = win.document.getElementById('stop-btn');
      assert.strictEqual(
        stop.style.display,
        'flex',
        'Stop shows two hops away from the running chat',
      );
      posted.length = 0;
      send(win, {type: 'triggerStop'});
      const stopMsg = posted.find(m => m.type === 'stop');
      assert.ok(stopMsg, 'the host command posts a stop');
      assert.strictEqual(
        stopMsg.tabId,
        chatId,
        'the stop lands on the chat at the root of the owner chain',
      );
      assert.ok(
        stop.classList.contains('stopping'),
        'the pending stop pulses on the file tab too',
      );
      win.close();
    },
  );

  await test(
    'Share chat clicked on a content tab exports the OWNING chat: the ' +
      'chat comes back on screen and the share names its id',
    () => {
      const {win, posted} = makeWebview('');
      const api = win._testApi;
      const chatId = api.getActiveTabId();
      send(win, {
        type: 'fileContent',
        tabId: chatId,
        name: 'notes_share.txt',
        path: '/tmp/notes_share.txt',
        content: 'share target body',
      });
      assert.notStrictEqual(api.getActiveTabId(), chatId);
      posted.length = 0;
      click(win, 'share-btn');
      assert.strictEqual(
        api.getActiveTabId(),
        chatId,
        'the owning chat is back on screen (the export splices the ' +
          'LIVE transcript, which only exists there)',
      );
      const share = posted.find(m => m.type === 'shareChatTasks');
      assert.ok(share, 'the click asks for the chat\'s persisted tasks');
      assert.strictEqual(share.tabId, chatId);
      assert.strictEqual(share.chatId, chatId);
    },
  );

  await test(
    'a voice submit landing on a content tab posts under the OWNING ' +
      'chat, never under the file view',
    () => {
      const {win, posted} = makeWebview('');
      const api = win._testApi;
      const chatId = api.getActiveTabId();
      send(win, {
        type: 'fileContent',
        tabId: chatId,
        name: 'notes_voice.txt',
        path: '/tmp/notes_voice.txt',
        content: 'voice target body',
      });
      assert.notStrictEqual(api.getActiveTabId(), chatId);
      assert.strictEqual(
        win.kissVoiceOwner().tabId,
        chatId,
        'an utterance started on the file tab is owned by the chat',
      );
      win.document.getElementById('task-input').value = 'voice prompt QX1';
      posted.length = 0;
      win.dispatchEvent(new win.CustomEvent('kiss-voice-submit'));
      assert.strictEqual(
        api.getActiveTabId(),
        chatId,
        'the submit brings the conversation back on screen',
      );
      const submit = posted.find(m => m.type === 'submit');
      assert.ok(submit, 'the prompt is submitted');
      assert.strictEqual(submit.tabId, chatId);
      assert.strictEqual(submit.prompt, 'voice prompt QX1');
    },
  );

  console.log(`\n${passed} passed, ${failures.length} failed`);
  process.exit(failures.length > 0 ? 1 : 0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
