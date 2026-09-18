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
//   drawer is inert, and the 1s getInfoFile poll runs only while the
//   drawer is open and a task runs;
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=sidebarchat-main.js',
  );
  return {win, posted};
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

function pollCount(wv) {
  return wv.posted.filter(m => m.type === 'getInfoFile').length;
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
    'getInfoFile polls only while the drawer is open and a task runs',
    async () => {
      const wv = makeWebview('');
      const win = wv.win;
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
      });
      send(win, {type: 'status', running: true});
      await sleep(1300);
      assert.strictEqual(pollCount(wv), 0, 'a closed drawer must not poll');
      click(win, 'meta-drawer-btn');
      await sleep(1300);
      const polls = wv.posted.filter(m => m.type === 'getInfoFile');
      assert.ok(polls.length >= 1, 'opening the drawer starts the poll');
      assert.strictEqual(polls[polls.length - 1].workDir, '/cfg/dir');
      click(win, 'meta-close');
      const count = pollCount(wv);
      await sleep(1500);
      assert.strictEqual(
        pollCount(wv),
        count,
        'closing the drawer stops the poll',
      );
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
