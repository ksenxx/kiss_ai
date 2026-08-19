// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the chat webview's share button (media/chat.html,
// media/main.js) and for the standalone shared-page script
// (media/share.js):
//
// * the share button sits immediately to the right of the mic button,
// * clicking it asks the daemon for ALL of the chat's persisted tasks
//   (`shareChatTasks`), and the `share_tasks` reply is assembled into a
//   `shareChat` command holding one .share-task section per task —
//   replayed tasks and the live on-screen transcript alike,
// * the daemon's `share_done` reply lands as a transcript banner (and
//   only in the conversation that was shared), and
// * share.js reproduces the webview's collapse / expand behaviour on a
//   page whose body is exactly what buildShareableHtml() serialized.

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
    console.log(`      ${e.message}`);
  }
}

function makeWebview(opts) {
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
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  win.cancelAnimationFrame = function () {};

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

  if (opts && opts.highlight) {
    // The REAL production renderers, so the deferred-highlight test
    // exercises the same hljs/marked paths the webview runs.
    win.eval(fs.readFileSync(path.join(MEDIA, 'highlight.min.js'), 'utf8'));
    win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  }
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=share-main.js',
  );

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function tabIdOf(wv) {
  const ready = wv.posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  return ready.tabId;
}

function click(el) {
  el.dispatchEvent(
    new el.ownerDocument.defaultView.MouseEvent('click', {
      bubbles: true,
      cancelable: true,
    }),
  );
}

/**
 * Drive one small task through the webview so #task-panel and #output
 * hold real panels, then return the tab id.  Every event carries
 * *taskId* so the webview adopts it as the on-screen task's id —
 * exactly what the daemon's taskId-stamped stream does.
 */
function runSmallTask(wv, chatId, taskId) {
  const win = wv.win;
  const TAB = tabIdOf(wv);
  const now = Date.now();
  send(win, {type: 'clear', chat_id: chatId, tabId: TAB});
  send(win, {type: 'status', running: true, tabId: TAB, startTs: now});
  send(win, {type: 'setTaskText', text: 'list the files', tabId: TAB});
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls -la',
    description: 'List files',
    tabId: TAB,
    taskId: taskId,
    ts: now,
  });
  send(win, {
    type: 'tool_result',
    content: 'file-one.txt\nfile-two.txt',
    tool_name: 'Bash',
    tabId: TAB,
    taskId: taskId,
    ts: now,
  });
  return TAB;
}

/**
 * Click the share button, assert the webview asked the daemon for the
 * chat's persisted tasks, answer with *tasks*, and return the
 * resulting shareChat command (or undefined when none was sent).
 */
function shareWithTasks(wv, tasks, replyExtra) {
  const win = wv.win;
  click(win.document.getElementById('share-btn'));
  const req = wv.posted.filter(m => m.type === 'shareChatTasks').pop();
  assert.ok(req, 'clicking share must ask the daemon for the chat tasks');
  const reply = Object.assign(
    {
      type: 'share_tasks',
      tabId: req.tabId,
      chatId: req.chatId,
      tasks: tasks || [],
      truncated: false,
    },
    replyExtra || {},
  );
  send(win, reply);
  return wv.posted.filter(m => m.type === 'shareChat').pop();
}

/**
 * Load share.js into a fresh page whose <body> is *bodyHtml* — the
 * exact document shape _build_share_page (web_server.py) writes.
 */
function makeSharePage(bodyHtml) {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body><div id="app">' +
      bodyHtml +
      '</div></body></html>',
    {runScripts: 'dangerously', pretendToBeVisual: true},
  );
  const win = dom.window;
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'share.js'), 'utf8') +
      '\n//# sourceURL=share-share.js',
  );
  return win;
}

async function run() {
  await test('share button sits immediately right of the mic button', () => {
    const {win} = makeWebview();
    const voiceBtn = win.document.getElementById('voice-btn');
    const shareBtn = win.document.getElementById('share-btn');
    assert.ok(voiceBtn, 'mic button must exist');
    assert.ok(shareBtn, 'share button must exist');
    assert.strictEqual(
      voiceBtn.nextElementSibling,
      shareBtn,
      'share button must be the element right after the mic button',
    );
    assert.strictEqual(
      shareBtn.closest('#input-footer'),
      win.document.getElementById('input-footer'),
      'share button must live in the footer below the input textbox',
    );
  });

  await test('click asks the daemon for the chat tasks, then shares', () => {
    const wv = makeWebview();
    const TAB = runSmallTask(wv, 'chat-777', 'task-a');
    click(wv.win.document.getElementById('share-btn'));
    const req = wv.posted.find(m => m.type === 'shareChatTasks');
    assert.ok(req, 'clicking share must post a shareChatTasks command');
    assert.strictEqual(req.tabId, TAB);
    assert.strictEqual(req.chatId, 'chat-777');
    assert.ok(
      !wv.posted.some(m => m.type === 'shareChat'),
      'nothing is shared before the daemon lists the chat tasks',
    );
    send(wv.win, {
      type: 'share_tasks',
      tabId: TAB,
      chatId: 'chat-777',
      tasks: [{task: 'list the files', task_id: 'task-a', events: []}],
      truncated: false,
    });
    const msg = wv.posted.find(m => m.type === 'shareChat');
    assert.ok(msg, 'the share_tasks reply must produce a shareChat command');
    assert.strictEqual(msg.tabId, TAB);
    assert.strictEqual(msg.chatId, 'chat-777');
    assert.ok(msg.html.includes('id="task-panel"'), 'task panel serialized');
    assert.ok(
      msg.html.includes('list the files'),
      'task panel text serialized',
    );
    assert.ok(msg.html.includes('id="output"'), 'transcript serialized');
    assert.ok(msg.html.includes('share-task'), 'task section serialized');
    assert.ok(msg.html.includes('ls -la'), 'tool panel serialized');
    assert.ok(
      !msg.html.includes('id="welcome"'),
      'the welcome screen is not a panel and must be dropped',
    );
  });

  await test('tab id is used as the chat id before the daemon names one', () => {
    const wv = makeWebview();
    const TAB = tabIdOf(wv);
    click(wv.win.document.getElementById('share-btn'));
    const req = wv.posted.find(m => m.type === 'shareChatTasks');
    assert.ok(req, 'share must work on a tab with no backend chat id yet');
    assert.strictEqual(req.chatId, TAB);
  });

  await test('an empty chat is refused with a visible error', () => {
    const wv = makeWebview();
    const msg = shareWithTasks(wv, []);
    assert.ok(
      msg === undefined,
      'a chat with no panels anywhere must not be shared',
    );
    const banners = wv.win.document.querySelectorAll('#output .ev.err');
    const texts = Array.from(banners, b => b.textContent);
    assert.ok(
      texts.some(t => t.includes('empty')),
      'the refusal must be visible in the transcript',
    );
    assert.ok(
      wv.win.document
        .getElementById('share-btn')
        .classList.contains('share-err'),
      'the button flashes red on refusal',
    );
  });

  await test('share_done ok shows a note banner and flashes the button', () => {
    const wv = makeWebview();
    const TAB = runSmallTask(wv, 'chat-1', 'task-1');
    send(wv.win, {
      type: 'share_done',
      ok: true,
      path: '/w/reports/chat-1.html',
      tabId: TAB,
    });
    const shareBtn = wv.win.document.getElementById('share-btn');
    assert.ok(shareBtn.classList.contains('share-ok'));
    assert.ok(!shareBtn.classList.contains('share-err'));
    const banners = wv.win.document.querySelectorAll('#output .ev.note');
    const texts = Array.from(banners, b => b.textContent);
    assert.ok(
      texts.some(t => t.includes('/w/reports/chat-1.html')),
      'the saved path must appear in a note banner',
    );
  });

  await test('the saved-page path in the banner becomes a file link', () => {
    // The path holds a space, which the prose regex of
    // linkifyFilePaths() would split in two: the banner must wrap the
    // EXACT path the daemon replied with.
    const SAVED = '/tmp/My Project/reports/chat-link.html';
    const wv = makeWebview();
    const TAB = runSmallTask(wv, 'chat-link', 'task-link');
    send(wv.win, {type: 'share_done', ok: true, path: SAVED, tabId: TAB});
    // The path starts as an inert candidate span and the webview asks
    // the host whether the file exists — the same round-trip every
    // transcript file path takes.
    const cand = wv.win.document.querySelector(
      '#output .ev.note [data-path-candidate]',
    );
    assert.ok(cand, 'the saved path must be wrapped in a candidate span');
    assert.strictEqual(cand.getAttribute('data-path-candidate'), SAVED);
    assert.strictEqual(cand.textContent, SAVED);
    const check = wv.posted.filter(m => m.type === 'checkPaths').pop();
    assert.ok(
      check && check.paths.includes(SAVED),
      'the webview must ask the host whether the saved page exists',
    );
    assert.strictEqual(check.tabId, TAB);
    // The host confirms (the daemon just wrote the file) and the span
    // is promoted to a clickable link.
    const results = {};
    results[SAVED] = true;
    send(wv.win, {
      type: 'pathsExist',
      results: results,
      workDir: check.workDir,
      tabId: check.tabId,
    });
    const link = wv.win.document.querySelector('#output .ev.note [data-path]');
    assert.ok(link, 'the confirmed path must become a clickable file link');
    assert.strictEqual(link.getAttribute('data-path'), SAVED);
    // Clicking the promoted link asks the host to open the saved page.
    click(link);
    const open = wv.posted.filter(m => m.type === 'openFile').pop();
    assert.ok(open, 'clicking the link must post an openFile command');
    assert.strictEqual(open.path, SAVED);
    assert.strictEqual(open.tabId, TAB);
  });

  await test('a share_done reply without a path is still a plain note', () => {
    const wv = makeWebview();
    const TAB = runSmallTask(wv, 'chat-nopath', 'task-nopath');
    const before = wv.posted.filter(m => m.type === 'checkPaths').length;
    send(wv.win, {type: 'share_done', ok: true, tabId: TAB});
    const banners = wv.win.document.querySelectorAll('#output .ev.note');
    const texts = Array.from(banners, b => b.textContent);
    assert.ok(
      texts.some(t => t.includes('Chat page saved to reports/')),
      'the note must fall back to the reports/ directory',
    );
    assert.ok(
      !wv.win.document.querySelector('#output .ev.note [data-path-candidate]'),
      'no candidate span exists without an exact path',
    );
    assert.strictEqual(
      wv.posted.filter(m => m.type === 'checkPaths').length,
      before,
      'no existence check is posted without an exact path',
    );
  });

  await test('share_done failure shows an error banner and a red flash', () => {
    const wv = makeWebview();
    const TAB = runSmallTask(wv, 'chat-2', 'task-2');
    send(wv.win, {
      type: 'share_done',
      ok: false,
      error: 'disk full',
      tabId: TAB,
    });
    const shareBtn = wv.win.document.getElementById('share-btn');
    assert.ok(shareBtn.classList.contains('share-err'));
    const banners = wv.win.document.querySelectorAll('#output .ev.err');
    const texts = Array.from(banners, b => b.textContent);
    assert.ok(
      texts.some(t => t.includes('disk full')),
      'the failure reason must appear in an error banner',
    );
  });

  await test('share_done for another tab never writes into this transcript', () => {
    const wv = makeWebview();
    runSmallTask(wv, 'chat-3', 'task-3');
    send(wv.win, {
      type: 'share_done',
      ok: true,
      path: '/elsewhere/reports/chat-9.html',
      tabId: 'some-other-tab',
    });
    const banners = wv.win.document.querySelectorAll('#output .ev.note');
    const texts = Array.from(banners, b => b.textContent);
    assert.ok(
      !texts.some(t => t.includes('chat-9.html')),
      'a background tab reply must not land in the visible transcript',
    );
    assert.ok(
      wv.win.document
        .getElementById('share-btn')
        .classList.contains('share-ok'),
      'the button flash still confirms the click landed',
    );
  });

  await test('a share_tasks reply for another tab is dropped', () => {
    const wv = makeWebview();
    runSmallTask(wv, 'chat-drop', 'task-drop');
    click(wv.win.document.getElementById('share-btn'));
    send(wv.win, {
      type: 'share_tasks',
      tabId: 'some-other-tab',
      chatId: 'chat-drop',
      tasks: [{task: 'foreign', task_id: 'task-x', events: []}],
      truncated: false,
    });
    assert.ok(
      !wv.posted.some(m => m.type === 'shareChat'),
      'a reply for a tab no longer highlighted must not serialize ' +
        'the screen of a different conversation',
    );
  });

  await test('share.js collapses and expands a serialized event panel', () => {
    const wv = makeWebview();
    runSmallTask(wv, 'chat-4', 'task-4');
    const msg = shareWithTasks(wv, []);
    const page = makeSharePage(msg.html);
    const panel = page.document.querySelector('#output .collapsible');
    assert.ok(panel, 'the exported page must hold a collapsible panel');
    const header = panel.querySelector('.collapse-header');
    assert.ok(header, 'the panel must keep its collapse header');
    const wasCollapsed = panel.classList.contains('collapsed');
    click(header);
    assert.strictEqual(
      panel.classList.contains('collapsed'),
      !wasCollapsed,
      'a header click must toggle the collapsed state',
    );
    if (panel.classList.contains('collapsed')) {
      const prev = panel.querySelector('.collapse-preview');
      assert.ok(
        prev.textContent.includes('file-one.txt'),
        'a collapsed panel must preview its content text',
      );
    }
    click(header);
    assert.strictEqual(
      panel.classList.contains('collapsed'),
      wasCollapsed,
      'a second click must restore the original state',
    );
    const prev = panel.querySelector('.collapse-preview');
    if (!panel.classList.contains('collapsed')) {
      assert.strictEqual(
        prev.textContent,
        '',
        'an expanded panel must show no preview',
      );
    }
  });

  await test('share.js toggles the static task panel drawer', () => {
    const wv = makeWebview();
    runSmallTask(wv, 'chat-5', 'task-5');
    const msg = shareWithTasks(wv, []);
    const page = makeSharePage(msg.html);
    const panel = page.document.getElementById('task-panel');
    const btn = page.document.getElementById('task-panel-drawer-btn');
    assert.ok(panel && btn, 'exported page keeps the task panel + drawer');
    const wasCollapsed = panel.classList.contains('drawer-collapsed');
    click(btn);
    assert.strictEqual(
      panel.classList.contains('drawer-collapsed'),
      !wasCollapsed,
      'the drawer button must toggle the task panel',
    );
    assert.strictEqual(
      btn.getAttribute('aria-expanded'),
      panel.classList.contains('drawer-collapsed') ? 'false' : 'true',
    );
    click(btn);
    assert.strictEqual(
      panel.classList.contains('drawer-collapsed'),
      wasCollapsed,
      'a second click must restore the drawer',
    );
  });

  await test('each exported task panel folds its own drawer', () => {
    const wv = makeWebview();
    runSmallTask(wv, 'chat-6', 'task-6b');
    const msg = shareWithTasks(wv, [
      {
        task: 'first task',
        task_id: 'task-6a',
        events: [
          {
            type: 'tool_call',
            name: 'Bash',
            command: 'whoami',
            ts: Date.now(),
          },
        ],
      },
      {task: 'list the files', task_id: 'task-6b', events: []},
    ]);
    const page = makeSharePage(msg.html);
    const panels = page.document.querySelectorAll('[id="task-panel"]');
    assert.strictEqual(
      panels.length,
      2,
      'one static task panel per task of the chat',
    );
    // Attribute selector, not '#task-panel-drawer-btn': jsdom's
    // subtree querySelector shortcuts id selectors through the
    // document-wide id map, which holds only the FIRST duplicate.
    // share.js itself is immune — closest()/matches() test the
    // element's own id attribute.
    const secondBtn = panels[1].querySelector(
      'button[aria-controls^="task-panel-text"]',
    );
    assert.ok(secondBtn, "the second task's panel keeps its drawer button");
    const firstBtn = panels[0].querySelector(
      'button[aria-controls^="task-panel-text"]',
    );
    assert.notStrictEqual(
      firstBtn.getAttribute('aria-controls'),
      secondBtn.getAttribute('aria-controls'),
      "each drawer must name ITS OWN text element (unique aria-controls)",
    );
    assert.strictEqual(
      panels[1].querySelector(
        '[id="' + secondBtn.getAttribute('aria-controls') + '"]',
      ),
      panels[1].querySelector('[id^="task-panel-text"]'),
      "the second drawer's aria-controls resolves inside its own panel",
    );
    click(secondBtn);
    assert.ok(
      panels[1].classList.contains('drawer-collapsed'),
      "the second task's drawer folds",
    );
    assert.ok(
      !panels[0].classList.contains('drawer-collapsed'),
      "the first task's panel must not fold with it",
    );
  });

  await test('share.js toggles a serialized Thinking section', () => {
    const page = makeSharePage(
      '<div id="output"><div class="ev think">' +
        '<div class="lbl" onclick="toggleThink(this)">' +
        '<span class="arrow">\u25BE</span> Thinking</div>' +
        '<div class="cnt">deep thought</div></div></div>',
    );
    const lbl = page.document.querySelector('.think .lbl');
    click(lbl);
    assert.ok(
      page.document.querySelector('.think .cnt').classList.contains('hidden'),
      'a think header click must hide the content',
    );
    assert.ok(
      page.document
        .querySelector('.think .arrow')
        .classList.contains('collapsed'),
      'the arrow must rotate with the collapse',
    );
    click(lbl);
    assert.ok(
      !page.document.querySelector('.think .cnt').classList.contains('hidden'),
      'a second click must show the content again',
    );
  });

  await test('share exports every task of the chat, not only the last', () => {
    const wv = makeWebview();
    const win = wv.win;
    const TAB = runSmallTask(wv, 'chat-multi', 'task-m1');
    send(win, {type: 'result', text: 'first done', success: true, tabId: TAB});
    send(win, {type: 'task_done', tabId: TAB});
    // A second task in the SAME chat replaces the transcript on screen.
    send(win, {type: 'clear', chat_id: 'chat-multi', tabId: TAB});
    send(win, {type: 'setTaskText', text: 'second task', tabId: TAB});
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'pwd',
      tabId: TAB,
      taskId: 'task-m2',
      ts: Date.now(),
    });
    assert.ok(
      !win.document.getElementById('output').textContent.includes('ls -la'),
      'precondition: the first task left the live transcript',
    );
    const msg = shareWithTasks(wv, [
      {
        task: 'list the files',
        task_id: 'task-m1',
        events: [
          {
            type: 'tool_call',
            name: 'Bash',
            command: 'ls -la',
            description: 'List files',
            ts: Date.now(),
          },
          {
            type: 'tool_result',
            content: 'file-one.txt',
            tool_name: 'Bash',
            ts: Date.now(),
          },
          {type: 'result', text: 'first done', success: true},
        ],
      },
      // The running second task's row: its persisted events lag the
      // live stream, so the export must use the screen for it.
      {task: 'second task', task_id: 'task-m2', events: []},
    ]);
    assert.ok(msg, 'the multi-task chat must be shared');
    assert.ok(msg.html.includes('ls -la'), 'first task panels exported');
    assert.ok(msg.html.includes('first done'), 'first task result exported');
    assert.ok(msg.html.includes('pwd'), 'second task panels exported');
    const firstIdx = msg.html.indexOf('ls -la');
    const secondIdx = msg.html.indexOf('pwd');
    assert.ok(firstIdx < secondIdx, 'tasks appear in chronological order');
    const sections = msg.html.match(/class="share-task"/g) || [];
    assert.strictEqual(
      sections.length,
      2,
      'one section per task — the live task must not be exported twice',
    );
    assert.ok(
      msg.html.includes('second task'),
      "the second task's text heads its own section",
    );
  });

  await test('after a reload the export still holds every task', () => {
    // THE regression this feature exists for: a reloaded webview
    // replays only the LATEST task into its DOM, and the shared page
    // used to show just that one task.
    const wv = makeWebview();
    const win = wv.win;
    const TAB = tabIdOf(wv);
    send(win, {
      type: 'task_events',
      tabId: TAB,
      task: 'second task',
      task_id: 'task-r2',
      chat_id: 'chat-reload',
      events: [
        {
          type: 'tool_call',
          name: 'Bash',
          command: 'pwd',
          ts: Date.now(),
        },
        {type: 'result', text: 'second done', success: true},
      ],
    });
    assert.ok(
      !win.document.getElementById('output').textContent.includes('ls -la'),
      'precondition: only the latest task is in the DOM',
    );
    const msg = shareWithTasks(wv, [
      {
        task: 'first task',
        task_id: 'task-r1',
        events: [
          {
            type: 'tool_call',
            name: 'Bash',
            command: 'ls -la',
            ts: Date.now(),
          },
          {type: 'result', text: 'first done', success: true},
        ],
      },
      {
        task: 'second task',
        task_id: 'task-r2',
        events: [
          {
            type: 'tool_call',
            name: 'Bash',
            command: 'pwd',
            ts: Date.now(),
          },
          {type: 'result', text: 'second done', success: true},
        ],
      },
    ]);
    assert.ok(msg, 'the reloaded chat must be shared');
    assert.ok(
      msg.html.includes('ls -la'),
      'the first task, absent from the DOM, is exported from the daemon',
    );
    assert.ok(msg.html.includes('pwd'), 'the on-screen task is exported');
    assert.ok(
      msg.html.indexOf('ls -la') < msg.html.indexOf('pwd'),
      'tasks appear in chronological order',
    );
    const sections = msg.html.match(/class="share-task"/g) || [];
    assert.strictEqual(
      sections.length,
      2,
      'one section per task — the replayed task must not be duplicated',
    );
  });

  await test('a new chat on the same tab shares only the new chat', () => {
    const wv = makeWebview();
    const win = wv.win;
    const TAB = runSmallTask(wv, 'chat-reset', 'task-old');
    send(win, {type: 'showWelcome', tabId: TAB});
    send(win, {type: 'clear', chat_id: 'chat-reset2', tabId: TAB});
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'pwd',
      tabId: TAB,
      taskId: 'task-new',
      ts: Date.now(),
    });
    click(win.document.getElementById('share-btn'));
    const req = wv.posted.filter(m => m.type === 'shareChatTasks').pop();
    assert.strictEqual(
      req.chatId,
      'chat-reset2',
      "share must ask for the NEW chat's tasks",
    );
    send(win, {
      type: 'share_tasks',
      tabId: TAB,
      chatId: 'chat-reset2',
      tasks: [{task: 'new task', task_id: 'task-new', events: []}],
      truncated: false,
    });
    const msg = wv.posted.filter(m => m.type === 'shareChat').pop();
    assert.ok(
      !msg.html.includes('ls -la'),
      'panels of the discarded chat must not leak into the new share',
    );
    assert.ok(msg.html.includes('pwd'), 'the new chat panels are exported');
  });

  await test('deferred-highlight code is highlighted in the export', () => {
    const wv = makeWebview({highlight: true});
    const win = wv.win;
    const TAB = tabIdOf(wv);
    send(win, {
      type: 'task_events',
      tabId: TAB,
      task: 'replayed task',
      task_id: 'task-hl2',
      chat_id: 'chat-hl',
      events: [
        {type: 'prompt', text: 'show some code'},
        {type: 'text_delta', text: '```js\nconst answer = 42;\n```'},
        {type: 'text_end'},
        {type: 'result', text: 'done', success: true},
      ],
    });
    assert.ok(
      win.document.querySelector('#output code.needs-hl'),
      'precondition: the replay left a deferred-highlight code block',
    );
    const msg = shareWithTasks(wv, [
      {
        task: 'earlier task',
        task_id: 'task-hl1',
        events: [
          {type: 'prompt', text: 'more code'},
          {type: 'text_delta', text: '```py\nvalue = 1\n```'},
          {type: 'text_end'},
          {type: 'result', text: 'done', success: true},
        ],
      },
      {task: 'replayed task', task_id: 'task-hl2', events: []},
    ]);
    assert.ok(
      !msg.html.includes('needs-hl'),
      'no code block may stay unhighlighted in the export — neither ' +
        "the screen's nor a replayed task's",
    );
    assert.ok(
      msg.html.includes('hljs'),
      'the exported code blocks carry real highlight markup',
    );
    assert.ok(
      win.document.querySelector('#output code.needs-hl'),
      'the live transcript is untouched: its block still awaits expansion',
    );
  });

  await test('collapsing an exported panel collapses nested fan-outs', () => {
    const page = makeSharePage(
      '<div id="output">' +
        '<div class="ev tc collapsible">' +
        '<div class="tc-h collapse-header"><span class="collapse-chv">' +
        '\u25BE</span>outer<span class="collapse-preview"></span></div>' +
        '<div class="tc-b">' +
        '<div class="ev tc tc-run-parallel collapsible user-pinned">' +
        '<div class="tc-h collapse-header"><span class="collapse-chv">' +
        '\u25BE</span>fan-out<span class="collapse-preview"></span></div>' +
        '<div class="tc-b">parallel work</div></div>' +
        '</div></div></div>',
    );
    const outer = page.document.querySelector('#output > .collapsible');
    const nested = page.document.querySelector('.tc-run-parallel');
    click(outer.querySelector(':scope > .collapse-header'));
    assert.ok(outer.classList.contains('collapsed'), 'outer collapsed');
    assert.ok(
      nested.classList.contains('collapsed'),
      'the swallowed fan-out panel must be collapsed too',
    );
    assert.ok(
      !nested.classList.contains('user-pinned'),
      'the fan-out loses its pin, exactly like the live webview',
    );
  });

  await test('the live view survives a shared replay untouched', () => {
    const wv = makeWebview();
    const win = wv.win;
    runSmallTask(wv, 'chat-live', 'task-l2');
    const statusSteps = win.document.getElementById('status-steps');
    const stepsBefore = statusSteps ? statusSteps.textContent : '';
    const outputBefore = win.document.getElementById('output').innerHTML;
    shareWithTasks(wv, [
      {
        task: 'first task',
        task_id: 'task-l1',
        events: [
          {
            type: 'tool_call',
            name: 'Bash',
            command: 'whoami',
            ts: Date.now(),
          },
          {type: 'step_count', count: 42},
          {type: 'result', text: 'done', success: true},
        ],
      },
      {task: 'list the files', task_id: 'task-l2', events: []},
    ]);
    assert.strictEqual(
      win.document.getElementById('output').innerHTML,
      outputBefore,
      "the replay of another task must not touch the screen's panels",
    );
    assert.strictEqual(
      statusSteps ? statusSteps.textContent : '',
      stepsBefore,
      "the replay must not walk the live task's status numbers",
    );
  });

  await test('a truncated share_tasks reply warns about dropped tasks', () => {
    const wv = makeWebview();
    runSmallTask(wv, 'chat-trunc', 'task-t2');
    const msg = shareWithTasks(
      wv,
      [{task: 'list the files', task_id: 'task-t2', events: []}],
      {truncated: true},
    );
    assert.ok(msg, 'the surviving tasks are still shared');
    const banners = wv.win.document.querySelectorAll('#output .ev.warn');
    const texts = Array.from(banners, b => b.textContent);
    assert.ok(
      texts.some(t => t.includes('most recent tasks')),
      'the user must see that older tasks were dropped',
    );
  });

  await test('a share_tasks error reply is shown and nothing is sent', () => {
    const wv = makeWebview();
    runSmallTask(wv, 'chat-dberr', 'task-dberr');
    click(wv.win.document.getElementById('share-btn'));
    const req = wv.posted.filter(m => m.type === 'shareChatTasks').pop();
    send(wv.win, {
      type: 'share_tasks',
      tabId: req.tabId,
      chatId: req.chatId,
      tasks: [],
      truncated: false,
      error: 'Failed to load the chat history: disk exploded',
    });
    assert.ok(
      !wv.posted.some(m => m.type === 'shareChat'),
      'a failed task listing must not produce a partial export',
    );
    const banners = wv.win.document.querySelectorAll('#output .ev.err');
    const texts = Array.from(banners, b => b.textContent);
    assert.ok(
      texts.some(t => t.includes('disk exploded')),
      'the daemon-side failure must be visible in the transcript',
    );
    assert.ok(
      wv.win.document
        .getElementById('share-btn')
        .classList.contains('share-err'),
      'the button flashes red on failure',
    );
  });

  await test('a task starting this instant is neither lost nor doubled', () => {
    // The startup window: setTaskText nulled currentTaskId and no
    // stamped event has adopted the new task's id yet, but the new
    // task's row (with no events) is already in the daemon's list.
    const wv = makeWebview();
    const win = wv.win;
    const TAB = runSmallTask(wv, 'chat-start', 'task-s1');
    send(win, {type: 'result', text: 'first done', success: true, tabId: TAB});
    send(win, {type: 'task_done', tabId: TAB});
    send(win, {type: 'clear', chat_id: 'chat-start', tabId: TAB});
    send(win, {type: 'setTaskText', text: 'second task', tabId: TAB});
    // No taskId on the event: the webview does not know the new id.
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'pwd',
      tabId: TAB,
      ts: Date.now(),
    });
    const msg = shareWithTasks(wv, [
      {
        task: 'list the files',
        task_id: 'task-s1',
        events: [
          {
            type: 'tool_call',
            name: 'Bash',
            command: 'ls -la',
            ts: Date.now(),
          },
          {type: 'result', text: 'first done', success: true},
        ],
      },
      {task: 'second task', task_id: 'task-s2', events: []},
    ]);
    assert.ok(msg, 'the chat must be shared');
    const sections = msg.html.match(/class="share-task"/g) || [];
    assert.strictEqual(
      sections.length,
      2,
      'the starting task must not be duplicated at the end of the page',
    );
    assert.ok(msg.html.includes('ls -la'), 'the finished task is kept');
    const secondAt = msg.html.indexOf('second task');
    assert.ok(
      msg.html.indexOf('pwd') > secondAt,
      "the live panels land in the starting task's OWN section",
    );
  });

  await test('a chat whose tasks recorded no output still exports', () => {
    const wv = makeWebview();
    const msg = shareWithTasks(wv, [
      {task: 'quiet task', task_id: 'task-q1', events: []},
    ]);
    assert.ok(msg, 'task descriptions are content in themselves');
    assert.ok(msg.html.includes('quiet task'), 'the task text is exported');
    assert.ok(
      msg.html.includes('(no output recorded)'),
      'an empty transcript says so instead of being dropped',
    );
  });

  await test('a share replay posts no checkPaths for detached panels', () => {
    const wv = makeWebview();
    runSmallTask(wv, 'chat-paths', 'task-p2');
    const before = wv.posted.filter(m => m.type === 'checkPaths').length;
    shareWithTasks(wv, [
      {
        task: 'first task',
        task_id: 'task-p1',
        events: [
          {
            type: 'tool_result',
            content: 'wrote src/kiss/server/web_server.py and left',
            tool_name: 'Bash',
            ts: Date.now(),
          },
          {type: 'result', text: 'done', success: true},
        ],
      },
      {task: 'list the files', task_id: 'task-p2', events: []},
    ]);
    const after = wv.posted.filter(m => m.type === 'checkPaths').length;
    assert.strictEqual(
      after,
      before,
      'detached export panels can never be clicked, so their file ' +
        'paths must not be verified with the daemon',
    );
  });

  await test("an old failure in the export never marks the tab failed", () => {
    const wv = makeWebview();
    const win = wv.win;
    const TAB = runSmallTask(wv, 'chat-flag', 'task-f2');
    send(win, {type: 'result', text: 'all good', success: true, tabId: TAB});
    send(win, {type: 'task_done', tabId: TAB});
    send(win, {type: 'status', running: false, tabId: TAB});
    assert.ok(
      win.document.querySelector('.chat-tab .chat-tab-ok'),
      'precondition: the tab shows its last task as successful',
    );
    shareWithTasks(wv, [
      {
        task: 'doomed task',
        task_id: 'task-f1',
        events: [
          {
            type: 'tool_call',
            name: 'Bash',
            command: 'false',
            ts: Date.now(),
          },
          {type: 'result', text: 'it broke', success: false},
        ],
      },
      {task: 'list the files', task_id: 'task-f2', events: []},
    ]);
    // Repaint the tab bar the way any status change would.
    send(win, {type: 'status', running: false, tabId: TAB});
    assert.ok(
      !win.document.querySelector('.chat-tab .chat-tab-fail'),
      "replaying an OLD task's failure must not flag the tab",
    );
    assert.ok(
      win.document.querySelector('.chat-tab .chat-tab-ok'),
      'the tab still shows its own last task as successful',
    );
  });

  await test('a sub-agent tab shares its own screen, not the chat', () => {
    const wv = makeWebview();
    const win = wv.win;
    const TAB = runSmallTask(wv, 'chat-parent', 'task-parent');
    send(win, {
      type: 'openSubagentTab',
      tab_id: 'tab-sub',
      parent_tab_id: TAB,
      description: 'help out',
      task_id: 'sub-1',
    });
    const subTab = win.document.querySelector(
      '.chat-tab[data-tab-id="tab-sub"]',
    );
    assert.ok(subTab, 'the sub-agent gets its own tab');
    click(subTab);
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'make sub-work',
      tabId: 'tab-sub',
      ts: Date.now(),
    });
    click(win.document.getElementById('share-btn'));
    assert.ok(
      !wv.posted.some(m => m.type === 'shareChatTasks'),
      "a sub-agent's rows are not chat tasks: nothing to ask the daemon",
    );
    const msg = wv.posted.filter(m => m.type === 'shareChat').pop();
    assert.ok(msg, 'the sub-agent screen is still exportable');
    assert.strictEqual(
      msg.chatId,
      'tab-sub',
      "the page gets the sub-agent tab's own file name, never the " +
        "parent chat's",
    );
    assert.ok(msg.html.includes('make sub-work'), 'its panels exported');
    assert.ok(
      !msg.html.includes('ls -la'),
      "the parent task's panels stay out of the sub-agent's page",
    );
  });

  await test('an oversized transcript is refused with a visible error', () => {
    const wv = makeWebview();
    const win = wv.win;
    const TAB = runSmallTask(wv, 'chat-huge', 'task-huge');
    send(win, {
      type: 'tool_result',
      content: 'x'.repeat(41 * 1024 * 1024),
      tool_name: 'Bash',
      tabId: TAB,
      taskId: 'task-huge',
      ts: Date.now(),
    });
    const msg = shareWithTasks(wv, []);
    assert.ok(
      msg === undefined,
      'a transcript beyond the transport frame limit must not be sent',
    );
    const banners = win.document.querySelectorAll('#output .ev.err');
    const texts = Array.from(banners, b => b.textContent);
    assert.ok(
      texts.some(t => t.includes('too large')),
      'the refusal must be visible in the transcript',
    );
    assert.ok(
      win.document
        .getElementById('share-btn')
        .classList.contains('share-err'),
      'the button flashes red on refusal',
    );
  });

  console.log(`\n${passed} passed, ${failures.length} failed`);
  // The webviews leave live timers behind (the running-task clock,
  // the button flash), so exit explicitly instead of draining them.
  process.exit(failures.length > 0 ? 1 : 0);
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
