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
// * clicking it posts a `shareChat` command carrying the highlighted
//   tab's chat id, the static task panel and every transcript panel,
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
 * hold real panels, then return the tab id.
 */
function runSmallTask(wv, chatId) {
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
    ts: now,
  });
  send(win, {
    type: 'tool_result',
    content: 'file-one.txt\nfile-two.txt',
    tool_name: 'Bash',
    tabId: TAB,
    ts: now,
  });
  return TAB;
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

  await test('click posts shareChat with chat id, task panel, panels', () => {
    const wv = makeWebview();
    const TAB = runSmallTask(wv, 'chat-777');
    click(wv.win.document.getElementById('share-btn'));
    const msg = wv.posted.find(m => m.type === 'shareChat');
    assert.ok(msg, 'clicking share must post a shareChat command');
    assert.strictEqual(msg.tabId, TAB);
    assert.strictEqual(msg.chatId, 'chat-777');
    assert.ok(msg.html.includes('id="task-panel"'), 'task panel serialized');
    assert.ok(
      msg.html.includes('list the files'),
      'task panel text serialized',
    );
    assert.ok(msg.html.includes('id="output"'), 'transcript serialized');
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
    const msg = wv.posted.find(m => m.type === 'shareChat');
    assert.ok(msg, 'share must work on a tab with no backend chat id yet');
    assert.strictEqual(msg.chatId, TAB);
  });

  await test('share_done ok shows a note banner and flashes the button', () => {
    const wv = makeWebview();
    const TAB = runSmallTask(wv, 'chat-1');
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

  await test('share_done failure shows an error banner and a red flash', () => {
    const wv = makeWebview();
    const TAB = runSmallTask(wv, 'chat-2');
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
    runSmallTask(wv, 'chat-3');
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

  await test('share.js collapses and expands a serialized event panel', () => {
    const wv = makeWebview();
    runSmallTask(wv, 'chat-4');
    click(wv.win.document.getElementById('share-btn'));
    const msg = wv.posted.find(m => m.type === 'shareChat');
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
    runSmallTask(wv, 'chat-5');
    click(wv.win.document.getElementById('share-btn'));
    const msg = wv.posted.find(m => m.type === 'shareChat');
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
    const TAB = runSmallTask(wv, 'chat-multi');
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
      ts: Date.now(),
    });
    assert.ok(
      !win.document.getElementById('output').textContent.includes('ls -la'),
      'precondition: the first task left the live transcript',
    );
    click(win.document.getElementById('share-btn'));
    const msg = wv.posted.find(m => m.type === 'shareChat');
    assert.ok(msg.html.includes('ls -la'), 'first task panels exported');
    assert.ok(msg.html.includes('first done'), 'first task result exported');
    assert.ok(msg.html.includes('pwd'), 'second task panels exported');
    const firstIdx = msg.html.indexOf('ls -la');
    const secondIdx = msg.html.indexOf('pwd');
    assert.ok(firstIdx < secondIdx, 'tasks appear in chronological order');
  });

  await test('a chat reset (showWelcome) drops the archived tasks', () => {
    const wv = makeWebview();
    const win = wv.win;
    const TAB = runSmallTask(wv, 'chat-reset');
    send(win, {type: 'showWelcome', tabId: TAB});
    send(win, {type: 'clear', chat_id: 'chat-reset2', tabId: TAB});
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'pwd',
      tabId: TAB,
      ts: Date.now(),
    });
    click(win.document.getElementById('share-btn'));
    const msg = wv.posted.find(m => m.type === 'shareChat');
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
    click(win.document.getElementById('share-btn'));
    const msg = wv.posted.find(m => m.type === 'shareChat');
    assert.ok(
      !msg.html.includes('needs-hl'),
      'no code block may stay unhighlighted in the export',
    );
    assert.ok(
      msg.html.includes('hljs'),
      'the exported code block carries real highlight markup',
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

  await test('an oversized transcript is refused with a visible error', () => {
    const wv = makeWebview();
    const win = wv.win;
    const TAB = runSmallTask(wv, 'chat-huge');
    send(win, {
      type: 'tool_result',
      content: 'x'.repeat(41 * 1024 * 1024),
      tool_name: 'Bash',
      tabId: TAB,
      ts: Date.now(),
    });
    click(win.document.getElementById('share-btn'));
    assert.ok(
      !wv.posted.some(m => m.type === 'shareChat'),
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
