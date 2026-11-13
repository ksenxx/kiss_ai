// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the chat page's share button (media/chat.html,
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
// * share.js reproduces the page's collapse / expand behaviour on a
//   page whose body is exactly what buildShareableHtml() serialized.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  readyTabId,
  pickMachine,
  withTeardown,
  statusMetrics,
} = require('./simplify2_harness');

const {makeRunner} = require('./ui_antipattern_harness');
const {click, shareWithTasks, makeSharePage} = require('./share_helpers');
const {trackWindow, closeWindows} = require('./workdir_helpers');

const {test: runTest, report} = makeRunner();
// Each test's windows are closed once it has settled.
const test = withTeardown(runTest, closeWindows);

function makePage(opts) {
  const wv = makeChatPage({
    beforeScripts: stubRaf,
    // The REAL production renderers, so the deferred-highlight test
    // exercises the same hljs/marked paths the page runs.
    extraScripts:
      opts && opts.highlight ? ['highlight.min.js', 'marked.min.js'] : [],
  });
  trackWindow(wv.win);
  return wv;
}

// The export size limit main.js enforces (SHARE_MAX_HTML_JSON_BYTES: 56
// MiB, the transport frame limit).  The oversized-transcript test drives
// the real limit with a 57-MiB panel (about two seconds, 350 MB).
const SHARE_LIMIT = 56 * 1024 * 1024;

/**
 * Drive one small task through the page so #task-panel and #output
 * hold real panels, then return the tab id.  Every event carries
 * *taskId* so the page adopts it as the on-screen task's id —
 * exactly what the daemon's taskId-stamped stream does.
 */
function runSmallTask(wv, chatId, taskId) {
  const win = wv.win;
  const TAB = readyTabId(wv.posted);
  const now = Date.now();
  // The daemon's order: the prompt echo, then the 'clear' that opens
  // the transcript with the task panel, then the live stream.
  send(win, {type: 'setTaskText', text: 'list the files', tabId: TAB});
  send(win, {type: 'clear', chat_id: chatId, tabId: TAB});
  send(win, {type: 'status', running: true, tabId: TAB, startTs: now});
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

async function run() {
  await test('share button sits immediately right of the mic button', () => {
    const {win} = makePage();
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
    const wv = makePage();
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
    assert.ok(
      msg.html.includes('class="ev task-panel'),
      'task panel serialized',
    );
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
    const wv = makePage();
    const TAB = readyTabId(wv.posted);
    click(wv.win.document.getElementById('share-btn'));
    const req = wv.posted.find(m => m.type === 'shareChatTasks');
    assert.ok(req, 'share must work on a tab with no backend chat id yet');
    assert.strictEqual(req.chatId, TAB);
  });

  await test('an empty chat is refused with a visible error', () => {
    const wv = makePage();
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
    // The share button sits in the closed '...' menu, so a page-side
    // refusal raises the same error toast a daemon-side one does.
    const toasts = Array.from(
      wv.win.document.querySelectorAll('.kiss-notification'),
      el => el.textContent,
    );
    assert.ok(
      toasts.some(t => t.includes('Share failed')),
      'the refusal raises an error toast; got toasts: ' +
        JSON.stringify(toasts),
    );
  });

  await test('share_done ok shows a note banner and flashes the button', () => {
    const wv = makePage();
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
    const wv = makePage();
    const TAB = runSmallTask(wv, 'chat-link', 'task-link');
    send(wv.win, {type: 'share_done', ok: true, path: SAVED, tabId: TAB});
    // The path starts as an inert candidate span and the page asks
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
      'the page must ask the host whether the saved page exists',
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
    const wv = makePage();
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
    const wv = makePage();
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

  // Audit 2026-11-11 (W1): the reply names only the tab, and the tab
  // may have moved on to another chat while the page was written (a
  // run from another surface rebinds it).  The result of chat A's
  // export used to be written into chat B's transcript.
  await test('share_done after the tab moved to another chat writes no banner', () => {
    const TASKS = [{task: 'list the files', task_id: 'task-a', events: []}];
    const outputText = wv => wv.win.document.getElementById('output').textContent;
    for (const reply of [
      {ok: true, path: '/w/reports/chat-a.html'},
      {ok: false, error: 'disk full'},
    ]) {
      const wv = makePage();
      const TAB = runSmallTask(wv, 'chat-a', 'task-a');
      const msg = shareWithTasks(wv, TASKS);
      assert.ok(msg && msg.chatId === 'chat-a', 'setup: chat A is exported');
      send(wv.win, {
        type: 'task_events',
        tabId: TAB,
        chat_id: 'chat-b',
        task_id: 'task-b',
        task: 'task b',
        extra: '{}',
        events: [{type: 'text_delta', text: 'chat b output'}, {type: 'text_end'}],
      });
      assert.ok(outputText(wv).includes('chat b output'), 'setup: chat B on screen');
      send(wv.win, Object.assign({type: 'share_done', tabId: msg.tabId}, reply));
      assert.ok(
        !outputText(wv).includes('chat-a.html') &&
          !outputText(wv).includes('Share failed'),
        "BUG: chat A's share result was written into chat B's transcript",
      );
      if (!reply.ok) {
        const toasts = Array.from(
          wv.win.document.querySelectorAll('.kiss-notification-message'),
          t => t.textContent,
        );
        assert.ok(
          toasts.some(t => t.includes('Share failed: disk full')),
          'the failure still raises its toast; got ' + JSON.stringify(toasts),
        );
      }
    }
    // The same export answered while its chat is still on screen.
    const wv = makePage();
    runSmallTask(wv, 'chat-a', 'task-a');
    const msg = shareWithTasks(wv, TASKS);
    send(wv.win, {
      type: 'share_done',
      tabId: msg.tabId,
      ok: true,
      path: '/w/reports/chat-a.html',
    });
    assert.ok(outputText(wv).includes('chat-a.html'), 'the banner is written');
  });

  await test('share_done for another tab never writes into this transcript', () => {
    const wv = makePage();
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
    const wv = makePage();
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

  await test('2026-11-04 W5 F3: a reply for a chat the tab has left is dropped', () => {
    // Another surface runs a new chat on a machine in this very tab
    // while the daemon loads the old chat's tasks: its reply would
    // splice the new chat's screen into the old chat's page.
    const wv = makePage();
    const win = wv.win;
    const TAB = runSmallTask(wv, 'old-chat-A', 'old-task-A');
    send(win, {type: 'status', running: false, tabId: TAB});
    click(win.document.getElementById('share-btn'));
    const req = wv.posted.filter(m => m.type === 'shareChatTasks').pop();
    assert.strictEqual(req.chatId, 'old-chat-A');
    send(win, {type: 'tabMachine', machine: 'remote-host', tabId: TAB});
    send(win, {type: 'setTaskText', text: 'new remote chat', tabId: TAB});
    send(win, {type: 'clear', chat_id: 'new-chat-B', tabId: TAB});
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'printf PRIVATE_NEW_CHAT_B',
      tabId: TAB,
      taskId: 'new-task-B',
      ts: Date.now(),
    });
    send(win, {
      type: 'share_tasks',
      tabId: TAB,
      chatId: 'old-chat-A',
      tasks: [{task: 'list the files', task_id: 'old-task-A', events: []}],
      truncated: false,
    });
    assert.ok(
      !wv.posted.some(m => m.type === 'shareChat'),
      'the stale reply exports nothing',
    );
    // The button is free again: a click shares the chat now shown.
    click(win.document.getElementById('share-btn'));
    const again = wv.posted.filter(m => m.type === 'shareChatTasks').pop();
    assert.notStrictEqual(again, req, 'Share works again');
    assert.strictEqual(again.chatId, 'new-chat-B');
    // Audit 2026-11-08 (W7 F1): the chat is bound to remote-host, whose
    // history holds its persisted tasks: the daemon is asked to fetch
    // them THERE, whatever machine the user has picked for the next
    // task; a reply that did not come from there (this daemon's own
    // empty history) exports nothing, visibly.
    assert.strictEqual(
      again.machine,
      'remote-host',
      'BUG - the bound chat\'s history is asked of this daemon',
    );
    send(win, {
      type: 'share_tasks',
      tabId: TAB,
      chatId: 'new-chat-B',
      tasks: [],
      truncated: false,
    });
    assert.ok(!wv.posted.some(m => m.type === 'shareChat'), 'not exported');
    const toasts = Array.from(
      win.document.querySelectorAll('.kiss-notification'),
      el => el.textContent,
    );
    assert.ok(
      toasts.some(t => /Share failed: .*remote-host/.test(t)),
      'the failure is reported; got toasts: ' + JSON.stringify(toasts),
    );
    const select = win.document.getElementById('chat-machine');
    const option = win.document.createElement('option');
    option.value = option.textContent = 'other-host';
    select.appendChild(option);
    pickMachine(win, 'other-host');
    click(win.document.getElementById('share-btn'));
    const picked = wv.posted.filter(m => m.type === 'shareChatTasks').pop();
    assert.notStrictEqual(picked, again, 'Share works again');
    assert.strictEqual(picked.machine, 'remote-host', 'the binding, not the pick');
    send(win, {
      type: 'share_tasks',
      tabId: TAB,
      chatId: 'new-chat-B',
      machine: 'remote-host',
      tasks: [{task: 'older remote task', task_id: 'old-task-B', events: []}],
      truncated: false,
    });
    const exported = wv.posted.filter(m => m.type === 'shareChat').pop();
    assert.ok(exported, 'the machine\'s reply exports');
    assert.ok(exported.html.includes('older remote task'), 'with its history');
  });

  await test('2026-11-04 W5 F2: a queued follow-up echo does not split the task in two', () => {
    // The daemon echoes a follow-up typed into the running task with
    // setTaskText (then prompt): the same task goes on, so the export
    // holds its persisted transcript once, with the live tail, not a
    // second "Original task" section.
    const wv = makePage();
    const win = wv.win;
    const TAB = runSmallTask(wv, 'chat-follow', 'task-follow');
    send(win, {type: 'setTaskText', text: 'also inspect the second file', tabId: TAB});
    send(win, {type: 'prompt', text: 'also inspect the second file', steer: true, tabId: TAB, taskId: 'task-follow'});
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'printf SAME_TASK_TAIL',
      tabId: TAB,
      taskId: 'task-follow',
      ts: Date.now(),
    });
    const msg = shareWithTasks(wv, [
      {task: 'list the files', task_id: 'task-follow', events: [
        {type: 'tool_call', name: 'Bash', command: 'ls -la', description: 'List files'},
        {type: 'tool_result', content: 'file-one.txt', tool_name: 'Bash'},
      ]},
    ]);
    const sections = makeSharePage(msg.html).document.querySelectorAll('.share-task');
    assert.strictEqual(sections.length, 1, 'one task, one section');
    assert.ok(msg.html.includes('SAME_TASK_TAIL'), 'with the live tail');
  });

  await test('share.js collapses and expands a serialized event panel', () => {
    const wv = makePage();
    runSmallTask(wv, 'chat-4', 'task-4');
    const msg = shareWithTasks(wv, []);
    const page = makeSharePage(msg.html);
    const panel = page.document.querySelector(
      '#output .collapsible:not(.task-panel)',
    );
    assert.ok(panel, 'the exported page must hold a collapsible panel');
    const header = panel.querySelector('.collapse-header');
    assert.ok(header, 'the panel must keep its collapse header');
    const wasCollapsed = panel.classList.contains('collapsed');
    // The first panel is the Bash call: collapsed, its preview is its
    // description alone ("Bash List files"); expanded, it shows none.
    // Two clicks visit both states, whichever the export starts in.
    for (const collapsed of [!wasCollapsed, wasCollapsed]) {
      click(header);
      assert.strictEqual(
        panel.classList.contains('collapsed'),
        collapsed,
        'a header click must toggle the collapsed state',
      );
      assert.strictEqual(
        panel.querySelector('.collapse-preview').textContent,
        collapsed ? 'List files' : '',
        collapsed
          ? 'a collapsed Bash panel must preview its description'
          : 'an expanded panel must show no preview',
      );
    }
  });

  await test('share.js folds a task panel like any event panel', () => {
    const wv = makePage();
    runSmallTask(wv, 'chat-5', 'task-5');
    const msg = shareWithTasks(wv, []);
    const page = makeSharePage(msg.html);
    const panel = page.document.querySelector('#output .task-panel');
    assert.ok(panel, 'exported page keeps the task panel');
    assert.ok(
      !panel.classList.contains('collapsed'),
      'the task panel is exported open: the thread reads task, then events',
    );
    const header = panel.querySelector('.collapse-header');
    click(header);
    assert.ok(
      panel.classList.contains('collapsed'),
      'a header click folds the task panel',
    );
    assert.strictEqual(
      panel.querySelector('.collapse-preview').textContent,
      'list the files',
      'a folded task panel previews the task text',
    );
    click(header);
    assert.ok(
      !panel.classList.contains('collapsed'),
      'a second click must restore the task panel',
    );
  });

  await test('each exported task opens with its own task panel', () => {
    const wv = makePage();
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
    const sections = page.document.querySelectorAll('.share-task');
    assert.strictEqual(sections.length, 2, 'one section per task');
    const panels = page.document.querySelectorAll('#output .task-panel');
    assert.strictEqual(panels.length, 2, 'one task panel per task of the chat');
    assert.strictEqual(
      sections[0].firstElementChild,
      panels[0],
      "the first section opens with the first task's panel",
    );
    assert.strictEqual(
      panels[0].querySelector('.task-panel-text').textContent,
      'first task',
    );
    assert.strictEqual(
      sections[1].firstElementChild,
      panels[1],
      "the second section opens with the second task's panel",
    );
    assert.strictEqual(
      panels[1].querySelector('.task-panel-text').textContent,
      'list the files',
    );
    click(panels[1].querySelector('.collapse-header'));
    assert.ok(
      panels[1].classList.contains('collapsed'),
      "the second task's panel folds",
    );
    assert.ok(
      !panels[0].classList.contains('collapsed'),
      "the first task's panel must not fold with it",
    );
  });

  await test('share exports every task of the chat, not only the last', () => {
    const wv = makePage();
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
    // THE regression this feature exists for: a reloaded page
    // replays only the LATEST task into its DOM, and the shared page
    // used to show just that one task.
    const wv = makePage();
    const win = wv.win;
    const TAB = readyTabId(wv.posted);
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
    const wv = makePage();
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
    const wv = makePage({highlight: true});
    const win = wv.win;
    const TAB = readyTabId(wv.posted);
    send(win, {
      type: 'task_events',
      tabId: TAB,
      task: 'replayed task',
      task_id: 'task-hl2',
      chat_id: 'chat-hl',
      // The code sits in the prompt: a replay folds that panel and
      // defers its highlighting (a Thoughts panel is never folded).
      events: [
        {
          type: 'prompt',
          text: 'show some code\n```js\nconst answer = 42;\n```',
        },
        {type: 'result', text: 'done', success: true},
      ],
    });
    assert.ok(
      win.document.querySelector('#output code.needs-hl'),
      'precondition: the replay left a deferred-highlight code block',
    );
    // A27_claude-1: the export's highlight pass re-linkifies every
    // code block; the paths it finds belong to a page already
    // serialized, so none is registered or asked about (the guard
    // stays up for that pass too).
    const checksBefore = wv.posted.filter(m => m.type === 'checkPaths').length;
    const spansBefore = win._testApi.pendingFileLinkCounts().spans;
    const msg = shareWithTasks(wv, [
      {
        task: 'earlier task',
        task_id: 'task-hl1',
        events: [
          {
            type: 'prompt',
            text: 'more code\n```py\nopen("src/kiss/core/agent.py")\n```',
          },
          {type: 'result', text: 'done', success: true},
        ],
      },
      {task: 'replayed task', task_id: 'task-hl2', events: []},
    ]);
    assert.strictEqual(
      wv.posted.filter(m => m.type === 'checkPaths').length,
      checksBefore,
      'BUG - the export asked the daemon about paths of a serialized page',
    );
    assert.strictEqual(
      win._testApi.pendingFileLinkCounts().spans,
      spansBefore,
      'no detached span of the export is registered',
    );
    assert.ok(
      !/data-path-(wd|tab|machine)=/.test(msg.html),
      'the export carries no file-link bookkeeping stamps',
    );
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
      'the fan-out loses its pin, exactly like the live page',
    );
  });

  await test('the live view survives a shared replay untouched', () => {
    const wv = makePage();
    const win = wv.win;
    const TAB = runSmallTask(wv, 'chat-live', 'task-l2');
    // The live task's own numbers are on the status row.
    send(win, {
      type: 'usage_info',
      tabId: TAB,
      taskId: 'task-l2',
      total_tokens: 1234,
      cost: 0.5,
      total_steps: 3,
    });
    const statusBefore = statusMetrics(win);
    assert.strictEqual(statusBefore.steps, 'Steps: 3', 'setup: the live count');
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
          // The replayed task's own numbers: its result paints them on
          // the status row the replay borrows.
          {
            type: 'result',
            text: 'done',
            success: true,
            total_tokens: 99999,
            cost: 9,
            step_count: 42,
          },
        ],
      },
      {task: 'list the files', task_id: 'task-l2', events: []},
    ]);
    assert.strictEqual(
      win.document.getElementById('output').innerHTML,
      outputBefore,
      "the replay of another task must not touch the screen's panels",
    );
    assert.deepStrictEqual(
      statusMetrics(win),
      statusBefore,
      "the replay must not walk the live task's status numbers",
    );
  });

  await test('a truncated share_tasks reply warns about dropped tasks', () => {
    const wv = makePage();
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
    const wv = makePage();
    runSmallTask(wv, 'chat-dberr', 'task-dberr');
    // A share that succeeded earlier on this page must not be mistaken
    // for the failed one's export.
    const first = shareWithTasks(wv, []);
    assert.ok(first, 'the first share exports');
    send(wv.win, {
      type: 'share_done',
      ok: true,
      path: '/tmp/first.html',
      tabId: first.tabId,
    });
    const exportsBefore = wv.posted.filter(m => m.type === 'shareChat').length;
    const failed = shareWithTasks(wv, [], {
      error: 'Failed to load the chat history: disk exploded',
    });
    assert.strictEqual(
      failed,
      undefined,
      'a failed task listing must not produce a partial export',
    );
    assert.strictEqual(
      wv.posted.filter(m => m.type === 'shareChat').length,
      exportsBefore,
      'no new export was sent',
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
    const wv = makePage();
    const win = wv.win;
    const TAB = runSmallTask(wv, 'chat-start', 'task-s1');
    send(win, {type: 'result', text: 'first done', success: true, tabId: TAB});
    send(win, {type: 'task_done', tabId: TAB});
    send(win, {type: 'clear', chat_id: 'chat-start', tabId: TAB});
    send(win, {type: 'setTaskText', text: 'second task', tabId: TAB});
    // No taskId on the event: the page does not know the new id.
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

  await test('W3 audit 2026-11-12: a list older than the task on screen leaves that task its own section', () => {
    // The daemon listed the chat's tasks before the task on screen
    // started: the list ends in an older task that recorded nothing,
    // which is not the task on screen mid-write (the ids differ).
    const wv = makePage();
    runSmallTask(wv, 'chat-late', 'task-new');
    const msg = shareWithTasks(wv, [
      {task: 'quiet older task', task_id: 'task-old', events: []},
    ]);
    const sections = Array.from(
      makeSharePage(msg.html).document.querySelectorAll('.share-task'),
      s => s.textContent,
    );
    assert.strictEqual(sections.length, 2, 'the older task, then the one on screen');
    assert.ok(sections[0].includes('quiet older task'));
    assert.ok(sections[0].includes('(no output recorded)'));
    assert.ok(!sections[0].includes('ls -la'), "not another task's output");
    assert.ok(sections[1].includes('list the files'));
    assert.ok(sections[1].includes('ls -la'));
  });

  await test('a chat whose tasks recorded no output still exports', () => {
    const wv = makePage();
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
    const wv = makePage();
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

  await test('an old failure in the export never marks the tab failed', () => {
    const wv = makePage();
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
    const wv = makePage();
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
    // Audit 2026-11-08 (W7 F2): the daemon's replay of the sub-agent's
    // transcript names the chat it belongs to (chat_id), which is the
    // PARENT chat's persistent id, not the sub-agent tab's -- the
    // export still goes under the tab's own name.
    send(win, {
      type: 'task_events',
      tabId: 'tab-sub',
      task: 'help out',
      task_id: 'sub-1',
      chat_id: 'chat-parent',
      events: [],
    });
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'make sub-work',
      tabId: 'tab-sub',
      ts: Date.now(),
    });
    click(win.document.getElementById('share-btn'));
    // A sub-agent's row is deliberately absent from the chat's task
    // list, but its OWN fan-outs' transcripts must ride along: the
    // daemon is asked for this one task (taskId) instead of the chat.
    const req = wv.posted.filter(m => m.type === 'shareChatTasks').pop();
    assert.ok(req, 'the daemon is asked for the sub-agent task');
    assert.strictEqual(req.taskId, 'sub-1');
    assert.strictEqual(
      req.chatId,
      'tab-sub',
      "the page gets the sub-agent tab's own file name, never the " +
        "parent chat's",
    );
    send(win, {
      type: 'share_tasks',
      tabId: req.tabId,
      chatId: req.chatId,
      tasks: [
        {
          task: 'help out',
          task_id: 'sub-1',
          events: [],
          subagents: [
            {
              task: 'nested worker',
              task_id: 'sub-1-child',
              parent_task_id: 'sub-1',
              events: [
                {type: 'tool_call', name: 'Bash', command: 'nested-work'},
              ],
            },
          ],
        },
      ],
      truncated: false,
    });
    const msg = wv.posted.filter(m => m.type === 'shareChat').pop();
    assert.ok(msg, 'the sub-agent screen is still exportable');
    assert.strictEqual(msg.chatId, 'tab-sub');
    assert.ok(msg.html.includes('make sub-work'), 'its panels exported');
    assert.ok(
      !msg.html.includes('ls -la'),
      "the parent task's panels stay out of the sub-agent's page",
    );
    assert.ok(
      msg.html.includes('nested-work'),
      "the sub-agent's own sub-agents ride along to its page",
    );
    assert.ok(
      msg.html.includes('data-task-id="sub-1-child"'),
      'the nested transcript lands in a tabbed section',
    );
  });

  await test('an oversized transcript is refused with a visible error', () => {
    const wv = makePage();
    const win = wv.win;
    const TAB = runSmallTask(wv, 'chat-huge', 'task-huge');
    // Bigger than SHARE_MAX_HTML_JSON_BYTES on its own: the panel is
    // open while the task streams, so its text is serialized once (a
    // collapsed panel repeats it in the collapse preview).
    send(win, {
      type: 'tool_result',
      content: 'x'.repeat(SHARE_LIMIT + 1024 * 1024),
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
      win.document.getElementById('share-btn').classList.contains('share-err'),
      'the button flashes red on refusal',
    );
  });

  report('shareChat');
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
