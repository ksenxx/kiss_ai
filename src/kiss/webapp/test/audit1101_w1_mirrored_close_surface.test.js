// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/* global require, console, process */

// Audit 2026-11-01 (W1 F4): a running chat closed from ANOTHER surface
// (the registry snapshot no longer lists it) while a file tab survives
// on a stacked surface must leave no running state behind, exactly as
// a local close onto a file does (closeTab -> clearChatSurface): the
// timer, the Stop button and the closed chat's transcript are gone
// once the file takes the screen.  In the split layout the chat pane
// shows the fresh placeholder chat instead, as before.
//
// Driven through the real chat.html + api.js + main.js in jsdom.

const assert = require('assert');
const {
  makePage,
  send,
  sleep,
  byId,
  output,
  stubRaf,
  stubMonaco,
  fakeMatchMedia,
} = require('./simplify2_harness');

function makeSurface(desktop) {
  let created;
  const {win} = makePage({
    bodyClass: 'remote-chat',
    beforeScripts(w) {
      stubRaf(w);
      fakeMatchMedia(w, desktop);
      created = stubMonaco(w);
    },
  });
  return {win, created};
}

/**
 * Chat a1 (the registry's only tab) runs a task whose transcript is on
 * screen; a file of its workspace is open in the background.
 */
async function runChatWithBackgroundFile(ctx) {
  const {win, created} = ctx;
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {work_dir: '/ws/a'}});
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 'a1', chatId: 'chat-1', workDir: '/ws/a'}],
  });
  win._testApi.endLaunch();
  send(win, {type: 'setTaskText', tabId: 'a1', text: 'closed chat task'});
  send(win, {
    type: 'status',
    tabId: 'a1',
    running: true,
    taskId: 't1',
    startTs: Date.now(),
  });
  send(win, {
    type: 'text_delta',
    tabId: 'a1',
    taskId: 't1',
    text: 'transcript of closed chat',
  });
  send(win, {type: 'text_end', tabId: 'a1', taskId: 't1'});
  send(win, {
    type: 'task_settings',
    tabId: 'a1',
    taskId: 't1',
    settings: {task_id: 't1', chat_id: 'chat-1', model: 'closed-model'},
  });
  send(win, {
    type: 'usage_info',
    tabId: 'a1',
    taskId: 't1',
    total_tokens: 1234,
    cost: 0.5,
    total_steps: 3,
  });
  assert.strictEqual(byId(win, 'meta-task-id').textContent, 't1');
  assert.strictEqual(byId(win, 'meta-model').textContent, 'closed-model');
  assert.match(byId(win, 'status-tokens').textContent, /^Tokens: /);
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/a/notes.txt',
    name: 'notes.txt',
    content: 'hello',
    version: 'v1',
    background: true,
  });
  // The stubbed editor is created on the next microtask.
  await sleep(0);
  assert.strictEqual(created.length, 1, 'the file got an editor');
  assert.strictEqual(win._testApi.getActiveTabId(), 'a1', 'the chat stays on');
  assert.match(
    byId(win, 'status-text').textContent,
    /^Running/,
    'setup: the timer ticks',
  );
  assert.ok(
    output(win).textContent.includes('transcript of closed chat'),
    'setup: the transcript is on screen',
  );
}

async function testStackedMirroredCloseOntoAFileClearsTheChatSurface() {
  const ctx = makeSurface(false);
  const {win} = ctx;
  try {
    await runChatWithBackgroundFile(ctx);
    // Another surface closes the running chat: the snapshot drops it.
    send(win, {type: 'tabs_state', tabs: []});
    const open = win._testApi.openTabs();
    assert.strictEqual(open.length, 1, 'only the file survives');
    assert.ok(open[0].isContentTab, 'the survivor is the file');
    assert.strictEqual(
      win._testApi.getActiveTabId(),
      open[0].id,
      'the file took the screen',
    );
    assert.doesNotMatch(
      byId(win, 'status-text').textContent,
      /^Running/,
      "BUG - the closed chat's timer must stop",
    );
    assert.strictEqual(
      output(win).children.length,
      0,
      "BUG - the closed chat's transcript must leave #output",
    );
    assert.strictEqual(
      byId(win, 'stop-btn').style.display,
      'none',
      'nothing is running: no Stop button',
    );
    // Audit 2026-11-11 (W1): the Task Info rows and the usage numbers
    // described the closed chat's task; they leave with it.
    assert.notStrictEqual(
      byId(win, 'meta-task-id').textContent,
      't1',
      "BUG - the closed chat's task id must leave Task Info",
    );
    assert.notStrictEqual(
      byId(win, 'meta-model').textContent,
      'closed-model',
      "BUG - the closed chat's model must leave Task Info",
    );
    assert.strictEqual(
      byId(win, 'status-tokens').textContent +
        byId(win, 'status-budget').textContent +
        byId(win, 'status-steps').textContent,
      '',
      "BUG - the closed chat's usage numbers must leave the status row",
    );
    // A chat opened afterwards starts clean.
    send(win, {
      type: 'tabs_state',
      tabs: [{tabId: 'b1', chatId: 'chat-2', workDir: '/ws/b'}],
    });
    win._testApi.switchToTab('b1');
    assert.strictEqual(win._testApi.getActiveTabId(), 'b1');
    assert.doesNotMatch(byId(win, 'status-text').textContent, /^Running/);
    assert.ok(
      !output(win).textContent.includes('transcript of closed chat'),
      "the closed chat's transcript never shows under another chat",
    );
  } finally {
    win.close();
  }
}

async function testSplitMirroredCloseKeepsThePlaceholderChat() {
  const ctx = makeSurface(true);
  const {win} = ctx;
  try {
    assert.ok(
      win.document.body.classList.contains('remote-desktop'),
      'setup: the split layout',
    );
    await runChatWithBackgroundFile(ctx);
    send(win, {type: 'tabs_state', tabs: []});
    const open = win._testApi.openTabs();
    assert.strictEqual(open.length, 2, 'the file and a placeholder chat');
    const chat = open.find(t => !t.isContentTab);
    assert.ok(chat, 'the chat pane got a placeholder chat');
    assert.strictEqual(
      win._testApi.getActiveTabId(),
      chat.id,
      'the placeholder is the active tab',
    );
    assert.doesNotMatch(
      byId(win, 'status-text').textContent,
      /^Running/,
      "the closed chat's timer stopped",
    );
    // The placeholder's welcome screen, not the closed chat's transcript.
    assert.ok(
      !output(win).textContent.includes('transcript of closed chat'),
      "the closed chat's transcript left #output",
    );
    const welcome = byId(win, 'welcome');
    assert.ok(
      output(win).contains(welcome) && welcome.style.display !== 'none',
      'the placeholder shows its welcome screen',
    );
    assert.strictEqual(byId(win, 'stop-btn').style.display, 'none');
  } finally {
    win.close();
  }
}

(async () => {
  await testStackedMirroredCloseOntoAFileClearsTheChatSurface();
  console.log(
    '  ok - stacked: a mirrored close onto a file clears the chat surface',
  );
  await testSplitMirroredCloseKeepsThePlaceholderChat();
  console.log(
    '  ok - split: a mirrored close keeps the placeholder chat clean',
  );
})().catch(e => {
  console.error(e);
  process.exit(1);
});
