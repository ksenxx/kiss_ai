// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// The share button runs ONE export at a time.  A second click while the
// daemon has not answered the first (`share_tasks`, then `share_done`)
// is ignored instead of serializing and writing the same page twice;
// the button re-arms when the export ends either way, when its reply is
// dropped (the tab lost the screen), and after a failsafe delay when no
// reply ever comes.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  readyTabId,
} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');

const {test, report} = makeRunner();

const FAILSAFE_MS = 120000;

/** A page whose long timers (the failsafe) are captured, not scheduled. */
function makePage() {
  const longTimers = [];
  const wv = makeChatPage({
    beforeScripts: win => {
      stubRaf(win);
      const realSetTimeout = win.setTimeout.bind(win);
      win.setTimeout = (fn, ms, ...args) => {
        if (ms >= FAILSAFE_MS) {
          longTimers.push(fn);
          return -longTimers.length;
        }
        return realSetTimeout(fn, ms, ...args);
      };
    },
  });
  wv.longTimers = longTimers;
  return wv;
}

function clickShare(wv) {
  const el = wv.win.document.getElementById('share-btn');
  el.dispatchEvent(new wv.win.MouseEvent('click', {bubbles: true}));
}

function requests(wv, type) {
  return wv.posted.filter(m => m.type === type);
}

/** Drive one small task through the page so the transcript has panels. */
function runSmallTask(wv) {
  const win = wv.win;
  const TAB = readyTabId(wv.posted);
  const now = Date.now();
  send(win, {type: 'setTaskText', text: 'list the files', tabId: TAB});
  send(win, {type: 'clear', chat_id: 'chat-1', tabId: TAB});
  send(win, {type: 'status', running: true, tabId: TAB, startTs: now});
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls -la',
    description: 'List files',
    tabId: TAB,
    taskId: 'task-1',
    ts: now,
  });
  send(win, {
    type: 'tool_result',
    content: 'file-one.txt',
    tool_name: 'Bash',
    tabId: TAB,
    taskId: 'task-1',
    ts: now,
  });
  return TAB;
}

async function run() {
  await test('a second click before the reply sends nothing', () => {
    const wv = makePage();
    const TAB = runSmallTask(wv);
    clickShare(wv);
    clickShare(wv);
    assert.strictEqual(requests(wv, 'shareChatTasks').length, 1);
    const req = requests(wv, 'shareChatTasks')[0];
    send(wv.win, {
      type: 'share_tasks',
      tabId: req.tabId,
      chatId: req.chatId,
      tasks: [],
      truncated: false,
    });
    assert.strictEqual(requests(wv, 'shareChat').length, 1, 'one export');
    // Still pending until the daemon has written the page.
    clickShare(wv);
    assert.strictEqual(requests(wv, 'shareChatTasks').length, 1);
    send(wv.win, {type: 'share_done', ok: true, path: '/w/c.html', tabId: TAB});
    clickShare(wv);
    assert.strictEqual(
      requests(wv, 'shareChatTasks').length,
      2,
      'the button re-arms once the export is done',
    );
    wv.win.close();
  });

  await test('a refused export re-arms the button at once', () => {
    const wv = makePage();
    clickShare(wv);
    const req = requests(wv, 'shareChatTasks')[0];
    send(wv.win, {
      type: 'share_tasks',
      tabId: req.tabId,
      chatId: req.chatId,
      error: 'no such chat',
    });
    clickShare(wv);
    assert.strictEqual(requests(wv, 'shareChatTasks').length, 2);
    wv.win.close();
    // An empty chat is refused by the page itself (a fresh page: the
    // refusal banner above would already make this transcript non-empty).
    const wv2 = makePage();
    clickShare(wv2);
    const req2 = requests(wv2, 'shareChatTasks')[0];
    send(wv2.win, {
      type: 'share_tasks',
      tabId: req2.tabId,
      chatId: req2.chatId,
      tasks: [],
      truncated: false,
    });
    assert.strictEqual(requests(wv2, 'shareChat').length, 0);
    clickShare(wv2);
    assert.strictEqual(requests(wv2, 'shareChatTasks').length, 2);
    wv2.win.close();
  });

  await test('a reply for a tab off screen is dropped and re-arms', () => {
    const wv = makePage();
    clickShare(wv);
    const req = requests(wv, 'shareChatTasks')[0];
    send(wv.win, {
      type: 'share_tasks',
      tabId: 'some-other-tab',
      chatId: req.chatId,
      tasks: [],
      truncated: false,
    });
    assert.strictEqual(requests(wv, 'shareChat').length, 0, 'dropped');
    clickShare(wv);
    assert.strictEqual(requests(wv, 'shareChatTasks').length, 2);
    wv.win.close();
  });

  await test('a reply that never comes re-arms after the failsafe', () => {
    const wv = makePage();
    clickShare(wv);
    clickShare(wv);
    assert.strictEqual(requests(wv, 'shareChatTasks').length, 1);
    assert.strictEqual(wv.longTimers.length, 1, 'one failsafe timer armed');
    wv.longTimers[0]();
    clickShare(wv);
    assert.strictEqual(requests(wv, 'shareChatTasks').length, 2);
    wv.win.close();
  });

  // The daemon answers a share on the connection that asked; a reply
  // swallowed by an outage never arrives, so the latch must not hold
  // Share shut for its whole failsafe once the page is back.
  await test('a lost connection re-arms the button for the reconnect', () => {
    const wv = makePage();
    runSmallTask(wv);
    clickShare(wv);
    assert.strictEqual(requests(wv, 'shareChatTasks').length, 1);
    send(wv.win, {type: 'daemonStatus', connected: false, reconnecting: true});
    send(wv.win, {type: 'daemonStatus', connected: true});
    clickShare(wv);
    assert.strictEqual(
      requests(wv, 'shareChatTasks').length,
      2,
      'the retry after the reconnect is sent',
    );
    const req = requests(wv, 'shareChatTasks')[1];
    send(wv.win, {
      type: 'share_tasks',
      tabId: req.tabId,
      chatId: req.chatId,
      tasks: [],
      truncated: false,
    });
    assert.strictEqual(requests(wv, 'shareChat').length, 1, 'one export');
    send(wv.win, {
      type: 'share_done',
      ok: true,
      path: '/w/c.html',
      tabId: req.tabId,
    });
    clickShare(wv);
    assert.strictEqual(requests(wv, 'shareChatTasks').length, 3);
    wv.win.close();
  });
}

run().then(
  () => report('audit1020_share_double_click'),
  err => {
    console.error(err);
    process.exit(1);
  },
);
