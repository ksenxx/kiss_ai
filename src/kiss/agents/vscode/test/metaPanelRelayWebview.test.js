// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the task-info relay between the chat surfaces
// of editor-tabs mode (media/chat.html + media/main.js):
//
// * a chat EDITOR PANEL (editor-tab-mode) posts coalesced `metaUpdate`
//   host messages carrying its #meta-list display strings — tokens,
//   cost, steps, time, machine, workdir, max budget — whenever they
//   change, and polls getTaskUpdate (every 5 s) for the running task's
//   task update, relaying the daemon's reply as the `taskUpdate` state
//   object {content, running, updatedAt, cost, error} (null once the
//   task ends, at which point the poll stops); the host's
//   `refreshTaskUpdate` message makes it repoll with `refresh: true`;
// * the TASK INFO view (meta-panel-mode) renders relayed `metaState`
//   messages into its own #meta-list and #meta-info, never posts
//   `metaUpdate` or getTaskUpdate itself, posts `metaRefresh` to the
//   host when its refresh button is pressed, and keeps the relayed
//   values through a local configData repaint; a null-values metaState
//   restores the placeholder dashes;
// * the HISTORY panel (history-panel-mode) posts no `metaUpdate`.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const META_INFO_POLL_MS = 5000;

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
 * Load chat.html + main.js into a jsdom window.  The window's
 * setInterval / clearInterval are wrapped so the test can see the
 * poll timer main.js installs (its delay, whether it is still alive)
 * and fire a tick on demand instead of waiting out the 5 s interval:
 * `intervals` holds the live timers, `tickPolls()` runs every live
 * timer registered with META_INFO_POLL_MS.
 */
function makeWebview(bodyAttrs) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  html = html.replace('<body', '<body' + bodyAttrs);
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
  const intervals = [];
  const realSetInterval = win.setInterval.bind(win);
  const realClearInterval = win.clearInterval.bind(win);
  win.setInterval = function (fn, ms, ...args) {
    const id = realSetInterval(fn, ms, ...args);
    intervals.push({id, fn, ms});
    return id;
  };
  win.clearInterval = function (id) {
    realClearInterval(id);
    const i = intervals.findIndex(entry => entry.id === id);
    if (i >= 0) intervals.splice(i, 1);
  };
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=metarelay-main.js',
  );
  function pollTimers() {
    return intervals.filter(entry => entry.ms === META_INFO_POLL_MS);
  }
  function tickPolls() {
    for (const entry of pollTimers()) entry.fn();
  }
  return {win, posted, pollTimers, tickPolls};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function lastMetaUpdate(wv) {
  const updates = wv.posted.filter(m => m.type === 'metaUpdate');
  return updates.length > 0 ? updates[updates.length - 1] : null;
}

/**
 * Copy a message posted from the jsdom realm into plain Node objects:
 * deepStrictEqual compares prototypes, and the webview's Object is not
 * this realm's.
 */
function plain(value) {
  return JSON.parse(JSON.stringify(value));
}

function polls(wv) {
  return wv.posted.filter(m => m.type === 'getTaskUpdate');
}

function rowText(win, id) {
  return win.document.getElementById(id).textContent;
}

function updatedLabel(ms) {
  return (
    'Updated ' +
    new Date(ms).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})
  );
}

const EMDASH = '\u2014';
const ROOT = 'panel-tab-1';
const PANEL_ATTRS = ` class="editor-tab-mode" data-kiss-tab-id="${ROOT}"`;
const META_ATTRS =
  ' class="editor-tab-mode meta-panel-mode" data-kiss-tab-id="meta-panel"';
const HISTORY_ATTRS =
  ' class="editor-tab-mode history-panel-mode"' +
  ' data-kiss-tab-id="history-panel"';

const CONFIG = {
  type: 'configData',
  config: {work_dir: '/cfg/dir', max_budget: 42},
  apiKeys: {},
};

async function main() {
  await test('a chat editor panel reports its task-info values as metaUpdate', async () => {
    const wv = makeWebview(PANEL_ATTRS);
    const win = wv.win;
    // The startup report seeds the host cache with the placeholders.
    await sleep(300);
    const first = lastMetaUpdate(wv);
    assert.ok(first, 'a metaUpdate must be posted at startup');
    assert.strictEqual(first.values.tokens, EMDASH);
    assert.strictEqual(first.values.workdir, EMDASH);
    assert.strictEqual(first.taskUpdate, null);
    assert.ok(
      !('progressMd' in first),
      'the retired progressMd field must not be relayed',
    );

    send(win, CONFIG);
    send(win, {
      type: 'usage_info',
      total_tokens: 12345,
      cost: 0.5,
      total_steps: 7,
      tabId: ROOT,
    });
    await sleep(300);
    const upd = lastMetaUpdate(wv);
    assert.strictEqual(upd.values.tokens, '12.3K');
    assert.strictEqual(upd.values.cost, '$0.50');
    assert.strictEqual(upd.values.steps, '7');
    assert.strictEqual(upd.values.workdir, '/cfg/dir');
    assert.strictEqual(upd.values.maxBudget, '$42.00');

    // The machine name (configData) is mirrored too.
    send(win, {...CONFIG, machine: 'buildbox'});
    await sleep(300);
    assert.strictEqual(lastMetaUpdate(wv).values.machine, 'buildbox');

    // The task's own settings rows ride along: date, base model, the
    // two modes, chat / task ids and (when the task has one) the parent.
    const startTs = Date.UTC(2026, 8, 19, 17, 40);
    send(win, {
      type: 'task_settings',
      tabId: ROOT,
      taskId: 'task-77',
      settings: {
        model: 'model-z',
        work_dir: '/task/dir',
        is_worktree: false,
        is_parallel: true,
        max_budget: 9,
        start_ts: startTs,
        chat_id: 'chat-9',
        task_id: 'task-77',
        is_subagent: true,
        parent_task_id: 'task-70',
      },
    });
    await sleep(300);
    const withSettings = lastMetaUpdate(wv).values;
    assert.strictEqual(
      withSettings.date,
      new Date(startTs).toLocaleString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      }),
    );
    assert.strictEqual(withSettings.model, 'model-z');
    assert.strictEqual(withSettings.worktree, 'no worktree');
    assert.strictEqual(withSettings.parallel, 'parallel');
    assert.strictEqual(withSettings.chatId, 'chat-9');
    assert.strictEqual(withSettings.taskId, 'task-77 (subagent)');
    assert.strictEqual(withSettings.parentTask, 'task-70');
    assert.strictEqual(withSettings.workdir, '/task/dir');
    assert.strictEqual(withSettings.maxBudget, '$9.00');
  });

  await test(
    'a running task starts the 5 s getTaskUpdate poll and relays the ' +
      'taskUpdate state; the end of the task relays null and stops the poll',
    async () => {
      const wv = makeWebview(PANEL_ATTRS);
      const win = wv.win;
      send(win, CONFIG);
      await sleep(250);
      assert.strictEqual(
        polls(wv).length,
        0,
        'an idle editor panel must not poll',
      );
      assert.strictEqual(
        wv.pollTimers().length,
        0,
        'an idle editor panel holds no poll timer',
      );

      send(win, {type: 'status', running: true});
      await sleep(300);
      assert.strictEqual(
        wv.pollTimers().length,
        1,
        'a running task installs one poll timer at META_INFO_POLL_MS',
      );
      const firstPolls = polls(wv);
      assert.ok(firstPolls.length >= 1, 'a running task starts the poll');
      const poll = firstPolls[firstPolls.length - 1];
      assert.deepStrictEqual(Object.keys(poll).sort(), [
        'knownSig',
        'refresh',
        'tabId',
        'token',
        'type',
      ]);
      assert.strictEqual(poll.tabId, ROOT);
      assert.strictEqual(poll.knownSig, '');
      assert.strictEqual(poll.refresh, false);
      assert.strictEqual(typeof poll.token, 'string');

      // The interval elapses: another poll, still with no known sig.
      wv.tickPolls();
      const second = polls(wv);
      assert.strictEqual(second.length, firstPolls.length + 1);
      assert.strictEqual(second[second.length - 1].knownSig, '');
      assert.strictEqual(second[second.length - 1].token, poll.token);

      const updatedAt = Date.UTC(2026, 8, 22, 10, 30);
      send(win, {
        type: 'taskUpdate',
        tabId: ROOT,
        token: poll.token,
        taskId: 'task-1',
        exists: true,
        sig: 'sig-1',
        content: '<h4>Progress</h4><p>step one <strong>done</strong></p>',
        error: '',
        running: false,
        cost: 0.12,
        updatedAt,
      });
      await sleep(300);
      const upd = lastMetaUpdate(wv);
      assert.deepStrictEqual(plain(upd.taskUpdate), {
        content: '<h4>Progress</h4><p>step one <strong>done</strong></p>',
        running: false,
        updatedAt,
        cost: 0.12,
        error: '',
      });
      // The panel's own (hidden) subpanel renders it too.
      const content = win.document.getElementById('meta-info-content');
      assert.ok(content.innerHTML.includes('<strong>done</strong>'));
      assert.ok(content.innerHTML.includes('<h4>Progress</h4>'));
      assert.strictEqual(
        rowText(win, 'meta-info-status'),
        updatedLabel(updatedAt) + ' \u00b7 $0.12',
      );

      // The next poll carries the reply's sig as knownSig.
      wv.tickPolls();
      const withSig = polls(wv);
      assert.strictEqual(withSig[withSig.length - 1].knownSig, 'sig-1');

      // An `unchanged` reply leaves the relayed state alone.
      const metaCount = wv.posted.filter(m => m.type === 'metaUpdate').length;
      send(win, {
        type: 'taskUpdate',
        tabId: ROOT,
        token: poll.token,
        exists: true,
        sig: 'sig-1',
        unchanged: true,
      });
      await sleep(300);
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'metaUpdate').length,
        metaCount,
        'an unchanged reply must not repost the relay',
      );
      assert.ok(content.innerHTML.includes('<strong>done</strong>'));

      // A reply for another generation (stale token) is ignored.
      send(win, {
        type: 'taskUpdate',
        tabId: ROOT,
        token: 'stale',
        exists: true,
        sig: 'sig-9',
        content: '<p>stale</p>',
        running: false,
        cost: 0,
        updatedAt,
        error: '',
      });
      await sleep(300);
      assert.ok(!content.innerHTML.includes('stale'));

      send(win, {type: 'status', running: false});
      await sleep(300);
      assert.strictEqual(
        lastMetaUpdate(wv).taskUpdate,
        null,
        'the end of the task relays a null task update',
      );
      assert.strictEqual(content.innerHTML, '');
      assert.strictEqual(rowText(win, 'meta-info-status'), '');
      assert.strictEqual(
        wv.pollTimers().length,
        0,
        'the poll timer must stop with the task',
      );
      const count = polls(wv).length;
      wv.tickPolls();
      await sleep(200);
      assert.strictEqual(
        polls(wv).length,
        count,
        'no further polls after the task',
      );
    },
  );

  await test('a task start fires exactly one immediate poll', async () => {
    // Starting the poll timer fires its own immediate request;
    // syncMetaInfoRunning must not add a second identical one.
    const wv = makeWebview(PANEL_ATTRS);
    const win = wv.win;
    send(win, CONFIG);
    await sleep(250);
    send(win, {type: 'status', running: true});
    // Well under the 5 s interval: only the immediate poll can exist.
    await sleep(500);
    assert.strictEqual(
      polls(wv).length,
      1,
      'the start of a task must fire exactly one immediate poll',
    );
  });

  await test(
    'a taskUpdate reply with no content for a running agent shows the ' +
      'pending state; exists:false hides the subpanel',
    async () => {
      const wv = makeWebview(PANEL_ATTRS);
      const win = wv.win;
      send(win, CONFIG);
      await sleep(250);
      send(win, {type: 'status', running: true});
      await sleep(300);
      const poll = polls(wv)[0];
      send(win, {
        type: 'taskUpdate',
        tabId: ROOT,
        token: poll.token,
        exists: true,
        sig: 'sig-0',
        content: '',
        error: '',
        running: true,
        cost: 0,
        updatedAt: 0,
      });
      await sleep(300);
      const info = win.document.getElementById('meta-info');
      const content = win.document.getElementById('meta-info-content');
      const btn = win.document.getElementById('meta-info-refresh');
      assert.ok(info.classList.contains('visible'));
      assert.ok(content.innerHTML.includes('meta-info-pending'));
      assert.ok(content.textContent.includes('Preparing the first update'));
      assert.strictEqual(rowText(win, 'meta-info-status'), 'Updating\u2026');
      assert.strictEqual(btn.disabled, true);
      assert.ok(btn.classList.contains('spinning'));
      assert.deepStrictEqual(plain(lastMetaUpdate(wv).taskUpdate), {
        content: '',
        running: true,
        updatedAt: 0,
        cost: 0,
        error: '',
      });

      // A failed run reports the error next to the last good time.
      const updatedAt = Date.UTC(2026, 8, 22, 11, 0);
      send(win, {
        type: 'taskUpdate',
        tabId: ROOT,
        token: poll.token,
        exists: true,
        sig: 'sig-1',
        content: '<p>old report</p>',
        error: 'boom',
        running: false,
        cost: 0,
        updatedAt,
      });
      await sleep(300);
      assert.strictEqual(
        rowText(win, 'meta-info-status'),
        updatedLabel(updatedAt) + ' \u00b7 Last run failed: boom',
      );
      assert.strictEqual(btn.disabled, false);
      assert.ok(!btn.classList.contains('spinning'));
      assert.strictEqual(lastMetaUpdate(wv).taskUpdate.error, 'boom');

      // Scripts and event handlers never reach the DOM.
      send(win, {
        type: 'taskUpdate',
        tabId: ROOT,
        token: poll.token,
        exists: true,
        sig: 'sig-2',
        content:
          '<p>safe</p><script>window.__pwned = 1</script>' +
          '<img src="x" onerror="window.__pwned = 2">',
        error: '',
        running: false,
        cost: 0,
        updatedAt,
      });
      await sleep(300);
      assert.ok(content.innerHTML.includes('<p>safe</p>'));
      assert.ok(!content.innerHTML.includes('<script'));
      assert.ok(!content.innerHTML.includes('onerror'));
      assert.strictEqual(win.__pwned, undefined);

      // Plain (markdown) text still renders through marked.
      send(win, {
        type: 'taskUpdate',
        tabId: ROOT,
        token: poll.token,
        exists: true,
        sig: 'sig-3',
        content: '# Plan\n\nnext is **tests**\n',
        error: '',
        running: false,
        cost: 0,
        updatedAt,
      });
      await sleep(300);
      assert.ok(content.innerHTML.includes('<h1'));
      assert.ok(content.innerHTML.includes('<strong>tests</strong>'));

      send(win, {
        type: 'taskUpdate',
        tabId: ROOT,
        token: poll.token,
        exists: false,
        sig: '',
      });
      await sleep(300);
      assert.ok(!info.classList.contains('visible'));
      assert.strictEqual(content.innerHTML, '');
      assert.strictEqual(lastMetaUpdate(wv).taskUpdate, null);
    },
  );

  await test(
    'the host message refreshTaskUpdate makes a running chat editor ' +
      'panel poll with refresh:true; its own refresh button does too',
    async () => {
      const wv = makeWebview(PANEL_ATTRS);
      const win = wv.win;
      send(win, CONFIG);
      await sleep(250);
      // Idle: the relayed refresh has no running task to poll for.
      send(win, {type: 'refreshTaskUpdate'});
      await sleep(100);
      assert.strictEqual(polls(wv).length, 0, 'an idle panel never polls');

      send(win, {type: 'status', running: true});
      await sleep(300);
      const before = polls(wv).length;
      assert.ok(before >= 1);
      send(win, {type: 'refreshTaskUpdate'});
      await sleep(100);
      const afterHost = polls(wv);
      assert.strictEqual(afterHost.length, before + 1);
      const refreshPoll = afterHost[afterHost.length - 1];
      assert.strictEqual(refreshPoll.refresh, true);
      assert.strictEqual(refreshPoll.tabId, ROOT);
      assert.strictEqual(refreshPoll.token, afterHost[0].token);

      // The panel's own (hidden) refresh button takes the same path.
      win.document.getElementById('meta-info-refresh').click();
      await sleep(100);
      const afterClick = polls(wv);
      assert.strictEqual(afterClick.length, before + 2);
      assert.strictEqual(afterClick[afterClick.length - 1].refresh, true);
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'metaRefresh').length,
        0,
        'a chat editor panel polls itself instead of asking the host',
      );
    },
  );

  await test(
    'the poll goes quiet when the owner chat stops behind a visible ' +
      'content tab',
    async () => {
      // A content tab (a file view) keeps the chat tab's task-info
      // target, and the owner's `status running:false` is NOT the
      // active tab's, so the module-level running flag never flips.
      // The poll must consult the POLLED tab's own flag and stop
      // issuing requests anyway.
      const wv = makeWebview(PANEL_ATTRS);
      const win = wv.win;
      send(win, CONFIG);
      await sleep(250);
      send(win, {type: 'status', running: true, tabId: ROOT});
      await sleep(300);
      const firstPolls = polls(wv);
      assert.ok(
        firstPolls.length >= 1,
        'the running task must be polling before the content tab opens',
      );
      const updatedAt = Date.UTC(2026, 8, 22, 12, 0);
      send(win, {
        type: 'taskUpdate',
        tabId: ROOT,
        token: firstPolls[firstPolls.length - 1].token,
        exists: true,
        sig: '9:9',
        content: '<p>live progress</p>',
        error: '',
        running: false,
        cost: 0.01,
        updatedAt,
      });
      await sleep(300);
      assert.strictEqual(
        lastMetaUpdate(wv).taskUpdate.content,
        '<p>live progress</p>',
      );

      // A directory listing opens (and focuses) a content tab.
      send(win, {
        type: 'fileContent',
        path: '/cfg/dir/sub',
        name: 'sub',
        isDirectory: true,
        content: 'a.txt\nb.txt\n',
      });
      // The owner chat's task ends while the content tab is visible.
      send(win, {type: 'status', running: false, tabId: ROOT});
      await sleep(300);
      assert.strictEqual(
        lastMetaUpdate(wv).taskUpdate,
        null,
        'the relayed task update must clear with the owner task',
      );
      assert.ok(
        !win.document
          .getElementById('status-text')
          .textContent.startsWith('Running'),
        'the mirrored clock must stop with the owner task',
      );
      const count = polls(wv).length;
      wv.tickPolls();
      wv.tickPolls();
      await sleep(200);
      assert.strictEqual(
        polls(wv).length,
        count,
        'no further polls once the polled tab itself stopped running',
      );
    },
  );

  await test(
    'the Task Info view renders metaState relays and never reports ' +
      'or polls itself',
    async () => {
      const wv = makeWebview(META_ATTRS);
      const win = wv.win;
      await sleep(400);
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'metaUpdate').length,
        0,
        'the Task Info view must not post metaUpdate',
      );

      const updatedAt = Date.UTC(2026, 8, 22, 13, 45);
      send(win, {
        type: 'metaState',
        values: {
          tokens: '9.99K',
          cost: '$1.25',
          steps: '42',
          time: '3m 2.0s',
          timeColor: 'var(--red)',
          machine: 'buildbox',
          workdir: '/task/dir',
          maxBudget: '$50.00',
          date: 'Sep 19, 2026, 05:40 PM',
          model: 'model-z',
          worktree: 'worktree',
          parallel: 'sequential',
          chatId: 'chat-9',
          taskId: 'task-77 (subagent)',
          parentTask: 'task-70',
        },
        taskUpdate: {
          content: '<h4>Done</h4><ul><li>x</li></ul>',
          running: false,
          updatedAt,
          cost: 0.05,
          error: '',
        },
      });
      assert.strictEqual(rowText(win, 'meta-tokens'), '9.99K');
      assert.strictEqual(rowText(win, 'meta-date'), 'Sep 19, 2026, 05:40 PM');
      assert.strictEqual(rowText(win, 'meta-model'), 'model-z');
      assert.strictEqual(rowText(win, 'meta-worktree'), 'worktree');
      assert.strictEqual(rowText(win, 'meta-parallel'), 'sequential');
      assert.strictEqual(rowText(win, 'meta-chat-id'), 'chat-9');
      assert.strictEqual(rowText(win, 'meta-task-id'), 'task-77 (subagent)');
      assert.strictEqual(rowText(win, 'meta-parent-id'), 'task-70');
      assert.strictEqual(
        win.document.getElementById('meta-parent-item').hidden,
        false,
        'the Parent task row shows for a relayed subagent task',
      );
      assert.strictEqual(rowText(win, 'meta-cost'), '$1.25');
      assert.strictEqual(rowText(win, 'meta-steps'), '42');
      assert.strictEqual(rowText(win, 'meta-time'), '3m 2.0s');
      assert.strictEqual(
        win.document.getElementById('meta-time').style.color,
        'var(--red)',
      );
      assert.strictEqual(rowText(win, 'meta-machine'), 'buildbox');
      assert.strictEqual(rowText(win, 'meta-workdir'), '/task/dir');
      assert.strictEqual(rowText(win, 'meta-max-budget'), '$50.00');
      const info = win.document.getElementById('meta-info');
      const content = win.document.getElementById('meta-info-content');
      const btn = win.document.getElementById('meta-info-refresh');
      assert.ok(info.classList.contains('visible'));
      assert.strictEqual(
        content.innerHTML,
        '<h4>Done</h4><ul><li>x</li></ul>',
        'the report HTML is rendered as-is',
      );
      assert.strictEqual(
        rowText(win, 'meta-info-status'),
        updatedLabel(updatedAt) + ' \u00b7 $0.05',
      );
      assert.strictEqual(btn.disabled, false);

      // A local configData repaint (the view's own daemon connection)
      // must not clobber the relayed rows.
      send(win, {
        type: 'configData',
        config: {work_dir: '/other/dir', max_budget: 3},
        apiKeys: {},
        machine: 'localhost',
      });
      await sleep(250);
      assert.strictEqual(rowText(win, 'meta-workdir'), '/task/dir');
      assert.strictEqual(rowText(win, 'meta-max-budget'), '$50.00');
      assert.strictEqual(rowText(win, 'meta-machine'), 'buildbox');

      // No poll, even while a task is reported running somewhere.
      send(win, {type: 'status', running: true});
      await sleep(300);
      wv.tickPolls();
      await sleep(100);
      assert.strictEqual(
        polls(wv).length,
        0,
        'the Task Info view must not poll getTaskUpdate',
      );
      assert.strictEqual(
        wv.pollTimers().length,
        0,
        'the Task Info view holds no poll timer',
      );

      // A relayed running state spins the button.
      send(win, {
        type: 'metaState',
        values: {tokens: '9.99K'},
        taskUpdate: {
          content: '<p>so far</p>',
          running: true,
          updatedAt,
          cost: 0.05,
          error: '',
        },
      });
      assert.strictEqual(rowText(win, 'meta-info-status'), 'Updating\u2026');
      assert.strictEqual(btn.disabled, true);
      assert.ok(btn.classList.contains('spinning'));

      // A relay from an older panel build (no settings keys) shows
      // dashes for the settings rows and hides the parent row; a null
      // task update hides the subpanel.
      send(win, {
        type: 'metaState',
        values: {tokens: '1', workdir: '/x'},
        taskUpdate: null,
      });
      assert.strictEqual(rowText(win, 'meta-model'), EMDASH);
      assert.strictEqual(rowText(win, 'meta-parent-id'), EMDASH);
      assert.strictEqual(
        win.document.getElementById('meta-parent-item').hidden,
        true,
      );
      assert.ok(!info.classList.contains('visible'));
      assert.strictEqual(content.innerHTML, '');
      assert.strictEqual(rowText(win, 'meta-info-status'), '');
      assert.strictEqual(btn.disabled, false);
      assert.ok(!btn.classList.contains('spinning'));

      // No panel reporting: back to the placeholders.
      send(win, {type: 'metaState', values: null, taskUpdate: null});
      assert.strictEqual(rowText(win, 'meta-tokens'), EMDASH);
      assert.strictEqual(rowText(win, 'meta-workdir'), EMDASH);
      assert.strictEqual(rowText(win, 'meta-date'), EMDASH);
      assert.strictEqual(rowText(win, 'meta-task-id'), EMDASH);
      assert.strictEqual(rowText(win, 'meta-time'), 'Ready');
      assert.strictEqual(
        win.document.getElementById('meta-time').style.color,
        '',
      );
      assert.ok(!info.classList.contains('visible'));
      assert.strictEqual(content.innerHTML, '');
    },
  );

  await test(
    'the Task Info view relays its refresh button to the host as ' +
      'metaRefresh and never polls getTaskUpdate',
    async () => {
      const wv = makeWebview(META_ATTRS);
      const win = wv.win;
      await sleep(300);
      send(win, {
        type: 'metaState',
        values: {tokens: '1'},
        taskUpdate: {
          content: '<p>report</p>',
          running: false,
          updatedAt: Date.UTC(2026, 8, 22, 14, 0),
          cost: 0,
          error: '',
        },
      });
      win.document.getElementById('meta-info-refresh').click();
      await sleep(100);
      assert.deepStrictEqual(
        plain(wv.posted.filter(m => m.type === 'metaRefresh')),
        [{type: 'metaRefresh'}],
        'the click posts exactly one metaRefresh to the host',
      );
      assert.strictEqual(
        polls(wv).length,
        0,
        'the Task Info view never posts getTaskUpdate',
      );
      // The host's refreshTaskUpdate is meant for chat editor panels
      // and does nothing here either.
      send(win, {type: 'refreshTaskUpdate'});
      await sleep(100);
      assert.strictEqual(polls(wv).length, 0);
    },
  );

  await test('the history panel posts no metaUpdate', async () => {
    const wv = makeWebview(HISTORY_ATTRS);
    await sleep(400);
    assert.strictEqual(
      wv.posted.filter(m => m.type === 'metaUpdate').length,
      0,
      'the history panel must not post metaUpdate',
    );
  });

  await test(
    'an ordinary sidebar webview (no editor-tab mode) posts no ' +
      'metaUpdate and starts no poll on a running task',
    async () => {
      const wv = makeWebview('');
      const win = wv.win;
      send(win, CONFIG);
      send(win, {type: 'status', running: true});
      await sleep(300);
      wv.tickPolls();
      await sleep(100);
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'metaUpdate').length,
        0,
      );
      assert.strictEqual(polls(wv).length, 0);
      assert.strictEqual(wv.pollTimers().length, 0);
    },
  );

  console.log(`\n${passed} passed, ${failures.length} failed`);
  process.exit(failures.length > 0 ? 1 : 0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
