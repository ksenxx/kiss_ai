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
//   change, and polls getInfoFile for the running task's
//   tmp/PROGRESS.md, relaying the file's markdown as `progressMd`
//   (cleared when the task ends, at which point the poll stops);
// * the TASK INFO view (meta-panel-mode) renders relayed `metaState`
//   messages into its own #meta-list and #meta-info, never posts
//   `metaUpdate` or getInfoFile itself, and keeps the relayed values
//   through a local configData repaint; a null-values metaState
//   restores the placeholder dashes;
// * the HISTORY panel (history-panel-mode) posts no `metaUpdate`.

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
  return {win, posted};
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

function rowText(win, id) {
  return win.document.getElementById(id).textContent;
}

const EMDASH = '\u2014';
const ROOT = 'panel-tab-1';
const PANEL_ATTRS = ` class="editor-tab-mode" data-kiss-tab-id="${ROOT}"`;
const META_ATTRS =
  ' class="editor-tab-mode meta-panel-mode" data-kiss-tab-id="meta-panel"';
const HISTORY_ATTRS =
  ' class="editor-tab-mode history-panel-mode"' +
  ' data-kiss-tab-id="history-panel"';

async function main() {
  await test(
    'a chat editor panel reports its task-info values as metaUpdate',
    async () => {
      const wv = makeWebview(PANEL_ATTRS);
      const win = wv.win;
      // The startup report seeds the host cache with the placeholders.
      await sleep(300);
      const first = lastMetaUpdate(wv);
      assert.ok(first, 'a metaUpdate must be posted at startup');
      assert.strictEqual(first.values.tokens, EMDASH);
      assert.strictEqual(first.values.workdir, EMDASH);
      assert.strictEqual(first.progressMd, '');

      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
      });
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
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
        machine: 'buildbox',
      });
      await sleep(300);
      assert.strictEqual(lastMetaUpdate(wv).values.machine, 'buildbox');
    },
  );

  await test(
    'a running task starts the getInfoFile poll and relays progressMd; ' +
      'the end of the task clears it and stops the poll',
    async () => {
      const wv = makeWebview(PANEL_ATTRS);
      const win = wv.win;
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
      });
      await sleep(250);
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'getInfoFile').length,
        0,
        'an idle editor panel must not poll',
      );

      send(win, {type: 'status', running: true});
      await sleep(1300);
      const polls = wv.posted.filter(m => m.type === 'getInfoFile');
      assert.ok(polls.length >= 1, 'a running task starts the poll');
      const poll = polls[polls.length - 1];
      assert.strictEqual(poll.workDir, '/cfg/dir');
      assert.strictEqual(poll.tabId, ROOT);

      send(win, {
        type: 'infoFile',
        workDir: '/cfg/dir',
        tabId: ROOT,
        token: poll.token,
        exists: true,
        sig: '100:20',
        content: '# Progress\n\nstep one **done**\n',
      });
      await sleep(300);
      const upd = lastMetaUpdate(wv);
      assert.strictEqual(upd.progressMd, '# Progress\n\nstep one **done**\n');
      // The panel's own (hidden) subpanel renders it too.
      assert.ok(
        win.document
          .getElementById('meta-info-content')
          .innerHTML.includes('<strong>done</strong>'),
      );

      send(win, {type: 'status', running: false});
      await sleep(300);
      assert.strictEqual(
        lastMetaUpdate(wv).progressMd,
        '',
        'the end of the task clears the relayed progress',
      );
      const count = wv.posted.filter(m => m.type === 'getInfoFile').length;
      await sleep(1500);
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'getInfoFile').length,
        count,
        'the poll must stop with the task',
      );
    },
  );

  await test('a task start fires exactly one immediate poll', async () => {
    // Starting the poll timer fires its own immediate request;
    // syncMetaInfoRunning must not add a second identical one.
    const wv = makeWebview(PANEL_ATTRS);
    const win = wv.win;
    send(win, {
      type: 'configData',
      config: {work_dir: '/cfg/dir', max_budget: 42},
      apiKeys: {},
    });
    await sleep(250);
    send(win, {type: 'status', running: true});
    // Well under the 1s interval: only the immediate poll can exist.
    await sleep(500);
    assert.strictEqual(
      wv.posted.filter(m => m.type === 'getInfoFile').length,
      1,
      'the start of a task must fire exactly one immediate poll',
    );
  });

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
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
      });
      await sleep(250);
      send(win, {type: 'status', running: true, tabId: ROOT});
      await sleep(300);
      const polls = wv.posted.filter(m => m.type === 'getInfoFile');
      assert.ok(
        polls.length >= 1,
        'the running task must be polling before the content tab opens',
      );
      send(win, {
        type: 'infoFile',
        workDir: '/cfg/dir',
        tabId: ROOT,
        token: polls[polls.length - 1].token,
        exists: true,
        sig: '9:9',
        content: '# live progress\n',
      });
      await sleep(300);
      assert.strictEqual(lastMetaUpdate(wv).progressMd, '# live progress\n');

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
        lastMetaUpdate(wv).progressMd,
        '',
        'the relayed progress must clear with the owner task',
      );
      assert.ok(
        !win.document
          .getElementById('status-text')
          .textContent.startsWith('Running'),
        'the mirrored clock must stop with the owner task',
      );
      const count = wv.posted.filter(m => m.type === 'getInfoFile').length;
      await sleep(2400);
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'getInfoFile').length,
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
        },
        progressMd: '# Plan\n\nnext is **tests**\n',
      });
      assert.strictEqual(rowText(win, 'meta-tokens'), '9.99K');
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
      assert.ok(info.classList.contains('visible'));
      assert.ok(content.innerHTML.includes('<h1'));
      assert.ok(content.innerHTML.includes('<strong>tests</strong>'));

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
      await sleep(1300);
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'getInfoFile').length,
        0,
        'the Task Info view must not poll getInfoFile',
      );

      // No panel reporting: back to the placeholders.
      send(win, {type: 'metaState', values: null, progressMd: ''});
      assert.strictEqual(rowText(win, 'meta-tokens'), EMDASH);
      assert.strictEqual(rowText(win, 'meta-workdir'), EMDASH);
      assert.strictEqual(rowText(win, 'meta-time'), 'Ready');
      assert.strictEqual(
        win.document.getElementById('meta-time').style.color,
        '',
      );
      assert.ok(!info.classList.contains('visible'));
      assert.strictEqual(content.innerHTML, '');
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
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
      });
      send(win, {type: 'status', running: true});
      await sleep(1300);
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'metaUpdate').length,
        0,
      );
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'getInfoFile').length,
        0,
      );
    },
  );

  console.log(`\n${passed} passed, ${failures.length} failed`);
  process.exit(failures.length > 0 ? 1 : 0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
