// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the docked task-info panel's new rows and info
// subpanel (media/chat.html + media/main.js, remote desktop mode):
//
// * #meta-workdir / #meta-max-budget rows fall back to configData
//   values (config.work_dir / config.max_budget) and adopt a live
//   task's task_settings (work_dir / max_budget) when they arrive,
// * the 2s getInfoFile poll runs only in remote desktop mode and only
//   while the visible tab has a RUNNING task, and carries the active
//   workdir plus the last known signature,
// * an infoFile reply paints #meta-info-content with the file's
//   MARKDOWN-FORMATTED contents (marked + kissSanitize) and shows the
//   subpanel (`visible` on #meta-info), hides the whole subpanel when
//   the file does not exist or the task ends, and ignores unchanged
//   or stale-workdir replies.

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

function makeWebview(opts) {
  const {desktopMatches = true} = opts || {};
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
      matches: query === '(min-width: 900px)' ? desktopMatches : false,
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
      '\n//# sourceURL=metainfo-main.js',
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

function rowText(win, id) {
  return win.document.getElementById(id).textContent;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/** Flip the visible tab's running state via a daemon status event. */
function setRunning(win, running) {
  send(win, {type: 'status', running});
}

/** Whether the #meta-info subpanel (header included) is shown. */
function infoVisible(win) {
  return win.document
    .getElementById('meta-info')
    .classList.contains('visible');
}

const EMDASH = '\u2014';

async function main() {
  await test(
    'workdir and max budget rows exist and fall back through config ' +
      'to task settings',
    () => {
      const wv = makeWebview();
      const win = wv.win;
      const TAB = tabIdOf(wv);
      assert.ok(
        win.document.body.classList.contains('remote-desktop'),
        'desktop remote webview must carry remote-desktop',
      );
      assert.strictEqual(rowText(win, 'meta-workdir'), EMDASH);
      assert.strictEqual(rowText(win, 'meta-max-budget'), EMDASH);

      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
      });
      assert.strictEqual(rowText(win, 'meta-workdir'), '/cfg/dir');
      assert.strictEqual(rowText(win, 'meta-max-budget'), '$42.00');

      send(win, {type: 'setTaskText', text: 'do it', tabId: TAB});
      send(win, {
        type: 'task_settings',
        settings: {
          model: 'model-x',
          work_dir: '/task/dir',
          max_budget: 7.5,
          chat_id: 'chat-1',
          task_id: 'task-1',
        },
        tabId: TAB,
        taskId: 'task-1',
      });
      assert.strictEqual(rowText(win, 'meta-workdir'), '/task/dir');
      assert.strictEqual(rowText(win, 'meta-max-budget'), '$7.50');

      // A new run clears the settings: the rows fall back to config.
      send(win, {type: 'clear', chat_id: 'chat-1', tabId: TAB});
      assert.strictEqual(rowText(win, 'meta-workdir'), '/cfg/dir');
      assert.strictEqual(rowText(win, 'meta-max-budget'), '$42.00');
    },
  );

  await test(
    'info subpanel polls getInfoFile and shows formatted markdown',
    async () => {
      const wv = makeWebview();
      const win = wv.win;
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
      });
      setRunning(win, true);
      await sleep(2400);
      const polls = wv.posted.filter(m => m.type === 'getInfoFile');
      assert.ok(polls.length >= 1, 'a getInfoFile poll must have fired');
      const poll = polls[polls.length - 1];
      assert.strictEqual(poll.workDir, '/cfg/dir');
      assert.strictEqual(poll.knownSig, '');
      const tok = poll.token;
      assert.ok(tok, 'polls must carry a request token');

      const content = win.document.getElementById('meta-info-content');
      assert.strictEqual(content.innerHTML, '');
      assert.ok(!infoVisible(win), 'subpanel starts hidden');

      send(win, {
        type: 'infoFile',
        workDir: '/cfg/dir',
        tabId: poll.tabId,
        token: tok,
        exists: true,
        sig: '100:20',
        content: '# Title\n\npara with **bold** text\n\n- item\n',
      });
      const html = content.innerHTML;
      assert.ok(html.includes('<h1'), 'markdown heading rendered: ' + html);
      assert.ok(html.includes('<strong>bold</strong>'), 'bold rendered');
      assert.ok(html.includes('<li>item'), 'list rendered');
      assert.ok(infoVisible(win), 'content shows the subpanel');

      // The next poll must carry the adopted signature.
      await sleep(2100);
      const later = wv.posted.filter(m => m.type === 'getInfoFile');
      assert.strictEqual(later[later.length - 1].knownSig, '100:20');

      // An unchanged reply (no content field) must not blank the panel.
      send(win, {
        type: 'infoFile',
        workDir: '/cfg/dir',
        token: tok,
        exists: true,
        sig: '100:20',
        unchanged: true,
      });
      assert.strictEqual(content.innerHTML, html);

      // A reply carrying a stale request token (an old poll answered
      // late) is ignored.
      send(win, {
        type: 'infoFile',
        workDir: '/cfg/dir',
        token: 'stale-token',
        exists: true,
        sig: '999:1',
        content: 'other project',
      });
      assert.strictEqual(content.innerHTML, html);

      // The file vanished: the subpanel empties AND hides entirely —
      // the tmp/info.md header must not linger over a blank body.
      send(win, {
        type: 'infoFile',
        workDir: '/cfg/dir',
        token: tok,
        exists: false,
        sig: '',
        content: '',
      });
      assert.strictEqual(content.innerHTML, '');
      assert.ok(!infoVisible(win), 'missing file hides the subpanel');

      // And it may reappear with new contents.
      send(win, {
        type: 'infoFile',
        workDir: '/cfg/dir',
        token: tok,
        exists: true,
        sig: '200:5',
        content: 'plain line',
      });
      assert.ok(content.textContent.includes('plain line'));
      assert.ok(infoVisible(win), 'new content shows the subpanel again');

      // A workdir switch (a task pinned elsewhere) clears the shown
      // contents SYNCHRONOUSLY and polls the new workdir immediately
      // with a reset signature — a late reply for the old workdir no
      // longer matches.
      const TAB = tabIdOf(wv);
      const before = wv.posted.filter(m => m.type === 'getInfoFile').length;
      send(win, {
        type: 'task_settings',
        settings: {work_dir: '/other/dir', task_id: 'task-9'},
        tabId: TAB,
        taskId: 'task-9',
      });
      assert.strictEqual(content.innerHTML, '', 'switch clears the panel');
      assert.ok(!infoVisible(win), 'switch hides the subpanel');
      const after = wv.posted.filter(m => m.type === 'getInfoFile');
      assert.strictEqual(after.length, before + 1, 'immediate poll fired');
      assert.strictEqual(after[after.length - 1].workDir, '/other/dir');
      assert.strictEqual(after[after.length - 1].knownSig, '');
      assert.notStrictEqual(
        after[after.length - 1].token,
        tok,
        'a workdir switch must start a new request generation',
      );
      send(win, {
        type: 'infoFile',
        workDir: '/cfg/dir',
        token: tok,
        exists: true,
        sig: '201:6',
        content: 'stale old-workdir reply',
      });
      assert.strictEqual(content.innerHTML, '', 'late old reply ignored');
    },
  );

  await test('a zero max budget is a real value, not unknown', () => {
    const wv = makeWebview();
    const win = wv.win;
    const TAB = tabIdOf(wv);
    send(win, {
      type: 'configData',
      config: {work_dir: '/cfg/dir', max_budget: 0},
      apiKeys: {},
    });
    assert.strictEqual(rowText(win, 'meta-max-budget'), '$0.00');
    send(win, {type: 'setTaskText', text: 'capped', tabId: TAB});
    send(win, {
      type: 'task_settings',
      settings: {work_dir: '/t', max_budget: 0, task_id: 'task-z'},
      tabId: TAB,
      taskId: 'task-z',
    });
    // A later positive config default must NOT replace the running
    // task's own zero cap.
    send(win, {
      type: 'configData',
      config: {work_dir: '/cfg/dir', max_budget: 42},
      apiKeys: {},
    });
    assert.strictEqual(rowText(win, 'meta-max-budget'), '$0.00');
  });

  await test(
    'markdown from info.md is sanitized before it hits the DOM',
    async () => {
      const wv = makeWebview();
      const win = wv.win;
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 1},
        apiKeys: {},
      });
      setRunning(win, true);
      await sleep(2400);
      const polls = wv.posted.filter(m => m.type === 'getInfoFile');
      assert.ok(polls.length > 0, 'poll must have fired');
      send(win, {
        type: 'infoFile',
        workDir: '/cfg/dir',
        token: polls[polls.length - 1].token,
        exists: true,
        sig: '5:5',
        content: 'hello <script>window.__pwned = 1;</script>\n',
      });
      const content = win.document.getElementById('meta-info-content');
      assert.ok(content.textContent.includes('hello'));
      assert.strictEqual(content.querySelector('script'), null);
      assert.strictEqual(win.__pwned, undefined);
    },
  );

  await test('a root work_dir falls back exactly like workDirForTab', () => {
    const wv = makeWebview();
    const win = wv.win;
    const TAB = tabIdOf(wv);
    send(win, {
      type: 'configData',
      config: {work_dir: '/cfg/dir', max_budget: 42},
      apiKeys: {},
    });
    send(win, {type: 'setTaskText', text: 'poisoned replay', tabId: TAB});
    send(win, {
      type: 'task_settings',
      settings: {work_dir: '/', max_budget: 3, task_id: 'task-r'},
      tabId: TAB,
      taskId: 'task-r',
    });
    assert.strictEqual(
      rowText(win, 'meta-workdir'),
      '/cfg/dir',
      'a filesystem root must never be adopted as the shown workdir',
    );
    assert.strictEqual(rowText(win, 'meta-max-budget'), '$3.00');
  });

  await test(
    'a configData repaint keeps the neighbour the panel is lent to',
    () => {
      const wv = makeWebview();
      const win = wv.win;
      const TAB = tabIdOf(wv);
      win._testApi.hideWelcome();
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
      });
      send(win, {
        type: 'task_events',
        tabId: TAB,
        chat_id: 'chat-n',
        task_id: 'task-2',
        task: 'live task',
        events: [
          {
            type: 'task_settings',
            settings: {
              work_dir: '/live/dir',
              max_budget: 5,
              task_id: 'task-2',
              chat_id: 'chat-n',
            },
          },
          {type: 'system_output', text: 'live\n'},
        ],
      });
      assert.strictEqual(rowText(win, 'meta-workdir'), '/live/dir');
      assert.strictEqual(rowText(win, 'meta-max-budget'), '$5.00');

      // Scroll onto a PREV neighbour: the panel is lent to it.
      send(win, {
        type: 'adjacent_task_events',
        tabId: TAB,
        direction: 'prev',
        task: 'older neighbour',
        task_id: 'task-1',
        events: [
          {
            type: 'task_settings',
            settings: {
              work_dir: '/old/dir',
              max_budget: 9,
              task_id: 'task-1',
              chat_id: 'chat-n',
            },
          },
          {type: 'system_output', text: 'older\n'},
        ],
      });
      assert.strictEqual(rowText(win, 'meta-workdir'), '/old/dir');
      assert.strictEqual(rowText(win, 'meta-max-budget'), '$9.00');

      // A settings reload must repaint the SAME lent-out task, not
      // jump back to the tab's own.
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
      });
      assert.strictEqual(rowText(win, 'meta-workdir'), '/old/dir');
      assert.strictEqual(rowText(win, 'meta-max-budget'), '$9.00');
    },
  );

  await test('no getInfoFile poll outside remote desktop mode', async () => {
    const wv = makeWebview({desktopMatches: false});
    const win = wv.win;
    assert.ok(!win.document.body.classList.contains('remote-desktop'));
    setRunning(win, true);
    await sleep(2400);
    assert.strictEqual(
      wv.posted.filter(m => m.type === 'getInfoFile').length,
      0,
      'a phone-sized remote webview must not poll even while running',
    );
  });

  await test('no getInfoFile poll while no task is running', async () => {
    const wv = makeWebview();
    const win = wv.win;
    send(win, {
      type: 'configData',
      config: {work_dir: '/cfg/dir', max_budget: 42},
      apiKeys: {},
    });
    await sleep(2400);
    assert.strictEqual(
      wv.posted.filter(m => m.type === 'getInfoFile').length,
      0,
      'an idle desktop webview must not poll',
    );
    assert.ok(!infoVisible(win), 'idle subpanel stays hidden');
  });

  await test(
    'task end empties and hides the subpanel and invalidates late ' +
      'replies; the next task repolls immediately',
    async () => {
      const wv = makeWebview();
      const win = wv.win;
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
      });
      setRunning(win, true);
      const polls = wv.posted.filter(m => m.type === 'getInfoFile');
      assert.ok(polls.length >= 1, 'a run start polls immediately');
      const tok = polls[polls.length - 1].token;
      send(win, {
        type: 'infoFile',
        workDir: '/cfg/dir',
        token: tok,
        exists: true,
        sig: '10:4',
        content: 'live notes',
      });
      const content = win.document.getElementById('meta-info-content');
      assert.ok(content.textContent.includes('live notes'));
      assert.ok(infoVisible(win));

      // The task ends: the subpanel empties and hides at once …
      setRunning(win, false);
      assert.strictEqual(content.innerHTML, '');
      assert.ok(!infoVisible(win), 'task end hides the subpanel');
      // … a poll answered after the end must not resurrect it …
      send(win, {
        type: 'infoFile',
        workDir: '/cfg/dir',
        token: tok,
        exists: true,
        sig: '11:9',
        content: 'post-mortem reply',
      });
      assert.strictEqual(content.innerHTML, '', 'late reply ignored');
      // … and the poll loop stays quiet while idle.
      const idleFrom = wv.posted.filter(m => m.type === 'getInfoFile').length;
      await sleep(2400);
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'getInfoFile').length,
        idleFrom,
        'no polls while idle',
      );

      // A new task starts: an immediate fresh-signature poll under a
      // NEW token, so only replies to it can paint the subpanel.
      setRunning(win, true);
      const fresh = wv.posted.filter(m => m.type === 'getInfoFile');
      assert.strictEqual(fresh.length, idleFrom + 1, 'restart repolls');
      assert.strictEqual(fresh[fresh.length - 1].knownSig, '');
      assert.notStrictEqual(fresh[fresh.length - 1].token, tok);
    },
  );

  console.log(`\n${passed} passed, ${failures.length} failed`);
  if (failures.length > 0) process.exit(1);
  process.exit(0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
