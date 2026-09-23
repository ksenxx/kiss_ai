// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the docked task-info panel's rows and its
// "Task update" info subpanel (media/chat.html + media/main.js, remote
// desktop mode). The subpanel shows the task-update agent's report
// (src/kiss/agents/seas/task_update_sea.py) about the visible tab's
// RUNNING task, fetched from the daemon through the getTaskUpdate /
// taskUpdate protocol described in tmp/task-update-protocol.md:
//
// * #meta-workdir / #meta-max-budget rows fall back to configData
//   values (config.work_dir / config.max_budget) and adopt a live
//   task's task_settings (work_dir / max_budget) when they arrive,
// * the 5 s getTaskUpdate poll runs only in remote desktop mode and only
//   while the visible tab has a RUNNING task; each poll carries the
//   polled chat tab's id, the last known report signature, the current
//   generation token and refresh:false (no workDir — a workdir change
//   alone neither retargets nor repolls),
// * a taskUpdate reply is matched against the generation token;
//   unchanged replies are ignored; exists:false, the task's end and a
//   tab switch empty and hide the whole subpanel (`visible` on
//   #meta-info) and invalidate late replies,
// * the report body is sanitized: HTML content (starting with '<') is
//   rendered as-is through kissSanitize, anything else through marked;
//   scripts and event-handler attributes never reach the DOM,
// * running:true without content shows the "Preparing the first
//   update…" paragraph, status "Updating…" and a disabled, spinning
//   #meta-info-refresh button; a finished run shows "Updated HH:MM" plus
//   " · $X.XX" when the run cost anything, and an error appends "Last
//   run failed: …" to #meta-info-status,
// * clicking #meta-info-refresh posts a getTaskUpdate with refresh:true
//   under the current token.
//
// The poll interval is driven by a fake setInterval installed on the
// jsdom window, so the tests advance virtual time instead of sleeping.

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
 * Replace the window's setInterval / clearInterval with a virtual
 * clock so a test fires the 5 s poll deterministically. Returns the
 * clock: advance(ms) runs every due interval callback in time order.
 */
function installFakeIntervals(win) {
  const intervals = new Map();
  let nextId = 1;
  let now = 0;
  win.setInterval = function (fn, ms, ...args) {
    const id = nextId++;
    const period = Math.max(1, Number(ms) || 0);
    intervals.set(id, {fn, period, due: now + period, args});
    return id;
  };
  win.clearInterval = function (id) {
    intervals.delete(id);
  };
  function nextDue(limit) {
    let best = null;
    for (const [id, t] of intervals) {
      if (t.due <= limit && (best === null || t.due < best.t.due)) {
        best = {id, t};
      }
    }
    return best;
  }
  return {
    advance(ms) {
      const end = now + ms;
      for (let hit = nextDue(end); hit; hit = nextDue(end)) {
        now = hit.t.due;
        hit.t.due += hit.t.period;
        hit.t.fn(...hit.t.args);
      }
      now = end;
    },
  };
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
  const clock = installFakeIntervals(win);
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
  return {win, posted, clock};
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

/** Every getTaskUpdate message posted so far, oldest first. */
function polls(wv) {
  return wv.posted.filter(m => m.type === 'getTaskUpdate');
}

function lastPoll(wv) {
  const all = polls(wv);
  assert.ok(all.length >= 1, 'a getTaskUpdate poll must have fired');
  return all[all.length - 1];
}

function contentEl(win) {
  return win.document.getElementById('meta-info-content');
}

function statusText(win) {
  return win.document.getElementById('meta-info-status').textContent;
}

function refreshBtn(win) {
  return win.document.getElementById('meta-info-refresh');
}

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/** The 'Updated HH:MM' prefix main.js renders for an epoch-ms stamp. */
function updatedLabel(win, epochMs) {
  return (
    'Updated ' +
    new win.Date(epochMs).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    })
  );
}

/** Boot a desktop webview with a running task and return its first poll. */
function runningWebview() {
  const wv = makeWebview();
  send(wv.win, {
    type: 'configData',
    config: {work_dir: '/cfg/dir', max_budget: 42},
    apiKeys: {},
  });
  setRunning(wv.win, true);
  return wv;
}

/** A complete taskUpdate reply; *fields* override the defaults. */
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
        sig: 'sig-1',
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
    'info subpanel polls getTaskUpdate every 5 s for the visible tab ' +
      'and renders a markdown report through marked',
    () => {
      const wv = runningWebview();
      const win = wv.win;
      const TAB = tabIdOf(wv);
      assert.strictEqual(polls(wv).length, 1, 'a run start polls at once');
      wv.clock.advance(POLL_MS - 1);
      assert.strictEqual(polls(wv).length, 1, 'no poll before 5 s');
      wv.clock.advance(1);
      assert.strictEqual(polls(wv).length, 2, 'the 5 s interval polls');
      wv.clock.advance(POLL_MS);
      assert.strictEqual(polls(wv).length, 3, 'and keeps polling');
      assert.strictEqual(
        win.document.querySelector('#meta-info .sidebar-hdr span').textContent,
        'Task update',
        'the subpanel header must read "Task update"',
      );
      const poll = lastPoll(wv);
      assert.deepStrictEqual(Object.keys(poll).sort(), [
        'knownSig',
        'refresh',
        'tabId',
        'token',
        'type',
      ]);
      assert.strictEqual(poll.tabId, TAB);
      assert.strictEqual(poll.knownSig, '');
      assert.strictEqual(poll.refresh, false);
      assert.ok(!('workDir' in poll), 'polls carry no workDir');
      const tok = poll.token;
      assert.ok(tok, 'polls must carry a request token');
      assert.strictEqual(
        polls(wv)[0].token,
        tok,
        'every poll of one running tab shares one generation token',
      );

      const content = contentEl(win);
      assert.strictEqual(content.innerHTML, '');
      assert.ok(!infoVisible(win), 'subpanel starts hidden');

      reply(win, poll, {
        sig: '100:20',
        content: '# Title\n\npara with **bold** text\n\n- item\n',
        updatedAt: 1700000000000,
      });
      const html = content.innerHTML;
      assert.ok(html.includes('<h1'), 'markdown heading rendered: ' + html);
      assert.ok(html.includes('<strong>bold</strong>'), 'bold rendered');
      assert.ok(html.includes('<li>item'), 'list rendered');
      assert.ok(infoVisible(win), 'content shows the subpanel');
      assert.strictEqual(
        statusText(win),
        updatedLabel(win, 1700000000000),
        'a free run shows the time without a cost suffix',
      );

      // The next poll must carry the adopted signature.
      wv.clock.advance(POLL_MS);
      assert.strictEqual(lastPoll(wv).knownSig, '100:20');
      assert.strictEqual(lastPoll(wv).token, tok);

      // An unchanged reply (no content field) must not blank the panel.
      send(win, {
        type: 'taskUpdate',
        tabId: TAB,
        token: tok,
        exists: true,
        sig: '100:20',
        unchanged: true,
      });
      assert.strictEqual(content.innerHTML, html);

      // A reply carrying a stale request token (an old poll answered
      // late) is ignored.
      reply(win, poll, {
        token: 'stale-token',
        sig: '999:1',
        content: 'other task',
      });
      assert.strictEqual(content.innerHTML, html);

      // No report any more: the subpanel empties AND hides entirely —
      // the header must not linger over a blank body.
      reply(win, poll, {exists: false, sig: ''});
      assert.strictEqual(content.innerHTML, '');
      assert.strictEqual(statusText(win), '');
      assert.ok(!infoVisible(win), 'exists:false hides the subpanel');
      wv.clock.advance(POLL_MS);
      assert.strictEqual(lastPoll(wv).knownSig, '', 'signature dropped');

      // And it may reappear with new contents.
      reply(win, poll, {sig: '200:5', content: 'plain line'});
      assert.ok(content.textContent.includes('plain line'));
      assert.ok(infoVisible(win), 'new content shows the subpanel again');

      // A workdir change alone (task_settings for the same tab) is not
      // a retarget: the report stays, no poll fires, the token holds.
      const before = polls(wv).length;
      send(win, {
        type: 'task_settings',
        settings: {work_dir: '/other/dir', task_id: 'task-9'},
        tabId: TAB,
        taskId: 'task-9',
      });
      assert.strictEqual(rowText(win, 'meta-workdir'), '/other/dir');
      assert.ok(content.textContent.includes('plain line'), 'report kept');
      assert.ok(infoVisible(win), 'subpanel still shown');
      assert.strictEqual(polls(wv).length, before, 'no repoll on workdir');
      wv.clock.advance(POLL_MS);
      assert.strictEqual(lastPoll(wv).token, tok, 'same generation');
      assert.strictEqual(lastPoll(wv).knownSig, '200:5');
    },
  );

  await test(
    'a running agent without a report shows the pending paragraph, ' +
      '"Updating…" and a disabled spinning refresh button',
    () => {
      const wv = runningWebview();
      const win = wv.win;
      const poll = lastPoll(wv);
      const btn = refreshBtn(win);
      assert.strictEqual(btn.disabled, false, 'button starts enabled');
      assert.ok(!btn.classList.contains('spinning'));

      reply(win, poll, {sig: 'run-1', content: '', running: true});
      const pending = contentEl(win).querySelector('p.meta-info-pending');
      assert.ok(pending, 'pending paragraph rendered');
      assert.strictEqual(
        pending.textContent,
        'Preparing the first update\u2026',
      );
      assert.ok(infoVisible(win), 'a pending report shows the subpanel');
      assert.strictEqual(statusText(win), 'Updating\u2026');
      assert.strictEqual(btn.disabled, true, 'button disabled while running');
      assert.ok(btn.classList.contains('spinning'), 'button spins');

      // A rerun over an EXISTING report keeps the report on screen.
      reply(win, poll, {
        sig: 'run-2',
        content: '<p>first report</p>',
        running: false,
        updatedAt: 1700000000000,
      });
      assert.ok(contentEl(win).textContent.includes('first report'));
      assert.strictEqual(btn.disabled, false);
      assert.ok(!btn.classList.contains('spinning'));
      reply(win, poll, {
        sig: 'run-3',
        content: '<p>first report</p>',
        running: true,
        updatedAt: 1700000000000,
      });
      assert.ok(contentEl(win).textContent.includes('first report'));
      assert.strictEqual(
        contentEl(win).querySelector('p.meta-info-pending'),
        null,
        'no pending paragraph while a report exists',
      );
      assert.strictEqual(statusText(win), 'Updating\u2026');
      assert.strictEqual(btn.disabled, true);
      assert.ok(btn.classList.contains('spinning'));
    },
  );

  await test(
    'HTML content is rendered as HTML, sanitized, and the status shows ' +
      'the time and cost of the last run',
    () => {
      const wv = runningWebview();
      const win = wv.win;
      const poll = lastPoll(wv);
      const at = 1700000000000;
      reply(win, poll, {
        sig: 'html-1',
        content:
          '  <h2>Report</h2><p>done <b>50%</b></p>' +
          '<script>window.__pwned = 1;</script>' +
          '<img src="x" onerror="window.__pwned = 2">',
        cost: 0.12,
        updatedAt: at,
      });
      const content = contentEl(win);
      const h2 = content.querySelector('h2');
      assert.ok(h2, 'HTML heading rendered as an element');
      assert.strictEqual(h2.textContent, 'Report');
      assert.ok(content.querySelector('b'), 'inline HTML kept');
      assert.ok(
        !content.innerHTML.includes('&lt;h2'),
        'leading whitespace must not demote the HTML to markdown text',
      );
      assert.strictEqual(content.querySelector('script'), null);
      assert.strictEqual(content.querySelector('[onerror]'), null);
      assert.ok(!content.innerHTML.includes('onerror'));
      assert.strictEqual(win.__pwned, undefined);
      assert.ok(infoVisible(win));
      assert.strictEqual(
        statusText(win),
        updatedLabel(win, at) + ' \u00b7 $0.12',
      );
      const btn = refreshBtn(win);
      assert.strictEqual(btn.disabled, false, 'button enabled after a run');
      assert.ok(!btn.classList.contains('spinning'));

      // A markdown report is sanitized too.
      reply(win, poll, {
        sig: 'md-1',
        content: 'hello <script>window.__pwned = 3;</script>\n',
        updatedAt: at,
      });
      assert.ok(content.textContent.includes('hello'));
      assert.strictEqual(content.querySelector('script'), null);
      assert.strictEqual(win.__pwned, undefined);
    },
  );

  await test(
    'a failed run shows "Last run failed: …" in the status line',
    () => {
      const wv = runningWebview();
      const win = wv.win;
      const poll = lastPoll(wv);
      const content = contentEl(win);

      // No report at all yet: the failure is the whole status line and
      // the body says so.
      reply(win, poll, {sig: 'err-1', error: 'budget exceeded'});
      assert.strictEqual(statusText(win), 'Last run failed: budget exceeded');
      assert.strictEqual(
        content.querySelector('p.meta-info-pending').textContent,
        'No update yet.',
      );
      assert.ok(infoVisible(win), 'an error alone shows the subpanel');
      assert.strictEqual(refreshBtn(win).disabled, false);

      // A failure after an earlier successful run: separator after the
      // time and cost, the earlier report stays on screen.
      const at = 1700000000000;
      reply(win, poll, {
        sig: 'err-2',
        content: '<p>earlier report</p>',
        error: 'model timed out',
        cost: 1.5,
        updatedAt: at,
      });
      assert.ok(content.textContent.includes('earlier report'));
      assert.strictEqual(
        statusText(win),
        updatedLabel(win, at) +
          ' \u00b7 $1.50 \u00b7 Last run failed: model timed out',
      );

      // Failure while a rerun is in flight: "Updating…" comes first.
      reply(win, poll, {
        sig: 'err-3',
        content: '<p>earlier report</p>',
        error: 'model timed out',
        running: true,
        updatedAt: at,
      });
      assert.strictEqual(
        statusText(win),
        'Updating\u2026 \u00b7 Last run failed: model timed out',
      );
      assert.strictEqual(refreshBtn(win).disabled, true);
    },
  );

  await test(
    'the refresh button posts getTaskUpdate with refresh:true under the ' +
      'current token; interval polls carry refresh:false',
    () => {
      const wv = runningWebview();
      const win = wv.win;
      const first = lastPoll(wv);
      assert.strictEqual(first.refresh, false, 'the start poll is normal');
      reply(win, first, {sig: 'r-1', content: '<p>report</p>'});

      const before = polls(wv).length;
      click(win, refreshBtn(win));
      assert.strictEqual(polls(wv).length, before + 1, 'click posts a poll');
      const manual = lastPoll(wv);
      assert.strictEqual(manual.refresh, true);
      assert.strictEqual(manual.token, first.token);
      assert.strictEqual(manual.tabId, first.tabId);
      assert.strictEqual(manual.knownSig, 'r-1');
      assert.ok(!('workDir' in manual));

      wv.clock.advance(POLL_MS);
      assert.strictEqual(polls(wv).length, before + 2);
      assert.strictEqual(lastPoll(wv).refresh, false, 'interval polls stay normal');

      // With no running task the button is inert: no poll is posted.
      setRunning(win, false);
      const idle = polls(wv).length;
      click(win, refreshBtn(win));
      assert.strictEqual(polls(wv).length, idle, 'idle click posts nothing');
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

  await test('no getTaskUpdate poll outside remote desktop mode', () => {
    const wv = makeWebview({desktopMatches: false});
    const win = wv.win;
    assert.ok(!win.document.body.classList.contains('remote-desktop'));
    setRunning(win, true);
    wv.clock.advance(POLL_MS * 3);
    assert.strictEqual(
      polls(wv).length,
      0,
      'a phone-sized remote webview must not poll even while running',
    );
  });

  await test('no getTaskUpdate poll while no task is running', () => {
    const wv = makeWebview();
    const win = wv.win;
    send(win, {
      type: 'configData',
      config: {work_dir: '/cfg/dir', max_budget: 42},
      apiKeys: {},
    });
    wv.clock.advance(POLL_MS * 3);
    assert.strictEqual(
      polls(wv).length,
      0,
      'an idle desktop webview must not poll',
    );
    assert.ok(!infoVisible(win), 'idle subpanel stays hidden');
  });

  await test(
    'task end empties and hides the subpanel and invalidates late ' +
      'replies; the next task repolls immediately',
    () => {
      const wv = runningWebview();
      const win = wv.win;
      const poll = lastPoll(wv);
      const tok = poll.token;
      reply(win, poll, {
        sig: '10:4',
        content: '<p>live notes</p>',
        running: true,
      });
      const content = contentEl(win);
      assert.ok(content.textContent.includes('live notes'));
      assert.ok(infoVisible(win));
      assert.strictEqual(refreshBtn(win).disabled, true);

      // The task ends: the subpanel empties and hides at once, the
      // status clears and the button is usable again …
      setRunning(win, false);
      assert.strictEqual(content.innerHTML, '');
      assert.ok(!infoVisible(win), 'task end hides the subpanel');
      assert.strictEqual(statusText(win), '');
      assert.strictEqual(refreshBtn(win).disabled, false);
      assert.ok(!refreshBtn(win).classList.contains('spinning'));
      // … a poll answered after the end must not resurrect it …
      reply(win, poll, {sig: '11:9', content: '<p>post-mortem reply</p>'});
      assert.strictEqual(content.innerHTML, '', 'late reply ignored');
      // … and the poll loop stays quiet while idle.
      const idleFrom = polls(wv).length;
      wv.clock.advance(POLL_MS * 3);
      assert.strictEqual(polls(wv).length, idleFrom, 'no polls while idle');

      // A new task starts: an immediate fresh-signature poll under a
      // NEW token, so only replies to it can paint the subpanel.
      setRunning(win, true);
      assert.strictEqual(polls(wv).length, idleFrom + 1, 'restart repolls');
      assert.strictEqual(lastPoll(wv).knownSig, '');
      assert.notStrictEqual(lastPoll(wv).token, tok);
    },
  );

  await test(
    'switching between two running tabs clears the subpanel, repolls ' +
      "for the newly visible tab and drops the other tab's late reply",
    () => {
      const wv = runningWebview();
      const win = wv.win;
      const TAB1 = tabIdOf(wv);
      const poll1 = lastPoll(wv);
      const tok1 = poll1.token;
      assert.strictEqual(poll1.tabId, TAB1);
      const content = contentEl(win);
      reply(win, poll1, {sig: 'one:1', content: '<p>tab one notes</p>'});
      assert.ok(content.textContent.includes('tab one notes'));

      // Open a second tab (it becomes visible) and run a task in it.
      click(win, win.document.querySelector('#new-chat-btn'));
      const newChat = wv.posted.filter(m => m.type === 'newChat').pop();
      assert.ok(newChat && newChat.tabId, 'new tab must be announced');
      const TAB2 = newChat.tabId;
      assert.notStrictEqual(TAB2, TAB1);
      assert.strictEqual(content.innerHTML, '', 'idle new tab shows nothing');
      send(win, {type: 'status', running: true, tabId: TAB2});
      const poll2 = lastPoll(wv);
      assert.strictEqual(poll2.tabId, TAB2, 'poll names the visible tab');
      assert.notStrictEqual(poll2.token, tok1);
      assert.strictEqual(poll2.knownSig, '');
      reply(win, poll2, {sig: 'two:2', content: '<p>tab two notes</p>'});
      assert.ok(content.textContent.includes('tab two notes'));

      // Back to the first tab while BOTH tabs are running: the same
      // running state, yet the subpanel must not keep showing tab
      // two's report for even one poll interval.
      const before = polls(wv).length;
      click(
        win,
        win.document.querySelector(
          `.chat-tab[data-tab-id=${JSON.stringify(TAB1)}]`,
        ),
      );
      assert.strictEqual(content.innerHTML, '', 'tab switch clears the panel');
      assert.ok(!infoVisible(win), 'tab switch hides the subpanel');
      assert.strictEqual(polls(wv).length, before + 1, 'immediate poll fired');
      const poll3 = lastPoll(wv);
      assert.strictEqual(poll3.tabId, TAB1);
      assert.strictEqual(poll3.knownSig, '', 'signature reset on switch');
      assert.notStrictEqual(poll3.token, poll2.token, 'new generation');
      // A late reply for tab two must not paint tab one's subpanel.
      reply(win, poll2, {sig: 'two:3', content: '<p>tab two, later</p>'});
      assert.strictEqual(content.innerHTML, '', 'late other-tab reply ignored');
      reply(win, poll3, {sig: 'one:4', content: '<p>tab one again</p>'});
      assert.ok(content.textContent.includes('tab one again'));
    },
  );

  await test(
    'a content tab (file view) keeps the subpanel on the chat tab it ' +
      "was lent to instead of polling under the content tab's id",
    () => {
      const wv = runningWebview();
      const win = wv.win;
      const TAB1 = tabIdOf(wv);
      const poll1 = lastPoll(wv);
      const tok1 = poll1.token;
      const content = contentEl(win);
      reply(win, poll1, {sig: 'chat:1', content: '<p>chat tab notes</p>'});
      assert.ok(content.textContent.includes('chat tab notes'));

      // Open a file view: it becomes the visible tab, but has no task.
      send(win, {
        type: 'fileContent',
        name: 'report.html',
        path: '/cfg/dir/reports/report.html',
        content: '<h1>report</h1>',
      });
      const contentTab = win.document.querySelector(
        `.chat-tab.active[data-tab-id]`,
      );
      assert.ok(contentTab, 'the content tab is the visible tab');
      assert.notStrictEqual(contentTab.getAttribute('data-tab-id'), TAB1);
      assert.ok(
        content.textContent.includes('chat tab notes'),
        'opening a file view keeps the report on screen',
      );
      const before = polls(wv).length;
      wv.clock.advance(POLL_MS);
      assert.ok(polls(wv).length > before, 'the poll keeps running');
      const poll = lastPoll(wv);
      assert.strictEqual(poll.tabId, TAB1, 'polls name the chat tab');
      assert.strictEqual(poll.token, tok1, 'no new generation');
      assert.strictEqual(poll.knownSig, 'chat:1');
      reply(win, poll, {sig: 'chat:2', content: '<p>chat tab, updated</p>'});
      assert.ok(content.textContent.includes('chat tab, updated'));

      // Back to the chat tab: same target, nothing is cleared.
      click(
        win,
        win.document.querySelector(
          `.chat-tab[data-tab-id=${JSON.stringify(TAB1)}]`,
        ),
      );
      assert.ok(content.textContent.includes('chat tab, updated'));
      assert.strictEqual(lastPoll(wv).token, tok1);
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
