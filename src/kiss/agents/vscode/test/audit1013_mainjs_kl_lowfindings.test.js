// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Audit 2026-10-13 (reports K and L on media/main.js), the low findings:
//
// * the Schedule and Machines `+` forms had no in-flight guard: a
//   repeated Enter sent a second identical create/add, which the daemon
//   refused as a duplicate, and that refusal was painted over the first
//   request's success;
// * `scheduleAdd.saving` never cleared when the connection dropped
//   before the reply, and a save sent while disconnected waited for a
//   reply that could never come;
// * the Machines / Recommendations / Apps refresh buttons started their
//   spin on a click made while already disconnected (the request is
//   never answered on its own);
// * the folder picker stayed on "Loading..." when the connection
//   dropped before its listing arrived: nothing re-listed on reconnect;
// * a double-click on an Explorer file row sent two openFile requests
//   and the second reply tore down and rebuilt the just-created editor;
// * the remote-URL bar's copy button re-implemented wireCopyButton's
//   copy-then-flash-a-check-mark pattern with its own timer.
//
// Round 2 (the reviewer's findings on the first round):
//
// * the double-click fix compared version stamps only: a reply with
//   the same `<mtime_ns>:<size>` but different bytes, or for a tab
//   whose editor had failed (Monaco fallback) or been replaced by a
//   report view, was dropped instead of re-rendered;
// * the Schedule editor's in-flight guard lived in the DOM only: the
//   poll's rebuild re-enabled Save before the update was answered;
// * a `history` reply's relabel repaint replayed the cached `machines`
//   reply, `action: 'add'` included, which the retry of a refused add
//   took as its own acknowledgement.
'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build the webview; `opts.noMonaco` leaves window.monaco unset (the
 * CDN loader path runs) and the returned `monaco` can be installed
 * later by the test.
 */
function makeWebview(opts) {
  opts = opts || {};
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
  // The sidebar poll is recorded, never run.
  win.setInterval = function () {
    return 1;
  };
  win.clearInterval = function () {};
  win.matchMedia = function (query) {
    return {
      matches: query === '(min-width: 900px)',
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    };
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
  const copied = [];
  Object.defineProperty(win.navigator, 'clipboard', {
    value: {
      writeText: text => {
        copied.push(text);
        return Promise.resolve();
      },
    },
    configurable: true,
  });
  // A fake Monaco rich enough for an editable content tab; every
  // created editor is recorded.
  const created = [];
  const monaco = {
    editor: {
      defineTheme: () => {},
      setTheme: () => {},
      create: (holder, opts) => {
        const value = String(opts.value === undefined ? '' : opts.value);
        const model = {
          getAlternativeVersionId: () => 1,
          getValue: () => value,
          getLineCount: () => value.split('\n').length,
        };
        const revealed = [];
        const editor = {
          getModel: () => model,
          onDidChangeModelContent: () => {},
          setPosition: pos => revealed.push(pos.lineNumber),
          revealLineInCenter: () => {},
          getDomNode: () => holder,
          dispose: () => {},
          layout: () => {},
          focus: () => {},
        };
        created.push({holder, editor, value, revealed});
        return editor;
      },
    },
  };
  if (!opts.noMonaco) win.monaco = monaco;
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'treeContextMenu.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=audit1013-kl-main.js',
  );
  return {win, posted, copied, created, monaco};
}

/** The content tab views on screen or hidden (one per open file). */
function contentViews(win) {
  return win.document.querySelectorAll('#content-tab-area .content-tab-view');
}

/** Deliver a fileContent reply for a text file read from disk. */
function openText(win, path, content, version, line) {
  send(win, {
    type: 'fileContent',
    path,
    name: path.slice(path.lastIndexOf('/') + 1),
    content,
    version,
    line,
  });
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function byId(win, id) {
  return win.document.getElementById(id);
}

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function ofType(posted, type) {
  return posted.filter(m => m.type === type);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/** Press Enter in a form input: sidebarInlineForm submits the form. */
function pressEnter(win, input) {
  input.dispatchEvent(
    new win.KeyboardEvent('keydown', {
      key: 'Enter',
      bubbles: true,
      cancelable: true,
    }),
  );
}

function openForm(win, list) {
  return list.querySelector('form.sidebar-inline-form');
}

function saveButton(form) {
  return form.querySelector('.sidebar-inline-form-save');
}

let failures = 0;

async function test(name, fn) {
  try {
    await fn();
    console.log(`  ok - ${name}`);
  } catch (e) {
    failures++;
    console.log(`  FAIL - ${name}`);
    console.log(`      ${e.stack || e.message}`);
  }
}

/** Open the Schedule `+` form with a job typed in; returns the form. */
function typedScheduleForm(win) {
  send(win, {type: 'cronJobs', jobs: []});
  click(win, byId(win, 'meta-schedule-add'));
  const form = openForm(win, byId(win, 'meta-schedule-list'));
  form.elements.name.value = 'Digest';
  form.elements.schedule.value = 'every 30m';
  form.elements.what.value = 'summarize my email';
  return form;
}

/** Open the Machines `+` form with a machine typed in; returns the form. */
function typedMachineForm(win) {
  send(win, {type: 'machines', machines: [], checkedAt: 1});
  click(win, byId(win, 'meta-machines-add'));
  const form = openForm(win, byId(win, 'meta-machines-list'));
  form.elements.name.value = 'gpu';
  form.elements.address.value = 'me@gpu';
  return form;
}

async function main() {
  await test('Schedule +: a repeated Enter sends one create; the reply closes or re-arms the form', () => {
    const {win, posted} = makeWebview();
    const list = byId(win, 'meta-schedule-list');
    const form = typedScheduleForm(win);
    posted.length = 0;
    pressEnter(win, form.elements.name);
    pressEnter(win, form.elements.name);
    click(win, saveButton(form));
    const creates = ofType(posted, 'cronAction');
    assert.strictEqual(creates.length, 1, 'exactly one create goes out');
    assert.strictEqual(creates[0].action, 'create');
    assert.strictEqual(creates[0].name, 'Digest');
    assert.ok(
      saveButton(form).disabled,
      'Save is disabled while the create is in flight',
    );
    // The periodic poll re-renders the list: the form comes back with
    // its draft and still refuses a second submit.
    send(win, {type: 'cronJobs', jobs: []});
    const again = openForm(win, list);
    assert.ok(again && again !== form, 'the poll rebuilt the form');
    assert.strictEqual(again.elements.name.value, 'Digest');
    assert.ok(saveButton(again).disabled, 'still in flight after a re-render');
    pressEnter(win, again.elements.name);
    assert.strictEqual(
      ofType(posted, 'cronAction').length,
      1,
      'no second create',
    );
    // The daemon refuses: the form re-opens, typed values kept, Save
    // enabled, and a new submit sends again.
    send(win, {
      type: 'cronJobs',
      jobs: [],
      action: 'create',
      error: 'bad schedule',
    });
    const reopened = openForm(win, list);
    assert.ok(reopened, 'a refused create keeps the form open');
    assert.strictEqual(reopened.elements.schedule.value, 'every 30m');
    assert.ok(!saveButton(reopened).disabled, 'a refusal re-enables Save');
    assert.strictEqual(
      byId(win, 'meta-schedule-status').textContent,
      'bad schedule',
    );
    pressEnter(win, reopened.elements.schedule);
    assert.strictEqual(ofType(posted, 'cronAction').length, 2);
    // The daemon accepts: the form closes.
    send(win, {
      type: 'cronJobs',
      action: 'create',
      jobs: [
        {
          id: 'a',
          name: 'Digest',
          schedule: 'every 30m',
          what: 'x',
          enabled: true,
        },
      ],
    });
    assert.strictEqual(
      openForm(win, list),
      null,
      'an accepted create closes the form',
    );
    win.close();
  });

  await test('Machines +: a repeated Enter sends one add; the reply closes or re-arms the form', () => {
    const {win, posted} = makeWebview();
    const list = byId(win, 'meta-machines-list');
    const form = typedMachineForm(win);
    posted.length = 0;
    pressEnter(win, form.elements.address);
    pressEnter(win, form.elements.address);
    const adds = ofType(posted, 'machineAction');
    assert.strictEqual(adds.length, 1, 'exactly one add goes out');
    assert.strictEqual(adds[0].action, 'add');
    assert.strictEqual(adds[0].address, 'me@gpu');
    assert.ok(saveButton(form).disabled);
    // A plain machines reply (the poll) rebuilds the form, still armed.
    send(win, {type: 'machines', machines: [], checkedAt: 2});
    const again = openForm(win, list);
    assert.ok(again && again !== form);
    assert.strictEqual(again.elements.address.value, 'me@gpu');
    assert.ok(saveButton(again).disabled, 'still in flight after a re-render');
    pressEnter(win, again.elements.address);
    assert.strictEqual(
      ofType(posted, 'machineAction').length,
      1,
      'no second add',
    );
    send(win, {
      type: 'machines',
      machines: [],
      checkedAt: 3,
      action: 'add',
      error: 'unreachable',
    });
    const reopened = openForm(win, list);
    assert.ok(reopened, 'a refused add keeps the form open');
    assert.ok(!saveButton(reopened).disabled, 'a refusal re-enables Save');
    assert.strictEqual(
      byId(win, 'meta-machines-status').textContent,
      'unreachable',
    );
    pressEnter(win, reopened.elements.address);
    assert.strictEqual(ofType(posted, 'machineAction').length, 2);
    send(win, {
      type: 'machines',
      machines: [{name: 'gpu', address: 'me@gpu', status: 'reachable'}],
      checkedAt: 4,
      action: 'add',
    });
    assert.strictEqual(
      openForm(win, list),
      null,
      'an accepted add closes the form',
    );
    win.close();
  });

  await test('a dropped connection re-arms the + forms; a save while disconnected is refused with a note', () => {
    const {win, posted} = makeWebview();
    const scheduleList = byId(win, 'meta-schedule-list');
    const machinesList = byId(win, 'meta-machines-list');
    const sched = typedScheduleForm(win);
    const mach = typedMachineForm(win);
    posted.length = 0;
    pressEnter(win, sched.elements.name);
    pressEnter(win, mach.elements.name);
    assert.ok(saveButton(sched).disabled && saveButton(mach).disabled);
    send(win, {type: 'daemonStatus', connected: false});
    assert.ok(
      !saveButton(sched).disabled,
      'the Schedule form is released: no reply is coming',
    );
    assert.ok(!saveButton(mach).disabled, 'the Machines form is released too');
    // Disconnected: the save is refused on the status line, nothing
    // is sent, the typed values stay.
    pressEnter(win, sched.elements.name);
    pressEnter(win, mach.elements.name);
    assert.strictEqual(ofType(posted, 'cronAction').length, 1);
    assert.strictEqual(ofType(posted, 'machineAction').length, 1);
    assert.match(
      byId(win, 'meta-schedule-status').textContent,
      /^Not connected/,
    );
    assert.match(
      byId(win, 'meta-machines-status').textContent,
      /^Not connected/,
    );
    assert.ok(!saveButton(sched).disabled && !saveButton(mach).disabled);
    // Back: the reconnect's own re-render keeps the drafts, Save works.
    send(win, {type: 'daemonStatus', connected: true});
    send(win, {type: 'cronJobs', jobs: []});
    send(win, {type: 'machines', machines: [], checkedAt: 5});
    const sched2 = openForm(win, scheduleList);
    const mach2 = openForm(win, machinesList);
    assert.strictEqual(sched2.elements.name.value, 'Digest');
    assert.strictEqual(mach2.elements.name.value, 'gpu');
    assert.ok(!saveButton(sched2).disabled && !saveButton(mach2).disabled);
    pressEnter(win, sched2.elements.name);
    pressEnter(win, mach2.elements.name);
    assert.strictEqual(ofType(posted, 'cronAction').length, 2);
    assert.strictEqual(ofType(posted, 'machineAction').length, 2);
    win.close();
  });

  await test('the refresh buttons neither send nor spin while disconnected', () => {
    const {win, posted} = makeWebview();
    send(win, {type: 'cronJobs', jobs: []});
    send(win, {type: 'daemonStatus', connected: false});
    posted.length = 0;
    const buttons = [
      byId(win, 'meta-machines-refresh'),
      byId(win, 'meta-recommendations-refresh'),
      byId(win, 'meta-apps-refresh'),
    ];
    for (const btn of buttons) {
      click(win, btn);
      assert.ok(!btn.disabled && !btn.classList.contains('spinning'), btn.id);
    }
    assert.deepStrictEqual(posted, [], 'nothing goes out while disconnected');
    send(win, {type: 'daemonStatus', connected: true});
    posted.length = 0;
    for (const btn of buttons) {
      click(win, btn);
      assert.ok(btn.disabled && btn.classList.contains('spinning'), btn.id);
    }
    // The Machines button pings the machines; the Apps button asks
    // for every panel again (requestSidebarPanels) with the re-probe.
    assert.deepStrictEqual(
      ofType(posted, 'getMachines').map(m => m.refresh),
      [true, undefined],
    );
    assert.strictEqual(ofType(posted, 'getRecommendations').length, 2);
    assert.deepStrictEqual(
      ofType(posted, 'getAppsStatus').map(m => m.refresh),
      [true],
    );
    win.close();
  });

  await test('the folder picker lists its folder again on reconnect', () => {
    const {win, posted} = makeWebview();
    send(win, {
      type: 'configData',
      config: {work_dir: '/ws/repo'},
      apiKeys: {},
    });
    click(win, byId(win, 'explorer-add-folder'));
    const picker = byId(win, 'folder-picker');
    assert.ok(picker && !picker.hidden, 'the picker is open');
    const pickerListings = () =>
      ofType(posted, 'listDir').filter(m =>
        String(m.token).startsWith('picker:'),
      );
    const first = pickerListings();
    assert.strictEqual(first.length, 1);
    const list = picker.querySelector('.folder-picker-list');
    assert.match(list.textContent, /Loading/);
    // The connection drops before the reply, then comes back.
    send(win, {type: 'daemonStatus', connected: false});
    send(win, {type: 'daemonStatus', connected: true});
    const after = pickerListings();
    assert.strictEqual(after.length, 2, 'the reconnect re-lists the folder');
    assert.strictEqual(after[1].path, first[0].path);
    assert.notStrictEqual(after[1].token, first[0].token);
    // The dead request's reply (should it ever come) is ignored; the
    // fresh one fills the list.
    send(win, {
      type: 'dirListing',
      token: first[0].token,
      path: '/ws/repo',
      entries: [{name: 'stale', isDir: true}],
    });
    assert.match(list.textContent, /Loading/);
    send(win, {
      type: 'dirListing',
      token: after[1].token,
      path: '/ws/repo',
      entries: [{name: 'sub', isDir: true}],
    });
    assert.strictEqual(list.querySelectorAll('.folder-picker-item').length, 1);
    assert.strictEqual(
      list.querySelector('.folder-picker-item span').textContent,
      'sub',
    );
    // A closed picker asks for nothing on a later reconnect.
    posted.length = 0;
    picker.querySelector('.folder-picker-cancel').click();
    assert.ok(picker.hidden);
    send(win, {type: 'daemonStatus', connected: false});
    send(win, {type: 'daemonStatus', connected: true});
    assert.strictEqual(pickerListings().length, 0);
    win.close();
  });

  await test('a second fileContent reply with the same version stamp reveals the line instead of rebuilding the editor', async () => {
    const {win, created} = makeWebview();
    send(win, {
      type: 'configData',
      config: {work_dir: '/ws/repo'},
      apiKeys: {},
    });
    const open = (content, version, line) =>
      openText(win, '/ws/repo/notes.txt', content, version, line);
    open('one\ntwo\nthree', '1:13');
    await sleep(20);
    assert.strictEqual(created.length, 1, 'the first reply creates the editor');
    // A double-click's second reply: the same bytes on disk.
    open('one\ntwo\nthree', '1:13', 2);
    await sleep(20);
    assert.strictEqual(created.length, 1, 'the second reply keeps the editor');
    assert.ok(
      created[0].revealed.includes(2),
      'and jumps to the requested line',
    );
    assert.strictEqual(
      win._testApi.openTabs().filter(t => t.isContentTab).length,
      1,
    );
    // The file changed on disk since: a fresh copy replaces the view.
    open('one\ntwo\nthree\nfour', '2:18');
    await sleep(20);
    assert.strictEqual(created.length, 2, 'changed bytes re-render');
    assert.strictEqual(created[1].value, 'one\ntwo\nthree\nfour');
    win.close();
  });

  await test('a reply with the same version stamp but other bytes replaces the view', async () => {
    const {win, created} = makeWebview();
    send(win, {type: 'configData', config: {work_dir: '/ws'}, apiKeys: {}});
    // The stamp is mtime + size: a same-sized rewrite whose mtime was
    // restored (cp -p, a tool preserving metadata) keeps it.
    openText(win, '/ws/a.txt', 'before', '7:6');
    await sleep(20);
    assert.strictEqual(created.length, 1);
    openText(win, '/ws/a.txt', 'after!', '7:6', 1);
    await sleep(20);
    assert.strictEqual(created.length, 2, 'different bytes re-render');
    assert.strictEqual(created[1].value, 'after!');
    assert.strictEqual(contentViews(win).length, 1, 'still one tab');
    // The same bytes under a new stamp (the file was touched): the
    // editor stays and the tab adopts the stamp, so a save is not
    // refused as a conflict with the touch.
    openText(win, '/ws/a.txt', 'after!', '8:6', 1);
    await sleep(20);
    assert.strictEqual(created.length, 2, 'same bytes keep the editor');
    assert.ok(created[1].revealed.includes(1));
    win.close();
  });

  await test('reopening a file whose Monaco load failed retries the editor', async () => {
    const {win, created, monaco} = makeWebview({noMonaco: true});
    send(win, {type: 'configData', config: {work_dir: '/ws'}, apiKeys: {}});
    openText(win, '/ws/b.py', 'print(1)\n', '100:8');
    const loader = win.document.head.querySelector(
      'script[src*="vs/loader.js"]',
    );
    assert.ok(loader, 'the CDN loader is requested');
    loader.dispatchEvent(new win.Event('error'));
    await sleep(20);
    const view = contentViews(win)[0];
    assert.ok(
      view.querySelector('.content-code-fallback'),
      'the read-only fallback shows the text',
    );
    assert.strictEqual(created.length, 0);
    // Monaco is reachable again (the CDN came back); opening the same
    // unchanged file must build the editor this time.
    win.monaco = monaco;
    openText(win, '/ws/b.py', 'print(1)\n', '100:8');
    await sleep(20);
    assert.strictEqual(created.length, 1, 'the reopen creates the editor');
    assert.strictEqual(contentViews(win).length, 1);
    assert.ok(!contentViews(win)[0].querySelector('.content-code-fallback'));
    win.close();
  });

  await test('reopening a Markdown file shown as a report restores its editable preview', async () => {
    const {win} = makeWebview();
    send(win, {type: 'configData', config: {work_dir: '/ws/a'}, apiKeys: {}});
    const file = '/ws/a/reports/test.md';
    const text = '# Title\n\nbody';
    send(win, {type: 'tool_call', name: 'Write', path: file, content: text});
    send(win, {
      type: 'tool_result',
      content: 'Successfully wrote ' + text.length + ' characters to ' + file,
      is_error: false,
      tool_name: 'Write',
      path: file,
    });
    // The file is opened (editable: it carries a stamp) before the task
    // finishes; task_done then renders it as a report in the same tab.
    openText(win, file, text, '5:13');
    await sleep(20);
    assert.ok(contentViews(win)[0].querySelector('.content-mode-btn'));
    send(win, {type: 'task_done', success: true});
    await sleep(20);
    assert.strictEqual(contentViews(win).length, 1, 'the report reuses the tab');
    assert.ok(
      contentViews(win)[0].querySelector('iframe.content-html-frame'),
      'the report view replaced the preview',
    );
    assert.ok(!contentViews(win)[0].querySelector('.content-mode-btn'));
    // Opening the unchanged file again brings the editable preview back.
    openText(win, file, text, '5:13');
    await sleep(20);
    assert.strictEqual(contentViews(win).length, 1);
    assert.ok(
      contentViews(win)[0].querySelector('.content-mode-btn'),
      'the Edit source toggle is back',
    );
    win.close();
  });

  await test('Schedule editor: the poll keeps an update in flight; a refusal or a drop releases it', () => {
    const {win, posted} = makeWebview();
    const list = byId(win, 'meta-schedule-list');
    const jobs = [
      {id: 'a', name: 'Digest', schedule: 'every 5m', what: 'x', enabled: true},
    ];
    send(win, {type: 'cronJobs', jobs});
    const editor = () => list.children[0].querySelector('form');
    click(win, list.querySelector('[aria-label="Edit the job"]'));
    editor().elements.schedule.value = 'every 2h';
    posted.length = 0;
    pressEnter(win, editor().elements.schedule);
    assert.strictEqual(ofType(posted, 'cronAction').length, 1);
    assert.ok(saveButton(editor()).disabled);
    // The poll rebuilds the editor: the update is still unanswered.
    send(win, {type: 'cronJobs', jobs});
    assert.strictEqual(editor().elements.schedule.value, 'every 2h');
    assert.ok(saveButton(editor()).disabled, 'still in flight after the poll');
    pressEnter(win, editor().elements.schedule);
    assert.strictEqual(ofType(posted, 'cronAction').length, 1, 'no resend');
    // The refusal re-arms it.
    send(win, {type: 'cronJobs', jobs, action: 'update', error: 'no'});
    assert.ok(!saveButton(editor()).disabled);
    pressEnter(win, editor().elements.schedule);
    assert.strictEqual(ofType(posted, 'cronAction').length, 2);
    assert.ok(saveButton(editor()).disabled);
    // So does a dropped connection (no reply is coming).
    send(win, {type: 'daemonStatus', connected: false});
    assert.ok(!saveButton(editor()).disabled, 'released by the drop');
    send(win, {type: 'daemonStatus', connected: true});
    send(win, {type: 'cronJobs', jobs});
    assert.ok(!saveButton(editor()).disabled);
    pressEnter(win, editor().elements.schedule);
    assert.strictEqual(ofType(posted, 'cronAction').length, 3);
    send(win, {type: 'cronJobs', jobs, action: 'update'});
    assert.strictEqual(editor(), null, 'the acknowledgement closes it');
    win.close();
  });

  await test('Machines +: a relabel repaint does not replay a refusal as the retry ack', () => {
    const {win, posted} = makeWebview();
    const list = byId(win, 'meta-machines-list');
    const form = typedMachineForm(win);
    posted.length = 0;
    pressEnter(win, form.elements.address);
    send(win, {
      type: 'machines',
      machines: [],
      checkedAt: 2,
      action: 'add',
      error: 'unreachable',
    });
    pressEnter(win, openForm(win, list).elements.address);
    assert.strictEqual(ofType(posted, 'machineAction').length, 2);
    assert.ok(saveButton(openForm(win, list)).disabled);
    // A history reply relabels a machine: the Machines list repaints
    // from the cached reply, which must not count as a second answer.
    send(win, {
      type: 'history',
      sessions: [],
      offset: 0,
      generation: 1,
      machineLabels: {gpu: 'gpu.local'},
    });
    const again = openForm(win, list);
    assert.ok(again, 'the form survives the repaint');
    assert.strictEqual(again.elements.address.value, 'me@gpu');
    assert.ok(saveButton(again).disabled, 'the retry is still in flight');
    pressEnter(win, again.elements.address);
    assert.strictEqual(ofType(posted, 'machineAction').length, 2, 'no third add');
    send(win, {
      type: 'machines',
      machines: [{name: 'gpu', address: 'me@gpu', status: 'reachable'}],
      checkedAt: 3,
      action: 'add',
    });
    assert.strictEqual(openForm(win, list), null, 'the retry ack closes it');
    win.close();
  });

  await test('the remote-URL bar copy button is a wireCopyButton', async () => {
    const {win, copied} = makeWebview();
    send(win, {
      type: 'remote_url',
      url: 'https://words.trycloudflare.com',
      tunnelActive: true,
    });
    const btn = win.document.querySelector('.remote-url-copy');
    assert.ok(btn, 'the bar has its copy button');
    // The icons as the DOM serializes them (self-closing tags expand).
    const probe = win.document.createElement('span');
    probe.innerHTML = win.PanelCopy.PANEL_COPY_SVG;
    const copySvg = probe.innerHTML;
    probe.innerHTML = win.PanelCopy.PANEL_CHECK_SVG;
    const checkSvg = probe.innerHTML;
    assert.strictEqual(btn.innerHTML, copySvg);
    assert.strictEqual(btn.getAttribute('aria-label'), 'Copy URL to clipboard');
    click(win, btn);
    await sleep(0);
    assert.deepStrictEqual(copied, ['https://words.trycloudflare.com']);
    assert.strictEqual(btn.innerHTML, checkSvg);
    assert.ok(btn.classList.contains('copied'));
    await sleep(1600);
    assert.strictEqual(btn.innerHTML, copySvg);
    assert.ok(!btn.classList.contains('copied'));
    win.close();
  });

  console.log(failures ? `\n${failures} failed` : '\nall passed');
  process.exit(failures ? 1 : 0);
}

main();
