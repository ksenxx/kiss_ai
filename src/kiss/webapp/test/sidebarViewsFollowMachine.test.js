// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (JSDOM) tests, remote webapp: the Explorer and Source
// Control views follow the MACHINE the active chat runs its tasks on.
//
// * a chat bound to a registered machine browses that machine's
//   working directory (the daemon's machineWorkDirs reply): the tree
//   is rebuilt on the machine's folder and every request of the views
//   (listDir, gitStatus, gitLog, openFile, fsAction, gitShow, ...) is
//   stamped `machine`, so the daemon forwards it there; this daemon's
//   added folders (Add Folder to Explorer) are left out and "Add
//   Folder" is withdrawn;
// * the views are back on this daemon's folders, unstamped, once the
//   chat is brought home, and follow a pick of another directory on
//   the machine (the `set` reply) and the daemon's tabMachine binding;
// * a file opened from the machine's Explorer is saved and reloaded on
//   the machine, and the views keep browsing the machine while that
//   content tab is shown.

/* global require, __dirname, console, process */

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const openWindows = [];

function makePage() {
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
  openWindows.push(win);
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  win.cancelAnimationFrame = function () {};
  // The desktop layout docks the task-info panel, so the views are on
  // screen as soon as their section is expanded.
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
  let state;
  win.acquireTransport = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  const editors = [];
  win.monaco = {
    editor: {
      defineTheme: () => {},
      setTheme: () => {},
      create: (holder, opts) => {
        let value = String(opts.value === undefined ? '' : opts.value);
        let version = 1;
        const listeners = [];
        const model = {
          getAlternativeVersionId: () => version,
          getValue: () => value,
          getLineCount: () => value.split('\n').length,
        };
        const editor = {
          getModel: () => model,
          onDidChangeModelContent: cb => listeners.push(cb),
          setPosition: () => {},
          revealLineInCenter: () => {},
          getDomNode: () => holder,
          dispose: () => {},
          layout: () => {},
          focus: () => {},
          _type: text => {
            value = String(text);
            version += 1;
            listeners.slice().forEach(cb => cb());
          },
        };
        editors.push(editor);
        return editor;
      },
    },
  };
  // A folder added to this daemon's Explorer earlier (Add Folder to
  // Explorer...), remembered across reloads.
  win.localStorage.setItem('kiss-explorer-roots', JSON.stringify(['/extra']));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'treeContextMenu.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted, editors};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function byId(win, id) {
  const node = win.document.getElementById(id);
  assert.ok(node, `#${id} must exist`);
  return node;
}

function click(win, el) {
  assert.ok(el, 'the element to click exists');
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function msgs(posted, type) {
  return posted.filter(m => m && m.type === type);
}

function all(win, sel) {
  return Array.from(win.document.querySelectorAll(sel));
}

function rootRows(win) {
  return all(win, '#explorer-tree > .explorer-row.is-root');
}

function rootPaths(win) {
  return rootRows(win).map(r => r.dataset.explorerPath);
}

function showSection(win, id) {
  const section = byId(win, id);
  if (section.classList.contains('collapsed'))
    click(win, section.querySelector('.meta-section-toggle'));
}

function pickMachine(win, value) {
  const select = byId(win, 'chat-machine');
  select.value = value;
  select.dispatchEvent(new win.Event('change', {bubbles: true}));
}

/** Answer the last listDir for *p* (stamped *machine*, '' for none). */
function answerListing(win, posted, p, machine, entries) {
  const reqs = msgs(posted, 'listDir').filter(
    m => m.path === p && (m.machine || '') === machine,
  );
  const req = reqs[reqs.length - 1];
  assert.ok(req, `a listDir for ${p} on '${machine || 'this machine'}' was sent`);
  const reply = {type: 'dirListing', token: req.token, path: p, root: req.workDir, entries};
  if (machine) reply.machine = machine;
  send(win, reply);
  return req;
}

const NOW = Date.now() / 1000;
const MACHINES = [
  {name: 'laptop', local: true, status: 'reachable', address: 'https://10.0.0.5:8787'},
  {name: 'gpu-box', label: 'gpu.lab', status: 'reachable', address: 'https://gpu:8787'},
];
const BOX_DIRS = {
  type: 'machineWorkDirs',
  machine: 'gpu-box',
  workDir: '/srv/proj',
  recentWorkDirs: [{path: '/srv/proj', ts: NOW}, {path: '/srv/other', ts: NOW - 50}],
  error: '',
};
const FILE = {name: 'main.py', path: '/srv/proj/main.py', isDir: false};
const DIR = {name: 'pkg', path: '/srv/proj/pkg', isDir: true};

/**
 * A page on this daemon's /work/ws (with /extra added to the Explorer
 * earlier), both views expanded and loaded.
 */
function boot() {
  const made = makePage();
  const {win, posted} = made;
  send(win, {
    type: 'configData',
    config: {work_dir: '/work/ws', recent_work_dirs: [{path: '/work/ws', ts: NOW}]},
    machine: 'laptop',
  });
  send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  showSection(win, 'meta-explorer');
  showSection(win, 'meta-scm');
  answerListing(win, posted, '/work/ws', '', [{name: 'home.txt', path: '/work/ws/home.txt', isDir: false}]);
  assert.deepStrictEqual(rootPaths(win), ['/work/ws', '/extra']);
  assert.strictEqual(msgs(posted, 'gitStatus')[0].machine, undefined);
  posted.length = 0;
  return made;
}

/** Bind the active chat to gpu-box and let its directories arrive. */
function goToBox(win, posted) {
  pickMachine(win, 'gpu-box');
  send(win, BOX_DIRS);
}

function rightClick(win, el) {
  el.dispatchEvent(
    new win.MouseEvent('contextmenu', {bubbles: true, cancelable: true, clientX: 10, clientY: 10}),
  );
}

function menuItem(win, label) {
  return all(win, '#sidebar-context-menu .tree-ctx-item').find(
    el => el.querySelector('.tree-ctx-label').textContent === label,
  );
}

function rowFor(win, p) {
  return all(win, '.explorer-row[data-explorer-path]').find(r => r.dataset.explorerPath === p);
}

/** The Save button of the content tab on screen (each tab's view has its own). */
function saveButton(win) {
  const view = all(win, '.content-tab-view').find(v => v.style.display !== 'none');
  assert.ok(view, 'a content tab view is shown');
  return view.querySelector('.content-save-btn');
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function waitFor(predicate, message) {
  const start = Date.now();
  while (!predicate()) {
    if (Date.now() - start >= 2000) throw new Error(message);
    await sleep(10);
  }
}

async function testTheExplorerShowsTheMachinesFolder() {
  const {win, posted} = boot();
  pickMachine(win, 'gpu-box');
  // Until the machine's directories arrive there is nothing to show;
  // this daemon's tree (and its added folder) is gone.
  assert.deepStrictEqual(rootPaths(win), []);
  assert.ok(win.document.querySelector('#explorer-tree .sidebar-empty'), 'No workspace folder');
  assert.strictEqual(msgs(posted, 'listDir').length, 0);
  send(win, BOX_DIRS);
  assert.deepStrictEqual(rootPaths(win), ['/srv/proj'], "the machine's working directory alone");
  const req = answerListing(win, posted, '/srv/proj', 'gpu-box', [DIR, FILE]);
  assert.strictEqual(req.workDir, '/srv/proj', "confined to the machine's folder");
  assert.ok(rowFor(win, '/srv/proj/main.py'), 'the listing rendered');
  assert.ok(byId(win, 'explorer-add-folder').hidden, 'no "Add Folder" for a machine');
  // Expanding a folder and opening a file ask the machine.
  posted.length = 0;
  click(win, rowFor(win, '/srv/proj/pkg'));
  assert.strictEqual(msgs(posted, 'listDir')[0].machine, 'gpu-box');
  assert.strictEqual(msgs(posted, 'listDir')[0].path, '/srv/proj/pkg');
  click(win, rowFor(win, '/srv/proj/main.py'));
  const opens = msgs(posted, 'openFile');
  assert.strictEqual(opens.length, 1);
  assert.strictEqual(opens[0].machine, 'gpu-box');
  assert.strictEqual(opens[0].workDir, '/srv/proj');
  // The top-level folder's menu offers no "Add Folder"; "New File..."
  // on a folder makes the file on the machine.
  rightClick(win, rootRows(win)[0]);
  assert.ok(menuItem(win, 'Add Folder to Explorer...').classList.contains('disabled'));
  win.document.dispatchEvent(new win.KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
  rightClick(win, rowFor(win, '/srv/proj/pkg'));
  click(win, menuItem(win, 'New File...'));
  const input = win.document.querySelector('input.explorer-input');
  assert.ok(input, 'the name box');
  input.value = 'new.py';
  input.dispatchEvent(new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true, cancelable: true}));
  const fs = msgs(posted, 'fsAction');
  assert.strictEqual(fs.length, 1);
  assert.strictEqual(fs[0].machine, 'gpu-box');
  assert.strictEqual(fs[0].action, 'newFile');
  assert.strictEqual(fs[0].path, '/srv/proj/pkg');
  posted.length = 0;
  send(win, {
    type: 'fsResult', token: fs[0].token, action: 'newFile', path: '/srv/proj/pkg/new.py',
    machine: 'gpu-box',
  });
  const reopened = msgs(posted, 'openFile');
  assert.strictEqual(reopened.length, 1, 'the new file opens at once');
  assert.strictEqual(reopened[0].machine, 'gpu-box');
  assert.strictEqual(reopened[0].path, '/srv/proj/pkg/new.py');
  // Home again: this daemon's folders, unstamped, "Add Folder" back.
  posted.length = 0;
  pickMachine(win, '');
  assert.deepStrictEqual(rootPaths(win), ['/work/ws', '/extra']);
  const home = answerListing(win, posted, '/work/ws', '', []);
  assert.strictEqual(home.machine, undefined);
  assert.ok(!byId(win, 'explorer-add-folder').hidden);
}

function testSourceControlShowsTheMachinesRepository() {
  const {win, posted} = boot();
  goToBox(win, posted);
  const status = msgs(posted, 'gitStatus');
  const log = msgs(posted, 'gitLog');
  assert.strictEqual(status.length, 1);
  assert.strictEqual(log.length, 1);
  assert.strictEqual(status[0].machine, 'gpu-box');
  assert.strictEqual(status[0].workDir, '/srv/proj');
  assert.strictEqual(log[0].machine, 'gpu-box');
  assert.strictEqual(log[0].token, status[0].token);
  send(win, {
    type: 'gitStatus', token: status[0].token, machine: 'gpu-box', branch: 'box-main',
    repo: '/srv/proj', changes: [],
  });
  send(win, {type: 'gitLog', token: log[0].token, machine: 'gpu-box', commits: []});
  assert.strictEqual(byId(win, 'scm-branch').textContent, 'box-main');
  // The refresh button asks the machine again.
  posted.length = 0;
  click(win, byId(win, 'scm-refresh'));
  assert.strictEqual(msgs(posted, 'gitStatus')[0].machine, 'gpu-box');
  // A pick of another directory on the machine (the daemon's `set`
  // reply) moves both views there.
  posted.length = 0;
  send(win, {...BOX_DIRS, workDir: '/srv/other', set: true});
  assert.strictEqual(msgs(posted, 'gitStatus')[0].workDir, '/srv/other');
  assert.strictEqual(msgs(posted, 'gitStatus')[0].machine, 'gpu-box');
  assert.deepStrictEqual(rootPaths(win), ['/srv/other']);
  assert.strictEqual(msgs(posted, 'listDir')[0].machine, 'gpu-box');
  // Home: this daemon's repository, unstamped.
  posted.length = 0;
  pickMachine(win, '');
  assert.strictEqual(msgs(posted, 'gitStatus')[0].machine, undefined);
  assert.strictEqual(msgs(posted, 'gitStatus')[0].workDir, '/work/ws');
}

function testTheDaemonsBindingMovesTheViews() {
  const {win, posted} = boot();
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'tabMachine', tabId: tab, machine: 'gpu-box'});
  assert.strictEqual(msgs(posted, 'getMachineWorkDirs').length, 1);
  assert.deepStrictEqual(rootPaths(win), []);
  send(win, BOX_DIRS);
  assert.deepStrictEqual(rootPaths(win), ['/srv/proj']);
  assert.strictEqual(msgs(posted, 'listDir')[0].machine, 'gpu-box');
  assert.strictEqual(msgs(posted, 'gitStatus')[0].machine, 'gpu-box');
  // A machine that cannot be asked leaves the views empty, not on
  // this daemon's folders.
  posted.length = 0;
  send(win, {type: 'tabMachine', tabId: tab, machine: 'office'});
  send(win, {type: 'machineWorkDirs', machine: 'office', workDir: '', recentWorkDirs: [], error: 'office: down'});
  assert.deepStrictEqual(rootPaths(win), []);
  assert.strictEqual(msgs(posted, 'listDir').length, 0);
  assert.strictEqual(msgs(posted, 'gitStatus').length, 0);
  send(win, {type: 'tabMachine', tabId: tab, machine: ''});
  assert.deepStrictEqual(rootPaths(win), ['/work/ws', '/extra']);
}

async function testAFileOfTheMachineIsSavedAndReloadedThere() {
  const {win, posted, editors} = boot();
  goToBox(win, posted);
  answerListing(win, posted, '/srv/proj', 'gpu-box', [FILE]);
  const chat = win._testApi.getActiveTabId();
  send(win, {
    type: 'fileContent', tabId: chat, machine: 'gpu-box', path: '/srv/proj/main.py',
    name: 'main.py', content: 'print(1)\n', version: 'v1',
  });
  await waitFor(() => editors.length === 1, 'the file opened as a content tab');
  const content = win._testApi.openTabs().find(t => t.isContentTab);
  assert.ok(content, 'the content tab exists');
  // Shown as the active tab (the desktop layout's content pane shows
  // it alongside the chat): the views keep browsing the machine.
  win._testApi.switchToTab(content.id);
  posted.length = 0;
  click(win, byId(win, 'explorer-refresh'));
  assert.strictEqual(msgs(posted, 'listDir')[0].machine, 'gpu-box');
  assert.strictEqual(msgs(posted, 'listDir')[0].path, '/srv/proj');
  editors[0]._type('print(2)\n');
  posted.length = 0;
  click(win, win.document.querySelector('.content-save-btn'));
  const saves = msgs(posted, 'saveFile');
  assert.strictEqual(saves.length, 1);
  assert.strictEqual(saves[0].machine, 'gpu-box');
  assert.strictEqual(saves[0].workDir, '/srv/proj');
  assert.strictEqual(saves[0].path, '/srv/proj/main.py');
  // The machine refuses the save (the file changed there): "Reload
  // from disk" reads the file from the machine again.
  send(win, {
    type: 'fileSaved', ok: false, conflict: true, error: 'changed on disk',
    token: saves[0].token, tabId: chat, path: '/srv/proj/main.py', name: 'main.py',
    machine: 'gpu-box',
  });
  posted.length = 0;
  const toast = win.document.querySelector('[data-notification-id^="file-save-conflict-"]');
  assert.ok(toast, 'the conflict toast');
  click(
    win,
    Array.from(toast.querySelectorAll('.kiss-notification-action')).find(
      b => b.textContent.trim() === 'Reload from disk',
    ),
  );
  const reloads = msgs(posted, 'openFile');
  assert.strictEqual(reloads.length, 1);
  assert.strictEqual(reloads[0].machine, 'gpu-box');
  assert.strictEqual(reloads[0].path, '/srv/proj/main.py');
  assert.strictEqual(reloads[0].workDir, '/srv/proj');
  send(win, {
    type: 'fileContent', tabId: chat, machine: 'gpu-box', path: '/srv/proj/main.py',
    name: 'main.py', content: 'print(9)\n', version: 'v3',
  });
  await waitFor(() => editors.length === 2, 'the reload rebuilt the editor');
  assert.strictEqual(editors[1].getModel().getValue(), 'print(9)\n');
  // Even once the chat is brought home, the file stays the machine's.
  win._testApi.switchToTab(chat);
  pickMachine(win, '');
  win._testApi.switchToTab(content.id);
  posted.length = 0;
  editors[1]._type('print(3)\n');
  click(win, win.document.querySelector('.content-save-btn'));
  assert.strictEqual(msgs(posted, 'saveFile')[0].machine, 'gpu-box');
}

async function testTheSamePathOnTwoMachinesIsTwoFiles() {
  const {win, posted, editors} = boot();
  const chat = win._testApi.getActiveTabId();
  // This daemon's /work/ws/notes.txt, then gpu-box's: the reply's
  // machine stamp tells them apart, whatever the chat's machine is
  // when it lands (the pick was made before the machine's reply came).
  send(win, {
    type: 'fileContent', tabId: chat, path: '/work/ws/notes.txt', name: 'notes.txt',
    content: 'home\n', version: 'h1',
  });
  await waitFor(() => editors.length === 1, 'the home file opened');
  goToBox(win, posted);
  win._testApi.switchToTab(chat);
  pickMachine(win, '');
  send(win, {
    type: 'fileContent', tabId: chat, machine: 'gpu-box', path: '/work/ws/notes.txt',
    name: 'notes.txt', content: 'box\n', version: 'b1',
  });
  await waitFor(() => editors.length === 2, "the machine's file opened as its own tab");
  const contentTabs = win._testApi.openTabs().filter(t => t.isContentTab);
  assert.strictEqual(contentTabs.length, 2, 'two tabs for the two files');
  assert.strictEqual(editors[0].getModel().getValue(), 'home\n');
  assert.strictEqual(editors[1].getModel().getValue(), 'box\n');
  // Each saves where it came from.
  win._testApi.switchToTab(contentTabs[1].id);
  posted.length = 0;
  editors[1]._type('box 2\n');
  click(win, saveButton(win));
  assert.strictEqual(msgs(posted, 'saveFile')[0].machine, 'gpu-box');
  assert.strictEqual(msgs(posted, 'saveFile')[0].workDir, '/srv/proj');
  win._testApi.switchToTab(contentTabs[0].id);
  posted.length = 0;
  editors[0]._type('home 2\n');
  click(win, saveButton(win));
  assert.strictEqual(msgs(posted, 'saveFile')[0].machine, undefined);
  assert.strictEqual(msgs(posted, 'saveFile')[0].workDir, '/work/ws');
  // A reply without a stamp is this daemon's own file even for a chat
  // bound (and picked) to a machine: the daemon stamps every reply a
  // machine answers (sidebar_panels.machine_workspace_reply), and a
  // transcript link of an earlier local task opens unstamped.  Its
  // save goes here, not to the machine (audit 2026-10-16, mainjs-1 F1).
  win._testApi.switchToTab(chat);
  pickMachine(win, 'gpu-box');
  send(win, {
    type: 'fileContent', tabId: chat, path: '/srv/proj/out.txt', name: 'out.txt',
    content: 'out\n', version: 'o1',
  });
  await waitFor(() => editors.length === 3, 'the local file opened');
  const local = win._testApi.openTabs().filter(t => t.isContentTab)[2];
  win._testApi.switchToTab(local.id);
  posted.length = 0;
  editors[2]._type('out 2\n');
  click(win, saveButton(win));
  assert.strictEqual(msgs(posted, 'saveFile')[0].machine, undefined);
}

async function testAnActionKeepsTheMachineItWasPickedOn() {
  const {win, posted} = boot();
  goToBox(win, posted);
  answerListing(win, posted, '/srv/proj', 'gpu-box', [DIR, FILE]);
  // Delete asks first; the chat comes home while the question is up.
  rightClick(win, rowFor(win, '/srv/proj/main.py'));
  click(win, menuItem(win, 'Delete'));
  pickMachine(win, '');
  answerListing(win, posted, '/work/ws', '', [
    {name: 'home.txt', path: '/work/ws/home.txt', isDir: false},
  ]);
  posted.length = 0;
  const toast = win.document.querySelector('[data-notification-id="fs-delete"]');
  click(
    win,
    Array.from(toast.querySelectorAll('.kiss-notification-action')).find(
      b => b.textContent.trim() === 'Delete',
    ),
  );
  const deletes = msgs(posted, 'fsAction');
  assert.strictEqual(deletes.length, 1);
  assert.strictEqual(deletes[0].machine, 'gpu-box', 'deleted where it was picked');
  assert.strictEqual(deletes[0].path, '/srv/proj/main.py');
  // An overwrite retry keeps it too.
  send(win, {
    type: 'fsResult', token: deletes[0].token, action: 'delete', path: '/srv/proj/main.py',
    machine: 'gpu-box', error: 'exists', exists: true,
  });
  // Cut on this machine, pasted on the machine: refused.
  rightClick(win, rowFor(win, '/work/ws/home.txt'));
  click(win, menuItem(win, 'Cut'));
  pickMachine(win, 'gpu-box');
  answerListing(win, posted, '/srv/proj', 'gpu-box', [DIR, FILE]);
  posted.length = 0;
  rightClick(win, rowFor(win, '/srv/proj/pkg'));
  click(win, menuItem(win, 'Paste'));
  assert.strictEqual(msgs(posted, 'fsAction').length, 0, 'nothing moved');
  assert.ok(
    Array.from(win.document.querySelectorAll('.kiss-notification-error')).some(el =>
      el.textContent.includes('another machine'),
    ),
    'the refusal is reported',
  );
  // Cut and pasted on the machine: moved there.
  rightClick(win, rowFor(win, '/srv/proj/main.py'));
  click(win, menuItem(win, 'Cut'));
  posted.length = 0;
  rightClick(win, rowFor(win, '/srv/proj/pkg'));
  click(win, menuItem(win, 'Paste'));
  const moves = msgs(posted, 'fsAction');
  assert.strictEqual(moves.length, 1);
  assert.strictEqual(moves[0].action, 'move');
  assert.strictEqual(moves[0].machine, 'gpu-box');
  assert.strictEqual(moves[0].dest, '/srv/proj/pkg');
}

function testAHomeChatSendsNothingStamped() {
  const {win, posted} = boot();
  click(win, byId(win, 'explorer-refresh'));
  click(win, byId(win, 'scm-refresh'));
  const stamped = posted.filter(m => m && m.machine !== undefined);
  assert.deepStrictEqual(stamped, []);
  assert.strictEqual(msgs(posted, 'listDir')[0].workDir, '/work/ws');
}

const tests = [
  ["the Explorer shows the machine's folder", testTheExplorerShowsTheMachinesFolder],
  ["Source Control shows the machine's repository", testSourceControlShowsTheMachinesRepository],
  ["the daemon's binding moves the views", testTheDaemonsBindingMovesTheViews],
  ['a file of the machine is saved and reloaded there', testAFileOfTheMachineIsSavedAndReloadedThere],
  ['the same path on two machines is two files', testTheSamePathOnTwoMachinesIsTwoFiles],
  ['an action keeps the machine it was picked on', testAnActionKeepsTheMachineItWasPickedOn],
  ['a home chat sends nothing stamped', testAHomeChatSendsNothingStamped],
];

(async () => {
  let failed = 0;
  for (const [name, fn] of tests) {
    try {
      await fn();
      console.log('ok - ' + name);
    } catch (e) {
      failed += 1;
      console.log('not ok - ' + name);
      console.log(e && e.stack ? e.stack : String(e));
    } finally {
      while (openWindows.length) openWindows.pop().close();
    }
  }
  console.log(`\n${tests.length - failed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
})();
