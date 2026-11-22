// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests of Projects outside the Chats list
// (media/projects.js, media/main.js "Projects", media/chat.html):
//
// * the sidebar's Projects row, under the refresh button, opens the
//   Projects page as a tab of the chat window;
// * "New project" shows a settings panel pre-filled with the product's
//   settings: the name first, then the description, the default working
//   directory with "New folder", the max budget, the check boxes, the
//   keys, then Cancel and Save, and a pin button in the corner;
// * Save sends what the user set, and what the project overrides
//   already; the daemon's answer closes the page, as Cancel does, and
//   takes the content window off the screen; the saved project is
//   listed under New project with its own settings over the product's;
// * a save that is out when the connection drops, or that an older
//   daemon does not know, is over, and the form of a new project takes
//   edits again;
// * a project's panel: its pin, its unsaved edits, Cancel, a refused
//   save, Delete;
// * an older daemon's refusal of the page's own getProjects is no error
//   to show;
// * "New folder" and "Choose ..." pick a folder through the folder
//   dialog, which can create one;
// * the chat header's project drop-down puts a chat in a project, and a
//   chat that enters a project starts in the project's default working
//   directory, which a pick in the header then overrides.
//
// historyProjects.test.js covers the project panels of the Chats list.

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  click,
  byId,
  lastMsg,
  ofType,
  typeInto,
  sleep,
  pickMachine,
  stubDesktop,
} = require('./simplify2_harness');
const {makeRunner, all, key} = require('./ui_antipattern_harness');
const {currentDir, pick: pickWorkDir} = require('./workdir_helpers');
const {MACHINES} = require('./settings_helpers');

const windows = [];
const {test, report} = makeRunner({windows});

const NOW = Date.now() / 1000;
// The product's settings, as the daemon's configData carries them.
const CONFIG = {
  work_dir: '/work/ws',
  recent_work_dirs: [
    {path: '/work/ws', ts: NOW},
    {path: '/work/other', ts: NOW - 60},
  ],
  max_budget: 25,
  use_web_browser: true,
  use_memory: false,
  compaction: true,
  chrome_bridge: false,
  claude_code_use_subscription: true,
  codex_use_subscription: false,
};
const KEYS = {ANTHROPIC_API_KEY: 'sk-product', GEMINI_API_KEY: ''};
// What a panel says when its save was out as the connection dropped.
const UNCONFIRMED =
  'The server connection dropped before this save was confirmed. ' +
  'Once it is back, save again if the project is not listed as you saved it.';

/** A project as the daemon's `projects` event lists it. */
function project(id, name, extra) {
  return Object.assign(
    {
      id,
      name,
      description: '',
      settings: {},
      isPinned: false,
      createdAt: NOW - 1000,
      updatedAt: NOW - 1000,
    },
    extra,
  );
}

// Names the page scripts for the coverage gate (projects.coverage.js).
const SOURCE_URLS = {'main.js': 'projects-main.js', 'projects.js': 'projects-page.js'};

/**
 * A page that has the product's settings; `scripts` adds page scripts,
 * `desktop` gives it the split layout (the content pane beside the chat).
 */
function openPage(scripts = ['projects.js'], desktop = false) {
  const page = makePage({
    bodyClass: 'remote-chat',
    extraScripts: scripts,
    sourceUrl: SOURCE_URLS,
    beforeScripts: desktop ? stubDesktop : undefined,
  });
  windows.push(page.win);
  send(page.win, {type: 'configData', config: CONFIG, apiKeys: KEYS});
  return page;
}

/** The open tabs that are the Projects page. */
function projectsTabs(win) {
  return win._testApi
    .openTabs()
    .filter(t => t.isContentTab && t.title === 'Projects');
}

/** Click the sidebar's Projects row; the page's host element. */
function openProjects(win) {
  click(win, byId(win, 'projects-row'));
  return win.document.querySelector('.projects-tab-view');
}

/** The New project panel of the page in `host`. */
function newPanel(host) {
  return host.querySelector('.projects-new-slot > .project-panel');
}

/** The panel of project `id` in the list of the page in `host`. */
function panelOf(host, id) {
  return host.querySelector(
    '.projects-list > .project-panel[data-project-id="' + id + '"]',
  );
}

/** Press the button `.cls` of `panel`. */
function press(win, panel, cls) {
  click(win, panel.querySelector('.' + cls));
}

/** Type `value` into the field `selector` of `panel`. */
function fill(win, panel, selector, value) {
  typeInto(win, panel.querySelector(selector), value);
}

/** Pick the option valued `value` in `select`, as the user does. */
function pickOption(win, select, value) {
  select.value = value;
  select.dispatchEvent(new win.Event('change', {bubbles: true}));
}

/** The labels of `select`'s options. */
function optionLabels(select) {
  return Array.from(select.options, o => o.textContent);
}

/** Tick or untick the check box of setting `key` in `panel`. */
function setBox(win, panel, key, on) {
  const box = panel.querySelector('input[data-setting="' + key + '"]');
  box.checked = on;
  box.dispatchEvent(new win.Event('change', {bubbles: true}));
}

/** The state of every check box of `panel`, by its setting. */
function boxes(panel) {
  const state = {};
  for (const box of panel.querySelectorAll('input[data-setting]')) {
    state[box.dataset.setting] = box.checked;
  }
  return state;
}

/** Plain copy of a value made in the page's realm. */
function plain(value) {
  return JSON.parse(JSON.stringify(value));
}

test('the Projects row, under the refresh button, opens the Projects page in a tab; New project shows a panel pre-filled with the product\u2019s settings', () => {
  const {win} = openPage();
  const chatTabId = win._testApi.getActiveTabId();
  const row = byId(win, 'projects-row');
  assert.strictEqual(row.textContent, 'Projects');
  assert.ok(row.querySelector('svg'), 'the row carries its icon');
  assert.ok(
    row.previousElementSibling.contains(byId(win, 'chats-refresh-btn')),
    'the refresh button sits in the row over it',
  );
  assert.ok(
    row.nextElementSibling.classList.contains('history-search-row'),
    'the search box follows: there is no Chats row',
  );
  assert.strictEqual(projectsTabs(win).length, 0);

  const host = openProjects(win);
  assert.strictEqual(projectsTabs(win).length, 1);
  assert.strictEqual(win._testApi.getActiveTabId(), projectsTabs(win)[0].id);
  assert.notStrictEqual(host.style.display, 'none', 'the page is on screen');
  assert.ok(
    win.document.querySelector('.chat-tab.content-tab .content-tab-icon svg'),
    'the tab wears the Projects icon',
  );
  openProjects(win);
  assert.strictEqual(projectsTabs(win).length, 1, 'one Projects tab a window');

  const panel = newPanel(host);
  assert.strictEqual(
    host.querySelector('.projects-new-btn').textContent,
    'New project',
  );
  assert.ok(panel.hidden, 'no panel until New project is pressed');
  assert.ok(!host.querySelector('.projects-empty').hidden);
  click(win, host.querySelector('.projects-new-btn'));
  assert.ok(!panel.hidden);
  assert.strictEqual(
    win.document.activeElement,
    panel.querySelector('.project-name'),
    'the keyboard is in the name box',
  );

  // The rows, in order: the pin (set in the corner by main.css), the
  // name, the description, the folder with "New folder", the budget,
  // the check boxes, the keys, then Cancel and Save.
  const rows = Array.from(panel.children);
  assert.ok(rows[0].classList.contains('project-pin-btn'));
  assert.strictEqual(rows[0].getAttribute('aria-pressed'), 'false');
  assert.ok(rows[1].querySelector('input.project-name'));
  assert.ok(rows[2].querySelector('textarea.project-description'));
  assert.ok(rows[3].textContent.startsWith('Default working directory'));
  assert.deepStrictEqual(
    Array.from(rows[3].querySelectorAll('select, button'), el => el.className),
    ['project-workdir', 'config-update-btn project-new-folder'],
  );
  assert.strictEqual(
    rows[3].querySelector('.project-new-folder').textContent,
    'New folder',
  );
  assert.ok(rows[4].textContent.startsWith('Max budget'));
  const settingsBoxes = all(win, '#settings-panel input[type="checkbox"]');
  assert.strictEqual(settingsBoxes.length, 7, 'the Settings sheet has seven');
  assert.deepStrictEqual(
    rows.slice(5, 12).map(r => r.textContent.trim()),
    settingsBoxes.map(box => box.closest('label').textContent.trim()),
    'the check boxes of the Settings sheet, in its order',
  );
  const keyRows = rows.filter(r => r.classList.contains('config-subpanel'));
  assert.ok(rows.indexOf(keyRows[0]) > 12, 'the keys follow the check boxes');
  assert.deepStrictEqual(
    Array.from(panel.querySelectorAll('input[data-key]'), i => i.dataset.key),
    all(win, '#settings-panel input[id^="cfg-key-"]').map(i =>
      i.id.slice('cfg-key-'.length),
    ),
    'every key field of the Settings sheet',
  );
  const actions = rows[rows.length - 1];
  assert.ok(actions.classList.contains('project-actions'));
  assert.deepStrictEqual(
    Array.from(actions.querySelectorAll('button:not([hidden])'), b =>
      b.textContent,
    ),
    ['Cancel', 'Save'],
  );

  // Pre-filled with the product's settings.
  assert.strictEqual(panel.querySelector('.project-name').value, '');
  assert.strictEqual(panel.querySelector('.project-max-budget').value, '25');
  assert.deepStrictEqual(boxes(panel), {
    use_web_browser: true,
    use_memory: false,
    compaction: true,
    chrome_bridge: false,
    victory_sound: true,
    claude_code_use_subscription: true,
    codex_use_subscription: false,
  });
  const folder = panel.querySelector('.project-workdir');
  assert.deepStrictEqual(optionLabels(folder), [
    'Same as KISS Sorcar',
    '/work/ws',
    '/work/other',
    'Choose ...',
  ]);
  assert.strictEqual(folder.selectedIndex, 0);
  const keyBox = panel.querySelector('input[data-key="ANTHROPIC_API_KEY"]');
  assert.strictEqual(keyBox.value, 'sk-product');

  // A key box is masked and has the eye toggle of the Settings sheet's,
  // named as there; the region is plain text and has none.
  for (const input of panel.querySelectorAll('input[data-key]')) {
    const sheetId = 'cfg-key-' + input.dataset.key;
    const eye = input.parentNode.querySelector('.config-password-toggle');
    const sheetEye = win.document.getElementById(sheetId + '-toggle');
    assert.strictEqual(input.type, byId(win, sheetId).type, sheetId);
    assert.strictEqual(
      eye && eye.getAttribute('aria-label'),
      sheetEye && sheetEye.getAttribute('aria-label'),
      sheetId,
    );
  }
  // Showing a key is no edit: the panel keeps following the product.
  const eye = keyBox.nextElementSibling;
  click(win, eye);
  assert.strictEqual(keyBox.type, 'text');
  assert.strictEqual(eye.getAttribute('aria-label'), 'Hide API key');
  send(win, {
    type: 'configData',
    config: CONFIG,
    apiKeys: {ANTHROPIC_API_KEY: 'sk-product-2'},
  });
  assert.strictEqual(keyBox.value, 'sk-product-2');
  click(win, eye);
  assert.strictEqual(keyBox.type, 'password');
  send(win, {type: 'configData', config: CONFIG, apiKeys: KEYS});

  // Cancel closes the page (its tab goes, the chat is back on screen)
  // and drops what was typed.
  fill(win, panel, '.project-name', 'Draft');
  press(win, panel, 'project-cancel');
  assert.strictEqual(projectsTabs(win).length, 0, 'the Projects tab closes');
  assert.strictEqual(win._testApi.getActiveTabId(), chatTabId);
  const fresh = newPanel(openProjects(win));
  assert.ok(fresh.hidden);
  assert.strictEqual(fresh.querySelector('.project-name').value, '');
});

test('Save sends what the user set with the pin and the daemon\u2019s answer closes the page; the saved project is listed under New project with its settings over the product\u2019s', () => {
  const {win, posted} = openPage();
  const chatTabId = win._testApi.getActiveTabId();
  const host = openProjects(win);
  click(win, host.querySelector('.projects-new-btn'));
  const panel = newPanel(host);

  // A project needs a name: nothing is sent without one.
  press(win, panel, 'project-save');
  assert.strictEqual(lastMsg(posted, 'saveProject'), null);
  const error = panel.querySelector('.project-error');
  assert.strictEqual(error.textContent, 'A project needs a name.');
  assert.ok(!error.hidden);

  fill(win, panel, '.project-name', '  Paper ');
  fill(win, panel, '.project-description', ' The draft ');
  pickOption(win, panel.querySelector('.project-workdir'), '/work/other');
  fill(win, panel, '.project-max-budget', '7');
  setBox(win, panel, 'use_web_browser', false);
  setBox(win, panel, 'victory_sound', false);
  fill(win, panel, 'input[data-key="GEMINI_API_KEY"]', ' g-proj ');
  press(win, panel, 'project-pin-btn');
  assert.strictEqual(
    panel.querySelector('.project-pin-btn').getAttribute('aria-pressed'),
    'true',
  );
  assert.strictEqual(lastMsg(posted, 'setProjectPinned'), null, 'not stored yet');
  // The product's settings change while the form is being filled: it
  // keeps what the user typed.
  send(win, {
    type: 'configData',
    config: Object.assign({}, CONFIG, {max_budget: 99}),
    apiKeys: KEYS,
  });
  assert.strictEqual(panel.querySelector('.project-max-budget').value, '7');

  press(win, panel, 'project-save');
  assert.ok(error.hidden);
  // What the user set is sent.  The five check boxes and eleven keys
  // left as they are, are not: they follow the product's settings.
  assert.deepStrictEqual(plain(lastMsg(posted, 'saveProject')), {
    type: 'saveProject',
    project: {
      name: 'Paper',
      description: 'The draft',
      settings: {
        default_work_dir: '/work/other',
        max_budget: 7,
        use_web_browser: false,
        // Unlike this browser's own choice (on), so the project keeps it.
        victory_sound: false,
        api_keys: {GEMINI_API_KEY: 'g-proj'},
      },
      isPinned: true,
    },
    requestId: 'project-save-1',
  });

  // A broadcast that comes before the daemon's answer leaves the form
  // as it was sent.
  send(win, {type: 'projects', projects: []});
  assert.strictEqual(panel.querySelector('.project-max-budget').value, '7');
  assert.strictEqual(projectsTabs(win).length, 1, 'open until the answer');
  // The daemon's answer closes the page: its tab goes and the chat is
  // back on screen.  Then comes its broadcast of the new list.
  send(win, {type: 'projectSaved', id: 'p1', requestId: 'project-save-1'});
  assert.strictEqual(projectsTabs(win).length, 0, 'the Projects tab closes');
  assert.strictEqual(win._testApi.getActiveTabId(), chatTabId);
  send(win, {
    type: 'projects',
    projects: [
      project('p1', 'Paper', {
        description: 'The draft',
        isPinned: true,
        settings: {
          default_work_dir: '/work/other',
          max_budget: 7,
          use_web_browser: false,
          victory_sound: false,
          api_keys: {GEMINI_API_KEY: 'g-proj'},
        },
      }),
    ],
  });
  // The page, opened again, lists the project.
  const reopened = openProjects(win);
  assert.ok(reopened.querySelector('.projects-empty').hidden);
  const saved = panelOf(reopened, 'p1');
  assert.ok(
    reopened.querySelector('.projects-new-slot').compareDocumentPosition(saved) &
      win.Node.DOCUMENT_POSITION_FOLLOWING,
    'the projects are listed after New project',
  );
  assert.strictEqual(saved.querySelector('.project-name').value, 'Paper');
  assert.strictEqual(
    saved.querySelector('.project-description').value,
    'The draft',
  );
  assert.strictEqual(saved.querySelector('.project-workdir').value, '/work/other');
  assert.strictEqual(saved.querySelector('.project-max-budget').value, '7');
  assert.deepStrictEqual(boxes(saved), {
    use_web_browser: false,
    use_memory: false,
    compaction: true,
    chrome_bridge: false,
    victory_sound: false,
    claude_code_use_subscription: true,
    codex_use_subscription: false,
  });
  assert.deepStrictEqual(
    ['GEMINI_API_KEY', 'ANTHROPIC_API_KEY'].map(
      k => saved.querySelector('input[data-key="' + k + '"]').value,
    ),
    ['g-proj', 'sk-product'],
    'its own key, and the product\u2019s for the one it leaves alone',
  );
  assert.strictEqual(
    saved.querySelector('.project-pin-btn').getAttribute('aria-pressed'),
    'true',
  );
  assert.ok(!saved.querySelector('.project-delete').hidden);
  // The new panel opens again pre-filled with the product's settings
  // (the newer ones), not with the last project's.
  const second = newPanel(reopened);
  click(win, reopened.querySelector('.projects-new-btn'));
  assert.strictEqual(second.querySelector('.project-name').value, '');
  assert.strictEqual(second.querySelector('.project-max-budget').value, '99');
  const newPin = second.querySelector('.project-pin-btn');
  assert.strictEqual(newPin.getAttribute('aria-pressed'), 'false');

  // A pin and nothing else is a draft too: a broadcast leaves it, and
  // Cancel drops it with the page.
  press(win, second, 'project-pin-btn');
  send(win, {type: 'configData', config: CONFIG, apiKeys: KEYS});
  assert.strictEqual(newPin.getAttribute('aria-pressed'), 'true');
  press(win, second, 'project-cancel');
  assert.strictEqual(projectsTabs(win).length, 0);
  const thirdHost = openProjects(win);
  const third = newPanel(thirdHost);
  assert.strictEqual(
    third.querySelector('.project-pin-btn').getAttribute('aria-pressed'),
    'false',
  );

  // Cancel does not wait for a save that is on its way, and lets go of
  // it: its answer, coming late, neither closes nor empties the draft
  // begun since.
  click(win, thirdHost.querySelector('.projects-new-btn'));
  fill(win, third, '.project-name', 'Thesis');
  press(win, third, 'project-save');
  const letGo = lastMsg(posted, 'saveProject').requestId;
  press(win, third, 'project-cancel');
  assert.strictEqual(projectsTabs(win).length, 0);
  const last = openProjects(win);
  click(win, last.querySelector('.projects-new-btn'));
  fill(win, newPanel(last), '.project-name', 'Site');
  send(win, {type: 'projectSaved', id: 'p2', requestId: letGo});
  assert.strictEqual(projectsTabs(win).length, 1, 'the page stays open');
  assert.ok(!newPanel(last).hidden);
  assert.strictEqual(newPanel(last).querySelector('.project-name').value, 'Site');
});

test('in the split layout, a new project saved or cancelled closes the Projects tab and takes the content window off the screen', () => {
  const {win, posted} = openPage(['projects.js'], true);
  const body = win.document.body;
  assert.ok(body.classList.contains('remote-desktop'));

  // The page alone in the content pane: the pane goes with its tab.
  let host = openProjects(win);
  assert.ok(body.classList.contains('content-pane-open'));
  click(win, host.querySelector('.projects-new-btn'));
  press(win, newPanel(host), 'project-cancel');
  assert.strictEqual(projectsTabs(win).length, 0);
  assert.ok(!body.classList.contains('content-pane-open'));
  assert.ok(!body.classList.contains('content-pane-folded'), 'nothing to fold');

  // Beside a file: the file stays open in the pane, which is folded
  // away once the daemon has answered the save.
  send(win, {
    type: 'fileContent',
    tabId: win._testApi.getActiveTabId(),
    path: '/work/ws/notes.md',
    name: 'notes.md',
    content: '# Notes',
    version: 'v1',
  });
  host = openProjects(win);
  click(win, host.querySelector('.projects-new-btn'));
  fill(win, newPanel(host), '.project-name', 'Paper');
  press(win, newPanel(host), 'project-save');
  assert.ok(body.classList.contains('content-pane-open'), 'until the answer');
  send(win, {
    type: 'projectSaved',
    id: 'p1',
    requestId: lastMsg(posted, 'saveProject').requestId,
  });
  assert.strictEqual(projectsTabs(win).length, 0);
  assert.ok(body.classList.contains('content-pane-folded'));
  assert.ok(!body.classList.contains('content-pane-open'));
  assert.deepStrictEqual(
    Array.from(win._testApi.openTabs().filter(t => t.isContentTab), t => t.title),
    ['notes.md'],
    'the file is still open',
  );
  // The pane's own button brings the file back.
  click(win, byId(win, 'content-pane-show'));
  assert.ok(body.classList.contains('content-pane-open'));
});

test('a project\u2019s panel: the pin is stored at once, unsaved edits outlive a broadcast, Cancel drops them, a refused save says why and so does one the connection dropped under, Delete asks first', () => {
  const {win, posted} = openPage();
  const first = project('p1', 'Paper', {settings: {max_budget: 7}});
  const second = project('p2', 'Site');
  send(win, {type: 'projects', projects: [first, second]});
  const host = openProjects(win);
  assert.deepStrictEqual(
    all(win, '.projects-list > .project-panel').map(p => p.dataset.projectId),
    ['p1', 'p2'],
  );
  const panel = panelOf(host, 'p1');
  const pin = panel.querySelector('.project-pin-btn');
  // The pin is named as Claude Code Desktop names its own, and holds
  // the pushpin's two drawings: main.css shows the outline while the
  // project is not pinned and the solid one while it is.
  assert.deepStrictEqual(
    [pin.title, pin.getAttribute('aria-label')],
    ['Pin project', 'Pin project'],
  );
  assert.deepStrictEqual(
    Array.from(pin.querySelectorAll('svg > path'), p => p.getAttribute('class')),
    ['project-pin-outline', 'project-pin-solid'],
  );

  press(win, panel, 'project-pin-btn');
  assert.deepStrictEqual(plain(lastMsg(posted, 'setProjectPinned')), {
    type: 'setProjectPinned',
    projectId: 'p1',
    pinned: true,
  });
  assert.strictEqual(pin.getAttribute('aria-pressed'), 'false', 'the daemon\u2019s word');

  // An edit, then a broadcast (the pin's, and another page's rename):
  // the edit stays, the pin follows.
  fill(win, panel, '.project-description', 'typed here');
  const pinned = Object.assign({}, first, {isPinned: true, name: 'Paper v2'});
  send(win, {type: 'projects', projects: [second, pinned]});
  assert.strictEqual(pin.getAttribute('aria-pressed'), 'true');
  assert.deepStrictEqual(
    [pin.title, pin.getAttribute('aria-label')],
    ['Unpin project', 'Unpin project'],
  );
  assert.strictEqual(panel.querySelector('.project-description').value, 'typed here');
  assert.strictEqual(panel.querySelector('.project-name').value, 'Paper');
  assert.deepStrictEqual(
    all(win, '.projects-list > .project-panel').map(p => p.dataset.projectId),
    ['p2', 'p1'],
    'the panels follow the daemon\u2019s order',
  );
  press(win, panel, 'project-pin-btn');
  assert.strictEqual(lastMsg(posted, 'setProjectPinned').pinned, false);

  press(win, panel, 'project-cancel');
  assert.ok(!panel.hidden, 'a project\u2019s panel stays');
  assert.strictEqual(panel.querySelector('.project-description').value, '');
  assert.strictEqual(panel.querySelector('.project-name').value, 'Paper v2');

  // A save the daemon refuses: its text is shown, the edit kept.
  fill(win, panel, '.project-name', 'Paper v3');
  press(win, panel, 'project-save');
  const save = lastMsg(posted, 'saveProject');
  assert.strictEqual(save.project.id, 'p1');
  assert.ok(!('isPinned' in save.project), 'the pin is not part of an update');
  assert.strictEqual(save.project.settings.max_budget, 7);
  const error = panel.querySelector('.project-error');
  for (const type of ['projectError', 'projectSaved']) {
    send(win, {type, requestId: 'not-a-request', text: 'someone else\u2019s'});
  }
  assert.ok(error.hidden, 'a reply to another request changes nothing');
  send(win, {
    type: 'projectError',
    requestId: save.requestId,
    projectId: 'p1',
    text: 'This project no longer exists.',
  });
  assert.strictEqual(error.textContent, 'This project no longer exists.');
  // What was not stored is an unsaved edit still: a broadcast keeps it.
  send(win, {type: 'projects', projects: [second, pinned]});
  assert.strictEqual(panel.querySelector('.project-name').value, 'Paper v3');

  // A save that is out when the connection drops is never answered:
  // its panel says so and keeps the edit, through the list the page is
  // sent once it is connected again.  A panel with no save out says
  // nothing.
  press(win, panel, 'project-save');
  assert.ok(error.hidden);
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  assert.strictEqual(error.textContent, UNCONFIRMED);
  assert.ok(panelOf(host, 'p2').querySelector('.project-error').hidden);
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'projects', projects: [second, pinned]});
  assert.strictEqual(panel.querySelector('.project-name').value, 'Paper v3');
  assert.ok(!error.hidden);

  // A save it takes.  A broadcast that comes before its answer (another
  // page's change) leaves the form as it was sent; the one that
  // follows the answer repaints the panel from what is stored.
  fill(win, panel, '.project-name', ' Paper v3 ');
  press(win, panel, 'project-save');
  assert.ok(error.hidden);
  send(win, {type: 'projects', projects: [second, pinned]});
  assert.strictEqual(panel.querySelector('.project-name').value, ' Paper v3 ');
  send(win, {
    type: 'projectSaved',
    id: 'p1',
    requestId: lastMsg(posted, 'saveProject').requestId,
  });
  const renamed = Object.assign({}, pinned, {name: 'Paper v3', isPinned: false});
  send(win, {type: 'projects', projects: [second, renamed]});
  assert.strictEqual(panel.querySelector('.project-name').value, 'Paper v3');
  assert.strictEqual(pin.getAttribute('aria-pressed'), 'false');

  // What is typed while a save is on its way is an edit of its own:
  // the save's answer and broadcast leave it in the form.
  fill(win, panel, '.project-name', 'Paper v4');
  press(win, panel, 'project-save');
  fill(win, panel, '.project-description', 'typed after Save');
  send(win, {
    type: 'projectSaved',
    id: 'p1',
    requestId: lastMsg(posted, 'saveProject').requestId,
  });
  const again = Object.assign({}, renamed, {name: 'Paper v4'});
  send(win, {type: 'projects', projects: [second, again]});
  assert.strictEqual(
    panel.querySelector('.project-description').value,
    'typed after Save',
  );
  press(win, panel, 'project-cancel');
  assert.strictEqual(panel.querySelector('.project-description').value, '');
  assert.strictEqual(panel.querySelector('.project-name').value, 'Paper v4');

  // A key group unfolds and folds.
  const toggle = panel.querySelector('.config-subpanel-toggle');
  click(win, toggle.querySelector('span'));
  assert.ok(!toggle.nextElementSibling.hidden);
  assert.strictEqual(toggle.getAttribute('aria-expanded'), 'true');
  click(win, toggle);
  assert.ok(toggle.nextElementSibling.hidden);

  // Delete asks first.
  const confirm = panel.querySelector('.project-confirm');
  press(win, panel, 'project-delete');
  assert.ok(!confirm.hidden);
  press(win, panel, 'project-delete-no');
  assert.ok(confirm.hidden);
  assert.strictEqual(lastMsg(posted, 'deleteProject'), null);
  press(win, panel, 'project-delete');
  press(win, panel, 'project-delete-yes');
  assert.deepStrictEqual(plain(lastMsg(posted, 'deleteProject')), {
    type: 'deleteProject',
    projectId: 'p1',
  });
  send(win, {type: 'projects', projects: [second]});
  assert.strictEqual(panelOf(host, 'p1'), null);
  assert.ok(panelOf(host, 'p2'));
  // A click that is on no button does nothing.
  click(win, host.querySelector('.projects-title'));
  click(win, panelOf(host, 'p2').querySelector('.project-name'));

  // Closing the tab takes the page down with it: a later broadcast
  // finds no page, and the row opens a fresh one.
  click(win, win.document.querySelector('.chat-tab.content-tab .chat-tab-close'));
  assert.strictEqual(projectsTabs(win).length, 0);
  send(win, {type: 'projects', projects: [second]});
  send(win, {type: 'projectSaved', id: 'p2', requestId: 'project-save-9'});
  send(win, {type: 'projectError', requestId: 'project-save-9', text: 'late'});
  const reopened = openProjects(win);
  assert.ok(panelOf(reopened, 'p2'));
  // The fresh page numbers its saves on from the closed one's, so an
  // answer still on its way to that one is not taken for its own.
  click(win, reopened.querySelector('.projects-new-btn'));
  const draft = newPanel(reopened);
  fill(win, draft, '.project-name', 'Later');
  press(win, draft, 'project-save');
  send(win, {type: 'projectSaved', id: 'p1', requestId: save.requestId});
  assert.strictEqual(projectsTabs(win).length, 1, 'the page stays open');
  assert.ok(!draft.hidden);
});

test('a new project\u2019s form is locked while its create is on its way: a second Save sends nothing and no field takes an edit, Cancel stays; a refusal opens it for another try, and so do an older daemon\u2019s \u201cunknown command\u201d and a dropped connection', () => {
  const {win, posted} = openPage();
  const host = openProjects(win);
  click(win, host.querySelector('.projects-new-btn'));
  const panel = newPanel(host);
  const controls = Array.from(
    panel.querySelectorAll('input, textarea, select, button'),
  );
  /** The class of every control of the form the browser lets the user work. */
  const open = () => controls.filter(c => !c.disabled).map(c => c.className);
  fill(win, panel, '.project-name', 'Paper');
  assert.strictEqual(open().length, controls.length);

  // A double click on Save: one create, not two projects.
  press(win, panel, 'project-save');
  press(win, panel, 'project-save');
  assert.strictEqual(ofType(posted, 'saveProject').length, 1);
  // The answer closes the page with what the form shows, so nothing is
  // let into it meanwhile: every control but Cancel is disabled, and a
  // click that reaches a disabled button all the same runs nothing.
  assert.deepStrictEqual(open(), ['config-update-btn project-cancel']);
  const pin = panel.querySelector('.project-pin-btn');
  press(win, panel, 'project-pin-btn');
  assert.strictEqual(pin.getAttribute('aria-pressed'), 'false');

  // The daemon refuses the create: the form is open again, and Save
  // sends it once more.
  send(win, {
    type: 'projectError',
    requestId: lastMsg(posted, 'saveProject').requestId,
    text: 'The disk is full.',
  });
  assert.strictEqual(
    panel.querySelector('.project-error').textContent,
    'The disk is full.',
  );
  assert.strictEqual(open().length, controls.length);
  fill(win, panel, '.project-description', 'Second try');
  press(win, panel, 'project-save');
  const retry = lastMsg(posted, 'saveProject');
  assert.deepStrictEqual(
    [ofType(posted, 'saveProject').length, retry.project.description, 'id' in retry.project],
    [2, 'Second try', false],
  );
  assert.deepStrictEqual(open(), ['config-update-btn project-cancel']);

  // A daemon that predates projects (one updated on disk and not yet
  // restarted serves this page) does not know the command.  Its answer
  // names no panel: the one that awaits an answer gets it, and the form
  // is open again.
  send(win, {type: 'error', text: 'Unknown command: saveProject'});
  assert.strictEqual(
    panel.querySelector('.project-error').textContent,
    'Unknown command: saveProject',
  );
  assert.strictEqual(open().length, controls.length);
  press(win, panel, 'project-save');
  assert.strictEqual(ofType(posted, 'saveProject').length, 3);
  assert.deepStrictEqual(open(), ['config-update-btn project-cancel']);
  // Any other error leaves the create on its way.
  send(win, {type: 'error', text: 'The update failed.'});
  assert.deepStrictEqual(open(), ['config-update-btn project-cancel']);

  // The connection drops before the daemon's answer, which comes on no
  // other: the form says so and is open again, as it was sent.  The
  // daemon made the project all the same: the list the page is sent
  // once it is connected again has it, under the form.
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  assert.strictEqual(panel.querySelector('.project-error').textContent, UNCONFIRMED);
  assert.strictEqual(open().length, controls.length);
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {
    type: 'projects',
    projects: [project('p1', 'Paper', {description: 'Second try'})],
  });
  assert.ok(panelOf(host, 'p1'));
  assert.ok(!panel.querySelector('.project-error').hidden);
  assert.strictEqual(panel.querySelector('.project-description').value, 'Second try');
  // The form makes another project when the user says so, and the
  // answer to that one create closes the page.
  fill(win, panel, '.project-name', 'Paper 2');
  press(win, panel, 'project-save');
  const last = lastMsg(posted, 'saveProject');
  assert.deepStrictEqual(
    [ofType(posted, 'saveProject').length, last.project.name],
    [4, 'Paper 2'],
  );
  send(win, {type: 'projectSaved', id: 'p2', requestId: last.requestId});
  assert.strictEqual(projectsTabs(win).length, 0);
});

test('a setting left as it is follows the product\u2019s settings while the panel has unsaved edits, and Save does not send it; one the user set is kept and sent', () => {
  // The page is opened, and a name typed, before the daemon has sent
  // its settings: the form is blank until they come.
  const {win, posted} = makePage({
    bodyClass: 'remote-chat',
    extraScripts: ['projects.js'],
    sourceUrl: SOURCE_URLS,
  });
  windows.push(win);
  const host = openProjects(win);
  click(win, host.querySelector('.projects-new-btn'));
  const panel = newPanel(host);
  const budget = panel.querySelector('.project-max-budget');
  const anthropic = panel.querySelector('input[data-key="ANTHROPIC_API_KEY"]');
  fill(win, panel, '.project-name', 'Paper');
  assert.deepStrictEqual([budget.value, boxes(panel).compaction], ['', false]);
  send(win, {type: 'configData', config: CONFIG, apiKeys: KEYS});
  assert.strictEqual(panel.querySelector('.project-name').value, 'Paper');
  assert.deepStrictEqual(
    [budget.value, boxes(panel).compaction, anthropic.value],
    ['25', true, 'sk-product'],
  );
  assert.strictEqual(optionLabels(panel.querySelector('.project-workdir')).length, 4);

  // The user sets the folder, the budget, a check box and a key, and
  // switches the sound off and on again; then the product's settings
  // change, a key rotated among them: what the user set stays and the
  // rest follows.
  const changed = {
    type: 'configData',
    config: Object.assign({}, CONFIG, {max_budget: 99, compaction: false}),
    apiKeys: {ANTHROPIC_API_KEY: 'sk-rotated', GEMINI_API_KEY: 'g-product'},
  };
  pickOption(win, panel.querySelector('.project-workdir'), '/work/other');
  fill(win, panel, '.project-max-budget', '7');
  setBox(win, panel, 'use_memory', true);
  setBox(win, panel, 'victory_sound', false);
  setBox(win, panel, 'victory_sound', true);
  fill(win, panel, 'input[data-key="GEMINI_API_KEY"]', 'g-proj');
  send(win, changed);
  assert.deepStrictEqual(
    [budget.value, boxes(panel).use_memory, boxes(panel).compaction, anthropic.value],
    ['7', true, false, 'sk-rotated'],
  );
  // Save sends what the user set.  A setting left alone is not sent,
  // as the page has it or otherwise: the page hears of no change made
  // elsewhere, and the daemon would keep the older value as the
  // project's own.  Nor is the sound, this browser's choice again.
  press(win, panel, 'project-save');
  const create = lastMsg(posted, 'saveProject');
  assert.deepStrictEqual(plain(create.project.settings), {
    default_work_dir: '/work/other',
    max_budget: 7,
    use_memory: true,
    api_keys: {GEMINI_API_KEY: 'g-proj'},
  });
  // The settings change while the save is out, and the daemon refuses
  // it: the form is an edit again, in which what was left alone
  // follows them.
  send(win, {type: 'configData', config: CONFIG, apiKeys: KEYS});
  assert.strictEqual(boxes(panel).compaction, false, 'as it was sent');
  send(win, {type: 'projectError', requestId: create.requestId, text: 'The disk is full.'});
  assert.deepStrictEqual(
    [budget.value, boxes(panel).use_memory, boxes(panel).compaction, anthropic.value],
    ['7', true, true, 'sk-product'],
  );

  // A project's panel: what the project overrides is sent as it is
  // stored, what it leaves to the product follows while its description
  // is edited and is not sent.  A budget box the user empties gives the
  // budget back to the product.
  const other = openPage();
  const settings = {max_budget: 7, api_keys: {GEMINI_API_KEY: 'g-own'}};
  send(other.win, {type: 'projects', projects: [project('p1', 'Paper', {settings})]});
  const own = panelOf(openProjects(other.win), 'p1');
  fill(other.win, own, '.project-description', 'typed here');
  send(other.win, changed);
  assert.deepStrictEqual(
    [boxes(own).compaction, own.querySelector('input[data-key="ANTHROPIC_API_KEY"]').value],
    [false, 'sk-rotated'],
  );
  press(other.win, own, 'project-save');
  assert.deepStrictEqual(plain(lastMsg(other.posted, 'saveProject').project), {
    id: 'p1',
    name: 'Paper',
    description: 'typed here',
    settings,
  });
  fill(other.win, own, '.project-max-budget', '');
  press(other.win, own, 'project-save');
  assert.deepStrictEqual(plain(lastMsg(other.posted, 'saveProject').project.settings), {
    api_keys: {GEMINI_API_KEY: 'g-own'},
  });
});

test('an older daemon\u2019s refusal of a read the page issues on its own is not shown as an error, nor taken for the answer to something else; the refusal of a read the user asked for is', () => {
  const {win, posted} = openPage([]);
  assert.strictEqual(ofType(posted, 'getProjects').length, 1, 'asked at start');
  const errors = () =>
    all(win, '#output div.ev.tr.err').map(banner => banner.textContent);
  // A promptlet add awaits the daemon's answer: an error that names no
  // command is taken for its refusal.
  click(win, byId(win, 'tricks-btn'));
  typeInto(win, byId(win, 'tricks-add-input'), 'Be brief');
  click(win, byId(win, 'tricks-add-btn'));
  const addError = byId(win, 'tricks-add-error');

  // A daemon that predates Projects (one updated on disk and not yet
  // restarted serves this page) does not know the reads the page issues
  // on its own: the projects at start, the composer's saved draft, the
  // tags of the task on show.
  for (const read of ['getProjects', 'getInputDraft', 'getTaskTags']) {
    send(win, {type: 'error', text: 'Unknown command: ' + read});
  }
  assert.deepStrictEqual(errors(), []);
  assert.ok(addError.hidden, 'the add is still awaiting its own answer');
  assert.ok(byId(win, 'chat-project-wrap').hidden, 'no project to offer');

  // The refusal of a read the user asked for (the task before the one
  // on show, by scrolling up) is shown in the chat that asked.
  send(win, {
    type: 'error',
    tabId: win._testApi.getActiveTabId(),
    text: 'Unknown command: getAdjacentTask',
  });
  assert.deepStrictEqual(errors(), ['Error: Unknown command: getAdjacentTask']);
  assert.ok(addError.hidden, 'an error of a chat answers no sheet');

  // The refusal of a command that is no read, and any other error, are
  // shown as before.
  send(win, {type: 'error', text: 'Unknown command: saveProject'});
  assert.strictEqual(errors()[1], 'Error: Unknown command: saveProject');
  assert.strictEqual(addError.textContent, 'Unknown command: saveProject');
  assert.ok(!addError.hidden);
  send(win, {type: 'error', text: 'getProjects: the disk is full'});
  assert.strictEqual(errors().length, 3);
});

/** The folder dialog's last listing request. */
function dialogListing(posted) {
  const listings = ofType(posted, 'listDir').filter(m =>
    String(m.token).startsWith('picker:'),
  );
  return listings[listings.length - 1];
}

const folderDialogTest = test('New folder and Choose ... pick a panel\u2019s folder through the folder dialog, which can create one', async () => {
  const {win, posted} = openPage();
  const host = openProjects(win);
  click(win, host.querySelector('.projects-new-btn'));
  const panel = newPanel(host);
  const folder = panel.querySelector('.project-workdir');
  fill(win, panel, '.project-name', 'Paper');

  press(win, panel, 'project-new-folder');
  const dialog = byId(win, 'folder-picker');
  const newRow = dialog.querySelector('.folder-picker-new');
  const nameBox = dialog.querySelector('.folder-picker-new-name');
  const pathBox = dialog.querySelector('.folder-picker-input');
  const note = dialog.querySelector('.folder-picker-note');
  /** The daemon lists `path`, the folder the dialog asked for last. */
  const listed = path =>
    send(win, {
      type: 'dirListing',
      token: dialogListing(posted).token,
      path,
      entries: [],
    });
  assert.ok(!dialog.hidden);
  assert.strictEqual(
    byId(win, 'folder-picker-title').textContent,
    'Choose the Default Working Directory',
  );
  assert.ok(!newRow.hidden, 'the New Folder row is open');
  await sleep(0);
  assert.strictEqual(win.document.activeElement, nameBox, 'and keeps the keyboard');
  // No folder is selected in the drop-down: the dialog starts in the
  // working directory.
  assert.strictEqual(dialogListing(posted).path, '/work/ws');

  // Nothing is created before the folder is listed, or without a name.
  typeInto(win, nameBox, 'drafts');
  click(win, dialog.querySelector('.folder-picker-new-create'));
  listed('/work/ws');
  typeInto(win, nameBox, '  ');
  click(win, dialog.querySelector('.folder-picker-new-create'));
  assert.strictEqual(lastMsg(posted, 'fsAction'), null);

  typeInto(win, nameBox, 'drafts');
  click(win, dialog.querySelector('.folder-picker-new-create'));
  const create = plain(lastMsg(posted, 'fsAction'));
  assert.deepStrictEqual(create, {
    type: 'fsAction',
    action: 'newFolder',
    path: '/work/ws',
    name: 'drafts',
    workDir: '/work/ws',
    tabId: win._testApi.getActiveTabId(),
    token: 'picker-new:2',
  });
  // A refusal is shown in the dialog, also when the folder was listed
  // again meanwhile; Enter in the box creates too.
  key(win, pathBox, 'Enter');
  send(win, {type: 'fsResult', token: create.token, error: 'drafts exists'});
  assert.strictEqual(note.textContent, 'drafts exists');
  assert.ok(!newRow.hidden);
  listed('/work/ws');
  typeInto(win, nameBox, 'drafts2');
  key(win, nameBox, 'a');
  key(win, nameBox, 'Enter');
  const retry = lastMsg(posted, 'fsAction');
  assert.deepStrictEqual([retry.name, retry.token], ['drafts2', 'picker-new:3']);
  // The answer to the earlier request is dropped; the one awaited steps
  // into the new folder, which Select Folder then picks.
  send(win, {type: 'fsResult', token: create.token, path: '/work/ws/drafts'});
  assert.strictEqual(dialogListing(posted).path, '/work/ws');
  send(win, {type: 'fsResult', token: retry.token, path: '/work/ws/drafts2'});
  assert.ok(newRow.hidden);
  assert.strictEqual(nameBox.value, '');
  assert.strictEqual(dialogListing(posted).path, '/work/ws/drafts2');
  // Nothing is created in the folder listed before while this one is
  // being listed, nor once its listing has failed.
  click(win, dialog.querySelector('.folder-picker-new-btn'));
  typeInto(win, nameBox, 'inside');
  click(win, dialog.querySelector('.folder-picker-new-create'));
  send(win, {
    type: 'dirListing',
    token: dialogListing(posted).token,
    error: 'Permission denied',
  });
  click(win, dialog.querySelector('.folder-picker-new-create'));
  assert.strictEqual(lastMsg(posted, 'fsAction'), retry);
  key(win, pathBox, 'Enter');
  listed('/work/ws/drafts2');
  click(win, dialog.querySelector('.folder-picker-select'));
  assert.ok(dialog.hidden);
  assert.strictEqual(folder.value, '/work/ws/drafts2');
  assert.deepStrictEqual(optionLabels(folder), [
    'Same as KISS Sorcar',
    '/work/ws/drafts2',
    '/work/ws',
    '/work/other',
    'Choose ...',
  ]);

  // "Choose ..." is not a value: the drop-down springs back and the
  // dialog opens at the folder selected, its New Folder row folded.
  Array.from(folder.options).find(o => o.dataset.choose).selected = true;
  folder.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.ok(!dialog.hidden);
  assert.strictEqual(folder.value, '/work/ws/drafts2');
  assert.ok(newRow.hidden);
  assert.strictEqual(dialogListing(posted).path, '/work/ws/drafts2');
  await sleep(0);
  assert.strictEqual(
    win.document.activeElement,
    dialog.querySelector('.folder-picker-input'),
  );
  // The dialog's own New Folder button opens the row.  The user moves
  // on to another folder before the answer: neither a refusal nor the
  // folder made takes that folder's place.
  listed('/work/ws/drafts2');
  click(win, dialog.querySelector('.folder-picker-new-btn'));
  assert.ok(!newRow.hidden);
  typeInto(win, nameBox, 'late');
  click(win, dialog.querySelector('.folder-picker-new-create'));
  const moved = lastMsg(posted, 'fsAction');
  click(win, dialog.querySelector('.folder-picker-up'));
  send(win, {type: 'fsResult', token: moved.token, error: 'late exists'});
  assert.strictEqual(note.textContent, '');
  const made = '/work/ws/drafts2/late';
  send(win, {type: 'fsResult', token: moved.token, path: made});
  assert.strictEqual(dialogListing(posted).path, '/work/ws');
  // Nor is a folder asked for and answered after the dialog closed
  // stepped into.
  listed('/work/ws');
  click(win, dialog.querySelector('.folder-picker-new-create'));
  const late = lastMsg(posted, 'fsAction');
  assert.strictEqual(late.path, '/work/ws');
  const listings = ofType(posted, 'listDir').length;
  click(win, dialog.querySelector('.folder-picker-cancel'));
  send(win, {type: 'fsResult', token: late.token, path: '/work/ws/late'});
  assert.ok(dialog.hidden);
  assert.strictEqual(ofType(posted, 'listDir').length, listings);
  assert.strictEqual(folder.value, '/work/ws/drafts2', 'Cancel picks nothing');

  // The folder picked is an edit of the form: a broadcast keeps it, and
  // Save sends it.
  send(win, {type: 'projects', projects: []});
  press(win, panel, 'project-save');
  assert.strictEqual(
    lastMsg(posted, 'saveProject').project.settings.default_work_dir,
    '/work/ws/drafts2',
  );
});

/** The folders the page asked the daemon to check before adopting them. */
function workDirChecks(posted) {
  return ofType(posted, 'listDir').filter(m =>
    String(m.token).startsWith('workdir:'),
  );
}

/** Answer the page's last working-directory check: the folder exists. */
function confirmWorkDir(win, posted) {
  const check = workDirChecks(posted).pop();
  send(win, {type: 'dirListing', token: check.token, path: check.path, entries: []});
}

/** Send `prompt` from the input box. */
function submit(win, posted, prompt) {
  typeInto(win, byId(win, 'task-input'), prompt);
  click(win, byId(win, 'send-btn'));
  return lastMsg(posted, 'submit');
}

test('the chat header\u2019s project drop-down puts the chat in a project; the chat then starts in the project\u2019s default folder, which a pick in the header overrides', () => {
  // No page script: the sidebar's row opens nothing, the rest works.
  const {win, posted, getState} = openPage([]);
  click(win, byId(win, 'projects-row'));
  assert.strictEqual(projectsTabs(win).length, 0);
  const wrap = byId(win, 'chat-project-wrap');
  const select = byId(win, 'chat-project');
  assert.ok(wrap.hidden && byId(win, 'chat-project-sep').hidden, 'no project');
  assert.strictEqual(ofType(posted, 'getProjects').length, 1, 'asked at start');

  const paper = project('p1', 'Paper', {
    settings: {default_work_dir: '/work/paper', use_web_browser: false, max_budget: 7},
  });
  const site = project('p2', 'Site');
  send(win, {type: 'projects', projects: [paper, site], chatProjects: {}});
  assert.ok(!wrap.hidden && !byId(win, 'chat-project-sep').hidden);
  assert.deepStrictEqual(optionLabels(select), ['Default', 'Paper', 'Site']);
  assert.strictEqual(select.value, '');
  assert.strictEqual(wrap.querySelector('.chat-header-select-label').textContent, 'Default');
  // Task Info names the budget the chat's next task runs under: the
  // Settings sheet's for a chat in no project.
  const budget = byId(win, 'meta-max-budget');
  assert.strictEqual(budget.textContent, '$25.00');

  // A chat the daemon has not created yet: the page keeps the pick and
  // sends it with the chat's first task.
  pickOption(win, select, 'p1');
  assert.strictEqual(lastMsg(posted, 'setChatProject'), null);
  assert.strictEqual(wrap.querySelector('.chat-header-select-label').textContent, 'Paper');
  const tabId = win._testApi.getActiveTabId();
  assert.deepStrictEqual(plain(getState().tabProjectPicks), {[tabId]: 'p1'});
  assert.strictEqual(budget.textContent, '$7.00', 'the budget of Paper');
  // The chat starts in the project's folder: checked, then adopted, as
  // a pick of that folder in the header is.
  assert.strictEqual(workDirChecks(posted).pop().path, '/work/paper');
  confirmWorkDir(win, posted);
  assert.strictEqual(lastMsg(posted, 'setWorkDir').workDir, '/work/paper');
  assert.strictEqual(currentDir(win), '/work/paper');
  // The folder picked in the header afterwards is the working
  // directory: a repaint of the projects does not put the default back.
  pickWorkDir(win, '/work/other');
  confirmWorkDir(win, posted);
  assert.strictEqual(lastMsg(posted, 'setWorkDir').workDir, '/work/other');
  const checks = workDirChecks(posted).length;
  send(win, {type: 'projects', projects: [paper, site]});
  assert.strictEqual(workDirChecks(posted).length, checks);
  assert.strictEqual(currentDir(win), '/work/other');
  assert.strictEqual(select.value, 'p1', 'the pick outlives a repaint');

  // Paper has a "Use web tools" of its own, which the daemon applies
  // to the run: the Settings sheet's is not sent over it.
  const first = submit(win, posted, 'write it');
  assert.strictEqual(first.projectId, 'p1');
  assert.ok(!('useWebTools' in first));
  send(win, {type: 'task_events', events: [], task: 'write it', tabId, chat_id: 'c9'});
  send(win, {type: 'chatProject', chatId: 'c9', projectId: 'p1'});
  assert.strictEqual(select.value, 'p1');
  assert.strictEqual(budget.textContent, '$7.00', 'the daemon\u2019s word for the chat');

  // A chat the daemon has: the pick is the daemon's to store, and the
  // chat moves to the new project's folder (Site has none: the folder
  // stays).
  pickOption(win, select, 'p2');
  assert.deepStrictEqual(plain(lastMsg(posted, 'setChatProject')), {
    type: 'setChatProject',
    chatId: 'c9',
    projectId: 'p2',
  });
  assert.strictEqual(select.value, 'p2');
  assert.strictEqual(workDirChecks(posted).length, checks);
  assert.strictEqual(budget.textContent, '$25.00', 'Site has no budget of its own');
  send(win, {type: 'chatProject', chatId: 'c9', projectId: 'p2'});
  // Site leaves "Use web tools" to the product: the chat's next task
  // is sent with the Settings sheet's.
  send(win, {type: 'task_done', tabId});
  const next = submit(win, posted, 'and publish it');
  assert.deepStrictEqual(
    [next.prompt, next.useWebTools, 'projectId' in next],
    ['and publish it', true, false],
  );
  // Another page takes the chat out of its project.
  send(win, {type: 'chatProject', chatId: 'c9', projectId: ''});
  assert.strictEqual(select.value, '');
  // A deleted project is no pick any more.
  send(win, {type: 'chatProject', chatId: 'c9', projectId: 'p2'});
  send(win, {type: 'projects', projects: [paper]});
  assert.deepStrictEqual(optionLabels(select), ['Default', 'Paper']);
  assert.strictEqual(select.value, '');
  send(win, {type: 'projects', projects: []});
  assert.ok(wrap.hidden);
});

test('a prompt sent while the project\u2019s folder is being checked starts behind the folder; a task for another machine does not wait', () => {
  const {win, posted} = openPage([]);
  const select = byId(win, 'chat-project');
  const box = byId(win, 'task-input');
  const inFolder = dir => ({settings: {default_work_dir: dir}});
  const paper = project('p1', 'Paper', inFolder('/work/paper'));
  const blog = project('p3', 'Blog', inFolder('/work/other'));
  send(win, {type: 'projects', projects: [paper, blog], chatProjects: {}});
  send(win, {type: 'machines', machines: MACHINES});

  // The prompt stays in the box until Paper's folder is adopted.
  pickOption(win, select, 'p1');
  typeInto(win, box, 'write it');
  click(win, byId(win, 'send-btn'));
  assert.strictEqual(lastMsg(posted, 'submit'), null);
  assert.strictEqual(box.value, 'write it');
  confirmWorkDir(win, posted);
  assert.deepStrictEqual(
    posted.map(m => m.type).filter(t => t === 'setWorkDir' || t === 'submit'),
    ['setWorkDir', 'submit'],
  );
  assert.strictEqual(lastMsg(posted, 'submit').projectId, 'p1');
  assert.strictEqual(box.value, '');
  // Blog's folder is this machine's: a task for another one starts.
  click(win, byId(win, 'new-chat-btn'));
  pickOption(win, select, 'p3');
  pickMachine(win, 'gpu-box');
  assert.strictEqual(submit(win, posted, 'run there').machine, 'gpu-box');
});

test('a new chat\u2019s project pick outlives a reload and goes with its tab; a chat of another machine is offered no project', () => {
  const paper = project('p1', 'Paper', {
    settings: {default_work_dir: '/work/paper'},
  });
  const before = openPage([]);
  send(before.win, {type: 'projects', projects: [paper], chatProjects: {}});
  pickOption(before.win, byId(before.win, 'chat-project'), 'p1');
  const tabId = before.win._testApi.getActiveTabId();
  const state = plain(before.getState());
  // What a page that had more tabs left behind, and a damaged entry.
  state.tabProjectPicks['closed-tab'] = 'p1';
  state.tabProjectPicks.damaged = 7;

  const {win, posted, getState} = makePage({
    bodyClass: 'remote-chat',
    initialState: state,
    sourceUrl: SOURCE_URLS,
  });
  windows.push(win);
  send(win, {type: 'configData', config: CONFIG, apiKeys: KEYS});
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId, chatId: '', title: 'new chat', workDir: ''}],
  });
  send(win, {type: 'projects', projects: [paper], chatProjects: {c1: 'p1'}});
  const select = byId(win, 'chat-project');
  assert.strictEqual(select.value, 'p1', 'the pick is back');
  pickOption(win, select, '');
  assert.deepStrictEqual(
    plain(getState().tabProjectPicks),
    {},
    'the picks of tabs that are gone went with them',
  );
  const unassigned = submit(win, posted, 'no project');
  assert.ok(!('projectId' in unassigned));
  assert.strictEqual(unassigned.useWebTools, true, 'the Settings sheet\u2019s');

  // A reload does not move the working directory, whatever the pick.
  assert.deepStrictEqual(workDirChecks(posted), []);

  // A chat of a registered machine: the projects are this daemon's.
  const remote = openPage([]);
  send(remote.win, {type: 'projects', projects: [paper], chatProjects: {}});
  send(remote.win, {type: 'machines', machines: MACHINES});
  pickOption(remote.win, byId(remote.win, 'chat-project'), 'p1');
  confirmWorkDir(remote.win, remote.posted);
  pickMachine(remote.win, 'gpu-box');
  assert.ok(byId(remote.win, 'chat-project-wrap').hidden);
  const sent = submit(remote.win, remote.posted, 'run there');
  assert.strictEqual(sent.machine, 'gpu-box');
  assert.ok(!('projectId' in sent), 'a task for another machine names none');
});

test('a panel names the product by its brand, and is blank while the daemon has sent no settings', () => {
  const {win} = makePage({
    bodyClass: 'remote-chat',
    extraScripts: ['projects.js'],
    sourceUrl: SOURCE_URLS,
    beforeScripts: w => {
      w.__BRAND__ = {productName: 'Acme Agent'};
    },
  });
  windows.push(win);
  send(win, {type: 'configData'});
  const host = openProjects(win);
  assert.ok(host.querySelector('.projects-intro').textContent.includes('Acme Agent'));
  click(win, host.querySelector('.projects-new-btn'));
  const panel = newPanel(host);
  assert.strictEqual(
    panel.querySelector('.project-workdir').options[0].textContent,
    'Same as Acme Agent',
  );
  assert.strictEqual(panel.querySelector('.project-max-budget').value, '');
  assert.ok(
    all(win, '.project-panel input[type="password"], .project-panel input[type="text"]').every(
      i => i.value === '',
    ),
    'no key and no name',
  );
});

folderDialogTest.then(() => report('projectsPage'));
