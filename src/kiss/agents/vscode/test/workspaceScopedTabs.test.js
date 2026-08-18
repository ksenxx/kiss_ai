// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the workspace-scoped tab bar: a client
// (VS Code webview or remote web app — both run this same main.js)
// only SHOWS the shared registry tabs whose current work dir matches
// the client's workspace directory (configWorkDir). Hiding is a
// RENDERING property only: hidden tabs keep their full local state
// (drafts, transcripts), stay in the local model (so restart recovery
// and one-tab-per-chat lookups still see them), are never activated,
// and no registry command is ever sent for a tab merely out of scope.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(opts) {
  const remote = !!(opts && opts.remote);
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  if (remote) {
    html = html.replace('{{BODY_CLASS_ATTR}}', ' class="remote-chat"');
  }
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  let state;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function setWorkspace(win, dir) {
  send(win, {type: 'configData', config: {work_dir: dir}, apiKeys: {}});
}

function tabBarIds(win) {
  return Array.from(win.document.querySelectorAll('.chat-tab'))
    .filter(el => !!el.dataset.tabId)
    .map(el => el.dataset.tabId);
}

function activeTabId(win) {
  const el = win.document.querySelector('.chat-tab.active');
  return el ? el.dataset.tabId : null;
}

function entry(tabId, workDir, chatId) {
  return {
    tabId: tabId,
    chatId: chatId || '',
    title: tabId,
    workDir: workDir || '',
  };
}

function clickEl(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function testForeignWorkspaceTabsAreHidden() {
  // Only tabs whose work dir is the workspace, a subdirectory of it
  // (a task that ran in a ".kiss-worktrees/kiss_wt-..." worktree), or
  // empty (unpinned) are shown; another workspace's tabs are not.
  const {win, posted} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [
      entry('a1', '/ws/a', 'chat-1'),
      entry('wt', '/ws/a/.kiss-worktrees/kiss_wt-7', 'chat-2'),
      entry('b1', '/ws/b', 'chat-3'),
      entry('un', '', 'chat-4'),
    ],
  });
  assert.deepStrictEqual(
    tabBarIds(win),
    ['a1', 'wt', 'un'],
    'only /ws/a tabs (plus unpinned) may be shown in workspace /ws/a',
  );
  // Hiding must be local-only: no closeTab may reach the registry.
  assert.strictEqual(
    posted.filter(m => m && m.type === 'closeTab').length,
    0,
    'an out-of-scope tab must never be closed in the shared registry',
  );
}

function testConfigArrivingAfterSnapshotRescopes() {
  // Boot ordering: the first `tabs_state` routinely lands before the
  // `configData` reply, so the workspace change must re-scope the
  // already-reconciled tab bar without waiting for a new broadcast.
  const {win} = makeWebview();
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1'), entry('b1', '/ws/b', 'chat-2')],
  });
  assert.deepStrictEqual(
    tabBarIds(win),
    ['a1', 'b1'],
    'without a workspace every shared tab is shown',
  );
  setWorkspace(win, '/ws/a');
  assert.deepStrictEqual(
    tabBarIds(win),
    ['a1'],
    'the late-arriving workspace must hide the foreign tab',
  );
  assert.strictEqual(activeTabId(win), 'a1');
}

function testWorkspaceSwitchPreservesHiddenTabState() {
  // Re-pinning the client to another workspace swaps which shared
  // tabs are visible; hidden tabs keep their local state — the draft
  // typed into a tab must survive its tab being hidden and re-shown.
  const {win} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1'), entry('b1', '/ws/b', 'chat-2')],
  });
  assert.deepStrictEqual(tabBarIds(win), ['a1']);
  const inp = win.document.getElementById('task-input');
  inp.value = 'UNSAVED DRAFT';
  setWorkspace(win, '/ws/b');
  assert.deepStrictEqual(
    tabBarIds(win),
    ['b1'],
    'switching workspace must show its tabs and hide the others',
  );
  assert.strictEqual(
    activeTabId(win),
    'b1',
    'the active tab may never be a hidden one',
  );
  assert.strictEqual(inp.value, '', 'b1 has no draft of its own');
  setWorkspace(win, '/ws/a');
  assert.deepStrictEqual(tabBarIds(win), ['a1']);
  assert.strictEqual(
    inp.value,
    'UNSAVED DRAFT',
    'the hidden tab must come back with its draft intact',
  );
}

function testPathNormalization() {
  // A "/" workspace contains every absolute path; trailing slashes
  // are trimmed; Windows paths compare case-insensitively with either
  // separator — the same normalization the history filter uses.
  const {win} = makeWebview();
  setWorkspace(win, '/');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1'), entry('b1', '/ws/b', 'chat-2')],
  });
  assert.deepStrictEqual(tabBarIds(win), ['a1', 'b1']);
  setWorkspace(win, '/ws/a/');
  assert.deepStrictEqual(
    tabBarIds(win),
    ['a1'],
    'a trailing slash on the workspace must not defeat the match',
  );
  const w = makeWebview();
  setWorkspace(w.win, 'C:\\Proj\\App');
  send(w.win, {
    type: 'tabs_state',
    tabs: [
      entry('w1', 'c:/proj/app/sub', 'chat-1'),
      entry('w2', 'D:\\other', 'chat-2'),
    ],
  });
  assert.deepStrictEqual(
    tabBarIds(w.win),
    ['w1'],
    'Windows paths must match case-insensitively across separators',
  );
}

function testCanonicalEmptyWorkDirWins() {
  // The registry's work dir is canonical INCLUDING the empty string
  // ("unpinned, belongs everywhere"): a stale local pin must not hide
  // a tab the registry says is unpinned.
  const {win} = makeWebview();
  send(win, {
    type: 'tabs_state',
    tabs: [entry('b1', '/ws/b', 'chat-1')],
  });
  assert.deepStrictEqual(tabBarIds(win), ['b1']);
  // The registry unpins the tab; the local tab.workDir still says
  // /ws/b, but the canonical empty value wins.
  send(win, {
    type: 'tabs_state',
    tabs: [entry('b1', '', 'chat-1')],
  });
  setWorkspace(win, '/ws/a');
  assert.deepStrictEqual(
    tabBarIds(win),
    ['b1'],
    'a canonically unpinned tab belongs to every workspace',
  );
}

function testScopeWorkDirPinsRunAgentTabToCaller() {
  // A run_agent sub-task runs in a channel/cron scratch directory
  // (workDir) OUTSIDE any workspace, but the daemon pins a separate
  // scopeWorkDir to the calling workspace so its tab shows there.
  const {win} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [
      // Scratch workDir is foreign; scopeWorkDir is the caller's ws.
      {
        tabId: 'api-1',
        chatId: 'chat-1',
        title: 'send hi',
        workDir: '/home/u/.kiss/channel_work',
        scopeWorkDir: '/ws/a',
      },
      // Same scratch workDir but scoped to ANOTHER workspace: hidden.
      {
        tabId: 'api-2',
        chatId: 'chat-2',
        title: 'send hi elsewhere',
        workDir: '/home/u/.kiss/channel_work',
        scopeWorkDir: '/ws/b',
      },
    ],
  });
  assert.deepStrictEqual(
    tabBarIds(win),
    ['api-1'],
    'the run_agent tab is shown in the workspace that dispatched it, ' +
      'and hidden in others, despite its foreign scratch workDir',
  );
  // Switching to /ws/b hides api-1 and shows api-2 — scope, not the
  // shared scratch workDir, decides visibility.
  setWorkspace(win, '/ws/b');
  assert.deepStrictEqual(
    tabBarIds(win),
    ['api-2'],
    'scopeWorkDir (not the scratch workDir) decides the tab bar',
  );
}

function testPendingLocalTabSurvivesForeignSnapshot() {
  // A locally created tab whose `openTab` echo has not arrived yet is
  // shielded from reconciliation even when every snapshot entry is
  // out of scope for this workspace.
  const {win, posted} = makeWebview();
  setWorkspace(win, '/ws/a');
  const plusBtn = win.document.querySelector('.chat-tab-add');
  assert.ok(plusBtn, 'the new-tab button must exist');
  clickEl(win, plusBtn);
  const opened = posted.filter(m => m && m.type === 'openTab');
  assert.ok(opened.length >= 1, 'the new tab must be announced');
  const newId = opened[opened.length - 1].tabId;
  send(win, {
    type: 'tabs_state',
    tabs: [entry('b1', '/ws/b', 'chat-1')],
  });
  assert.ok(
    tabBarIds(win).indexOf(newId) !== -1,
    'the pending local tab must survive a foreign-only snapshot',
  );
  assert.ok(
    tabBarIds(win).indexOf('b1') === -1,
    'the foreign tab must stay hidden',
  );
}

function testRemoteWorkDirSaveRescopesImmediately() {
  // Remote web app: pinning a new work_dir from the settings panel
  // re-scopes the tab bar right away (no configData round-trip).
  const {win, posted} = makeWebview({remote: true});
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1'), entry('b1', '/ws/b', 'chat-2')],
  });
  assert.deepStrictEqual(tabBarIds(win), ['a1']);

  const settingsBtn = win.document.querySelector('.chat-tab-settings');
  assert.ok(settingsBtn, 'the settings button must exist');
  clickEl(win, settingsBtn);
  // The daemon answers the panel's getConfig with the current config.
  setWorkspace(win, '/ws/a');

  const wdInp = win.document.getElementById('cfg-work-dir');
  wdInp.value = '/ws/b';
  wdInp.dispatchEvent(new win.Event('input', {bubbles: true}));
  clickEl(win, win.document.getElementById('settings-panel-close'));

  const pins = posted.filter(m => m && m.type === 'setWorkDir');
  assert.ok(pins.length >= 1, 'closing settings must pin the new dir');
  assert.strictEqual(pins[pins.length - 1].workDir, '/ws/b');
  assert.deepStrictEqual(
    tabBarIds(win),
    ['b1'],
    'the re-pinned workspace must swap the visible tabs immediately',
  );
}

function testWorkspaceWorkDirMessageRescopes() {
  // The extension host reports a live workspace-folder change with a
  // `workspaceWorkDir` message (its daemon `setWorkDir` produces no
  // configData); the tab bar must re-scope on it.
  const {win} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1'), entry('b1', '/ws/b', 'chat-2')],
  });
  assert.deepStrictEqual(tabBarIds(win), ['a1']);
  send(win, {type: 'workspaceWorkDir', workDir: '/ws/b'});
  assert.deepStrictEqual(
    tabBarIds(win),
    ['b1'],
    'a live workspace-folder change must re-scope the tab bar',
  );
  assert.strictEqual(activeTabId(win), 'b1');
}

function testReadyRestoresHiddenTabsToo() {
  // Hidden tabs stay in the local model, so a daemon-restart recovery
  // (`ready.restoredTabs`) must offer the FULL shared registry, not
  // just this workspace's visible subset — otherwise the first
  // reconnecting client would permanently drop every other
  // workspace's tabs from a re-seeded registry.
  const {win, posted} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1'), entry('b1', '/ws/b', 'chat-2')],
  });
  assert.deepStrictEqual(tabBarIds(win), ['a1']);
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  const readies = posted.filter(m => m && m.type === 'ready');
  const last = readies[readies.length - 1];
  // Array.from re-realms the JSDOM-side array so deepStrictEqual's
  // prototype check compares against THIS realm's Array.
  const restoredIds = Array.from(last.restoredTabs || [], t => t.tabId).sort();
  assert.deepStrictEqual(
    restoredIds,
    ['a1', 'b1'],
    'restart recovery must include hidden tabs of other workspaces',
  );
}

function testHistoryClickOnHiddenChatOpensFreshTabHere() {
  // Clicking a history row whose chat lives in a HIDDEN tab is an
  // explicit "open this chat in THIS workspace": a fresh tab resumes
  // it here (the daemon's one-tab-per-chat displacement then retires
  // the old tab) instead of silently activating an invisible tab.
  const {win, posted} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1'), entry('b1', '/ws/b', 'chat-2')],
  });
  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [
      {
        id: 'chat-2',
        task_id: 't-9',
        title: 'foreign chat',
        preview: 'foreign chat',
        timestamp: 1000,
        has_events: true,
        work_dir: '/ws/b',
      },
    ],
  });
  const row = win.document.querySelector('.sidebar-item');
  assert.ok(row, 'the history row must render');
  clickEl(win, row);
  const resumes = posted.filter(m => m && m.type === 'resumeSession');
  assert.ok(resumes.length >= 1, 'the click must resume the chat');
  const resume = resumes[resumes.length - 1];
  assert.notStrictEqual(
    resume.tabId,
    'b1',
    'the hidden foreign tab must not be reused/activated',
  );
  assert.strictEqual(
    activeTabId(win),
    resume.tabId,
    'the chat must open in a fresh visible tab of this workspace',
  );
}

function testBulkCloseSparesHiddenTabs() {
  // "Close All" (and friends) act on visible tabs only: closing a tab
  // this user cannot see would destroy another workspace's tab in the
  // shared registry.
  const {win, posted} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [
      entry('a1', '/ws/a', 'chat-1'),
      entry('b1', '/ws/b', 'chat-2'),
      entry('a2', '/ws/a', 'chat-3'),
    ],
  });
  assert.deepStrictEqual(tabBarIds(win), ['a1', 'a2']);
  const strip = win.document.querySelector('.chat-tab[data-tab-id="a1"]');
  strip.dispatchEvent(
    new win.MouseEvent('contextmenu', {bubbles: true, clientX: 5, clientY: 5}),
  );
  const items = Array.from(
    win.document.querySelectorAll('#tab-context-menu .tab-ctx-item'),
  );
  const closeAll = items.find(el => el.textContent === 'Close All');
  assert.ok(closeAll, 'the Close All menu item must exist');
  clickEl(win, closeAll);
  const closedIds = posted
    .filter(m => m && m.type === 'closeTab')
    .map(m => m.tabId)
    .sort();
  assert.deepStrictEqual(
    closedIds,
    ['a1', 'a2'],
    'Close All must never close a hidden foreign tab',
  );
}

function testCloseActiveTabSkipsHiddenSuccessor() {
  // Closing the active tab must hand over to the nearest VISIBLE tab,
  // never to a hidden one sitting adjacent in the tab array.
  const {win} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [
      entry('a1', '/ws/a', 'chat-1'),
      entry('b1', '/ws/b', 'chat-2'),
      entry('a2', '/ws/a', 'chat-3'),
    ],
  });
  assert.strictEqual(activeTabId(win), 'a1');
  const closeBtn = win.document.querySelector(
    '.chat-tab[data-tab-id="a1"] .chat-tab-close',
  );
  clickEl(win, closeBtn);
  assert.strictEqual(
    activeTabId(win),
    'a2',
    'the successor must be the nearest visible tab, not the hidden one',
  );
}

function testForeignFileContentOpensHiddenNotFocused() {
  // A file/report produced by a HIDDEN tab's task inherits that
  // owner's workspace scope: it opens as a hidden content tab and
  // never steals focus in an unrelated workspace.
  const {win} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1'), entry('b1', '/ws/b', 'chat-2')],
  });
  assert.strictEqual(activeTabId(win), 'a1');
  send(win, {
    type: 'fileContent',
    tabId: 'b1',
    path: '/ws/b/report.html',
    name: 'report.html',
    content: '<p>foreign report</p>',
  });
  assert.strictEqual(
    activeTabId(win),
    'a1',
    'a foreign file open must not steal focus',
  );
  assert.deepStrictEqual(
    tabBarIds(win),
    ['a1'],
    'the foreign-owned content tab must be hidden, not rendered',
  );
  // In the owning workspace the content tab is there, waiting.
  setWorkspace(win, '/ws/b');
  assert.ok(
    tabBarIds(win).length === 2,
    'the content tab must surface in its own workspace',
  );
}

function testPlaceholderDraftSurvivesForeignSnapshots() {
  // With only foreign registry tabs, the local placeholder carries
  // the composer; repeated foreign-only snapshots must not destroy
  // and recreate it (that would erase the draft).
  const {win} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('b1', '/ws/b', 'chat-1')],
  });
  const placeholderId = activeTabId(win);
  assert.ok(placeholderId, 'a placeholder must exist');
  const inp = win.document.getElementById('task-input');
  inp.value = 'PLACEHOLDER DRAFT';
  send(win, {
    type: 'tabs_state',
    tabs: [entry('b1', '/ws/b', 'chat-1')],
  });
  assert.strictEqual(
    activeTabId(win),
    placeholderId,
    'a foreign-only snapshot must not replace the placeholder',
  );
  assert.strictEqual(
    inp.value,
    'PLACEHOLDER DRAFT',
    'the placeholder draft must survive foreign-only snapshots',
  );
}

function testReadySendsCanonicalWorkDir() {
  // Restart recovery must serialize the registry's CANONICAL work dir
  // (including '' = unpinned), never the stale local pin, or an empty
  // registry would be re-seeded with wrong pins.
  const {win, posted} = makeWebview();
  send(win, {
    type: 'tabs_state',
    tabs: [entry('b1', '/ws/b', 'chat-1')],
  });
  // The registry unpins the tab; the local tab.workDir keeps /ws/b.
  send(win, {
    type: 'tabs_state',
    tabs: [entry('b1', '', 'chat-1')],
  });
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  const readies = posted.filter(m => m && m.type === 'ready');
  const last = readies[readies.length - 1];
  const restored = Array.from(last.restoredTabs || []).find(
    t => t.tabId === 'b1',
  );
  assert.ok(restored, 'the tab must be offered for recovery');
  assert.strictEqual(
    restored.workDir,
    '',
    'recovery must carry the canonical (unpinned) work dir',
  );
}

function testHostNeverPointedAtHiddenChat() {
  // When the reported chat tab dies, the surviving chat reported to
  // the host must be a VISIBLE one — the host routes commit/merge
  // actions to it.
  const {win, posted} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1'), entry('b1', '/ws/b', 'chat-2')],
  });
  assert.strictEqual(activeTabId(win), 'a1');
  const closeBtn = win.document.querySelector(
    '.chat-tab[data-tab-id="a1"] .chat-tab-close',
  );
  clickEl(win, closeBtn);
  const reported = posted
    .filter(m => m && m.type === 'activeTabChanged')
    .map(m => m.tabId);
  assert.ok(
    reported.indexOf('b1') === -1,
    'the host must never be pointed at a hidden foreign chat',
  );
}

function testOpenTabRejectedKeepsSoleVisibleChat() {
  // The registry cap rejection must count VISIBLE chats: a hidden
  // foreign tab must not make the only visible chat look expendable
  // (closing it would loop open/reject forever).
  const {win, posted} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('b1', '/ws/b', 'chat-1')],
  });
  const placeholderId = activeTabId(win);
  const plusBtn = win.document.querySelector('.chat-tab-add');
  clickEl(win, plusBtn);
  const newId = activeTabId(win);
  assert.notStrictEqual(newId, placeholderId);
  // Close the old placeholder so the registered tab is the only
  // visible chat, then the daemon rejects its registration.
  const oldClose = win.document.querySelector(
    '.chat-tab[data-tab-id="' + placeholderId + '"] .chat-tab-close',
  );
  clickEl(win, oldClose);
  const opensBefore = posted.filter(m => m && m.type === 'openTab').length;
  send(win, {type: 'openTabRejected', tabId: newId});
  assert.strictEqual(
    activeTabId(win),
    newId,
    'the sole visible chat must survive its registration rejection',
  );
  const opensAfter = posted.filter(m => m && m.type === 'openTab').length;
  assert.strictEqual(
    opensAfter,
    opensBefore,
    'a rejection must not trigger another openTab (no loop)',
  );
}

function testStaleContextMenuCannotCloseHiddenTab() {
  // A context menu opened before a workspace switch must not act on
  // its now-hidden anchor tab.
  const {win, posted} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1'), entry('b1', '/ws/b', 'chat-2')],
  });
  const strip = win.document.querySelector('.chat-tab[data-tab-id="a1"]');
  strip.dispatchEvent(
    new win.MouseEvent('contextmenu', {bubbles: true, clientX: 5, clientY: 5}),
  );
  send(win, {type: 'workspaceWorkDir', workDir: '/ws/b'});
  const items = Array.from(
    win.document.querySelectorAll('#tab-context-menu .tab-ctx-item'),
  );
  const closeItem = items.find(el => el.textContent === 'Close');
  assert.ok(closeItem, 'the Close menu item must exist');
  clickEl(win, closeItem);
  const closed = posted
    .filter(m => m && m.type === 'closeTab')
    .map(m => m.tabId);
  assert.ok(
    closed.indexOf('a1') === -1,
    'a stale menu must not close the now-hidden tab',
  );
}

function testOrphanContentTabKeepsOwnerScope() {
  // A content tab freezes its owner's workspace scope at creation:
  // when the owner closes (e.g. a finished sub-agent tab), the orphan
  // must stay pinned to that workspace instead of appearing
  // everywhere as unpinned.
  const {win} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1'), entry('b1', '/ws/b', 'chat-2')],
  });
  send(win, {
    type: 'fileContent',
    tabId: 'b1',
    path: '/ws/b/report.html',
    name: 'report.html',
    content: '<p>foreign report</p>',
  });
  assert.deepStrictEqual(tabBarIds(win), ['a1']);
  // The owner disappears from the registry (its chat tab closed).
  send(win, {
    type: 'tabs_state',
    tabs: [entry('a1', '/ws/a', 'chat-1')],
  });
  assert.deepStrictEqual(
    tabBarIds(win),
    ['a1'],
    'the orphaned foreign content tab must stay hidden here',
  );
  setWorkspace(win, '/ws/b');
  const contentStrips = win.document.querySelectorAll(
    '.chat-tab.content-tab',
  );
  assert.strictEqual(
    contentStrips.length,
    1,
    'the orphaned content tab must still surface in its own workspace',
  );
  assert.ok(
    tabBarIds(win).indexOf('a1') === -1,
    "workspace A's chat must stay hidden in workspace B",
  );
}

function testContentTabsNotSharedAcrossForeignScopes() {
  // Two different foreign workspaces are both "hidden" here, but they
  // must never share one content tab for the same path.
  const {win} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [
      entry('a1', '/ws/a', 'chat-1'),
      entry('b1', '/ws/b', 'chat-2'),
      entry('c1', '/ws/c', 'chat-3'),
    ],
  });
  const open = ownerId => {
    send(win, {
      type: 'fileContent',
      tabId: ownerId,
      path: '/shared/notes.md',
      name: 'notes.md',
      content: 'from ' + ownerId,
    });
  };
  open('b1');
  open('c1');
  setWorkspace(win, '/ws/b');
  assert.strictEqual(
    tabBarIds(win).length,
    2,
    "workspace B must show b1 plus B's own content tab",
  );
  setWorkspace(win, '/ws/c');
  assert.strictEqual(
    tabBarIds(win).length,
    2,
    "workspace C must show c1 plus C's own content tab",
  );
}

function testHostRepointedWhenReportedChatHides() {
  // While an everywhere-visible content tab is active, a workspace
  // switch must still re-point the host's chat binding away from the
  // now-hidden chat (commit/merge actions route to it).
  const {win, posted} = makeWebview();
  setWorkspace(win, '/ws/a');
  send(win, {
    type: 'tabs_state',
    tabs: [
      entry('a1', '/ws/a', 'chat-1'),
      entry('u1', '', 'chat-2'),
      entry('b1', '/ws/b', 'chat-3'),
    ],
  });
  assert.strictEqual(activeTabId(win), 'a1');
  // Open a content tab owned by the unpinned chat and activate it.
  send(win, {
    type: 'fileContent',
    tabId: 'u1',
    path: '/shared/readme.md',
    name: 'readme.md',
    content: 'shared file',
  });
  const contentStrip = Array.from(
    win.document.querySelectorAll('.chat-tab.content-tab'),
  )[0];
  assert.ok(contentStrip, 'the content tab must render');
  clickEl(win, contentStrip);
  send(win, {type: 'workspaceWorkDir', workDir: '/ws/b'});
  const reported = posted
    .filter(m => m && m.type === 'activeTabChanged')
    .map(m => m.tabId);
  assert.strictEqual(
    reported[reported.length - 1],
    'u1',
    'the host must be re-pointed at a visible chat (the content ' +
      "tab's owner), never left on the hidden one",
  );
}

const tests = [
  testForeignWorkspaceTabsAreHidden,
  testConfigArrivingAfterSnapshotRescopes,
  testWorkspaceSwitchPreservesHiddenTabState,
  testPathNormalization,
  testCanonicalEmptyWorkDirWins,
  testScopeWorkDirPinsRunAgentTabToCaller,
  testPendingLocalTabSurvivesForeignSnapshot,
  testRemoteWorkDirSaveRescopesImmediately,
  testWorkspaceWorkDirMessageRescopes,
  testReadyRestoresHiddenTabsToo,
  testHistoryClickOnHiddenChatOpensFreshTabHere,
  testBulkCloseSparesHiddenTabs,
  testCloseActiveTabSkipsHiddenSuccessor,
  testForeignFileContentOpensHiddenNotFocused,
  testPlaceholderDraftSurvivesForeignSnapshots,
  testReadySendsCanonicalWorkDir,
  testHostNeverPointedAtHiddenChat,
  testOpenTabRejectedKeepsSoleVisibleChat,
  testStaleContextMenuCannotCloseHiddenTab,
  testOrphanContentTabKeepsOwnerScope,
  testContentTabsNotSharedAcrossForeignScopes,
  testHostRepointedWhenReportedChatHides,
];

let failures = 0;
for (const t of tests) {
  try {
    t();
    console.log('PASS', t.name);
  } catch (err) {
    failures += 1;
    console.error('FAIL', t.name);
    console.error(err && err.stack ? err.stack : err);
  }
}
if (failures > 0) {
  console.error(`${failures} test(s) failed`);
  process.exit(1);
}
console.log(`All ${tests.length} workspaceScopedTabs tests passed`);
