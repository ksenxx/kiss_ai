// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// JSDOM end-to-end tests for the desktop remote page's content pane
// and task-info panel (media/chat.html, main.js, main.css,
// remote-codex.css):
//
//   * the content pane (its tab row, area and handle) exists only while
//     a file, browser or terminal tab is open: body.content-pane-open
//     follows the open content tabs, and without it #app is the chat
//     column alone, clear of the docked panel;
//   * with the pane open the panel lies on top of it; a press anywhere
//     in the pane slides the panel off (body.meta-hidden, the panel
//     inert) and the #meta-drawer tab brings it back; closing the last
//     content tab brings it back too;
//   * the Explorer and Source Control sections catch up on task news
//     that arrived while the panel was hidden;
//   * the machine name from configData shows above the transcript;
//   * a tool-call panel starts folded and is active (panel.panel-active,
//     main.css pulses its header text) until its tool_result, a
//     Question and a fan-out panel start open.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {inlineDesignTokens} = require('./designTokens');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(opts) {
  const {remote = true, desktop = true} = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(
    '{{BODY_CLASS_ATTR}}',
    remote ? ' class="remote-chat"' : '',
  );
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
  win.Element.prototype.setPointerCapture = function () {};
  win.Element.prototype.releasePointerCapture = function () {};
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
  const listeners = [];
  const mql = {
    matches: desktop,
    media: '(min-width: 900px)',
    addEventListener: (ev, fn) => {
      if (ev === 'change') listeners.push(fn);
    },
    removeEventListener: () => {},
    addListener: fn => listeners.push(fn),
    removeListener: () => {},
  };
  win.matchMedia = function (query) {
    if (query === '(min-width: 900px)') return mql;
    return {
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  function fireChange(matches) {
    mql.matches = matches;
    listeners.forEach(fn => fn(mql));
  }
  return {win, posted, fireChange};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function byId(win, id) {
  return win.document.getElementById(id);
}

function press(win, el) {
  el.dispatchEvent(new win.Event('pointerdown', {bubbles: true}));
}

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function ofType(posted, type) {
  return posted.filter(m => m.type === type);
}

function hasClass(win, cls) {
  return win.document.body.classList.contains(cls);
}

/** Open a chat tab `a1` on /ws and a file in the content pane. */
function openChatAndFile(win) {
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 'a1', chatId: 'chat-1', title: 'a1', workDir: '/ws'}],
  });
  send(win, {type: 'task_events', tabId: 'a1', task: 'the task', events: []});
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/notes.md',
    name: 'notes.md',
    content: '# Notes',
    version: 'v1',
  });
  const fileTab = win._testApi.openTabs().find(t => t.isContentTab);
  assert.ok(fileTab, 'the file opened as a content tab');
  return fileTab;
}

function closeContentTab(win, tabId) {
  const btn = win.document.querySelector(
    '#content-tab-list .chat-tab[data-tab-id="' + tabId + '"] .chat-tab-close',
  );
  assert.ok(btn, 'the content tab has a close button');
  click(win, btn);
}

const TESTS = [];
function test(name, fn) {
  TESTS.push({name, fn});
}

test('the content pane exists only while a content tab is open', () => {
  const {win} = makeWebview();
  assert.ok(hasClass(win, 'remote-desktop'));
  assert.ok(!hasClass(win, 'content-pane-open'), 'no content tab: no pane');
  const fileTab = openChatAndFile(win);
  assert.ok(hasClass(win, 'content-pane-open'), 'a file opened the pane');
  closeContentTab(win, fileTab.id);
  assert.ok(
    !win._testApi.openTabs().some(t => t.isContentTab),
    'the file tab closed',
  );
  assert.ok(!hasClass(win, 'content-pane-open'), 'the pane went away');
  win.close();
});

test('a press in the pane hides the panel; the drawer and the pane closing bring it back', () => {
  const {win} = makeWebview();
  const panel = byId(win, 'meta-panel');
  const drawer = byId(win, 'meta-drawer');
  // Without a content pane the panel is docked beside the chat: a
  // press in the (absent) pane changes nothing.
  press(win, byId(win, 'content-tab-bar'));
  assert.ok(!hasClass(win, 'meta-hidden'));

  const fileTab = openChatAndFile(win);
  assert.ok(!hasClass(win, 'meta-hidden'), 'the panel starts on top');
  assert.strictEqual(drawer.getAttribute('aria-expanded'), 'true');
  press(win, byId(win, 'content-tab-area'));
  assert.ok(hasClass(win, 'meta-hidden'), 'a press in the pane hides it');
  assert.ok(panel.hasAttribute('inert'), 'the hidden panel is inert');
  assert.strictEqual(drawer.getAttribute('aria-expanded'), 'false');

  click(win, drawer);
  assert.ok(!hasClass(win, 'meta-hidden'), 'the drawer brings it back');
  assert.ok(!panel.hasAttribute('inert'));
  assert.strictEqual(drawer.getAttribute('aria-expanded'), 'true');

  // The pane's tab row counts as the pane.
  press(win, byId(win, 'content-tab-bar'));
  assert.ok(hasClass(win, 'meta-hidden'));
  // A second file keeps the user's choice.
  send(win, {
    type: 'fileContent',
    tabId: 'a1',
    path: '/ws/other.md',
    name: 'other.md',
    content: '# Other',
    version: 'v1',
  });
  assert.ok(hasClass(win, 'meta-hidden'), 'another file keeps it hidden');
  // Closing every content tab docks the panel beside the chat again.
  for (const t of win._testApi.openTabs().filter(t => t.isContentTab))
    closeContentTab(win, t.id);
  assert.ok(!hasClass(win, 'content-pane-open'));
  assert.ok(!hasClass(win, 'meta-hidden'), 'docked: never hidden');
  assert.ok(!panel.hasAttribute('inert'));
  assert.strictEqual(fileTab.id !== undefined, true);
  win.close();
});

test('focus taken by a frame of the pane counts as a press in it', () => {
  const {win} = makeWebview();
  openChatAndFile(win);
  const area = byId(win, 'content-tab-area');
  const frame = win.document.createElement('iframe');
  area.appendChild(frame);
  // The window blurring for any other reason (another window, a frame
  // elsewhere) changes nothing.
  win.dispatchEvent(new win.Event('blur'));
  assert.ok(!hasClass(win, 'meta-hidden'));
  const outside = win.document.createElement('iframe');
  win.document.body.appendChild(outside);
  outside.focus();
  win.dispatchEvent(new win.Event('blur'));
  assert.ok(!hasClass(win, 'meta-hidden'), 'a frame outside the pane');
  frame.focus();
  assert.strictEqual(win.document.activeElement, frame);
  win.dispatchEvent(new win.Event('blur'));
  assert.ok(hasClass(win, 'meta-hidden'), 'the preview frame took the press');
  win.close();
});

test('leaving the desktop layout drops the hidden state', () => {
  const {win, fireChange} = makeWebview();
  openChatAndFile(win);
  press(win, byId(win, 'content-tab-area'));
  assert.ok(hasClass(win, 'meta-hidden'));
  fireChange(false);
  assert.ok(!hasClass(win, 'remote-desktop'));
  assert.ok(!hasClass(win, 'meta-hidden'), 'the phone drawer is never hidden');
  assert.ok(byId(win, 'meta-panel').hasAttribute('inert'), 'closed drawer');
  win.close();
});

test('the workspace views catch up on task news once the panel is back', async () => {
  const {win, posted} = makeWebview();
  send(win, {type: 'configData', config: {work_dir: '/ws'}});
  const listed = ofType(posted, 'listDir').length;
  assert.ok(listed >= 1, 'the Explorer listed the pinned workspace');
  // Answer the root listing so task news has a loaded folder to re-list.
  const req = ofType(posted, 'listDir').pop();
  send(win, {
    type: 'dirListing',
    token: req.token,
    path: req.path,
    root: req.path,
    entries: [{name: 'a.txt', type: 'file'}],
  });
  openChatAndFile(win);
  press(win, byId(win, 'content-tab-area'));
  assert.ok(hasClass(win, 'meta-hidden'));
  const before = ofType(posted, 'listDir').length;
  send(win, {type: 'tasks_updated'});
  await new Promise(r => setTimeout(r, 700));
  assert.strictEqual(
    ofType(posted, 'listDir').length,
    before,
    'hidden: the Explorer does not reload yet',
  );
  click(win, byId(win, 'meta-drawer'));
  assert.ok(
    ofType(posted, 'listDir').length > before,
    'back on screen: the Explorer reloads',
  );
  win.close();
});

test('the machine name shows above the transcript', () => {
  const {win} = makeWebview();
  const el = byId(win, 'chat-machine');
  assert.strictEqual(el.textContent, '', 'empty until configData');
  send(win, {type: 'configData', config: {}, machine: 'devbox-7'});
  assert.strictEqual(el.textContent, 'devbox-7');
  assert.strictEqual(byId(win, 'status-machine').textContent, 'devbox-7');
  win.close();
});

test('a tool-call panel starts folded and pulses until its result', () => {
  const {win} = makeWebview({remote: false});
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab, startTs: 1000});
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls -la',
    description: 'list files',
    tabId: tab,
    ts: 1000,
  });
  const O = byId(win, 'output');
  const tc = O.querySelector('.ev.tc');
  assert.ok(tc.classList.contains('collapsed'), 'folded at birth');
  assert.ok(tc.classList.contains('panel-active'), 'active until the result');
  const hdr = tc.querySelector(':scope > .tc-h');
  const name = hdr.querySelector('.tc-h-name');
  const prev = hdr.querySelector('.collapse-preview');
  assert.strictEqual(name.textContent, 'Bash');
  assert.strictEqual(name.children.length, 0, 'the name is plain text');
  assert.strictEqual(prev.textContent, 'list files', 'the folded preview');
  assert.strictEqual(prev.children.length, 0, 'the preview is plain text');
  assert.strictEqual(hdr.getAttribute('aria-expanded'), 'false');
  // The user may unfold it; the preview empties, the header keeps pulsing.
  click(win, hdr);
  assert.ok(!tc.classList.contains('collapsed'));
  assert.strictEqual(prev.textContent, '');
  assert.ok(tc.classList.contains('panel-active'));
  click(win, hdr);
  assert.strictEqual(prev.textContent, 'list files');

  send(win, {type: 'tool_result', name: 'Bash', content: 'ok', tabId: tab, ts: 2000});
  assert.ok(!tc.classList.contains('panel-active'), 'the result ends the pulse');
  assert.strictEqual(name.textContent, 'Bash');
  assert.strictEqual(prev.textContent, 'list files');
  win.close();
});

test('the Thoughts panel pulses while the model thinks and writes', () => {
  const {win} = makeWebview({remote: false});
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab, startTs: 1000});
  send(win, {type: 'thinking_start', tabId: tab, ts: 1000});
  send(win, {type: 'thinking_delta', text: 'hmm', tabId: tab});
  const panel = byId(win, 'output').querySelector('.llm-panel');
  assert.ok(panel.classList.contains('panel-active'), 'active from the first token');
  assert.ok(panel.querySelector(':scope > .llm-panel-hdr'), 'the pulsing header');
  send(win, {type: 'thinking_end', tabId: tab});
  send(win, {type: 'text_delta', text: 'Done.', tabId: tab});
  send(win, {type: 'text_end', tabId: tab, ts: 1500});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: tab, ts: 2000});
  assert.ok(!panel.classList.contains('panel-active'), 'the next event ends it');
  win.close();
});

test('Question and fan-out panels start open; the task end stops every pulse', () => {
  const {win} = makeWebview({remote: false});
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab, startTs: 1000});
  send(win, {
    type: 'tool_call',
    name: 'ask_user_question',
    question: 'Proceed?',
    tabId: tab,
    ts: 1000,
  });
  send(win, {type: 'tool_result', name: 'ask_user_question', content: 'yes', tabId: tab, ts: 1100});
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tasks: ['one', 'two'],
    tabId: tab,
    ts: 1200,
  });
  const O = byId(win, 'output');
  const q = O.querySelector('.tc-question');
  const rp = O.querySelector('.tc-run-parallel');
  assert.ok(q && !q.classList.contains('collapsed'), 'a Question is read, not folded');
  assert.ok(rp && !rp.classList.contains('collapsed'), 'a fan-out keeps its sub-agent tabs open');
  assert.ok(rp.classList.contains('panel-active'));
  assert.ok(!q.classList.contains('panel-active'), 'the answered Question is done');
  // The task stops without a tool_result (or task_done) for the
  // fan-out: the bare status flip ends the pulse.
  send(win, {type: 'status', running: false, tabId: tab});
  assert.strictEqual(O.querySelectorAll('.panel-active').length, 0);
  // Not sealed, though: a refused submit on a busy tab sends the same
  // status while the task runs on, so a late tool_result still times it.
  assert.ok(!rp.dataset.timeDone, 'paused, not sealed');
  send(win, {type: 'tool_result', name: 'run_parallel', content: 'ok', tabId: tab, ts: 9000});
  assert.strictEqual(rp.dataset.timeDone, '1', 'the result seals it');
  assert.ok(rp.querySelector('.panel-elapsed'), 'with its duration');
  win.close();
});

test('the panel close button hides the docked panel; focus follows the controls', () => {
  const {win, fireChange} = makeWebview();
  const panel = byId(win, 'meta-panel');
  const drawer = byId(win, 'meta-drawer');
  const close = byId(win, 'meta-close');
  openChatAndFile(win);
  // A panel wide enough to cover the whole pane leaves nothing to
  // press there: its close button hides it.
  close.focus();
  assert.strictEqual(win.document.activeElement, close);
  click(win, close);
  assert.ok(hasClass(win, 'meta-hidden'), 'the close button hides the panel');
  assert.strictEqual(
    win.document.activeElement,
    drawer,
    'focus moves to the drawer (the panel went inert)',
  );
  click(win, drawer);
  assert.ok(!hasClass(win, 'meta-hidden'));
  assert.strictEqual(
    win.document.activeElement,
    close,
    'focus leaves the faded drawer for the panel',
  );
  // Focus elsewhere stays where it is.
  byId(win, 'task-input').focus();
  press(win, byId(win, 'content-tab-area'));
  assert.ok(hasClass(win, 'meta-hidden'));
  assert.strictEqual(win.document.activeElement, byId(win, 'task-input'));
  click(win, drawer);
  assert.strictEqual(win.document.activeElement, byId(win, 'task-input'));
  // On the phone the same button closes the drawer.
  fireChange(false);
  click(win, byId(win, 'meta-drawer-btn'));
  assert.ok(panel.classList.contains('open'), 'the phone drawer opened');
  click(win, close);
  assert.ok(!panel.classList.contains('open'), 'the phone drawer closed');
  assert.ok(!hasClass(win, 'meta-hidden'));
  win.close();
});

test('a copied running panel reads the tool name as one word', () => {
  const {win} = makeWebview({remote: false});
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tab, startTs: 1000});
  send(win, {type: 'tool_call', name: 'Bash', command: 'ls', tabId: tab, ts: 1000});
  const tc = byId(win, 'output').querySelector('.ev.tc');
  assert.ok(win.PanelCopy, 'panelCopy.js loaded');
  const copied = win.PanelCopy.getRawText(tc.querySelector(':scope > .tc-h'));
  assert.strictEqual(copied.split('\n')[0], 'Bash', 'copied as a word');
  win.close();
});

test('replaying a finished task leaves no header pulsing', () => {
  const {win} = makeWebview({remote: false});
  const tab = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: false, tabId: tab});
  // A call that never got its result (the task was stopped); replay
  // skips the terminal event that would have closed it.
  send(win, {
    type: 'task_events',
    tabId: tab,
    task: 'the task',
    events: [
      {type: 'tool_call', name: 'Bash', command: 'sleep 99', ts: 1000},
      {type: 'task_stopped', ts: 2000},
    ],
  });
  const O = byId(win, 'output');
  assert.ok(O.querySelector('.tc'), 'the call replayed');
  assert.strictEqual(O.querySelectorAll('.panel-active').length, 0);
  win.close();
});

test('the stylesheets carry the layout, the pulse and the machine strip', () => {
  const remote = inlineDesignTokens(
    fs.readFileSync(path.join(MEDIA, 'remote-codex.css'), 'utf8'),
  );
  const main = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  function rule(css, selector) {
    const src = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    // Anchored at a line start, so `#x` does not match `body.y #x`.
    const re = new RegExp(
      String.raw`(?:^|\n)\s*` + src + String.raw`\s*(?:,[^{]*)?\{([^}]*)\}`,
      'g',
    );
    // Every rule with exactly this selector, joined (a media query
    // may restate one).
    const bodies = [];
    let m;
    while ((m = re.exec(css)) !== null) bodies.push(m[1]);
    assert.ok(bodies.length > 0, 'CSS rule missing: ' + selector);
    return bodies.join('\n');
  }
  // No content tab: one chat column, clear of the docked panel.
  const app = rule(remote, 'body.remote-chat.remote-desktop #app');
  assert.ok(/grid-template-columns:\s*minmax\(0, 1fr\);/.test(app));
  assert.ok(/margin-right:\s*var\(--meta-w/.test(app));
  // A content tab: the split grid runs under the panel.
  const split = rule(remote, 'body.remote-chat.remote-desktop.content-pane-open #app');
  assert.ok(/margin-right:\s*0;/.test(split));
  assert.ok(/'machine\s+handle ctabs'/.test(split));
  const hiddenPane = rule(
    remote,
    'body.remote-chat.remote-desktop:not(.content-pane-open) #pane-resizer',
  );
  assert.ok(/display:\s*none/.test(hiddenPane));
  // The slide and the drawer.
  const panel = rule(remote, 'body.remote-chat.remote-desktop #meta-panel');
  assert.ok(/position:\s*fixed/.test(panel));
  assert.ok(/transition:\s*transform/.test(panel));
  const slid = rule(
    remote,
    'body.remote-chat.remote-desktop.content-pane-open.meta-hidden #meta-panel',
  );
  assert.ok(/transform:\s*translateX\(100%\)/.test(slid));
  const drawer = rule(
    remote,
    'body.remote-chat.remote-desktop.content-pane-open #meta-drawer',
  );
  assert.ok(/position:\s*fixed/.test(drawer) && /right:\s*0/.test(drawer));
  assert.ok(/opacity:\s*0/.test(drawer) && /transition:\s*opacity/.test(drawer));
  assert.ok(/visibility:\s*hidden/.test(drawer), 'out of the tab order while faded');
  const drawerShown = rule(
    remote,
    'body.remote-chat.remote-desktop.content-pane-open.meta-hidden #meta-drawer',
  );
  assert.ok(/opacity:\s*1/.test(drawerShown) && /visibility:\s*visible/.test(drawerShown));
  // The panel's close button doubles as its "hide" over the pane.
  const closeShown = rule(
    remote,
    'body.remote-chat.remote-desktop.content-pane-open #meta-close',
  );
  assert.ok(/display:\s*block/.test(closeShown));
  assert.ok(/display:\s*none/.test(rule(main, '#meta-drawer')));
  // The machine strip: centred, bold, green, gone while empty.
  const machine = rule(main, '#chat-machine');
  assert.ok(/text-align:\s*center/.test(machine));
  assert.ok(/font-weight:\s*700/.test(machine));
  assert.ok(/color:\s*var\(--green\)/.test(machine));
  assert.ok(/display:\s*none/.test(rule(main, '#chat-machine:empty')));
  // The pulse: an active panel's header text, tool call and Thoughts
  // alike, breathes; it holds still under prefers-reduced-motion.
  const pulse = rule(main, '.panel-active > .tc-h');
  assert.ok(/animation:\s*panel-pulse/.test(pulse));
  assert.ok(/animation:\s*none/.test(pulse));
  assert.ok(/\.panel-active > \.llm-panel-hdr\s*\{/.test(main), 'Thoughts pulse too');
  assert.ok(/@keyframes panel-pulse/.test(main));
  assert.ok(!/wave-ch|tc-running/.test(main), 'the wave is gone');
  // The activity bar is gone from the markup.
  const html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  assert.ok(!/id="activity-bar"/.test(html));
  assert.ok(/id="meta-explorer"/.test(html) && /id="meta-scm"/.test(html));
});

async function main() {
  let failed = 0;
  for (const {name, fn} of TESTS) {
    try {
      await fn();
      console.log('  ok - ' + name);
    } catch (e) {
      failed++;
      console.log('  FAIL - ' + name);
      console.log('      ' + (e.stack || e.message));
    }
  }
  console.log(`${TESTS.length - failed} passed, ${failed} failed`);
  if (failed) process.exit(1);
}

main();
