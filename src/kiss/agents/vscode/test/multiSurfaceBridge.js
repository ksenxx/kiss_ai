// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Test helper: host several REAL chat webviews (media/main.js under
// jsdom), each connected to a running kiss daemon over its local WSS
// endpoint, and expose them to a driving process over stdin/stdout.
//
// Every webview is one "surface" — a VS Code sidebar view, a VS Code
// editor-tab panel, or a browser tab of the remote web app — and each
// gets its OWN daemon connection, exactly like production (the
// extension host forwards webview commands verbatim over its local
// connection; the remote page's acquireVsCodeApi shim posts them over
// its WebSocket).  The connection is the extension's own `WsClient`
// (out/wsClient.js) authenticated with the token from the endpoint
// file named on the command line, like AgentClient.ts.  Everything a
// webview posts is sent as one JSON frame; every event frame the
// daemon sends is dispatched to the webview as a `message` event.
//
// Protocol (one JSON object per line):
//   in : {op:'open', name, bodyAttrs, state}   -> out {op:'opened', name}
//   in : {op:'close', name}                    -> out {op:'closed', name}
//   in : {op:'tabs', name}                     -> out {op:'tabs', name, tabs, errors}
//   in : {op:'submit', name, text}             -> out {op:'submitted', name}
//   in : {op:'closeTab', name, tabId}          -> out {op:'tabClosed', name}
//   in : {op:'events', name}                   -> out {op:'events', name, events}
//   in : {op:'activateTab', name, tabId}       -> out {op:'tabActivated', name, found}
//   in : {op:'post', name, msg}                -> out {op:'posted', name}
//   in : {op:'click', name, selector}          -> out {op:'clicked', name, found}
//   in : {op:'hold', name}                     -> out {op:'held', name}
//   in : {op:'release', name}                  -> out {op:'released', name, delivered}
//   in : {op:'ask', name}                      -> out {op:'ask', name, activeTabId,
//                                                     answering, question, attention}
//   in : {op:'answer', name, text}             -> out {op:'answered', name, answering}
//   in : {op:'browser', name, tabId}           -> out {op:'browser', name, info}
//   in : {op:'browserKeys', name, keys}        -> out {op:'browserKeys', name, found}
//   in : {op:'browserClick', name, x, y}       -> out {op:'browserClick', name, found}
//   in : {op:'disconnect', name}               -> out {op:'disconnected', name}
//   in : {op:'reconnect', name}                -> out {op:'reconnected', name}
//   in : {op:'quit'}
//
// `ask` reports the ask_user_question state as the user sees it: whether
// the composer of the ACTIVE tab is the answer box (body.ask-answering),
// the text of that tab's pending question panel, and the ids of the
// background tabs waiting for an answer (drawn as the "?" mark on their
// strip entry, or on their chat's main-row entry while that group is off
// screen).
// `answer` types into the composer and presses Send — the same path a
// user's answer takes — and reports whether the composer was the answer
// box at that moment.
//
// `tabs` describes every open tab: the webview's own tab records
// (_testApi.openTabs) joined with what the rendered tab rows show of
// each (title, spinner, done tick).

const fs = require('fs');
const path = require('path');
const readline = require('readline');
const {JSDOM, VirtualConsole} = require('jsdom');
const {WsClient} = require(path.join(__dirname, '..', 'out', 'wsClient.js'));

const MEDIA = path.join(__dirname, '..', 'media');
const ENDPOINT_FILE = process.argv[2];
if (!ENDPOINT_FILE) {
  process.stderr.write(
    'usage: node multiSurfaceBridge.js <sorcar-local.json>\n',
  );
  process.exit(2);
}
const CONNECT_TIMEOUT_MS = 10000;

// The daemon's endpoint file (`kiss.agents.sorcar.local_endpoint`):
// WSS URL, per-start local token and the CA to trust.  Read per
// connection, as the extension does, so a restarted daemon's fresh
// token is picked up by `reconnect`.
function readEndpoint() {
  const rec = JSON.parse(fs.readFileSync(ENDPOINT_FILE, 'utf8'));
  return {
    url: rec.url,
    token: rec.token,
    ca: rec.ca ? fs.readFileSync(rec.ca, 'utf8') : undefined,
  };
}

const surfaces = new Map();
const DETAILED_EVENTS = new Set([
  'new_tab',
  'openSubagentTab',
  'subagentDone',
  'closeSubagentTab',
  'askUser',
  'askUserDone',
  'openBrowserTab',
  'browserTabs',
  'browserState',
  'closeBrowserTab',
  'browserError',
  'error',
]);

// jsdom lays nothing out, so a streamed browser surface would report a
// 0x0 viewport and never ask for frames: give .browser-screen a size.
const BROWSER_SCREEN = {width: 640, height: 480};

function out(obj) {
  process.stdout.write(JSON.stringify(obj) + '\n');
}

function makeWebview(bodyAttrs, initialState, onPost, errors) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace('{{BODY_CLASS_ATTR}}', bodyAttrs || '');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const virtualConsole = new VirtualConsole();
  virtualConsole.on('jsdomError', e => {
    errors.push(String((e && e.stack) || e));
  });
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
    virtualConsole,
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  Object.defineProperty(win.HTMLElement.prototype, 'clientWidth', {
    get() {
      return this.classList.contains('browser-screen')
        ? BROWSER_SCREEN.width
        : 0;
    },
  });
  Object.defineProperty(win.HTMLElement.prototype, 'clientHeight', {
    get() {
      return this.classList.contains('browser-screen')
        ? BROWSER_SCREEN.height
        : 0;
    },
  });

  let state = initialState || null;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: onPost,
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'browserTab.js'), 'utf8'));
  // The sourceURL names the script in V8 coverage output
  // (NODE_V8_COVERAGE) so a driver can measure main.js branches.
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=multiSurfaceBridge-main.js',
  );
  return {win, getState: () => state};
}

// One daemon connection for `surface`: the webview's posts go out as
// JSON frames once the local token is accepted, the daemon's event
// frames come back as `message` events.
function connectSocket(surface, onConnect) {
  const endpoint = readEndpoint();
  const socket = new WsClient({
    url: endpoint.url,
    ca: endpoint.ca,
    connectTimeoutMs: CONNECT_TIMEOUT_MS,
  });
  const pending = [];
  let authenticated = false;
  socket.on('open', () => {
    socket.send(JSON.stringify({type: 'auth', token: endpoint.token}));
  });
  surface.socket = socket;
  surface.post = msg => {
    const line = JSON.stringify(msg);
    if (authenticated) socket.send(line);
    else pending.push(line);
  };
  // What the webview posted before this socket existed (main.js sends
  // `ready` while it is still being evaluated).
  surface.queued.splice(0).forEach(surface.post);
  socket.on('message', text => {
    let ev;
    try {
      ev = JSON.parse(text);
    } catch (_e) {
      return;
    }
    if (!authenticated) {
      if (ev.type === 'auth_ok' && ev.local === true) {
        authenticated = true;
        pending.splice(0).forEach(line => socket.send(line));
        onConnect();
      } else {
        surface.errors.push('auth: ' + JSON.stringify(ev));
      }
      return;
    }
    // A held surface (a slow client: a phone tab in the background)
    // keeps the daemon's frames until `release` delivers them in order.
    if (surface.held) surface.held.push(ev);
    else deliver(surface, ev);
  });
  socket.on('error', e => surface.errors.push('socket: ' + String(e)));
  socket.connect();
}

function deliver(surface, ev) {
  if (ev.type === 'browserFrame') surface.frameCount += 1;
  // Tab-lifecycle events are kept whole (the driver asserts on their
  // fields); everything else only by type and routing keys.
  surface.events.push(
    DETAILED_EVENTS.has(ev.type)
      ? ev
      : {type: ev.type, tabId: ev.tabId, taskId: ev.taskId},
  );
  dispatchToWebview(surface, ev);
}

function dispatchToWebview(surface, ev) {
  try {
    surface.view.win.dispatchEvent(
      new surface.view.win.MessageEvent('message', {data: ev}),
    );
  } catch (e) {
    surface.errors.push(
      'dispatch ' + ev.type + ': ' + String((e && e.stack) || e),
    );
  }
}

function openSurface(cmd) {
  const surface = {
    view: null,
    socket: null,
    queued: [],
    post: msg => surface.queued.push(msg),
    errors: [],
    events: [],
    held: null,
    frameCount: 0,
    frames: () => surface.frameCount,
  };
  surface.view = makeWebview(
    cmd.bodyAttrs,
    cmd.state,
    msg => surface.post(msg),
    surface.errors,
  );
  surfaces.set(cmd.name, surface);
  // Like the shim after `auth_ok` (and the VS Code host on connect):
  // tell the app the daemon is up, so a later drop counts as an
  // outage whose reconnect re-sends `ready`.
  connectSocket(surface, () => {
    dispatchToWebview(surface, {type: 'daemonStatus', connected: true});
    out({op: 'opened', name: cmd.name});
  });
}

// The remote webapp's shim keeps the page up while its WebSocket is
// down (daemonStatus connected:false, reconnecting:true), then
// re-authenticates a fresh socket and tells the app it is back, upon
// which main.js re-sends `ready`.  Mirror the two halves so a driver
// can act on other surfaces while this one is away.
function disconnectSurface(cmd, surface) {
  surface.socket.destroy();
  surface.post = () => {}; // dropped, as on a dead WebSocket
  dispatchToWebview(surface, {
    type: 'daemonStatus',
    connected: false,
    reconnecting: true,
  });
  out({op: 'disconnected', name: cmd.name});
}

function reconnectSurface(cmd, surface) {
  connectSocket(surface, () => {
    dispatchToWebview(surface, {type: 'daemonStatus', connected: true});
    out({op: 'reconnected', name: cmd.name});
  });
}

// The streamed browser tab `tabId` as the webview shows it: tab-strip
// entry, address bar, badge, whether a frame has been painted, and how
// many frames this surface received in total.
function describeBrowser(surface, tabId) {
  const {win} = surface.view;
  const strip = tabEl(win.document, tabId);
  const holder = win.document.querySelector(
    `#content-tab-area .browser-tab-view[data-tab-id="${tabId}"]`,
  );
  const img = holder && holder.querySelector('.browser-frame');
  return {
    inTabBar: !!strip,
    isBrowserTab: !!(
      strip &&
      strip.querySelector('.content-tab-icon') &&
      strip.querySelector('.content-tab-icon').textContent === '\uD83C\uDF10'
    ),
    title: strip
      ? (strip.querySelector('.chat-tab-label') || {}).textContent || ''
      : '',
    url: holder ? holder.querySelector('.browser-url').value : null,
    badge: holder ? holder.querySelector('.browser-badge').textContent : null,
    hasFrame: !!(img && img.src && img.src.indexOf('data:image/jpeg') === 0),
    visible: !!(holder && holder.style.display !== 'none'),
    frames: surface.frames(),
  };
}

function shownBrowserScreen(surface) {
  const holders = Array.from(
    surface.view.win.document.querySelectorAll(
      '#content-tab-area .browser-tab-view',
    ),
  );
  const shown = holders.find(h => h.style.display !== 'none');
  return shown ? shown.querySelector('.browser-screen') : null;
}

// There is no row of chat tabs: the group strip (#tab-list) renders
// the chat on screen with its sub-agents (and, on a stacked surface,
// every content tab), and the desktop remote's split layout lists the
// content tabs on #content-tab-list instead.  A background chat and its
// sub-agents are open but not rendered, so the webview's own tab records
// (_testApi.openTabs) are the list of open tabs and the rows supply what
// the user sees of each.
function allTabEls(doc) {
  const seen = new Set();
  return Array.from(
    doc.querySelectorAll(
      '#tab-list [data-tab-id], #content-tab-list [data-tab-id]',
    ),
  ).filter(el => {
    if (seen.has(el.dataset.tabId)) return false;
    seen.add(el.dataset.tabId);
    return true;
  });
}

function renderedTabEl(doc, tabId) {
  return (
    doc.querySelector(`#tab-list [data-tab-id="${tabId}"]`) ||
    doc.querySelector(`#content-tab-list [data-tab-id="${tabId}"]`)
  );
}

// The element of tab `tabId` on whichever row shows it.  A tab of a
// background group is reached the way the user reaches it: its chat is
// picked (the Chats panel's click, _testApi.switchToTab) so the group
// comes on screen first and the strip renders the tab.
function tabEl(doc, tabId) {
  const el = renderedTabEl(doc, tabId);
  if (el) return el;
  const win = doc.defaultView;
  const record = (win._testApi ? win._testApi.openTabs() : []).find(
    t => t.id === tabId,
  );
  if (!record) return null;
  win._testApi.switchToTab(record.rootId);
  return renderedTabEl(doc, tabId);
}

// One entry per open tab.  A sub-agent tab's id is
// `${parentTabId}__sub_${taskId}` (main.js subagentTabIdFor), so the
// driver derives parent and task from the id.
function describeTabs(surface) {
  const {win} = surface.view;
  const doc = win.document;
  return win._testApi.openTabs().map(t => {
    const el = renderedTabEl(doc, t.id);
    const indicator = el && el.querySelector('.subagent-indicator');
    return {
      id: t.id,
      title: el
        ? (el.querySelector('.chat-tab-label') || {}).textContent || ''
        : t.title,
      isSubagentTab: t.isSubagentTab,
      subagentDone: el
        ? !!(indicator && indicator.classList.contains('done'))
        : t.isSubagentTab && t.isDone,
      running: el
        ? !!el.querySelector('.status-spinner')
        : t.isSubagentTab
          ? !t.isDone
          : t.isRunning,
    };
  });
}

function describeAsk(surface) {
  const doc = surface.view.win.document;
  const active = doc.querySelector('#tab-list [data-tab-id].active');
  const panel = doc.querySelector('#output .tc-question.tc-question-pending');
  return {
    activeTabId: active ? active.dataset.tabId : '',
    answering: doc.body.classList.contains('ask-answering'),
    question: panel ? panel.textContent : '',
    // The tabs waiting for an answer other than the one on screen: the
    // mark is drawn on the asking tab's strip entry, or on its chat's
    // main-row entry while that group is off screen, so the webview's
    // records name the asking tabs themselves.
    attention: doc.defaultView._testApi
      .openTabs()
      .filter(t => t.askPending && !(active && active.dataset.tabId === t.id))
      .map(t => t.id),
  };
}

function handle(cmd) {
  if (cmd.op === 'quit') {
    surfaces.forEach(s => s.socket.destroy());
    process.exit(0);
  }
  if (cmd.op === 'open') {
    openSurface(cmd);
    return;
  }
  const surface = surfaces.get(cmd.name);
  if (!surface) {
    out({op: 'error', name: cmd.name, error: 'no such surface'});
    return;
  }
  const doc = surface.view.win.document;
  switch (cmd.op) {
    case 'close':
      surface.socket.destroy();
      surface.view.win.close();
      surfaces.delete(cmd.name);
      out({op: 'closed', name: cmd.name});
      break;
    case 'tabs':
      out({
        op: 'tabs',
        name: cmd.name,
        tabs: describeTabs(surface),
        errors: surface.errors,
      });
      break;
    case 'submit':
      doc.getElementById('task-input').value = cmd.text;
      doc.getElementById('send-btn').click();
      out({op: 'submitted', name: cmd.name});
      break;
    case 'closeTab': {
      const tab = tabEl(doc, cmd.tabId);
      const el = tab && tab.querySelector('.chat-tab-close');
      if (el) el.click();
      out({op: 'tabClosed', name: cmd.name, found: !!el});
      break;
    }
    case 'events':
      out({op: 'events', name: cmd.name, events: surface.events});
      break;
    case 'activateTab': {
      const el = tabEl(doc, cmd.tabId);
      if (el) el.click();
      out({op: 'tabActivated', name: cmd.name, found: !!el});
      break;
    }
    case 'disconnect':
      disconnectSurface(cmd, surface);
      break;
    case 'reconnect':
      reconnectSurface(cmd, surface);
      break;
    case 'post':
      surface.post(cmd.msg);
      out({op: 'posted', name: cmd.name});
      break;
    case 'click': {
      // `text` narrows the match to the first element whose text
      // contains it (history rows carry no id attribute).
      const el = cmd.text
        ? Array.from(doc.querySelectorAll(cmd.selector)).find(e =>
            (e.textContent || '').includes(cmd.text),
          )
        : doc.querySelector(cmd.selector);
      if (el) el.click();
      out({op: 'clicked', name: cmd.name, found: !!el});
      break;
    }
    case 'hold':
      if (!surface.held) surface.held = [];
      out({op: 'held', name: cmd.name});
      break;
    case 'release': {
      const queued = surface.held || [];
      surface.held = null;
      queued.forEach(ev => deliver(surface, ev));
      out({op: 'released', name: cmd.name, delivered: queued.length});
      break;
    }
    case 'ask':
      out({op: 'ask', name: cmd.name, ...describeAsk(surface)});
      break;
    case 'answer': {
      const answering = doc.body.classList.contains('ask-answering');
      doc.getElementById('task-input').value = cmd.text;
      doc.getElementById('send-btn').click();
      out({op: 'answered', name: cmd.name, answering});
      break;
    }
    case 'browser':
      out({
        op: 'browser',
        name: cmd.name,
        info: describeBrowser(surface, cmd.tabId),
      });
      break;
    case 'browserKeys': {
      // Type into the SHOWN browser surface the way a user would: one
      // keydown/keyup pair per key (single characters, 'Enter', ...).
      const screen = shownBrowserScreen(surface);
      if (screen) {
        const {win} = surface.view;
        cmd.keys.forEach(k => {
          const init = {
            key: k,
            code: k.length === 1 ? 'Key' + k.toUpperCase() : k,
            keyCode: k === 'Enter' ? 13 : k.toUpperCase().charCodeAt(0),
            bubbles: true,
          };
          screen.dispatchEvent(new win.KeyboardEvent('keydown', init));
          screen.dispatchEvent(new win.KeyboardEvent('keyup', init));
        });
      }
      out({op: 'browserKeys', name: cmd.name, found: !!screen});
      break;
    }
    case 'browserClick': {
      // A left click at page coordinates (x, y): jsdom has no layout, so
      // browserTab.js maps client coordinates 1:1 onto the page.
      const screen = shownBrowserScreen(surface);
      if (screen) {
        const {win} = surface.view;
        const init = {clientX: cmd.x, clientY: cmd.y, button: 0, bubbles: true};
        screen.dispatchEvent(
          new win.MouseEvent('pointerdown', {...init, buttons: 1}),
        );
        screen.dispatchEvent(
          new win.MouseEvent('pointerup', {...init, buttons: 0}),
        );
      }
      out({op: 'browserClick', name: cmd.name, found: !!screen});
      break;
    }
    default:
      out({op: 'error', name: cmd.name, error: 'unknown op ' + cmd.op});
  }
}

readline.createInterface({input: process.stdin}).on('line', line => {
  if (!line.trim()) return;
  handle(JSON.parse(line));
});
