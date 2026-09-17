// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// JSDOM integration test: a chat panel that shows an image (a tool
// result's embedded screenshot / chart) is never collapsed or hidden by
// the webview's automatic passes, on any surface:
//
//   * collapseOlderPanels  -- while a task streams, older panels fold as
//     new ones arrive; an image panel must stay open.
//   * collapseAllExceptResult -- a replayed transcript folds everything
//     but the result; an image panel must stay open.
//   * applyChevronState -- a finished task's plain tool panels are taken
//     off screen (chv-hidden); an image panel must stay on screen.
//
// Only the user's own click on the chevron folds such a panel.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// A real, valid 1x1 transparent PNG.
const PNG_B64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg' +
  'YGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==';

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
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

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function startTask(wv, chatId) {
  const ready = wv.posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  send(wv.win, {type: 'clear', chat_id: chatId, tabId: ready.tabId});
  send(wv.win, {
    type: 'status',
    running: true,
    tabId: ready.tabId,
    startTs: Date.now(),
  });
  return ready.tabId;
}

function toolPanels(win) {
  return Array.from(win.document.querySelectorAll('#output > .ev.tc'));
}

const IMAGE = {path: '/w/shot.png', mime: 'image/png', b64: PNG_B64};

function testLiveStreamKeepsImagePanelOpen() {
  const wv = makeWebview();
  const win = wv.win;
  const TAB = startTask(wv, 'chat-imgpanel-1');

  // Panel 1: a plain Read.  Panel 2: a screenshot with an image.
  send(win, {type: 'tool_call', name: 'Read', path: 'a.txt', tabId: TAB});
  send(win, {
    type: 'tool_result',
    content: 'aaa',
    is_error: false,
    tool_name: 'Read',
    tabId: TAB,
  });
  send(win, {
    type: 'tool_call',
    name: 'screenshot',
    path: 'shot.png',
    tabId: TAB,
  });
  send(win, {
    type: 'tool_result',
    content: 'saved',
    is_error: false,
    tool_name: 'screenshot',
    images: [IMAGE],
    tabId: TAB,
  });
  // Panel 3 arrives: the automatic pass folds older panels.
  send(win, {type: 'tool_call', name: 'Read', path: 'b.txt', tabId: TAB});
  send(win, {
    type: 'tool_result',
    content: 'bbb',
    is_error: false,
    tool_name: 'Read',
    tabId: TAB,
  });
  // Panel 4 arrives too, so panel 3 (no image) is also folded.
  send(win, {type: 'tool_call', name: 'Read', path: 'c.txt', tabId: TAB});

  const panels = toolPanels(win);
  assert.strictEqual(panels.length, 4, 'four tool panels must render');
  assert.ok(panels[1].querySelector('img.tr-img'), 'panel 2 shows the image');
  assert.ok(
    panels[0].classList.contains('collapsed'),
    'a plain older panel is folded as the stream moves on',
  );
  assert.ok(
    panels[2].classList.contains('collapsed'),
    'a plain older panel is folded as the stream moves on',
  );
  assert.ok(
    !panels[1].classList.contains('collapsed'),
    'BUG: the panel showing an image was auto-collapsed while streaming',
  );

  // The task finishes: plain panels go off screen, the image stays.
  send(win, {
    type: 'result',
    text: 'done',
    summary: 'done',
    success: true,
    tabId: TAB,
  });
  send(win, {type: 'status', running: false, tabId: TAB});
  assert.ok(
    !panels[1].classList.contains('chv-hidden'),
    'BUG: the panel showing an image was hidden when the task finished',
  );
  assert.ok(
    !panels[1].classList.contains('collapsed'),
    'the image panel stays open after the task finishes',
  );

  // Only the user's own chevron click folds it.
  const chevron = panels[1].querySelector(':scope > .collapse-header');
  assert.ok(chevron, 'the panel has a clickable header');
  chevron.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    panels[1].classList.contains('collapsed'),
    'the user can still fold an image panel by hand',
  );
  console.log('ok: a streaming image panel is never auto-collapsed or hidden');
}

function testReplayKeepsImagePanelOpenAndVisible() {
  const wv = makeWebview();
  const win = wv.win;
  const events = [{type: 'prompt', text: 'replayed task'}];
  for (let i = 0; i < 3; i++) {
    events.push({type: 'tool_call', name: 'Read', path: '/tmp/r' + i});
    events.push({type: 'tool_result', name: 'Read', content: 'x' + i});
  }
  events.push({type: 'tool_call', name: 'screenshot', path: 'shot.png'});
  events.push({
    type: 'tool_result',
    tool_name: 'screenshot',
    content: 'saved',
    images: [IMAGE],
  });
  events.push({type: 'tool_call', name: 'Read', path: '/tmp/last'});
  events.push({type: 'tool_result', name: 'Read', content: 'last'});
  events.push({type: 'result', text: 'done', summary: 'done', success: true});
  send(win, {type: 'task_events', task: 'replayed task', task_id: 7, events});

  const panels = toolPanels(win);
  assert.strictEqual(panels.length, 5, 'five tool panels replay');
  const imagePanel = panels[3];
  assert.ok(
    imagePanel.querySelector('img.tr-img'),
    'the screenshot panel shows its image',
  );
  const plain = [panels[0], panels[1], panels[2], panels[4]];
  assert.ok(
    plain.every(p => p.classList.contains('collapsed')),
    'plain replayed tool panels are folded',
  );
  assert.ok(
    plain.every(p => p.classList.contains('chv-hidden')),
    'plain replayed tool panels of a finished task are taken off screen',
  );
  assert.ok(
    !imagePanel.classList.contains('collapsed'),
    'BUG: the replayed panel showing an image was auto-collapsed',
  );
  assert.ok(
    !imagePanel.classList.contains('chv-hidden'),
    'BUG: the replayed panel showing an image was hidden',
  );
  console.log('ok: a replayed image panel stays open and on screen');
}

// A `summary` tool call adopts every panel before it into its own
// collapsed digest.  When one of those shows an image, the digest must
// stay open (the picture would otherwise vanish behind the fold).
function testSummaryAdoptionKeepsImageVisible() {
  const wv = makeWebview();
  const win = wv.win;
  const TAB = startTask(wv, 'chat-imgpanel-3');
  send(win, {
    type: 'tool_call',
    name: 'screenshot',
    path: 'shot.png',
    tabId: TAB,
  });
  send(win, {
    type: 'tool_result',
    content: 'saved',
    is_error: false,
    tool_name: 'screenshot',
    images: [IMAGE],
    tabId: TAB,
  });
  send(win, {
    type: 'tool_call',
    name: 'summary',
    description: 'so far',
    tabId: TAB,
  });
  const summary = win.document.querySelector('#output .tc-summary');
  assert.ok(summary, 'the summary panel renders');
  assert.ok(
    summary.querySelector('.summary-sub img.tr-img'),
    'the summary adopted the image panel',
  );
  assert.ok(
    !summary.classList.contains('collapsed'),
    'BUG: a summary that adopted an image panel was auto-collapsed',
  );
  // Without an image the digest folds as before.
  send(win, {type: 'tool_call', name: 'Read', path: 'x.txt', tabId: TAB});
  send(win, {
    type: 'tool_result',
    content: 'x',
    is_error: false,
    tool_name: 'Read',
    tabId: TAB,
  });
  send(win, {
    type: 'tool_call',
    name: 'summary',
    description: 'later',
    tabId: TAB,
  });
  const summaries = win.document.querySelectorAll('#output .tc-summary');
  assert.strictEqual(summaries.length, 2);
  assert.ok(
    summaries[1].classList.contains('collapsed'),
    'a summary adopting only text panels still folds',
  );
  console.log('ok: a summary that adopts an image panel stays open');
}

function main() {
  testLiveStreamKeepsImagePanelOpen();
  testReplayKeepsImagePanelOpenAndVisible();
  testSummaryAdoptionKeepsImageVisible();
  console.log('All tests passed');
  // The webview keeps ticking panel-time intervals; exit explicitly.
  process.exit(0);
}

try {
  main();
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
