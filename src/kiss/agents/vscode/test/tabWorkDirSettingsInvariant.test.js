// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

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
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function lastMsg(posted, type) {
  for (let i = posted.length - 1; i >= 0; i -= 1) {
    if (posted[i] && posted[i].type === type) return posted[i];
  }
  return null;
}

function initialTabId(posted) {
  const ready = posted.find(m => m && m.type === 'ready');
  return ready ? ready.tabId : '';
}

function workDirOnMentionAfterSettingsChange(initialWd, newWd, bindKind) {
  const {win, posted} = makeWebview();
  const tabId = initialTabId(posted);
  assert.ok(tabId, 'main.js must announce the initial tab id');

  send(win, {
    type: 'configData',
    config: {work_dir: initialWd},
    apiKeys: {},
  });

  if (bindKind === 'clear') {
    send(win, {type: 'clear', tabId: tabId, chat_id: 'chat-real-task'});
  } else if (bindKind === 'task_events_no_extra_workdir') {
    send(win, {
      type: 'task_events',
      tabId: tabId,
      chat_id: 'chat-real-task',
      task_id: 42,
      task: 'Real persisted task',
      events: [],
      extra: JSON.stringify({startTs: 1_700_000_000_000}),
    });
  } else {
    throw new Error('unknown bindKind: ' + bindKind);
  }

  send(win, {
    type: 'configData',
    config: {work_dir: newWd},
    apiKeys: {},
  });

  const inp = win.document.getElementById('task-input');
  inp.value = '@readm';
  inp.dispatchEvent(new win.window.Event('input', {bubbles: true}));

  const msg = lastMsg(posted, 'getFiles');
  assert.ok(msg, 'typing an @-mention must post a getFiles command');
  win.close();
  return msg.workDir;
}

function testInvariantHoldsAfterSettingsChange_ClearBind() {
  const wd = workDirOnMentionAfterSettingsChange(
    '/path/initial',
    '/path/new',
    'clear',
  );
  assert.strictEqual(
    wd,
    '/path/initial',
    'INVARIANT: after the user changes work_dir in settings, a tab ' +
      'whose backend chat-id was bound via a "clear" event MUST keep ' +
      'routing commands to the work_dir it had when the chat-id was ' +
      'bound — observed workDir = ' +
      JSON.stringify(wd),
  );
  console.log('  ok - clear-bound tab keeps its original work_dir after settings change');
}

function testInvariantHoldsAfterSettingsChange_TaskEventsBindNoExtraWorkdir() {
  const wd = workDirOnMentionAfterSettingsChange(
    '/path/initial',
    '/path/new',
    'task_events_no_extra_workdir',
  );
  assert.strictEqual(
    wd,
    '/path/initial',
    'INVARIANT: after the user changes work_dir in settings, a tab ' +
      'whose backend chat-id was bound via a "task_events" replay (and ' +
      'whose persisted "extra" carries no ``work_dir`` — older rows) ' +
      'MUST keep routing commands to the work_dir it had when the ' +
      'chat-id was bound — observed workDir = ' +
      JSON.stringify(wd),
  );
  console.log(
    '  ok - task_events-bound tab keeps its original work_dir after settings change',
  );
}

function testTaskEventsExtraWorkDirStillWinsOverConfig() {
  const {win, posted} = makeWebview();
  const tabId = initialTabId(posted);

  send(win, {
    type: 'configData',
    config: {work_dir: '/path/initial'},
    apiKeys: {},
  });

  send(win, {
    type: 'task_events',
    tabId: tabId,
    chat_id: 'chat-real-task',
    task_id: 42,
    task: 'Real persisted task',
    events: [],
    extra: JSON.stringify({work_dir: '/path/task-recorded'}),
  });

  send(win, {
    type: 'configData',
    config: {work_dir: '/path/new'},
    apiKeys: {},
  });

  const inp = win.document.getElementById('task-input');
  inp.value = '@readm';
  inp.dispatchEvent(new win.window.Event('input', {bubbles: true}));
  const msg = lastMsg(posted, 'getFiles');
  assert.strictEqual(
    msg.workDir,
    '/path/task-recorded',
    'extra.work_dir must pin the tab even when configWorkDir later changes',
  );
  win.close();
  console.log('  ok - extra.work_dir pin survives settings change');
}

function main() {
  testInvariantHoldsAfterSettingsChange_ClearBind();
  testInvariantHoldsAfterSettingsChange_TaskEventsBindNoExtraWorkdir();
  testTaskEventsExtraWorkDirStillWinsOverConfig();
  console.log('tabWorkDirSettingsInvariant.test.js: all assertions passed.');
}

main();
