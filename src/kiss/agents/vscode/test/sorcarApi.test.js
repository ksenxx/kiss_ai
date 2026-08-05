// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

const assert = require('assert');
const {test} = require('node:test');
const path = require('path');

const {createSorcarApi, SORCAR_API_COMMANDS} = require('../media/api.js');
const {makeWebview} = require('./simplify2_harness.js');

test('every api.js method posts its catalog command', () => {
  const posted = [];
  const api = createSorcarApi(msg => posted.push(msg));
  SORCAR_API_COMMANDS.forEach(name => {
    assert.strictEqual(typeof api[name], 'function', name);
    api[name]();
    assert.deepStrictEqual(posted.pop(), {type: name});
    api[name]({tabId: 't1', extra: 42});
    assert.deepStrictEqual(posted.pop(), {type: name, tabId: 't1', extra: 42});
  });
  assert.strictEqual(posted.length, 0);
});

test('api.send forwards catalog commands and rejects the rest', () => {
  const posted = [];
  const api = createSorcarApi(msg => posted.push(msg));
  api.send({type: 'stop', tabId: 'tab-9'});
  assert.deepStrictEqual(posted, [{type: 'stop', tabId: 'tab-9'}]);
  assert.throws(() => api.send({type: 'notAThing'}), /unknown command/);
  assert.throws(() => api.send(null), /unknown command/);
  assert.strictEqual(posted.length, 1);
});

test('catalog contains the core task-lifecycle commands', () => {
  ['run', 'submit', 'stop', 'userAnswer', 'newChat', 'closeTab',
   'getHistory', 'getConfig', 'setWorkDir', 'auth',
  ].forEach(name => assert.ok(SORCAR_API_COMMANDS.includes(name), name));
});

test('the real chat webview only sends API commands', async () => {
  const {win, posted} = makeWebview();
  const inp = win.document.getElementById('task-input');
  inp.value = 'do something';
  win.document.getElementById('send-btn').click();
  await new Promise(r => setTimeout(r, 50));
  assert.ok(posted.length > 0, 'webview posted no messages');
  for (const msg of posted) {
    assert.ok(
      SORCAR_API_COMMANDS.includes(msg.type),
      `non-API message sent by webview: ${JSON.stringify(msg)}`,
    );
  }
  assert.ok(
    posted.some(m => m.type === 'submit' && m.prompt === 'do something'),
    'submit command missing',
  );
});

test('SorcarApi (extension host) emits correct wire commands', () => {
  const {SorcarApi} = require(path.join('..', 'out', 'SorcarApi.js'));
  const sent = [];
  const api = new SorcarApi({sendCommand: cmd => sent.push(cmd)});

  api.run({prompt: 'p', model: 'm', workDir: '/w', attachments: [],
           useWorktree: false, useParallel: true, autoCommit: false,
           tabId: 't'});
  api.stop('t');
  api.appendUserMessage('more', 't');
  api.userAnswer('yes', 't');
  api.resumeSession({chatId: 'c1', taskId: 'task1', tabId: 't'});
  api.setWorkDir('/w');
  api.selectModel('m', 't');
  api.getModels();
  api.getInputHistory();
  api.getConfig();
  api.complete({query: 'q', tabId: 't'});
  api.recordFileUsage('/f', '/w');
  api.worktreeAction('merge', 't');
  api.autocommitAction('commit', 't', '/w');
  api.mergeAction('all-done', 't', '/w');
  api.generateCommitMessage('m', 't', '/w');
  api.closeTab('t');
  api.serverReset();
  api.forward({type: 'getHistory', query: 'x'});

  const types = sent.map(c => c.type);
  assert.deepStrictEqual(types, [
    'run', 'stop', 'appendUserMessage', 'userAnswer', 'resumeSession',
    'setWorkDir', 'selectModel', 'getModels', 'getInputHistory',
    'getConfig', 'complete', 'recordFileUsage', 'worktreeAction',
    'autocommitAction', 'mergeAction', 'generateCommitMessage',
    'closeTab', 'serverReset', 'getHistory',
  ]);
  assert.deepStrictEqual(sent[0], {
    type: 'run', prompt: 'p', model: 'm', workDir: '/w', attachments: [],
    useWorktree: false, useParallel: true, autoCommit: false, tabId: 't',
  });
  assert.deepStrictEqual(sent[1], {type: 'stop', tabId: 't'});
  assert.deepStrictEqual(sent[4], {
    type: 'resumeSession', chatId: 'c1', taskId: 'task1', tabId: 't',
  });
  assert.deepStrictEqual(sent[14], {
    type: 'mergeAction', action: 'all-done', tabId: 't', workDir: '/w',
  });
});
