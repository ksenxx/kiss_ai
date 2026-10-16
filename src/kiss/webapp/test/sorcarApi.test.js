// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

const assert = require('assert');
const {test} = require('node:test');

const {createSorcarApi, SORCAR_API_COMMANDS} = require('../media/api.js');
const {makePage} = require('./simplify2_harness.js');

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
  ['submit', 'stop', 'userAnswer', 'newChat', 'closeTab',
   'getHistory', 'getConfig', 'setWorkDir',
  ].forEach(name => assert.ok(SORCAR_API_COMMANDS.includes(name), name));
});

test('catalog omits the commands only other clients send', () => {
  // ``run`` is the Python API's (daemon_client.py), ``getTabsState``
  // a daemon link's on reconnect (commands.py _cmd_get_tabs_state),
  // ``getModels``/``getSeaCommands`` the daemon's own ``ready``
  // fan-out: no page message carries them.  ``auth`` is the WebSocket
  // handshake transport.js sends raw before the api exists, and
  // ``generateCommitMessage`` was the VS Code SCM input's command.
  ['run', 'getTabsState', 'getModels', 'getSeaCommands', 'activeTasksQuery',
   'auth', 'generateCommitMessage',
  ].forEach(name => assert.ok(!SORCAR_API_COMMANDS.includes(name), name));
});

test('the real chat page only sends API commands', async () => {
  const {win, posted} = makePage();
  const inp = win.document.getElementById('task-input');
  inp.value = 'do something';
  win.document.getElementById('send-btn').click();
  assert.ok(posted.length > 0, 'page posted no messages');
  for (const msg of posted) {
    assert.ok(
      SORCAR_API_COMMANDS.includes(msg.type),
      `non-API message sent by the page: ${JSON.stringify(msg)}`,
    );
  }
  assert.ok(
    posted.some(m => m.type === 'submit' && m.prompt === 'do something'),
    'submit command missing',
  );
  win.close();
});
