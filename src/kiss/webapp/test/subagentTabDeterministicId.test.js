// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Every client must mint the SAME tab id for the same run_parallel
// sub-agent.
//
// Tab ids are global across clients: the shared tab registry mirrors
// one tab to every window.  Before this invariant existed, each client
// answered the sub-agent's `new_tab` broadcast by minting its own
// RANDOM tab id and resuming the sub-agent on it.  With two or more
// clients connected (two browser windows on the same chat),
// the daemon then announced one `openSubagentTab` per client-minted id,
// every client "deduped" the other clients' ids via retagSubagentTab ->
// api.closeTab(oldId), and the daemon's cleanup_tab() unsubscribed
// EVERY id from the sub-agent's event stream: the sub-agent tabs froze
// on the head of the transcript (system prompt + prompt, nothing else).
//
// The fix: media/main.js mints the deterministic id
// `${parentTabId}__sub_${taskId}` (the daemon's own replay-id
// convention from AgentServer._open_persisted_subagent_tabs), so all
// clients converge on ONE shared tab id — no retag, no closeTab, one
// live subscription.  This file pins the id minting; a converged
// announcement landing on that tab without a closeTab is pinned by
// runParallelSubagentTabDedupe.test.js.
//
// The page runs on the REAL webapp transport (media/transport.js
// driving a fake WebSocket), so the frames the daemon would see are
// exactly the ones the assertions inspect.

/* global require, console */

'use strict';

const assert = require('assert');
const {send, readyTabId} = require('./simplify2_harness');
const {subagentTabEls} = require('./subagent_tab_helpers');
const {makeWebappClient} = require('./transport_page_helpers');

/** Boot a client, start the parent task, and return the parent tab id. */
function bootParent(client) {
  const parentId = readyTabId(client.drain());
  send(client.win, {
    type: 'status',
    running: true,
    tabId: parentId,
    startTs: Date.now(),
  });
  send(client.win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: parentId,
    extras: {tasks: JSON.stringify(['sub 1'])},
  });
  return parentId;
}

// ---------------------------------------------------------------------------
// 1. new_tab must resume the sub-agent on the deterministic shared id.
// ---------------------------------------------------------------------------

function testDeterministicId() {
  const client = makeWebappClient();
  const parentId = bootParent(client);
  const taskId = 'sub-task-det-1';

  send(client.win, {
    type: 'new_tab',
    task_id: taskId,
    parent_tab_id: parentId,
    taskId: '',
  });

  const resume = client.drain()
    .filter(m => m.type === 'resumeSession' && m.taskId === taskId)
    .pop();
  assert.ok(resume, 'new_tab must make the client resume the sub-agent');
  assert.strictEqual(
    resume.tabId,
    parentId + '__sub_' + taskId,
    'the sub-agent must be resumed on the deterministic ' +
      'shared tab id, not a client-local random id',
  );
  const tabs = subagentTabEls(client.win);
  assert.strictEqual(tabs.length, 1, 'exactly one sub-agent tab');
  assert.strictEqual(
    tabs[0].dataset.tabId,
    parentId + '__sub_' + taskId,
    'the open tab must carry the deterministic id',
  );
  console.log('PASS deterministic new_tab id');
}

// ---------------------------------------------------------------------------
// 2. A parent-less spawn still mints a deterministic (task-keyed) id.
// ---------------------------------------------------------------------------

function testParentlessDeterministicId() {
  const client = makeWebappClient();
  const taskId = 'sub-task-det-3';
  send(client.win, {
    type: 'new_tab',
    task_id: taskId,
    parent_tab_id: '',
    taskId: '',
  });
  const resume = client.drain()
    .filter(m => m.type === 'resumeSession' && m.taskId === taskId)
    .pop();
  assert.ok(resume, 'a parent-less new_tab must still resume the sub-agent');
  assert.strictEqual(
    resume.tabId,
    'task__sub_' + taskId,
    'parent-less spawns must also converge across clients',
  );
  console.log('PASS parent-less deterministic id');
}

testDeterministicId();
testParentlessDeterministicId();
console.log('subagentTabDeterministicId.test.js: all tests passed');
// JSDOM windows keep timers alive (running-task UI timer, shim
// reconnect); an assertion failure above throws and exits non-zero, so
// reaching this line means success.
process.exit(0);
