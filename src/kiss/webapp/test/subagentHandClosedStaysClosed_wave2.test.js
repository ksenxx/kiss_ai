// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// A running sub-agent's tab stays open unless the user closes it.  Once
// the user has closed it -- on this surface, or on another one whose
// close the daemon mirrors here (`closeSubagentTab`) -- no later live
// announcement of that sub-agent may reopen it: neither a re-delivered
// spawn (`new_tab`) nor a live `openSubagentTab`, under any of the tab
// ids the daemon addresses the sub-agent by -- whether the user closed
// it with its run_parallel panel expanded or collapsed, or closed every
// tab of the fan-out so that the panel folded itself.  The sub-agent
// stays on its panel, so expanding the panel (the user asking for the
// fan-out again) reopens it, and a live announcement then lands on
// that one tab.

const assert = require('assert');
const {send} = require('./simplify2_harness');
const {makeWebappClient} = require('./transport_page_helpers');
const {
  togglePanel,
  clickClose,
  openSubTabIds,
  bootParallelRun,
} = require('./subagent_tab_helpers');
const {test, report} = require('./ui_antipattern_harness').makeRunner();

/** The daemon re-delivers the spawn of *taskId* under *parentId*. */
function respawn(win, parentId, taskId) {
  send(win, {
    type: 'new_tab',
    task_id: taskId,
    parent_tab_id: parentId,
    taskId: '',
  });
}

/** A live (not finished) announcement of *taskId* under tab id *tabId*. */
function announceLive(win, parentId, taskId, tabId) {
  send(win, {
    type: 'openSubagentTab',
    tab_id: tabId,
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: taskId,
    taskIndex: 0,
    isSubagentTab: true,
  });
}

/** The tab ids of the *type* frames the page posted after frame *from*. */
function postedTabIds(client, from, type) {
  return client
    .drain()
    .slice(from)
    .filter(m => m.type === type)
    .map(m => m.tabId);
}

test('a running sub-agent tab the user closed stays closed', () => {
  const client = makeWebappClient();
  const {win, panel, parentId, taskIds, subTabIds} = bootParallelRun(
    2,
    client,
    {replay: true},
  );
  const booted = client.drain().length;
  clickClose(win, subTabIds[0]);
  assert.deepStrictEqual(openSubTabIds(win), [subTabIds[1]]);
  assert.deepStrictEqual(postedTabIds(client, booted, 'closeTab'), [
    subTabIds[0],
  ]);

  const closed = client.drain().length;
  respawn(win, parentId, taskIds[0]);
  announceLive(win, parentId, taskIds[0], subTabIds[0]);
  announceLive(win, parentId, taskIds[0], parentId + '__sub_replay-1');
  assert.deepStrictEqual(
    openSubTabIds(win),
    [subTabIds[1]],
    'no live announcement reopens the tab the user closed',
  );
  assert.deepStrictEqual(
    postedTabIds(client, closed, 'closeTab'),
    [],
    'nothing to close again',
  );
  assert.deepStrictEqual(
    postedTabIds(client, closed, 'resumeSession'),
    [],
    'no replay is asked for the closed sub-agent',
  );

  // Collapsing and expanding the panel asks for the whole fan-out again.
  togglePanel(win, panel);
  togglePanel(win, panel);
  assert.strictEqual(openSubTabIds(win).length, 2, 'expanding reopens it');
  const reopened = openSubTabIds(win).find(id => id !== subTabIds[1]);
  respawn(win, parentId, taskIds[0]);
  announceLive(win, parentId, taskIds[0], reopened);
  assert.strictEqual(
    openSubTabIds(win).length,
    2,
    'a live announcement lands on the reopened tab: ' +
      JSON.stringify(openSubTabIds(win)),
  );
  win.close();
});

test("another surface's close of a running sub-agent tab holds here too", () => {
  const client = makeWebappClient();
  const {win, parentId, taskIds, subTabIds} = bootParallelRun(2, client, {
    replay: true,
  });
  const booted = client.drain().length;
  send(win, {type: 'closeSubagentTab', tab_id: subTabIds[0]});
  assert.deepStrictEqual(openSubTabIds(win), [subTabIds[1]]);
  assert.deepStrictEqual(
    postedTabIds(client, booted, 'closeTab'),
    [],
    'a mirrored close is not echoed',
  );
  respawn(win, parentId, taskIds[0]);
  announceLive(win, parentId, taskIds[0], subTabIds[0]);
  assert.deepStrictEqual(openSubTabIds(win), [subTabIds[1]]);
  assert.deepStrictEqual(postedTabIds(client, booted, 'resumeSession'), []);
  win.close();
});

test('a running sub-agent tab closed under a folded panel stays closed', () => {
  const client = makeWebappClient();
  const {win, panel, parentId, taskIds, subTabIds} = bootParallelRun(
    2,
    client,
    {replay: true},
  );
  togglePanel(win, panel);
  assert.ok(panel.classList.contains('collapsed'));
  assert.strictEqual(openSubTabIds(win).length, 2, 'running tabs stay open');
  clickClose(win, subTabIds[0]);
  respawn(win, parentId, taskIds[0]);
  announceLive(win, parentId, taskIds[0], parentId + '__sub_replay-1');
  assert.deepStrictEqual(openSubTabIds(win), [subTabIds[1]]);
  togglePanel(win, panel);
  assert.strictEqual(openSubTabIds(win).length, 2, 'expanding reopens it');
  win.close();
});

test('closing every running tab folds the panel; the tabs stay closed', () => {
  const client = makeWebappClient();
  const {win, panel, parentId, taskIds, subTabIds} = bootParallelRun(
    2,
    client,
    {replay: true},
  );
  clickClose(win, subTabIds[0]);
  clickClose(win, subTabIds[1]);
  assert.ok(panel.classList.contains('collapsed'), 'the empty panel folds');
  for (const taskId of taskIds) {
    respawn(win, parentId, taskId);
    announceLive(win, parentId, taskId, parentId + '__sub_' + taskId);
  }
  assert.deepStrictEqual(openSubTabIds(win), []);
  togglePanel(win, panel);
  assert.strictEqual(openSubTabIds(win).length, 2, 'expanding reopens both');
  win.close();
});

test('a sub-agent nobody closed still gets its tab from a live spawn', () => {
  const client = makeWebappClient();
  const {win, parentId, subTabIds} = bootParallelRun(1, client, {
    replay: true,
  });
  respawn(win, parentId, 'sub-task-late');
  assert.strictEqual(
    openSubTabIds(win).length,
    2,
    'a new sub-agent opens its tab: ' + JSON.stringify(openSubTabIds(win)),
  );
  assert.ok(openSubTabIds(win).includes(subTabIds[0]));
  win.close();
});

report('subagentHandClosedStaysClosed_wave2');
