// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/* global require, console */

// File links of a tab bound to a registered machine (main.js against
// the real chat.html):
//   * a `path:line` link of a bound tab keeps the machine stamp on its
//     check and its `openFile` (the plain path case is pinned by
//     fileLinksOfMachineBoundTab.test.js);
//   * a tab of this machine sends neither stamp, also when the tab is
//     in the background;
//   * the checks re-issued after a reconnect keep the machine stamp;
//   * the machine is the daemon's binding (where the shown task ran),
//     never the drop-down's unsubmitted pick, and a reply stamped with
//     another machine does not resolve the tab's spans.
// The daemon side (forwarding to the machine) is covered by
// src/kiss/tests/server/test_remote_check_paths.py.

const assert = require('assert');
const {makePage, send, pickMachine} = require('./simplify2_harness');
const {findLinks, checkPathCommands: checks} = require('./file_link_helpers');

const BOX_TAB = 'tab-on-box';
const HOME_TAB = 'tab-at-home';

function makeDomPage() {
  const {win, posted} = makePage({
    initialState: {
      tabs: [
        {
          title: 'box',
          chatId: BOX_TAB,
          backendChatId: 'chat-box',
          parentTabId: '',
        },
        {
          title: 'home',
          chatId: HOME_TAB,
          backendChatId: 'chat-home',
          parentTabId: '',
        },
      ],
      activeTabIndex: 0,
      chatId: BOX_TAB,
    },
  });
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: BOX_TAB, chatId: 'chat-box', title: 'box', workDir: '/box/work'},
      {
        tabId: HOME_TAB,
        chatId: 'chat-home',
        title: 'home',
        workDir: '/home/work',
      },
    ],
  });
  send(win, {type: 'tabMachine', tabId: BOX_TAB, machine: 'box'});
  return {win, posted, close: () => win.close()};
}

function answer(win, check) {
  const results = {};
  for (const p of check.paths) results[p] = true;
  const reply = {
    type: 'pathsExist',
    results,
    workDir: check.workDir,
    tabId: check.tabId,
  };
  if (check.machine) reply.machine = check.machine;
  send(win, reply);
}

function linkFor(win, p) {
  return findLinks(win, p)[0];
}

function testBoundTabLineSuffixKeepsTheMachine() {
  const wv = makeDomPage();
  send(wv.win, {
    type: 'prompt',
    text: 'error at src/box.py:12',
    tabId: BOX_TAB,
  });
  send(wv.win, {type: 'text_end', tabId: BOX_TAB});
  const sent = checks(wv.posted);
  assert.strictEqual(sent.length, 1);
  assert.strictEqual(sent[0].machine, 'box', 'the check carries the machine');
  assert.deepStrictEqual(sent[0].paths, ['src/box.py']);
  answer(wv.win, sent[0]);
  const link = linkFor(wv.win, 'src/box.py:12');
  assert.ok(link, 'the path:line link is promoted');
  link.dispatchEvent(new wv.win.MouseEvent('click', {bubbles: true}));
  const open = wv.posted.filter(m => m.type === 'openFile')[0];
  assert.ok(open);
  assert.strictEqual(open.path, 'src/box.py');
  assert.strictEqual(open.line, 12);
  assert.strictEqual(open.machine, 'box');
  wv.close();
  console.log('  ok - a path:line link of a bound tab opens on its machine');
}

function testHomeTabSendsNoMachine() {
  const wv = makeDomPage();
  send(wv.win, {
    type: 'prompt',
    text: 'see src/home.py please',
    tabId: HOME_TAB,
  });
  send(wv.win, {type: 'text_end', tabId: HOME_TAB});
  const sent = checks(wv.posted);
  assert.strictEqual(sent.length, 1);
  assert.strictEqual(sent[0].tabId, HOME_TAB);
  assert.ok(!('machine' in sent[0]), 'a tab of this machine sends no machine');
  // The home tab is in the background: its spans live in its saved
  // transcript fragment, which handlePathsExist still resolves.
  answer(wv.win, sent[0]);
  // Switch to the home tab and click its link.
  wv.win._testApi.switchToTab(HOME_TAB);
  assert.strictEqual(wv.win._testApi.getActiveTabId(), HOME_TAB);
  const link = linkFor(wv.win, 'src/home.py');
  assert.ok(link, 'the home link is promoted');
  assert.strictEqual(link.getAttribute('data-path-machine'), null);
  link.dispatchEvent(new wv.win.MouseEvent('click', {bubbles: true}));
  const open = wv.posted.filter(m => m.type === 'openFile')[0];
  assert.ok(open);
  assert.ok(!('machine' in open), 'the openFile carries no machine');
  wv.close();
  console.log('  ok - a tab of this machine sends no machine stamp');
}

function testReissueAfterReconnectKeepsTheMachine() {
  const wv = makeDomPage();
  send(wv.win, {type: 'prompt', text: 'see src/box.py please', tabId: BOX_TAB});
  send(wv.win, {type: 'text_end', tabId: BOX_TAB});
  assert.strictEqual(checks(wv.posted).length, 1);
  // The daemon goes away before answering and comes back: the check
  // is re-issued, still stamped with the machine.
  send(wv.win, {type: 'daemonStatus', connected: false});
  send(wv.win, {type: 'daemonStatus', connected: true});
  const sent = checks(wv.posted);
  assert.strictEqual(sent.length, 2, 'the unanswered check is re-issued');
  assert.strictEqual(sent[1].machine, 'box');
  assert.deepStrictEqual(sent[1].paths, ['src/box.py']);
  wv.close();
  console.log('  ok - re-issued checks keep the machine stamp');
}

/**
 * Pick *name* ('' = this machine) in the chat header's machine
 * drop-down, as a user would: this only says where the tab's NEXT task
 * goes; the daemon's binding (`tabMachine`) stays.
 */
function pickInDropDown(win, name) {
  send(win, {
    type: 'machines',
    machines: [
      {name: 'home', local: true, status: 'reachable'},
      {name: 'box', local: false, status: 'reachable'},
      {name: 'other', local: false, status: 'reachable'},
    ],
  });
  pickMachine(win, name);
  assert.strictEqual(
    win.document.getElementById('chat-machine').value,
    name,
    'the drop-down shows the pick',
  );
}

function testLinksFollowTheBindingNotTheDropDown() {
  const wv = makeDomPage();
  // The box tab's task ran on "box"; the user has since picked
  // "other" in the drop-down for the next task but not submitted.
  pickInDropDown(wv.win, 'other');
  send(wv.win, {type: 'prompt', text: 'see src/box.py please', tabId: BOX_TAB});
  send(wv.win, {type: 'text_end', tabId: BOX_TAB});
  const sent = checks(wv.posted);
  assert.strictEqual(sent.length, 1);
  assert.strictEqual(
    sent[0].machine,
    'box',
    "the check goes to the task's machine",
  );
  const span = wv.win.document.querySelector('#output [data-path-candidate]');
  assert.strictEqual(span.getAttribute('data-path-machine'), 'box');
  // A reply stamped with another machine (the same path and work dir)
  // does not resolve the span: the check stays outstanding.
  send(wv.win, {
    type: 'pathsExist',
    results: {'src/box.py': true},
    workDir: sent[0].workDir,
    tabId: BOX_TAB,
    machine: 'other',
  });
  assert.ok(
    !linkFor(wv.win, 'src/box.py'),
    "another machine's reply is ignored",
  );
  // The user picks this machine for the next task, then the daemon
  // drops and comes back before the box answers: the re-issued check,
  // the link and the click still belong to the box.
  pickInDropDown(wv.win, '');
  send(wv.win, {type: 'daemonStatus', connected: false});
  send(wv.win, {type: 'daemonStatus', connected: true});
  const reissued = checks(wv.posted);
  assert.strictEqual(reissued.length, 2, 'the unanswered check is re-issued');
  assert.strictEqual(
    reissued[1].machine,
    'box',
    'the re-issued check names the box, not the drop-down pick',
  );
  answer(wv.win, reissued[1]);
  const link = linkFor(wv.win, 'src/box.py');
  assert.ok(link, "the box's reply promoted the link");
  link.dispatchEvent(new wv.win.MouseEvent('click', {bubbles: true}));
  const open = wv.posted.filter(m => m.type === 'openFile')[0];
  assert.ok(open);
  assert.strictEqual(open.machine, 'box', 'the openFile goes to the box');
  wv.close();
  console.log(
    '  ok - links follow the machine the task ran on, not the drop-down',
  );
}

function testHomeTabLinksIgnoreADropDownPick() {
  const wv = makeDomPage();
  wv.win._testApi.switchToTab(HOME_TAB);
  // The home tab's task ran here; "box" is picked for the next one.
  pickInDropDown(wv.win, 'box');
  send(wv.win, {
    type: 'prompt',
    text: 'see src/home.py please',
    tabId: HOME_TAB,
  });
  send(wv.win, {type: 'text_end', tabId: HOME_TAB});
  const sent = checks(wv.posted);
  assert.strictEqual(sent.length, 1);
  assert.ok(!('machine' in sent[0]), 'the check stays on this machine');
  // A reply from the box for the same path does not resolve it.
  send(wv.win, {
    type: 'pathsExist',
    results: {'src/home.py': true},
    workDir: sent[0].workDir,
    tabId: HOME_TAB,
    machine: 'box',
  });
  assert.ok(!linkFor(wv.win, 'src/home.py'), "the box's reply is ignored");
  answer(wv.win, sent[0]);
  const link = linkFor(wv.win, 'src/home.py');
  assert.ok(link, "this machine's reply promoted the link");
  link.dispatchEvent(new wv.win.MouseEvent('click', {bubbles: true}));
  const open = wv.posted.filter(m => m.type === 'openFile')[0];
  assert.ok(open);
  assert.ok(!('machine' in open), 'the openFile stays on this machine');
  wv.close();
  console.log("  ok - a home tab's links ignore the drop-down pick");
}

testBoundTabLineSuffixKeepsTheMachine();
testHomeTabSendsNoMachine();
testReissueAfterReconnectKeepsTheMachine();
testLinksFollowTheBindingNotTheDropDown();
testHomeTabLinksIgnoreADropDownPick();
console.log('\nAll fileLinkMachine tests passed.');
