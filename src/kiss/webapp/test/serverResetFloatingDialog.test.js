// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The settings panel's "Restart server" button: with a task running it
// opens an in-page floating confirmation box (#server-reset-confirm-modal
// inside #settings-panel); OK posts exactly one `serverReset` command,
// Cancel posts none, Escape closes the box and posts none, a second click
// while the box is open neither closes nor re-stacks it, and with no
// running task (or a task that has since stopped) the reset goes out at
// once without a box and without an in-page confirmation toast.

'use strict';

const assert = require('assert');
const {makePage, send, waitFor} = require('./simplify2_harness.js');

function clickInWindow(win, id) {
  const el = win.document.getElementById(id);
  assert.ok(el, `element #${id} must exist`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function getFloatingModal(win) {
  return win.document.getElementById('server-reset-confirm-modal');
}

function isFloatingModalOpen(win) {
  const modal = getFloatingModal(win);
  return !!(modal && modal.classList.contains('open'));
}

function resets(posted) {
  return posted.filter(m => m.type === 'serverReset');
}

function pageWithRunningTask() {
  const page = makePage();
  send(page.win, {
    type: 'status',
    tabId: page.win._testApi.getActiveTabId(),
    running: true,
    startTs: Date.now(),
  });
  return page;
}

async function testMarkup() {
  const {win} = makePage();
  const settingsPanel = win.document.getElementById('settings-panel');
  assert.ok(settingsPanel, '#settings-panel must exist');
  const modal = getFloatingModal(win);
  assert.ok(modal, 'settings panel must contain #server-reset-confirm-modal');
  assert.ok(
    settingsPanel.contains(modal),
    'the floating modal must live INSIDE #settings-panel',
  );
  assert.ok(win.document.getElementById('server-reset-confirm-ok'));
  assert.ok(win.document.getElementById('server-reset-confirm-cancel'));
  assert.strictEqual(isFloatingModalOpen(win), false, 'closed by default');
  assert.strictEqual(modal.getAttribute('role'), 'dialog');
  win.close();
}

async function testOkPostsOneReset() {
  const {win, posted} = pageWithRunningTask();
  clickInWindow(win, 'cfg-server-reset-btn');
  await waitFor(
    () => isFloatingModalOpen(win),
    'click must open the floating confirmation box',
  );
  assert.strictEqual(
    resets(posted).length,
    0,
    'the page must NOT post serverReset until the user clicks OK',
  );
  clickInWindow(win, 'server-reset-confirm-ok');
  await waitFor(() => !isFloatingModalOpen(win), 'OK must close the box');
  assert.strictEqual(resets(posted).length, 1, 'OK posts exactly one reset');
  win.close();
}

async function testCancelPostsNothing() {
  const {win, posted} = pageWithRunningTask();
  clickInWindow(win, 'cfg-server-reset-btn');
  await waitFor(() => isFloatingModalOpen(win), 'click must open the box');
  clickInWindow(win, 'server-reset-confirm-cancel');
  await waitFor(() => !isFloatingModalOpen(win), 'Cancel must close the box');
  assert.strictEqual(resets(posted).length, 0, 'Cancel posts no reset');
  win.close();
}

async function testNoRunningTaskResetsAtOnce() {
  const {win, posted} = makePage();
  clickInWindow(win, 'cfg-server-reset-btn');
  assert.strictEqual(
    isFloatingModalOpen(win),
    false,
    'with no running agent, the floating confirmation box must NOT open',
  );
  await waitFor(
    () => resets(posted).length === 1,
    'no-agent click must post serverReset immediately',
  );
  assert.strictEqual(
    win.document.querySelectorAll('.kiss-notification').length,
    0,
    'the page must NOT render any in-page confirmation toast',
  );
  win.close();
}

async function testStoppedTaskResetsAtOnce() {
  const {win, posted} = pageWithRunningTask();
  send(win, {
    type: 'status',
    tabId: win._testApi.getActiveTabId(),
    running: false,
  });
  clickInWindow(win, 'cfg-server-reset-btn');
  await waitFor(
    () => resets(posted).length === 1,
    'click after the agent stops must post serverReset at once',
  );
  assert.strictEqual(
    isFloatingModalOpen(win),
    false,
    'a click after the agent stopped must NOT open the floating dialog',
  );
  win.close();
}

async function testEscapeClosesWithoutPosting() {
  const {win, posted} = pageWithRunningTask();
  clickInWindow(win, 'cfg-server-reset-btn');
  await waitFor(() => isFloatingModalOpen(win), 'click must open the box');
  win.document.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Escape', bubbles: true}),
  );
  await waitFor(() => !isFloatingModalOpen(win), 'Escape must close the box');
  assert.strictEqual(resets(posted).length, 0, 'Escape posts no reset');
  win.close();
}

async function testDoubleClickThenOk() {
  const {win, posted} = pageWithRunningTask();
  clickInWindow(win, 'cfg-server-reset-btn');
  await waitFor(() => isFloatingModalOpen(win), 'first click opens the box');
  clickInWindow(win, 'cfg-server-reset-btn');
  assert.strictEqual(
    isFloatingModalOpen(win),
    true,
    'second click must NOT close or re-stack the floating dialog',
  );
  clickInWindow(win, 'server-reset-confirm-ok');
  await waitFor(() => !isFloatingModalOpen(win), 'OK must close the dialog');
  assert.strictEqual(
    resets(posted).length,
    1,
    'double-click + OK must produce exactly ONE serverReset post',
  );
  win.close();
}

async function testCancelTwice() {
  const {win, posted} = pageWithRunningTask();
  clickInWindow(win, 'cfg-server-reset-btn');
  await waitFor(() => isFloatingModalOpen(win), 'first click opens the box');
  clickInWindow(win, 'server-reset-confirm-cancel');
  await waitFor(() => !isFloatingModalOpen(win), 'Cancel must close the box');
  clickInWindow(win, 'cfg-server-reset-btn');
  await waitFor(
    () => isFloatingModalOpen(win),
    'after the previous box closed, a fresh click must re-open it',
  );
  clickInWindow(win, 'server-reset-confirm-cancel');
  await waitFor(() => !isFloatingModalOpen(win), 'second Cancel closes again');
  assert.strictEqual(
    resets(posted).length,
    0,
    'two Cancel cycles must produce zero serverReset posts',
  );
  win.close();
}

// A cancelled box must leave no confirmation listener behind: reopening it
// and clicking OK posts exactly one reset, not one per earlier Cancel.
async function testCancelThenOkPostsOneReset() {
  const {win, posted} = pageWithRunningTask();
  clickInWindow(win, 'cfg-server-reset-btn');
  await waitFor(() => isFloatingModalOpen(win), 'first click opens the box');
  clickInWindow(win, 'server-reset-confirm-cancel');
  await waitFor(() => !isFloatingModalOpen(win), 'Cancel must close the box');
  assert.strictEqual(resets(posted).length, 0, 'Cancel posts no reset');
  clickInWindow(win, 'cfg-server-reset-btn');
  await waitFor(
    () => isFloatingModalOpen(win),
    'a fresh click after Cancel must re-open the box',
  );
  clickInWindow(win, 'server-reset-confirm-ok');
  await waitFor(() => !isFloatingModalOpen(win), 'OK must close the box');
  assert.strictEqual(
    resets(posted).length,
    1,
    'OK after a Cancel must produce exactly ONE serverReset post',
  );
  win.close();
}

async function main() {
  await testMarkup();
  await testOkPostsOneReset();
  await testCancelPostsNothing();
  await testNoRunningTaskResetsAtOnce();
  await testStoppedTaskResetsAtOnce();
  await testEscapeClosesWithoutPosting();
  await testDoubleClickThenOk();
  await testCancelTwice();
  await testCancelThenOkPostsOneReset();
  console.log('serverResetFloatingDialog: all passed.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
