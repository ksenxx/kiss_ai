// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// The tests the two CLI-agent subpanels of the settings sheet (Claude
// Code, Codex) share, written once over an agent description:
//
// * The subpanel starts collapsed and holds the subscription switch (on
//   by default), the masked token box with a reveal toggle and the
//   status line; opening the sheet asks the daemon for the login status.
// * Every status reply shape is painted in plain words.
// * `configData` sets the switch from its config key; closing the sheet
//   saves the switch and the token.
// * A reply for another machine, or a forwarded link error, never
//   repaints this sheet's line (the error becomes a notification).
//
// An agent is `{subpanel, cli, statusCmd, statusReply, tokenKey,
// configKey, loggedInAuth, statusCases}`: `subpanel` is the id stem
// (`claude-code` -> #claude-code-toggle, #claude-code-body,
// #cfg-claude-code-status, #cfg-claude-code-use-subscription and the
// `claude-code-status-ok/-warn` classes), `statusCases` a list of
// `[statusFields, wordingRegExp, 'ok' | 'warn']`.

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');
const {
  lastMsg,
  click,
  byId,
  openSettings,
  closeSettings,
  statusLine,
} = require('./settings_helpers');

function testSubpanelLayoutAndDefaults(agent) {
  const {win, posted} = makePage();
  openSettings(win);

  const toggle = byId(win, `${agent.subpanel}-toggle`);
  const body = byId(win, `${agent.subpanel}-body`);
  assert.strictEqual(toggle.getAttribute('aria-expanded'), 'false');
  assert.strictEqual(body.hidden, true, 'starts collapsed');
  click(win, toggle);
  assert.strictEqual(body.hidden, false, 'the header expands it');

  const sw = byId(win, `cfg-${agent.subpanel}-use-subscription`);
  assert.strictEqual(sw.checked, true, 'the switch is on by default');
  const token = byId(win, `cfg-key-${agent.tokenKey}`);
  assert.ok(body.contains(token), 'the token box lives in the subpanel');
  assert.strictEqual(token.type, 'password', 'the token is masked');
  assert.ok(
    byId(win, `cfg-key-${agent.tokenKey}-toggle`),
    'the token box has a reveal toggle',
  );
  const line = statusLine(win, agent.subpanel);
  assert.ok(body.contains(line), 'the status line is in the subpanel');
  assert.strictEqual(line.textContent, `Checking the ${agent.cli} CLI\u2026`);

  assert.ok(
    lastMsg(posted, agent.statusCmd),
    `opening the panel asks for the ${agent.cli} login status`,
  );

  // Re-open: collapsed again like the other subpanels.
  closeSettings(win);
  openSettings(win);
  assert.strictEqual(
    byId(win, `${agent.subpanel}-body`).hidden,
    true,
    'reopen re-collapses',
  );
  win.close();
  console.log('  ok - subpanel layout, defaults and status request');
}

function testStatusLineCases(agent) {
  const {win} = makePage();
  openSettings(win);
  const line = statusLine(win, agent.subpanel);
  for (const [status, re, kind] of agent.statusCases) {
    send(win, {type: agent.statusReply, ...status});
    assert.ok(
      re.test(line.textContent),
      `${re} should match "${line.textContent}"`,
    );
    const cls = `${agent.subpanel}-status-${kind}`;
    assert.ok(line.classList.contains(cls), `${cls} on "${line.textContent}"`);
    const other = `${agent.subpanel}-status-${kind === 'ok' ? 'warn' : 'ok'}`;
    assert.ok(!line.classList.contains(other), `${other} cleared`);
  }
  win.close();
  console.log('  ok - status line wording for every reply shape');
}

function testConfigLoadAndSave(agent) {
  const {win, posted} = makePage();
  openSettings(win);
  const sw = byId(win, `cfg-${agent.subpanel}-use-subscription`);

  // Missing key keeps the default (on); an explicit false turns it off.
  send(win, {type: 'configData', config: {max_budget: 100}, apiKeys: {}});
  assert.strictEqual(sw.checked, true, 'a config without the key leaves it on');
  send(win, {
    type: 'configData',
    config: {max_budget: 100, [agent.configKey]: false},
    apiKeys: {[agent.tokenKey]: 'tok-stored'},
  });
  assert.strictEqual(sw.checked, false, 'configData false unchecks the switch');
  assert.strictEqual(
    byId(win, `cfg-key-${agent.tokenKey}`).value,
    'tok-stored',
    'the stored token populates its box',
  );

  // The user flips the switch; a later configData must not undo it.
  sw.checked = true;
  sw.dispatchEvent(new win.Event('change', {bubbles: true}));
  send(win, {
    type: 'configData',
    config: {max_budget: 100, [agent.configKey]: false},
    apiKeys: {},
  });
  assert.strictEqual(
    sw.checked,
    true,
    'a user edit survives a configData repaint',
  );

  const token = byId(win, `cfg-key-${agent.tokenKey}`);
  token.value = 'tok-new';
  token.dispatchEvent(new win.Event('input', {bubbles: true}));
  closeSettings(win);
  const msg = lastMsg(posted, 'saveConfig');
  assert.ok(msg, 'closing the panel saves the form');
  assert.strictEqual(msg.config[agent.configKey], true, 'switch saved');
  assert.strictEqual(msg.apiKeys[agent.tokenKey], 'tok-new', 'token saved');

  // Off is saved as an explicit false, never dropped.
  openSettings(win);
  sw.checked = false;
  sw.dispatchEvent(new win.Event('change', {bubbles: true}));
  closeSettings(win);
  assert.strictEqual(
    lastMsg(posted, 'saveConfig').config[agent.configKey],
    false,
    'an unchecked switch is saved as false',
  );
  win.close();
  console.log(
    '  ok - configData loads and saveConfig stores the switch and token',
  );
}

function testRepliesForOtherMachinesAreIgnored(agent) {
  const {win} = makePage();
  openSettings(win);
  const line = statusLine(win, agent.subpanel);
  const before = line.textContent;

  // A forwarded reply while this sheet is for the local daemon: ignored.
  send(win, {
    type: agent.statusReply,
    forwarded: true,
    machine: 'other',
    installed: true,
    loggedIn: true,
    authMethod: agent.loggedInAuth,
    useSubscription: true,
  });
  assert.strictEqual(
    line.textContent,
    before,
    "another machine's status is not painted",
  );

  // A forwarded link error (no status fields at all) is a notification.
  send(win, {
    type: agent.statusReply,
    forwarded: true,
    machine: 'other',
    error: 'machine unreachable',
  });
  assert.strictEqual(
    line.textContent,
    before,
    'a link error does not repaint the line',
  );
  const notice = Array.from(win.document.querySelectorAll('*')).find(
    el =>
      el.children.length === 0 &&
      /other.*machine unreachable/.test(el.textContent),
  );
  assert.ok(notice, 'the link error is shown as a notification');
  win.close();
  console.log(
    '  ok - replies for other machines and link errors are kept apart',
  );
}

/**
 * Run the shared subpanel tests for `agent` (layout and defaults, status
 * line wording, config load/save, replies for other machines ignored).
 */
function runAgentSubpanelTests(agent) {
  testSubpanelLayoutAndDefaults(agent);
  testStatusLineCases(agent);
  testConfigLoadAndSave(agent);
  testRepliesForOtherMachinesAreIgnored(agent);
}

module.exports = {runAgentSubpanelTests};
