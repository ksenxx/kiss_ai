// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests, driving the real chat.html + api.js +
// main.js, for:
//
// * the Bedrock settings subpanel (after Custom Models, collapsed on
//   every open, holding the five AWS inputs that left API Keys);
// * the "..." menu's appearance items: "Dark mode" / "Light mode",
//   "High contrast" / "Low contrast" (low by default) and "Auto mode"
//   (dark from 7 pm to 7 am on this device's clock, re-checked every
//   minute), all persisted in localStorage;
// * the settings sheet opening for the machine the chat header shows:
//   its loads and saves carry `machine`, only that machine's forwarded
//   replies fill the form, and closing the sheet repaints this
//   daemon's values;
// * the CHATS row details line with the machine, max budget and agent.

'use strict';

const assert = require('assert');
const {makePage, send, pickMachine} = require('./simplify2_harness');
const {lastMsg, click, byId} = require('./settings_helpers');

function label(win, id) {
  return byId(win, id).querySelector('.more-item-label').textContent;
}

const MACHINES = [
  {name: 'laptop', status: 'reachable', local: true},
  {name: 'gpu-box', status: 'reachable', address: 'ssh me@gpu'},
];

/** A page whose chat header lists `laptop` (this daemon) and `gpu-box`. */
function pageWithMachines(opts) {
  const page = makePage(opts);
  send(page.win, {
    type: 'configData',
    config: {},
    apiKeys: {},
    machine: 'laptop',
  });
  send(page.win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  return page;
}

// ---- Bedrock subpanel ----

function testBedrockSubpanel() {
  const {win} = makePage();
  const subpanels = Array.from(
    win.document.querySelectorAll('#config-form .config-subpanel'),
  ).map(el => el.id);
  assert.deepStrictEqual(
    subpanels,
    [
      'api-keys-subpanel',
      'custom-models-subpanel',
      'bedrock-subpanel',
      'claude-code-subpanel',
      'codex-subpanel',
    ],
    'Bedrock is the subpanel after Custom Models, Claude Code after it, Codex last',
  );
  const bedrockBody = byId(win, 'bedrock-body');
  const apiKeysBody = byId(win, 'api-keys-body');
  for (const key of [
    'AWS_BEARER_TOKEN_BEDROCK',
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
    'AWS_SESSION_TOKEN',
    'AWS_REGION',
  ]) {
    assert.ok(
      bedrockBody.querySelector('#cfg-key-' + key),
      key + ' lives in the Bedrock subpanel',
    );
    assert.ok(
      !apiKeysBody.querySelector('#cfg-key-' + key),
      key + ' left the API Keys subpanel',
    );
  }
  assert.ok(
    apiKeysBody.querySelector('#cfg-key-ANTHROPIC_API_KEY'),
    'the other keys stay in API Keys',
  );

  click(win, byId(win, 'settings-btn'));
  const toggle = byId(win, 'bedrock-toggle');
  assert.strictEqual(toggle.textContent.trim(), 'Bedrock');
  assert.strictEqual(bedrockBody.hidden, true, 'Bedrock starts collapsed');
  assert.strictEqual(toggle.getAttribute('aria-expanded'), 'false');
  click(win, toggle);
  assert.strictEqual(bedrockBody.hidden, false, 'the header click expands it');
  assert.strictEqual(toggle.getAttribute('aria-expanded'), 'true');
  click(win, toggle);
  assert.strictEqual(bedrockBody.hidden, true, 'a second click collapses it');

  // Left expanded, it is collapsed again on the next open, and the
  // saved region still lands in its box.
  click(win, toggle);
  click(win, byId(win, 'settings-panel-close'));
  click(win, byId(win, 'settings-btn'));
  assert.strictEqual(bedrockBody.hidden, true, 'reopen re-collapses Bedrock');
  send(win, {
    type: 'configData',
    config: {},
    apiKeys: {AWS_REGION: 'eu-west-1'},
    machine: 'laptop',
  });
  assert.strictEqual(byId(win, 'cfg-key-AWS_REGION').value, 'eu-west-1');
  win.close();
  console.log(
    '  ok - Bedrock is its own collapsible subpanel after Custom Models',
  );
}

// ---- Appearance items ----

/** A page whose Date reports `hour` and whose setInterval is recorded. */
function pageAtHour(hour, intervals) {
  return makePage({
    beforeScripts: win => {
      const RealDate = win.Date;
      class FixedDate extends RealDate {
        getHours() {
          return hour.value;
        }
      }
      win.Date = FixedDate;
      const realSetInterval = win.setInterval;
      win.setInterval = (fn, ms) => {
        intervals.push({fn, ms});
        return realSetInterval(fn, ms);
      };
    },
  });
}

function testThemeAndContrastItems() {
  const {win} = makePage();
  const body = win.document.body;
  assert.ok(body.classList.contains('light-theme'), 'light by default');
  assert.strictEqual(label(win, 'theme-btn'), 'Dark mode');
  assert.strictEqual(
    byId(win, 'theme-btn').getAttribute('aria-label'),
    'Dark mode',
  );
  assert.strictEqual(label(win, 'contrast-btn'), 'High contrast');
  assert.ok(
    !body.classList.contains('high-contrast'),
    'low contrast by default',
  );
  assert.strictEqual(
    byId(win, 'theme-auto-btn').getAttribute('aria-checked'),
    'false',
  );

  click(win, byId(win, 'theme-btn'));
  assert.ok(!body.classList.contains('light-theme'), 'Dark mode paints dark');
  assert.strictEqual(label(win, 'theme-btn'), 'Light mode');
  assert.strictEqual(win.localStorage.getItem('kissRemoteTheme'), 'dark');

  click(win, byId(win, 'contrast-btn'));
  assert.ok(body.classList.contains('high-contrast'), 'High contrast turns on');
  assert.ok(!body.classList.contains('light-theme'), 'the theme is unchanged');
  assert.strictEqual(label(win, 'contrast-btn'), 'Low contrast');
  assert.strictEqual(win.localStorage.getItem('kissRemoteContrast'), 'high');

  click(win, byId(win, 'theme-btn'));
  assert.ok(body.classList.contains('light-theme'), 'Light mode paints light');
  assert.ok(
    body.classList.contains('high-contrast'),
    'contrast survives a theme flip',
  );

  click(win, byId(win, 'contrast-btn'));
  assert.ok(
    !body.classList.contains('high-contrast'),
    'Low contrast turns it off',
  );
  assert.strictEqual(label(win, 'contrast-btn'), 'High contrast');
  assert.strictEqual(win.localStorage.getItem('kissRemoteContrast'), 'low');
  win.close();

  // Both choices come back on the next load.
  const again = makePage({
    beforeScripts: w => {
      w.localStorage.setItem('kissRemoteTheme', 'dark');
      w.localStorage.setItem('kissRemoteContrast', 'high');
    },
  });
  assert.ok(!again.win.document.body.classList.contains('light-theme'));
  assert.ok(again.win.document.body.classList.contains('high-contrast'));
  assert.strictEqual(label(again.win, 'contrast-btn'), 'Low contrast');
  again.win.close();
  console.log('  ok - Dark/Light mode and High/Low contrast flip and persist');
}

function testAutoMode() {
  const hour = {value: 20};
  const intervals = [];
  const {win} = pageAtHour(hour, intervals);
  const body = win.document.body;
  assert.ok(
    body.classList.contains('light-theme'),
    'light until Auto mode is picked',
  );

  click(win, byId(win, 'theme-auto-btn'));
  assert.strictEqual(win.localStorage.getItem('kissRemoteTheme'), 'auto');
  assert.strictEqual(
    byId(win, 'theme-auto-btn').getAttribute('aria-checked'),
    'true',
  );
  assert.ok(!body.classList.contains('light-theme'), 'dark at 8 pm');
  assert.strictEqual(label(win, 'theme-btn'), 'Light mode');

  // The minute check flips the palette when the clock crosses 7 am.
  const check = intervals.find(i => i.ms === 60 * 1000);
  assert.ok(check, 'auto mode re-checks the clock every minute');
  hour.value = 7;
  check.fn();
  assert.ok(body.classList.contains('light-theme'), 'light from 7 am');
  hour.value = 18;
  check.fn();
  assert.ok(body.classList.contains('light-theme'), 'still light at 6 pm');
  hour.value = 19;
  check.fn();
  assert.ok(!body.classList.contains('light-theme'), 'dark from 7 pm');
  assert.strictEqual(
    byId(win, 'theme-auto-btn').getAttribute('aria-checked'),
    'true',
    'auto stays on across the flips',
  );

  // Pinning a palette leaves auto mode; the check then does nothing.
  click(win, byId(win, 'theme-btn'));
  assert.ok(body.classList.contains('light-theme'), 'Light mode pins light');
  assert.strictEqual(win.localStorage.getItem('kissRemoteTheme'), 'light');
  assert.strictEqual(
    byId(win, 'theme-auto-btn').getAttribute('aria-checked'),
    'false',
  );
  hour.value = 23;
  check.fn();
  assert.ok(
    body.classList.contains('light-theme'),
    'pinned light stays at 11 pm',
  );
  win.close();

  // Auto mode comes back on the next load and reads the clock at once.
  const reload = makePage({
    beforeScripts: w => {
      w.localStorage.setItem('kissRemoteTheme', 'auto');
      const RealDate = w.Date;
      w.Date = class extends RealDate {
        getHours() {
          return 2;
        }
      };
    },
  });
  assert.ok(
    !reload.win.document.body.classList.contains('light-theme'),
    'dark at 2 am',
  );
  assert.strictEqual(
    byId(reload.win, 'theme-auto-btn').getAttribute('aria-checked'),
    'true',
  );
  reload.win.close();
  console.log('  ok - Auto mode follows the clock: dark 7 pm to 7 am');
}

// ---- Settings for the machine the chat header shows ----

function testSettingsOpenForTheSelectedMachine() {
  const {win, posted} = pageWithMachines();
  const api = win._testApi;
  api.createNewTab();
  pickMachine(win, 'gpu-box');

  const before = posted.length;
  click(win, byId(win, 'settings-btn'));
  const getConfig = lastMsg(posted.slice(before), 'getConfig');
  const getModels = lastMsg(posted.slice(before), 'getMyModels');
  assert.strictEqual(
    getConfig && getConfig.machine,
    'gpu-box',
    'getConfig is for gpu-box',
  );
  assert.strictEqual(
    getModels && getModels.machine,
    'gpu-box',
    'getMyModels is for gpu-box',
  );
  const line = byId(win, 'settings-machine');
  assert.strictEqual(line.hidden, false, 'the sheet names its machine');
  assert.strictEqual(line.textContent, 'Settings of gpu-box');

  // This daemon's own poll does not touch the form; gpu-box's
  // forwarded reply fills it; another machine's is ignored.
  send(win, {
    type: 'configData',
    config: {remote_password: 'laptop-pw', max_budget: 3, work_dir: '/laptop'},
    apiKeys: {OPENAI_API_KEY: 'sk-laptop'},
    machine: 'laptop',
  });
  assert.strictEqual(byId(win, 'cfg-remote-password').value, '');
  assert.strictEqual(byId(win, 'cfg-key-OPENAI_API_KEY').value, '');
  assert.strictEqual(
    byId(win, 'meta-workdir').textContent,
    '/laptop',
    'the work dir is still adopted (the TASK INFO Workdir row shows it)',
  );
  send(win, {
    type: 'configData',
    forwarded: true,
    machine: 'other-box',
    config: {remote_password: 'other-pw'},
    apiKeys: {},
  });
  assert.strictEqual(byId(win, 'cfg-remote-password').value, '');
  send(win, {
    type: 'configData',
    forwarded: true,
    machine: 'gpu-box',
    config: {remote_password: 'gpu-pw', max_budget: 42, use_web_browser: false},
    apiKeys: {OPENAI_API_KEY: 'sk-gpu', AWS_REGION: 'us-west-2'},
  });
  assert.strictEqual(byId(win, 'cfg-remote-password').value, 'gpu-pw');
  assert.strictEqual(byId(win, 'cfg-max-budget').value, '42');
  assert.strictEqual(byId(win, 'cfg-key-OPENAI_API_KEY').value, 'sk-gpu');
  assert.strictEqual(byId(win, 'cfg-key-AWS_REGION').value, 'us-west-2');
  assert.strictEqual(byId(win, 'cfg-use-web-tools').checked, false);
  assert.strictEqual(
    byId(win, 'meta-workdir').textContent,
    '/laptop',
    "gpu-box's config does not move this daemon's work dir",
  );

  // Custom models: only gpu-box's forwarded list is painted, and
  // the Add / Delete commands carry the machine.
  send(win, {type: 'myModelsData', models: [{name: 'laptop-model'}]});
  send(win, {
    type: 'myModelsData',
    forwarded: true,
    machine: 'gpu-box',
    models: [{name: 'gpu-model', endpoint: '', api_key: '', headers: ''}],
  });
  const rows = Array.from(
    win.document.querySelectorAll('#custom-models-list .custom-model-name'),
  ).map(el => el.textContent);
  assert.deepStrictEqual(rows, ['gpu-model']);
  byId(win, 'cfg-custom-model-name').value = 'new-model';
  byId(win, 'cfg-custom-model-name').dispatchEvent(
    new win.Event('input', {bubbles: true}),
  );
  click(win, byId(win, 'custom-model-add-btn'));
  const save = lastMsg(posted, 'saveMyModel');
  assert.strictEqual(
    save && save.machine,
    'gpu-box',
    'saveMyModel is for gpu-box',
  );
  assert.strictEqual(save.name, 'new-model');

  // A machine that does not answer is reported, not painted as blank.
  send(win, {
    type: 'configData',
    forwarded: true,
    machine: 'gpu-box',
    error: 'gpu-box (ssh me@gpu) did not answer',
  });
  assert.strictEqual(byId(win, 'cfg-remote-password').value, 'gpu-pw');
  const notice = win.document.querySelector(
    '.kiss-notification[data-notification-id="settings-machine-error"]',
  );
  assert.ok(notice, 'the error is shown as a notification');
  assert.ok(/did not answer/.test(notice.textContent), notice.textContent);

  // Closing saves to gpu-box, then asks this daemon for its own
  // values back and drops the machine line.
  byId(win, 'cfg-remote-password').value = 'gpu-pw-2';
  byId(win, 'cfg-remote-password').dispatchEvent(
    new win.Event('input', {bubbles: true}),
  );
  const beforeClose = posted.length;
  click(win, byId(win, 'settings-panel-close'));
  const saved = lastMsg(posted.slice(beforeClose), 'saveConfig');
  assert.strictEqual(
    saved && saved.machine,
    'gpu-box',
    'saveConfig is for gpu-box',
  );
  assert.strictEqual(saved.config.remote_password, 'gpu-pw-2');
  const reload = lastMsg(posted.slice(beforeClose), 'getConfig');
  assert.ok(
    reload && !('machine' in reload),
    'this daemon is asked for its config',
  );
  assert.ok(
    lastMsg(posted.slice(beforeClose), 'getMyModels') &&
      !('machine' in lastMsg(posted.slice(beforeClose), 'getMyModels')),
    'and for its custom models',
  );
  assert.strictEqual(line.hidden, true);
  send(win, {
    type: 'configData',
    config: {remote_password: 'laptop-pw'},
    apiKeys: {},
    machine: 'laptop',
  });
  assert.strictEqual(byId(win, 'cfg-remote-password').value, 'laptop-pw');

  // A gpu-box reply that arrives after the close is ignored.
  send(win, {
    type: 'configData',
    forwarded: true,
    machine: 'gpu-box',
    config: {remote_password: 'late'},
    apiKeys: {},
  });
  assert.strictEqual(byId(win, 'cfg-remote-password').value, 'laptop-pw');

  // A tab on this daemon opens this daemon's settings: no machine.
  pickMachine(win, '');
  const beforeLocal = posted.length;
  click(win, byId(win, 'settings-btn'));
  const local = lastMsg(posted.slice(beforeLocal), 'getConfig');
  assert.ok(local && !('machine' in local), 'a local tab asks this daemon');
  assert.strictEqual(line.hidden, true);
  win.close();
  console.log("  ok - the settings sheet opens for the chat header's machine");
}

function testRemoteSheetKeepsThisDaemonsStateApart() {
  const {win, posted} = pageWithMachines();
  const api = win._testApi;
  const panel = byId(win, 'settings-panel');
  // This daemon: web tools off, two custom models.
  send(win, {
    type: 'configData',
    config: {use_web_browser: false, use_memory: true},
    apiKeys: {},
    machine: 'laptop',
  });
  send(win, {
    type: 'myModelsData',
    models: [
      {
        name: 'laptop-model',
        endpoint: 'http://laptop:1/v1',
        api_key: 'sk-laptop',
      },
    ],
  });
  assert.strictEqual(byId(win, 'cfg-use-web-tools').checked, false);

  api.createNewTab();
  pickMachine(win, 'gpu-box');
  click(win, byId(win, 'settings-btn'));
  assert.ok(
    panel.classList.contains('for-machine'),
    'the sheet is marked as a machine sheet',
  );
  // This daemon's model list is gone before gpu-box's arrives: no row
  // to edit, so nothing of this daemon's can be posted to gpu-box.
  assert.strictEqual(
    win.document.querySelectorAll('#custom-models-list .custom-model-name')
      .length,
    0,
    "this daemon's custom models are not on offer for gpu-box",
  );
  send(win, {
    type: 'configData',
    forwarded: true,
    machine: 'gpu-box',
    config: {use_web_browser: true, use_memory: false},
    apiKeys: {},
  });
  assert.strictEqual(
    byId(win, 'cfg-use-web-tools').checked,
    true,
    "gpu-box's toggle",
  );
  assert.strictEqual(
    byId(win, 'cfg-use-memory').checked,
    false,
    "gpu-box's toggle",
  );

  // Closing restores this daemon's toggles at once, before its reply.
  const beforeClose = posted.length;
  click(win, byId(win, 'settings-panel-close'));
  assert.ok(!panel.classList.contains('for-machine'));
  assert.strictEqual(
    byId(win, 'cfg-use-web-tools').checked,
    false,
    "this daemon's toggle is back",
  );
  assert.strictEqual(
    byId(win, 'cfg-use-memory').checked,
    true,
    "this daemon's toggle is back",
  );
  assert.strictEqual(
    win.document.querySelectorAll('#custom-models-list .custom-model-name')
      .length,
    0,
    "gpu-box's models are not on offer for this daemon",
  );
  assert.ok(
    lastMsg(posted.slice(beforeClose), 'saveConfig').machine === 'gpu-box',
  );
  // The machine refused the close's save: still reported.
  send(win, {
    type: 'configData',
    forwarded: true,
    machine: 'gpu-box',
    error: 'gpu-box (ssh me@gpu) did not answer',
  });
  const notice = win.document.querySelector(
    '.kiss-notification[data-notification-id="settings-machine-error"]',
  );
  assert.ok(notice, 'a refusal after the close is still shown');
  win.close();
  console.log("  ok - a machine's sheet never mixes with this daemon's state");
}

function testAppearanceSurvivesRefusedStorage() {
  const {win} = makePage({
    beforeScripts: w => {
      w.Storage.prototype.setItem = () => {
        throw new Error('QuotaExceededError');
      };
    },
  });
  const body = win.document.body;
  click(win, byId(win, 'contrast-btn'));
  assert.ok(
    body.classList.contains('high-contrast'),
    'contrast switches without storage',
  );
  assert.strictEqual(label(win, 'contrast-btn'), 'Low contrast');
  click(win, byId(win, 'theme-auto-btn'));
  assert.strictEqual(
    byId(win, 'theme-auto-btn').getAttribute('aria-checked'),
    'true',
    'auto mode switches on without storage',
  );
  assert.ok(
    body.classList.contains('high-contrast'),
    'contrast survives the mode change',
  );
  win.close();
  console.log(
    '  ok - appearance choices work for the session when storage refuses writes',
  );
}

// ---- CHATS row details: machine, max budget, agent ----

function testHistoryRowSettingsLine() {
  const {win, posted} = pageWithMachines();
  win.document.getElementById('sidebar').classList.add('open');
  const req = posted.filter(m => m && m.type === 'getHistory').pop();
  const row = overrides =>
    Object.assign(
      {
        id: 'chat-' + overrides.task_id,
        task_id: overrides.task_id,
        title: 'task ' + overrides.task_id,
        timestamp: 1_700_000_000,
        preview: 'task ' + overrides.task_id,
        has_events: false,
        failed: false,
        is_running: false,
        awaiting_answer: false,
        tokens: 1,
        cost: 0,
        steps: 1,
        is_favorite: false,
        work_dir: '',
        startTs: 1_700_000_000_000,
        endTs: 0,
      },
      overrides,
    );
  send(win, {
    type: 'history',
    offset: 0,
    generation: req ? req.generation : undefined,
    sessions: [
      row({
        task_id: 1,
        machine: 'gpu-box',
        max_budget: 12.5,
        sea: 'write_paper',
        tool_profile: 'review',
        timeout: 60,
      }),
      row({task_id: 2, max_budget: 0}),
    ],
  });
  const lines = {};
  win.document.querySelectorAll('#history-list .sidebar-item').forEach(item => {
    const text = item.querySelector('.sidebar-item-text').textContent.trim();
    const settings = item.querySelector('.running-item-settings');
    lines[text] = settings ? settings.textContent : null;
  });
  assert.strictEqual(
    lines['task 1'],
    'machine gpu-box • max budget $12.50 • agent write_paper • tools review • timeout 60s',
  );
  assert.strictEqual(
    lines['task 2'],
    'machine laptop',
    'a local row without a budget or agent names this daemon only',
  );
  win.close();
  console.log(
    '  ok - CHATS row details list the machine, max budget and agent',
  );
}

try {
  testBedrockSubpanel();
  testThemeAndContrastItems();
  testAutoMode();
  testSettingsOpenForTheSelectedMachine();
  testHistoryRowSettingsLine();
  testRemoteSheetKeepsThisDaemonsStateApart();
  testAppearanceSurvivesRefusedStorage();
  console.log('\n7 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
