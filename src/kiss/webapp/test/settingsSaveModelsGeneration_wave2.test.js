// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// This daemon answers a settings save with its model list, like a
// getModels (new keys or endpoints change the list).  The user can pick
// a model in the picker before that reply comes back, and the reply's
// `selected` is the default the daemon had when it read the save, from
// before the pick.  The save therefore carries a `generation` that the
// daemon echoes on the reply (server/commands.py `_cmd_save_config`), so
// main.js (noteMachineModels) can tell the reply is older than the pick
// and keep the pick.  A registered machine's save brings back no model
// list, so it is not numbered; the picker asks that machine anew.

const assert = require('assert');
const {
  makePage,
  send,
  stubRaf,
  pickMachine,
  click,
  byId,
  ofType,
} = require('./simplify2_harness');
const {MACHINES} = require('./settings_helpers');
const {test, report} = require('./ui_antipattern_harness').makeRunner();

const MODELS = [
  {name: 'model-a', inp: 1, out: 2, uses: 0, vendor: 'anthropic'},
  {name: 'model-b', inp: 3, out: 4, uses: 0, vendor: 'openai'},
];

/** A connected page with this daemon's model list (default model-a). */
function pickerPage() {
  const page = makePage({modelName: 'model-a', beforeScripts: stubRaf});
  send(page.win, {type: 'daemonStatus', connected: true});
  send(page.win, {
    type: 'configData',
    config: {},
    apiKeys: {},
    machine: 'laptop',
  });
  send(page.win, {type: 'machines', machines: MACHINES, checkedAt: 1});
  send(page.win, {type: 'models', models: MODELS, selected: 'model-a'});
  return page;
}

/** The model the picker's pill shows. */
function label(win) {
  return byId(win, 'model-name').textContent.replace(/\u200e/g, '');
}

/** Open the picker and click *model*. */
function pick(win, model) {
  click(win, byId(win, 'model-btn'));
  const item = Array.from(
    win.document.querySelectorAll('#model-list .model-item'),
  ).find(
    el =>
      el
        .querySelector('.model-item-name')
        .textContent.replace(/\u200e/g, '') === model,
  );
  assert.ok(item, 'the picker offers ' + model);
  click(win, item);
}

/**
 * Open the settings sheet, change the budget and close the sheet, as
 * the user does; *config* is the sheet's configData reply.
 *
 * @returns {object} The one saveConfig the close posted.
 */
function saveSettings(win, posted, config) {
  click(win, byId(win, 'settings-btn'));
  send(win, {apiKeys: {}, ...config});
  const budget = byId(win, 'cfg-max-budget');
  budget.value = '7';
  budget.dispatchEvent(new win.Event('input', {bubbles: true}));
  const from = posted.length;
  click(win, byId(win, 'settings-panel-close'));
  const saves = ofType(posted.slice(from), 'saveConfig');
  assert.strictEqual(saves.length, 1, 'closing the edited sheet saves once');
  assert.strictEqual(saves[0].config.max_budget, 7);
  return saves[0];
}

const LOCAL_CONFIG = {type: 'configData', config: {max_budget: 3}};

/** This daemon's reply to *save*: its model list, echoing the save's generation. */
function answerSave(win, save, selected) {
  send(win, {
    type: 'models',
    models: MODELS,
    selected,
    generation: save.generation,
  });
}

test("a pick made before the save's model list arrives stands", () => {
  const {win, posted} = pickerPage();
  const save = saveSettings(win, posted, LOCAL_CONFIG);
  assert.ok(
    Number.isInteger(save.generation) && save.generation > 0,
    'the save of this daemon is numbered: ' + JSON.stringify(save),
  );
  pick(win, 'model-b');
  assert.strictEqual(label(win), 'model-b');
  answerSave(win, save, 'model-a');
  assert.strictEqual(
    label(win),
    'model-b',
    "the save's reply carries the default from before the pick",
  );
  byId(win, 'task-input').value = 'go';
  click(win, byId(win, 'send-btn'));
  const submits = ofType(posted, 'submit');
  assert.strictEqual(submits[submits.length - 1].model, 'model-b');
  win.close();
});

test('the model list of a save made after the pick is current', () => {
  const {win, posted} = pickerPage();
  pick(win, 'model-b');
  const save = saveSettings(win, posted, LOCAL_CONFIG);
  // Another surface made model-a the default since the pick.
  answerSave(win, save, 'model-a');
  assert.strictEqual(label(win), 'model-a');
  win.close();
});

test("a registered machine's save is not numbered and asks for its list", () => {
  const {win, posted} = pickerPage();
  pickMachine(win, 'gpu-box');
  const from = posted.length;
  const save = saveSettings(win, posted, {
    type: 'configData',
    forwarded: true,
    machine: 'gpu-box',
    config: {max_budget: 42},
  });
  assert.strictEqual(save.machine, 'gpu-box');
  assert.ok(!('generation' in save), JSON.stringify(save));
  const asks = ofType(posted.slice(from), 'getModels').filter(
    m => m.machine === 'gpu-box',
  );
  assert.ok(asks.length >= 1, 'the picker asks gpu-box for its list again');
  win.close();
});

report('settingsSaveModelsGeneration_wave2');
