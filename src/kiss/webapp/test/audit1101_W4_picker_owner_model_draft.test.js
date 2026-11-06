// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) regressions for the 2026-11-01 audit of
// media/main.js (W4, lines 22800-end).
//
// * F1: the native file picker the Upload button opens belongs to the
//   chat it was opened in.  A websocket `tabs_state` may close that
//   chat, or the user switch chats, while the picker is open: the files
//   it then delivers go nowhere for a closed chat, and into the
//   opening chat's own (parked) attachments for a chat in the
//   background -- never into whichever chat is on screen.
// * F2: a custom-model Save puts the boxes back at once (the list
//   repaints on the daemon's myModelsData), but the daemon may still
//   refuse the save (the name was taken by another surface since this
//   one's last list, the file could not be written): the refusal
//   brings the draft back into the editor instead of losing it.  The
//   acknowledged save, and a refusal landing on a newer edit or after
//   the sheet moved to another machine, change nothing.
// * Round 2 (F1, F2): a refusal landing after the user typed into the
//   boxes again (a newer Add draft) leaves that typing alone, for a
//   refused edit as for a refused Add; and a save is acknowledged only
//   by a list that holds the entry as sent (every value, a renamed
//   model's old name gone), not by another surface's broadcast that
//   merely names the same model.

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  twoTabs,
  waitFor,
  sleep,
  byId,
  pickMachine,
} = require('./simplify2_harness');
const {openSettings, typeInto} = require('./settings_helpers');
const {makeRunner} = require('./ui_antipattern_harness');

const {test, report} = makeRunner();
const pages = [];

/**
 * A page whose file pickers are captured instead of opened natively and
 * whose FileReader reads are observed: `reads` holds one promise per
 * read the page started, settled on its `loadend`, so a negative check
 * ("the file went nowhere") can wait for the real end of every read
 * instead of a fixed sleep.  The platform reader itself is untouched.
 */
function pageWithPicker() {
  const pickers = [];
  const reads = [];
  const page = makePage({
    beforeScripts(w) {
      const click = w.HTMLInputElement.prototype.click;
      w.HTMLInputElement.prototype.click = function () {
        if (this.type === 'file') {
          pickers.push(this);
          return undefined;
        }
        return click.call(this);
      };
      const readAsDataURL = w.FileReader.prototype.readAsDataURL;
      w.FileReader.prototype.readAsDataURL = function (...args) {
        reads.push(
          new Promise(resolve => this.addEventListener('loadend', resolve)),
        );
        return readAsDataURL.apply(this, args);
      };
    },
  });
  pages.push(page.win);
  return Object.assign(page, {pickers, reads});
}

/**
 * Resolve once every read the page started has ended and the awaited
 * chain behind it (prepareAttachment -> fillAttachmentSlot -> chips) has
 * run: the moment a leaked file would be on screen.
 */
async function readsSettled(page) {
  await Promise.all(page.reads);
  await sleep(0);
}

function pick(win, picker, name) {
  Object.defineProperty(picker, 'files', {
    value: [new win.File(['bytes'], name, {type: 'application/pdf'})],
  });
  picker.dispatchEvent(new win.Event('change'));
}

function chipNames(win) {
  return Array.from(
    win.document.querySelectorAll('#file-chips .file-chip'),
  ).map(c => c.querySelector('span:not(.fc-icon)').textContent.trim());
}

function openPickerIn(page, tabId) {
  const {win, pickers} = page;
  page.win._testApi.switchToTab(tabId);
  const before = pickers.length;
  byId(win, 'upload-btn').click();
  assert.strictEqual(
    pickers.length,
    before + 1,
    'the Upload button opens a picker',
  );
  return pickers[before];
}

function boxValues(win) {
  return {
    name: byId(win, 'cfg-custom-model-name').value,
    endpoint: byId(win, 'cfg-custom-endpoint').value,
    apiKey: byId(win, 'cfg-custom-api-key').value,
    headers: byId(win, 'cfg-custom-headers').value,
  };
}

function editVisible(win) {
  return byId(win, 'custom-model-save-btn').style.display !== 'none';
}

const OLD = {
  name: 'old-model',
  endpoint: 'https://model.example/v1',
  api_key: 'old-key',
  headers: 'X-Old: 1',
};
const DRAFT = {
  name: 'concurrent-model',
  endpoint: 'https://new-unsaved.example/v1',
  apiKey: 'new-unsaved-key',
  headers: 'X-Unsaved: unique',
};

/** Open the sheet on one model and save an edit of all four boxes. */
function editAndSave(win, posted) {
  openSettings(win);
  send(win, {
    type: 'configData',
    config: {
      custom_endpoint: 'https://config-default.example/v1',
      custom_api_key: 'config-key',
      custom_headers: 'X-Config: before',
    },
  });
  send(win, {type: 'myModelsData', models: [OLD]});
  win.document.querySelector('.custom-model-edit-btn').click();
  typeInto(win, 'cfg-custom-model-name', DRAFT.name);
  typeInto(win, 'cfg-custom-endpoint', DRAFT.endpoint);
  typeInto(win, 'cfg-custom-api-key', DRAFT.apiKey);
  typeInto(win, 'cfg-custom-headers', DRAFT.headers);
  byId(win, 'custom-model-save-btn').click();
  const msg = posted.filter(m => m.type === 'saveMyModel').pop();
  assert.strictEqual(msg.originalName, OLD.name);
  assert.strictEqual(msg.name, DRAFT.name);
  // As before: Save puts the pre-edit values back at once.
  assert.strictEqual(editVisible(win), false, 'the editor closes on Save');
  assert.strictEqual(
    boxValues(win).endpoint,
    'https://config-default.example/v1',
  );
  return msg;
}

async function main() {
  await test('F1: a picker opened in a chat closed meanwhile attaches nothing to its successor', async () => {
    const page = pageWithPicker();
    const {win, posted} = page;
    const {api, first, second} = twoTabs(win);
    api.endLaunch();
    send(win, {
      type: 'tabs_state',
      tabs: [
        {tabId: first, chatId: '', title: 'A'},
        {tabId: second, chatId: '', title: 'B'},
      ],
    });
    const picker = openPickerIn(page, first);
    send(win, {
      type: 'tabs_state',
      tabs: [{tabId: second, chatId: '', title: 'B'}],
    });
    assert.strictEqual(api.getActiveTabId(), second);
    assert.ok(
      !api.openTabs().some(t => t.id === first),
      'A was closed remotely',
    );
    pick(win, picker, 'A-private.pdf');
    await readsSettled(page);
    assert.deepStrictEqual(chipNames(win), [], "BUG: A's file appeared in B");
    byId(win, 'task-input').value = 'Task intended for B';
    byId(win, 'send-btn').click();
    const frame = posted.filter(m => m.type === 'submit').pop();
    assert.strictEqual(frame.tabId, second);
    assert.strictEqual(
      (frame.attachments || []).length,
      0,
      "BUG: B's submit carries A's file",
    );
  });

  await test('F1: a picker opened in a chat left for another parks its file in the opening chat', async () => {
    const page = pageWithPicker();
    const {win} = page;
    const {api, first, second} = twoTabs(win);
    api.endLaunch();
    const picker = openPickerIn(page, first);
    api.switchToTab(second);
    pick(win, picker, 'A-private.pdf');
    // The read ends while B is still on screen: a slot that landed in
    // whichever chat is active when it completes would show here.
    await readsSettled(page);
    assert.strictEqual(page.reads.length, 1, 'setup: the file was read');
    assert.deepStrictEqual(chipNames(win), [], "BUG: A's file appeared in B");
    api.switchToTab(first);
    await waitFor(
      () => win.document.querySelector('#file-chips .file-chip:not(.pending)'),
      "A's chip is ready",
    );
    assert.deepStrictEqual(
      chipNames(win),
      ['A-private.pdf'],
      "the file is A's",
    );
    api.switchToTab(second);
    assert.deepStrictEqual(chipNames(win), [], 'B still has none');
  });

  await test('F1: a picker completed in the chat it was opened in attaches as before', async () => {
    const page = pageWithPicker();
    const {win, posted} = page;
    const {api, first} = twoTabs(win);
    api.endLaunch();
    const picker = openPickerIn(page, first);
    pick(win, picker, 'mine.pdf');
    await waitFor(
      () => win.document.querySelector('#file-chips .file-chip:not(.pending)'),
      'the chip is ready',
    );
    assert.deepStrictEqual(chipNames(win), ['mine.pdf']);
    byId(win, 'task-input').value = 'Task with a file';
    byId(win, 'send-btn').click();
    const frame = posted.filter(m => m.type === 'submit').pop();
    assert.strictEqual(frame.tabId, first);
    assert.strictEqual(
      frame.attachments.map(f => f.name).join(','),
      'mine.pdf',
    );
  });

  await test('F2: a refused rename brings the draft back into the editor', () => {
    const {win, posted} = makePage();
    pages.push(win);
    editAndSave(win, posted);
    // Another surface created the target name since this surface's last
    // myModelsData: save_custom_model refuses the rename (model_info.py).
    send(win, {
      type: 'error',
      text: "A custom model named 'concurrent-model' already exists",
    });
    assert.strictEqual(
      editVisible(win),
      true,
      'BUG: the refused edit is not back',
    );
    assert.deepStrictEqual(boxValues(win), DRAFT, 'with every typed value');
    // It is still an edit of the old model: Save posts the same rename.
    byId(win, 'custom-model-save-btn').click();
    const retry = posted.filter(m => m.type === 'saveMyModel').pop();
    assert.strictEqual(retry.originalName, OLD.name);
    assert.strictEqual(retry.name, DRAFT.name);
    assert.strictEqual(retry.endpoint, DRAFT.endpoint);
    // Cancel restores the pre-edit values, as for any edit.
    send(win, {
      type: 'error',
      text: 'Could not write ~/.s10s/MY_MODELS.json: disk full',
    });
    assert.strictEqual(
      editVisible(win),
      true,
      'a write failure brings it back too',
    );
    byId(win, 'custom-model-cancel-btn').click();
    assert.strictEqual(editVisible(win), false);
    assert.strictEqual(
      boxValues(win).endpoint,
      'https://config-default.example/v1',
    );
  });

  await test('F2: the acknowledged save is done; a later error reopens nothing', () => {
    const {win, posted} = makePage();
    pages.push(win);
    editAndSave(win, posted);
    send(win, {
      type: 'myModelsData',
      models: [
        {
          name: DRAFT.name,
          endpoint: DRAFT.endpoint,
          api_key: DRAFT.apiKey,
          headers: DRAFT.headers,
        },
      ],
    });
    assert.strictEqual(editVisible(win), false);
    send(win, {type: 'error', text: 'Update failed: no network'});
    assert.strictEqual(
      editVisible(win),
      false,
      'an unrelated error reopens no editor',
    );
    assert.strictEqual(boxValues(win).name, '');
  });

  await test("A20-3: another command's refusal leaves the pending save alone", () => {
    const {win, posted} = makePage();
    pages.push(win);
    editAndSave(win, posted);
    // The daemon stamps a refused mutation with the command it answers
    // (commands.py _send_error_to_sender): a promptlet add refused
    // while this save awaits its answer is not the save's refusal.
    send(win, {
      type: 'error',
      command: 'addTrick',
      text: 'That promptlet is already in MY_INJECTION.md',
    });
    assert.strictEqual(
      editVisible(win),
      false,
      "another command's refusal reopens no editor",
    );
    // The save's own refusal still brings the draft back.
    send(win, {
      type: 'error',
      command: 'saveMyModel',
      text: "A custom model named 'concurrent-model' already exists",
    });
    assert.strictEqual(editVisible(win), true, 'the refused edit is back');
    assert.deepStrictEqual(boxValues(win), DRAFT);
  });

  await test('F2: a refusal landing on a newer edit leaves that edit alone', () => {
    const {win, posted} = makePage();
    pages.push(win);
    editAndSave(win, posted);
    send(win, {
      type: 'myModelsData',
      models: [
        OLD,
        {name: 'other', endpoint: 'https://o/v1', api_key: 'k', headers: ''},
      ],
    });
    win.document.querySelectorAll('.custom-model-edit-btn')[1].click();
    typeInto(win, 'cfg-custom-endpoint', 'https://other-draft/v1');
    send(win, {
      type: 'error',
      text: "A custom model named 'concurrent-model' already exists",
    });
    assert.strictEqual(editVisible(win), true);
    assert.strictEqual(boxValues(win).name, 'other', 'the newer edit stays');
    assert.strictEqual(boxValues(win).endpoint, 'https://other-draft/v1');
  });

  await test('F2 (F10): two saves in flight: the first refusal brings back the first draft, not the second', () => {
    const {win, posted} = makePage();
    pages.push(win);
    editAndSave(win, posted);
    // Before the daemon answers that save, the list (another surface's
    // change) shows a second model, which the user edits and saves as
    // it is: two saves await their answers, in order.
    const other = {
      name: 'other',
      endpoint: 'https://o/v1',
      api_key: 'k',
      headers: '',
    };
    send(win, {type: 'myModelsData', models: [OLD, other]});
    assert.strictEqual(editVisible(win), false, 'nothing acknowledged yet');
    win.document.querySelectorAll('.custom-model-edit-btn')[1].click();
    byId(win, 'custom-model-save-btn').click();
    const second = posted.filter(m => m.type === 'saveMyModel').pop();
    assert.strictEqual(second.originalName, other.name, 'setup: second save');
    // The first save is refused: its draft comes back, not the second's.
    send(win, {
      type: 'error',
      text: "A custom model named 'concurrent-model' already exists",
    });
    assert.strictEqual(editVisible(win), true, 'BUG: the first draft is not back');
    assert.deepStrictEqual(
      boxValues(win),
      DRAFT,
      "BUG: the second save's draft came back instead of the refused one",
    );
    // The second save is acknowledged by the list: the editor on the
    // first draft is left alone.
    send(win, {type: 'myModelsData', models: [OLD, other]});
    assert.strictEqual(editVisible(win), true);
    assert.deepStrictEqual(boxValues(win), DRAFT);
    byId(win, 'custom-model-save-btn').click();
    const retry = posted.filter(m => m.type === 'saveMyModel').pop();
    assert.strictEqual(retry.originalName, OLD.name, 'Save retries the rename');
    assert.strictEqual(retry.name, DRAFT.name);
  });

  await test('F1 (round 2): a refused edit leaves a newer Add draft typed since alone', () => {
    const {win, posted} = makePage();
    pages.push(win);
    editAndSave(win, posted);
    const next = {
      name: 'next-add',
      endpoint: 'https://newer/v1',
      apiKey: 'newer-key',
      headers: 'Newer: 1',
    };
    typeInto(win, 'cfg-custom-model-name', next.name);
    typeInto(win, 'cfg-custom-endpoint', next.endpoint);
    typeInto(win, 'cfg-custom-api-key', next.apiKey);
    typeInto(win, 'cfg-custom-headers', next.headers);
    send(win, {
      type: 'error',
      text: "A custom model named 'concurrent-model' already exists",
    });
    assert.strictEqual(editVisible(win), false, 'BUG: the old edit reopened');
    assert.deepStrictEqual(boxValues(win), next, 'the newer draft stands');
    // What Add now submits is that draft, not the refused one.
    byId(win, 'custom-model-add-btn').click();
    const msg = posted.filter(m => m.type === 'saveMyModel').pop();
    assert.strictEqual(msg.name, next.name);
    assert.strictEqual(msg.originalName, undefined);
  });

  await test('F1 (round 2): typing only the endpoint since also keeps a refused edit away', () => {
    const {win, posted} = makePage();
    pages.push(win);
    editAndSave(win, posted);
    typeInto(win, 'cfg-custom-endpoint', 'https://typed-since/v1');
    send(win, {
      type: 'error',
      text: "A custom model named 'concurrent-model' already exists",
    });
    assert.strictEqual(editVisible(win), false);
    assert.strictEqual(boxValues(win).endpoint, 'https://typed-since/v1');
    assert.strictEqual(boxValues(win).name, '');
  });

  await test("F2 (round 2): another surface's list naming the target does not acknowledge the rename", () => {
    const {win, posted} = makePage();
    pages.push(win);
    editAndSave(win, posted);
    // Another surface won the same name; the original is still listed:
    // this list cannot be the answer to the rename.
    send(win, {
      type: 'myModelsData',
      models: [
        OLD,
        {
          name: DRAFT.name,
          endpoint: 'https://competitor/v1',
          api_key: 'competitor-key',
          headers: '',
        },
      ],
    });
    assert.strictEqual(editVisible(win), false);
    send(win, {
      type: 'error',
      text: "A custom model named 'concurrent-model' already exists",
    });
    assert.strictEqual(
      editVisible(win),
      true,
      'BUG: the competing list was taken for the acknowledgment',
    );
    assert.deepStrictEqual(boxValues(win), DRAFT);
    byId(win, 'custom-model-save-btn').click();
    const retry = posted.filter(m => m.type === 'saveMyModel').pop();
    assert.strictEqual(retry.originalName, OLD.name);
  });

  await test('F2 (round 2): an edit keeping its name is acknowledged by its values, not its name', () => {
    const {win, posted} = makePage();
    pages.push(win);
    openSettings(win);
    send(win, {type: 'myModelsData', models: [OLD]});
    win.document.querySelector('.custom-model-edit-btn').click();
    typeInto(win, 'cfg-custom-endpoint', 'https://edited/v1');
    byId(win, 'custom-model-save-btn').click();
    const msg = posted.filter(m => m.type === 'saveMyModel').pop();
    assert.strictEqual(msg.originalName, OLD.name);
    assert.strictEqual(msg.name, OLD.name);
    // Another surface added a model meanwhile: its broadcast still
    // lists old-model with the old endpoint.
    send(win, {
      type: 'myModelsData',
      models: [
        OLD,
        {name: 'other', endpoint: 'https://o/v1', api_key: 'k', headers: ''},
      ],
    });
    send(win, {
      type: 'error',
      text: 'Could not write ~/.s10s/MY_MODELS.json: disk full',
    });
    assert.strictEqual(
      editVisible(win),
      true,
      'BUG: the unrelated list was taken for the acknowledgment',
    );
    assert.strictEqual(boxValues(win).endpoint, 'https://edited/v1');
    // Saved again and listed as sent: done; a later error reopens nothing.
    byId(win, 'custom-model-save-btn').click();
    send(win, {
      type: 'myModelsData',
      models: [
        {
          name: OLD.name,
          endpoint: 'https://edited/v1',
          api_key: OLD.api_key,
          headers: OLD.headers,
        },
        {name: 'other', endpoint: 'https://o/v1', api_key: 'k', headers: ''},
      ],
    });
    send(win, {type: 'error', text: 'Update failed: no network'});
    assert.strictEqual(editVisible(win), false);
  });

  await test("F2 (round 2): a competing Add of the same name does not acknowledge this surface's Add", () => {
    const {win, posted} = makePage();
    pages.push(win);
    openSettings(win);
    send(win, {type: 'myModelsData', models: [OLD]});
    typeInto(win, 'cfg-custom-model-name', 'added-model');
    typeInto(win, 'cfg-custom-endpoint', 'https://added/v1');
    byId(win, 'custom-model-add-btn').click();
    assert.strictEqual(posted.filter(m => m.type === 'saveMyModel').pop().name, 'added-model');
    send(win, {
      type: 'myModelsData',
      models: [
        OLD,
        {name: 'added-model', endpoint: 'https://elsewhere/v1', api_key: '', headers: ''},
      ],
    });
    send(win, {
      type: 'error',
      text: "A custom model named 'added-model' already exists; use its Edit button to change it",
    });
    assert.strictEqual(
      boxValues(win).name,
      'added-model',
      'BUG: the competing list was taken for the acknowledgment',
    );
    assert.strictEqual(boxValues(win).endpoint, 'https://added/v1');
  });

  await test('F2: a refused Add puts the cleared name back', () => {
    const {win, posted} = makePage();
    pages.push(win);
    openSettings(win);
    send(win, {type: 'myModelsData', models: [OLD]});
    typeInto(win, 'cfg-custom-model-name', 'added-model');
    typeInto(win, 'cfg-custom-endpoint', 'https://added/v1');
    byId(win, 'custom-model-add-btn').click();
    const msg = posted.filter(m => m.type === 'saveMyModel').pop();
    assert.strictEqual(msg.name, 'added-model');
    assert.strictEqual(
      boxValues(win).name,
      '',
      'Add clears the name box, as before',
    );
    send(win, {
      type: 'error',
      text: "A custom model named 'added-model' already exists; use its Edit button to change it",
    });
    assert.strictEqual(
      boxValues(win).name,
      'added-model',
      'BUG: the refused Add lost its name',
    );
    assert.strictEqual(boxValues(win).endpoint, 'https://added/v1');
    assert.strictEqual(editVisible(win), false, 'still an Add, not an edit');
  });

  await test("F2: a remote machine's refusal brings the draft back while its sheet shows", () => {
    const {win, posted} = makePage();
    pages.push(win);
    send(win, {
      type: 'machines',
      machines: [
        {name: 'local', local: true, status: 'reachable'},
        {name: 'gpu-box', status: 'reachable'},
      ],
      checkedAt: 1,
    });
    win._testApi.createNewTab();
    pickMachine(win, 'gpu-box');
    openSettings(win);
    send(win, {
      type: 'configData',
      forwarded: true,
      machine: 'gpu-box',
      config: {custom_endpoint: 'https://gpu-default/v1'},
      apiKeys: {},
    });
    send(win, {
      type: 'myModelsData',
      forwarded: true,
      machine: 'gpu-box',
      models: [OLD],
    });
    win.document.querySelector('.custom-model-edit-btn').click();
    typeInto(win, 'cfg-custom-model-name', DRAFT.name);
    typeInto(win, 'cfg-custom-endpoint', DRAFT.endpoint);
    byId(win, 'custom-model-save-btn').click();
    const msg = posted.filter(m => m.type === 'saveMyModel').pop();
    assert.strictEqual(msg.machine, 'gpu-box');
    assert.strictEqual(editVisible(win), false);
    // Another machine's forwarded error is not this save's.
    send(win, {
      type: 'myModelsData',
      forwarded: true,
      machine: 'other-box',
      error: 'nope',
    });
    assert.strictEqual(editVisible(win), false);
    send(win, {
      type: 'myModelsData',
      forwarded: true,
      machine: 'gpu-box',
      error: "A custom model named 'concurrent-model' already exists",
    });
    assert.strictEqual(
      editVisible(win),
      true,
      "BUG: gpu-box's refusal did not bring the draft back",
    );
    assert.strictEqual(boxValues(win).name, DRAFT.name);
    assert.strictEqual(boxValues(win).endpoint, DRAFT.endpoint);
    // Once the sheet shows another machine, a refusal is only toasted.
    byId(win, 'custom-model-cancel-btn').click();
    byId(win, 'custom-model-save-btn');
    win.document.querySelector('.custom-model-edit-btn').click();
    typeInto(win, 'cfg-custom-model-name', 'second-try');
    byId(win, 'custom-model-save-btn').click();
    pickMachine(win, '');
    send(win, {
      type: 'myModelsData',
      forwarded: true,
      machine: 'gpu-box',
      error: 'nope',
    });
    assert.strictEqual(
      editVisible(win),
      false,
      "another machine's sheet shows no gpu-box draft",
    );
  });

  report('audit1101_W4_picker_owner_model_draft');
  await sleep(0);
  for (const win of pages) win.close();
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
