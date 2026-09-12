// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the settings panel's collapsible
// subpanels and the Custom Models (~/.kiss/MY_MODELS.json) CRUD UI,
// driving the real chat.html + api.js + main.js:
//
// * Opening the settings panel requests the custom-model list
//   (getMyModels) and starts with BOTH subpanels (API Keys, Custom
//   Models) collapsed — every time, not just the first.
// * The header buttons expand/collapse their subpanel.
// * `myModelsData` paints one row per model with Edit / Delete buttons.
// * Add posts `saveMyModel` and clears the name box; an empty name
//   posts nothing.
// * Edit loads the model into the boxes and swaps Add for Save +
//   Cancel; Save posts `saveMyModel` with `originalName`; Cancel (and
//   Save) restore the pre-edit box values.
// * Delete posts `deleteMyModel`; deleting the model being edited
//   abandons the edit.
// * While an edit holds the boxes, a `configData` repaint must not
//   clobber them, and closing the panel mid-edit must flush the USER's
//   values (not the edited model's) into `saveConfig`.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  let state;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function lastMsg(posted, type) {
  for (let i = posted.length - 1; i >= 0; i -= 1) {
    if (posted[i] && posted[i].type === type) return posted[i];
  }
  return null;
}

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function openSettings(win) {
  click(win, win.document.getElementById('settings-btn'));
}

function closeSettings(win) {
  click(win, win.document.getElementById('settings-panel-close'));
}

function typeInto(win, id, value) {
  const node = win.document.getElementById(id);
  assert.ok(node, `#${id} must exist`);
  node.value = value;
  node.dispatchEvent(new win.Event('input', {bubbles: true}));
}

function boxValues(win) {
  const el = id => win.document.getElementById(id);
  return {
    name: el('cfg-custom-model-name').value,
    endpoint: el('cfg-custom-endpoint').value,
    apiKey: el('cfg-custom-api-key').value,
    headers: el('cfg-custom-headers').value,
  };
}

const MODELS = [
  {
    name: 'my-org/model-a',
    endpoint: 'http://localhost:8080/v1',
    api_key: 'sk-a',
    headers: 'X-A: 1',
  },
  {name: 'model-b', endpoint: '', api_key: '', headers: ''},
];

function subpanelState(win, toggleId, bodyId) {
  const toggle = win.document.getElementById(toggleId);
  const body = win.document.getElementById(bodyId);
  assert.ok(toggle && body, `${toggleId}/${bodyId} must exist`);
  return {
    expanded: toggle.getAttribute('aria-expanded') === 'true',
    hidden: body.hidden,
  };
}

function testSubpanelsCollapsedByDefaultAndToggle() {
  const {win, posted} = makeWebview();
  openSettings(win);

  assert.ok(
    lastMsg(posted, 'getMyModels'),
    'opening the settings panel must request the custom-model list',
  );
  for (const [toggleId, bodyId] of [
    ['api-keys-toggle', 'api-keys-body'],
    ['custom-models-toggle', 'custom-models-body'],
  ]) {
    const st = subpanelState(win, toggleId, bodyId);
    assert.strictEqual(st.expanded, false, `${toggleId} starts collapsed`);
    assert.strictEqual(st.hidden, true, `${bodyId} starts hidden`);
  }

  // A click expands; a second click collapses again.
  const toggle = win.document.getElementById('api-keys-toggle');
  click(win, toggle);
  let st = subpanelState(win, 'api-keys-toggle', 'api-keys-body');
  assert.strictEqual(st.expanded, true, 'click expands the subpanel');
  assert.strictEqual(st.hidden, false, 'body shows after the click');
  assert.ok(
    toggle.classList.contains('expanded'),
    'the chevron rotation class follows the state',
  );
  click(win, toggle);
  st = subpanelState(win, 'api-keys-toggle', 'api-keys-body');
  assert.strictEqual(st.expanded, false, 'second click collapses');

  // Left expanded, the subpanel must be collapsed again on REOPEN.
  click(win, win.document.getElementById('custom-models-toggle'));
  closeSettings(win);
  openSettings(win);
  st = subpanelState(win, 'custom-models-toggle', 'custom-models-body');
  assert.strictEqual(st.expanded, false, 'reopen re-collapses the subpanel');
  assert.strictEqual(st.hidden, true, 'reopened body is hidden');

  win.close();
  console.log('  ok - subpanels start collapsed and toggle/re-collapse');
}

function testSettingsPanelIsCenteredNotSlidIn() {
  const html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  assert.ok(html.includes('id="settings-panel"'), 'panel exists');
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const rule = css.match(/#settings-panel \{[^}]*\}/);
  assert.ok(rule, 'main.css must style #settings-panel');
  assert.ok(
    rule[0].includes('translate(-50%, -50%)') &&
      rule[0].includes('left: 50%') &&
      rule[0].includes('top: 50%'),
    'the settings panel is centered in the window',
  );
  assert.ok(
    !rule[0].includes('translateX(100%)') && !rule[0].includes('transition:'),
    'the settings panel no longer slides in from the right',
  );
  console.log('  ok - settings panel CSS centers without entry animation');
}

function testMyModelsListRenderAndAdd() {
  const {win, posted} = makeWebview();
  openSettings(win);
  send(win, {type: 'myModelsData', models: MODELS});

  const rows = win.document.querySelectorAll('.custom-model-row');
  assert.strictEqual(rows.length, 2, 'one row per custom model');
  assert.strictEqual(
    rows[0].querySelector('.custom-model-name').textContent,
    'my-org/model-a',
  );
  assert.ok(rows[0].querySelector('.custom-model-edit-btn'), 'edit button');
  assert.ok(
    rows[0].querySelector('.custom-model-delete-btn'),
    'delete button',
  );

  // Empty list paints the placeholder.
  send(win, {type: 'myModelsData', models: []});
  assert.ok(
    win.document.querySelector('.custom-models-empty'),
    'an empty list shows the placeholder',
  );
  send(win, {type: 'myModelsData', models: MODELS});

  // Add with an empty name posts nothing.
  const before = posted.length;
  click(win, win.document.getElementById('custom-model-add-btn'));
  assert.strictEqual(
    lastMsg(posted.slice(before), 'saveMyModel'),
    null,
    'an empty name must not be sent',
  );

  typeInto(win, 'cfg-custom-model-name', 'new-model');
  typeInto(win, 'cfg-custom-endpoint', 'http://h:1/v1');
  typeInto(win, 'cfg-custom-api-key', 'sk-new');
  typeInto(win, 'cfg-custom-headers', 'X-N: 2');
  click(win, win.document.getElementById('custom-model-add-btn'));
  const msg = lastMsg(posted, 'saveMyModel');
  assert.ok(msg, 'Add posts saveMyModel');
  assert.deepStrictEqual(
    {
      name: msg.name,
      endpoint: msg.endpoint,
      apiKey: msg.apiKey,
      headers: msg.headers,
    },
    {
      name: 'new-model',
      endpoint: 'http://h:1/v1',
      apiKey: 'sk-new',
      headers: 'X-N: 2',
    },
    'Add sends all four fields',
  );
  assert.strictEqual(
    msg.originalName,
    undefined,
    'Add carries no originalName',
  );
  assert.strictEqual(
    win.document.getElementById('cfg-custom-model-name').value,
    '',
    'Add clears the name box',
  );

  win.close();
  console.log('  ok - list renders and Add posts saveMyModel');
}

function testEditSaveCancelRestoreBoxes() {
  const {win, posted} = makeWebview();
  openSettings(win);
  send(win, {type: 'myModelsData', models: MODELS});
  typeInto(win, 'cfg-custom-endpoint', 'http://mine:9/v1');

  const addBtn = win.document.getElementById('custom-model-add-btn');
  const saveBtn = win.document.getElementById('custom-model-save-btn');
  const cancelBtn = win.document.getElementById('custom-model-cancel-btn');

  const editBtn = win.document.querySelectorAll('.custom-model-edit-btn')[0];
  click(win, editBtn);
  assert.deepStrictEqual(
    boxValues(win),
    {
      name: 'my-org/model-a',
      endpoint: 'http://localhost:8080/v1',
      apiKey: 'sk-a',
      headers: 'X-A: 1',
    },
    'Edit loads the model into the boxes',
  );
  assert.strictEqual(addBtn.style.display, 'none', 'Add hides during edit');
  assert.notStrictEqual(saveBtn.style.display, 'none', 'Save shows');
  assert.notStrictEqual(cancelBtn.style.display, 'none', 'Cancel shows');
  assert.ok(
    win.document
      .querySelectorAll('.custom-model-row')[0]
      .classList.contains('editing'),
    'the edited row is highlighted',
  );

  // A configData poll mid-edit must not clobber the loaded values.
  send(win, {
    type: 'configData',
    config: {custom_endpoint: 'http://cfg/v1', custom_api_key: 'sk-cfg'},
    apiKeys: {},
  });
  assert.strictEqual(
    boxValues(win).endpoint,
    'http://localhost:8080/v1',
    'configData must not repaint the boxes during an edit',
  );

  // Cancel restores what the user had typed and swaps the buttons back.
  click(win, cancelBtn);
  assert.strictEqual(boxValues(win).endpoint, 'http://mine:9/v1');
  assert.notStrictEqual(addBtn.style.display, 'none', 'Add shows again');
  assert.strictEqual(saveBtn.style.display, 'none', 'Save hides');
  assert.strictEqual(cancelBtn.style.display, 'none', 'Cancel hides');

  // Edit again, rename, Save: originalName identifies the old entry.
  click(win, win.document.querySelectorAll('.custom-model-edit-btn')[0]);
  typeInto(win, 'cfg-custom-model-name', 'renamed-model');
  click(win, saveBtn);
  const msg = lastMsg(posted, 'saveMyModel');
  assert.strictEqual(msg.name, 'renamed-model');
  assert.strictEqual(msg.originalName, 'my-org/model-a');
  assert.strictEqual(
    boxValues(win).endpoint,
    'http://mine:9/v1',
    'Save restores the pre-edit box values',
  );

  win.close();
  console.log('  ok - Edit/Save/Cancel load and restore the boxes');
}

function testDeletePostsAndAbandonsEdit() {
  const {win, posted} = makeWebview();
  openSettings(win);
  send(win, {type: 'myModelsData', models: MODELS});

  click(win, win.document.querySelectorAll('.custom-model-edit-btn')[1]);
  click(win, win.document.querySelectorAll('.custom-model-delete-btn')[1]);
  const msg = lastMsg(posted, 'deleteMyModel');
  assert.ok(msg, 'Delete posts deleteMyModel');
  assert.strictEqual(msg.name, 'model-b');
  assert.strictEqual(
    win.document.getElementById('custom-model-save-btn').style.display,
    'none',
    'deleting the edited model abandons the edit',
  );

  // The daemon's refreshed list no longer holds an edited model that
  // another window deleted: the edit is abandoned then too.
  click(win, win.document.querySelectorAll('.custom-model-edit-btn')[0]);
  send(win, {type: 'myModelsData', models: [MODELS[1]]});
  assert.strictEqual(
    win.document.getElementById('custom-model-save-btn').style.display,
    'none',
    'a myModelsData without the edited model cancels the edit',
  );

  win.close();
  console.log('  ok - Delete posts deleteMyModel and abandons edits');
}

function testCollisionAndReservedNamesAreRejectedLocally() {
  const {win, posted} = makeWebview();
  openSettings(win);
  send(win, {type: 'myModelsData', models: MODELS});

  // Add with an existing name must not post (it would silently
  // overwrite that model) and must surface an error notification.
  typeInto(win, 'cfg-custom-model-name', 'model-b');
  let before = posted.length;
  click(win, win.document.getElementById('custom-model-add-btn'));
  assert.strictEqual(
    lastMsg(posted.slice(before), 'saveMyModel'),
    null,
    'a colliding Add must not be sent',
  );
  assert.ok(
    win.document.querySelector('.kiss-notification-error'),
    'a colliding Add shows an error notification',
  );

  // A reserved (underscore) name is rejected before posting too.
  typeInto(win, 'cfg-custom-model-name', '_reserved');
  before = posted.length;
  click(win, win.document.getElementById('custom-model-add-btn'));
  assert.strictEqual(lastMsg(posted.slice(before), 'saveMyModel'), null);
  assert.strictEqual(
    win.document.getElementById('cfg-custom-model-name').value,
    '_reserved',
    'a rejected Add keeps the draft in the box',
  );

  // Renaming one model onto another existing one is rejected.
  click(win, win.document.querySelectorAll('.custom-model-edit-btn')[0]);
  typeInto(win, 'cfg-custom-model-name', 'model-b');
  before = posted.length;
  click(win, win.document.getElementById('custom-model-save-btn'));
  assert.strictEqual(lastMsg(posted.slice(before), 'saveMyModel'), null);
  assert.notStrictEqual(
    win.document.getElementById('custom-model-save-btn').style.display,
    'none',
    'a rejected rename keeps the edit open',
  );

  win.close();
  console.log('  ok - collisions and reserved names are rejected locally');
}

function testCancelRestoresAuthoritativeConfigValues() {
  const {win, posted} = makeWebview();
  openSettings(win);
  send(win, {type: 'myModelsData', models: MODELS});

  // Edit starts BEFORE this panel open's configData reply arrived: the
  // snapshot holds stale (empty) boxes.
  click(win, win.document.querySelectorAll('.custom-model-edit-btn')[0]);
  send(win, {
    type: 'configData',
    config: {custom_endpoint: 'http://cfg/v1', custom_api_key: 'sk-cfg'},
    apiKeys: {},
  });
  click(win, win.document.getElementById('custom-model-cancel-btn'));
  assert.strictEqual(
    boxValues(win).endpoint,
    'http://cfg/v1',
    'Cancel restores the authoritative config value, not the stale snapshot',
  );
  assert.strictEqual(boxValues(win).apiKey, 'sk-cfg');

  // Typing DURING the edit must not leave a stray edited-mark behind:
  // after Cancel, a later configData still repaints the box.
  click(win, win.document.querySelectorAll('.custom-model-edit-btn')[0]);
  typeInto(win, 'cfg-custom-endpoint', 'http://model-edit/v1');
  click(win, win.document.getElementById('custom-model-cancel-btn'));
  send(win, {
    type: 'configData',
    config: {custom_endpoint: 'http://cfg2/v1'},
    apiKeys: {},
  });
  assert.strictEqual(
    boxValues(win).endpoint,
    'http://cfg2/v1',
    'an edit-time keystroke must not block later config repaints',
  );

  // Closing now must flush the authoritative value, not a stale one.
  closeSettings(win);
  const msg = lastMsg(posted, 'saveConfig');
  assert.strictEqual(msg.config.custom_endpoint, 'http://cfg2/v1');

  win.close();
  console.log('  ok - Cancel restores authoritative config values');
}

function testCloseMidEditFlushesUserValues() {
  const {win, posted} = makeWebview();
  openSettings(win);
  send(win, {
    type: 'configData',
    config: {custom_endpoint: 'http://cfg/v1'},
    apiKeys: {},
  });
  send(win, {type: 'myModelsData', models: MODELS});
  typeInto(win, 'cfg-custom-endpoint', 'http://typed/v1');
  click(win, win.document.querySelectorAll('.custom-model-edit-btn')[0]);
  closeSettings(win);

  const msg = lastMsg(posted, 'saveConfig');
  assert.ok(msg, 'closing the panel flushes the form');
  assert.strictEqual(
    msg.config.custom_endpoint,
    'http://typed/v1',
    "the edit's loaded endpoint must NOT leak into the saved config",
  );

  win.close();
  console.log('  ok - closing mid-edit saves the user values, not the edit');
}

function main() {
  testSubpanelsCollapsedByDefaultAndToggle();
  testSettingsPanelIsCenteredNotSlidIn();
  testMyModelsListRenderAndAdd();
  testEditSaveCancelRestoreBoxes();
  testDeletePostsAndAbandonsEdit();
  testCollisionAndReservedNamesAreRejectedLocally();
  testCancelRestoresAuthoritativeConfigValues();
  testCloseMidEditFlushesUserValues();
  console.log('settingsCustomModels: all tests passed');
}

main();
