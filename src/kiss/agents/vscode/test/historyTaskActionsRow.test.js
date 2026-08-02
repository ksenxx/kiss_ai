// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (jsdom) tests for the action-button strip of a task panel
// inside the task-history panel.
//
// A task panel (``.sidebar-item.running-item`` inside ``#history-list``)
// used to squeeze its favourite / copy / collapse buttons onto the same
// line as the task text, which left the text almost no room on a narrow
// sidebar and made the 12px icons hard to hit.  The buttons must now sit
// on their own full-width line *below* the task text, and each button
// must be 50% bigger than the old 12px icon / 16px box.
//
// The same ``media/`` bundle is served to the VS Code webview and to the
// remote webapp (``web_server.py`` renders ``chat.html`` with
// ``<body class="remote-chat">`` plus ``remote-codex.css``), so every
// assertion below is run against both surfaces.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// Old geometry (before the fix) and the 50%-bigger geometry required now.
// The compact button is a 12x16 box: a 12px icon with "2px 0" padding.
// Both axes have to grow by exactly half, so 12x16 becomes 18x24.
const OLD_ICON_PX = 12;
const NEW_ICON_PX = 18;
const OLD_BUTTON_W_PX = 12;
const NEW_BUTTON_W_PX = 18;
const OLD_BUTTON_H_PX = 16;
const NEW_BUTTON_H_PX = 24;

const ACTION_BUTTON_SELECTORS = [
  '.sidebar-item-favorite',
  '.sidebar-item-copy',
  '.sidebar-item-collapse',
];

function makeWebview(remote) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  if (remote) html = html.replace('<body', '<body class="remote-chat"');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const style = win.document.createElement('style');
  style.textContent = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  win.document.head.appendChild(style);
  if (remote) {
    const remoteStyle = win.document.createElement('style');
    remoteStyle.textContent = fs.readFileSync(
      path.join(MEDIA, 'remote-codex.css'),
      'utf8',
    );
    win.document.head.appendChild(remoteStyle);
  }

  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
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

function makeSession(overrides) {
  return Object.assign(
    {
      id: 'chat-1',
      task_id: 'task-1',
      title: 'refactor the parser',
      preview: 'refactor the parser',
      has_events: true,
      tokens: 1234,
      cost: 0.5678,
      steps: 7,
      timestamp: 1700000000,
      work_dir: '/home/user/proj',
      model: 'test-model',
      is_worktree: true,
      is_parallel: false,
      auto_commit_mode: true,
      is_favorite: false,
    },
    overrides || {},
  );
}

function loadHistory(win, sessions) {
  win.dispatchEvent(
    new win.MessageEvent('message', {
      data: {type: 'history', offset: 0, generation: 0, sessions: sessions},
    }),
  );
}

function historyRow(win) {
  return win.document.querySelector('#history-list .sidebar-item');
}

function px(value) {
  const n = parseFloat(String(value));
  return isNaN(n) ? 0 : n;
}

// ---------------------------------------------------------------------------
// 1. The buttons live on their own line.
// ---------------------------------------------------------------------------

function testActionsOnSeparateLine(remote) {
  const label = remote ? 'webapp' : 'extension';
  const {win} = makeWebview(remote);
  loadHistory(win, [makeSession()]);

  const row = historyRow(win);
  assert.ok(row, `${label}: history row rendered`);

  const actions = row.querySelector('.sidebar-item-actions');
  assert.ok(actions, `${label}: action strip rendered`);

  const text = row.querySelector('.sidebar-item-text');
  assert.ok(text, `${label}: task text rendered`);

  // The row is a wrapping flex container; a child only gets its own line
  // when it is forced to span the full content width.
  const rowStyle = win.getComputedStyle(row);
  assert.strictEqual(
    rowStyle.display,
    'flex',
    `${label}: the task panel must stay a flex container`,
  );
  assert.strictEqual(
    rowStyle.flexWrap,
    'wrap',
    `${label}: the task panel must wrap so the actions can break to a new line`,
  );

  const actionsStyle = win.getComputedStyle(actions);
  assert.strictEqual(
    actionsStyle.flexBasis,
    '100%',
    `${label}: the action strip must claim a full-width line of its own; ` +
      `got flex-basis=${actionsStyle.flexBasis}`,
  );
  assert.strictEqual(
    px(actionsStyle.marginLeft),
    0,
    `${label}: the action strip must not keep the inline left gutter; ` +
      `got margin-left=${actionsStyle.marginLeft}`,
  );

  // The task text must no longer have to share its line with the buttons:
  // it keeps flex:1 while the actions wrap away onto the next line.
  const textStyle = win.getComputedStyle(text);
  assert.ok(
    px(textStyle.flexGrow) >= 1,
    `${label}: the task text must still grow to fill its line`,
  );
  // Regression guard: the title needs its own full-width basis too. With the
  // default flex-basis of 0 it is squeezed to zero width beside the
  // full-width action strip and the task title becomes invisible.
  assert.strictEqual(
    textStyle.flexBasis,
    '100%',
    `${label}: the task text must claim a full-width line of its own, ` +
      `otherwise the full-width action strip collapses it; ` +
      `got flex-basis=${textStyle.flexBasis}`,
  );

  // DOM order decides which line comes first when nothing reorders them:
  // text -> actions -> info.
  const children = Array.prototype.slice.call(row.children);
  assert.ok(
    children.indexOf(text) < children.indexOf(actions),
    `${label}: the action strip must come after the task text`,
  );
  const info = row.querySelector('.running-item-info');
  assert.ok(
    children.indexOf(actions) < children.indexOf(info),
    `${label}: the action strip must come before the metadata block`,
  );
  assert.strictEqual(
    win.getComputedStyle(actions).order,
    win.getComputedStyle(text).order,
    `${label}: no CSS 'order' trickery may re-shuffle the action strip`,
  );

  win.close();
  console.log(`  ok - ${label}: action buttons sit on their own line`);
}

// ---------------------------------------------------------------------------
// 2. The buttons are 50% bigger.
// ---------------------------------------------------------------------------

function testActionsFiftyPercentBigger(remote) {
  const label = remote ? 'webapp' : 'extension';
  const {win} = makeWebview(remote);
  loadHistory(win, [makeSession()]);
  const row = historyRow(win);

  ACTION_BUTTON_SELECTORS.forEach(sel => {
    const btn = row.querySelector(sel);
    assert.ok(btn, `${label}: ${sel} rendered`);

    const btnStyle = win.getComputedStyle(btn);
    assert.strictEqual(
      px(btnStyle.minWidth),
      NEW_BUTTON_W_PX,
      `${label}: ${sel} must be ${NEW_BUTTON_W_PX}px wide (50% wider than ` +
        `the old ${OLD_BUTTON_W_PX}px); got min-width=${btnStyle.minWidth}`,
    );
    assert.strictEqual(
      px(btnStyle.minHeight),
      NEW_BUTTON_H_PX,
      `${label}: ${sel} must be ${NEW_BUTTON_H_PX}px tall (50% taller than ` +
        `the old ${OLD_BUTTON_H_PX}px); got min-height=${btnStyle.minHeight}`,
    );
    assert.strictEqual(
      btnStyle.justifyContent,
      'center',
      `${label}: ${sel} must centre its icon in the bigger box`,
    );

    const svg = btn.querySelector('svg');
    assert.ok(svg, `${label}: ${sel} renders an icon`);
    const svgStyle = win.getComputedStyle(svg);
    assert.strictEqual(
      px(svgStyle.width),
      NEW_ICON_PX,
      `${label}: ${sel} icon must be ${NEW_ICON_PX}px wide (50% bigger than ` +
        `the old ${OLD_ICON_PX}px); got width=${svgStyle.width}`,
    );
    assert.strictEqual(
      px(svgStyle.height),
      NEW_ICON_PX,
      `${label}: ${sel} icon must be ${NEW_ICON_PX}px tall; ` +
        `got height=${svgStyle.height}`,
    );
  });

  win.close();
  console.log(`  ok - ${label}: action buttons are 50% bigger`);
}

// ---------------------------------------------------------------------------
// 3. The enlarged icons survive every innerHTML rewrite.
// ---------------------------------------------------------------------------

function testIconSizeSurvivesStateChanges(remote) {
  const label = remote ? 'webapp' : 'extension';
  const {win, posted} = makeWebview(remote);
  loadHistory(win, [makeSession()]);
  const row = historyRow(win);

  // Favouriting swaps the outline star for the filled star via innerHTML.
  const fav = row.querySelector('.sidebar-item-favorite');
  fav.click();
  assert.ok(
    fav.classList.contains('favorited'),
    `${label}: clicking the star favourites the task`,
  );
  assert.ok(
    posted.some(m => m.type === 'setFavorite'),
    `${label}: favouriting still reaches the backend`,
  );
  assert.strictEqual(
    px(win.getComputedStyle(fav.querySelector('svg')).width),
    NEW_ICON_PX,
    `${label}: the filled star stays ${NEW_ICON_PX}px`,
  );

  fav.click();
  assert.ok(
    !fav.classList.contains('favorited'),
    `${label}: clicking again unfavourites the task`,
  );
  assert.strictEqual(
    px(win.getComputedStyle(fav.querySelector('svg')).width),
    NEW_ICON_PX,
    `${label}: the outline star stays ${NEW_ICON_PX}px`,
  );

  // The copy button swaps in a check mark after a successful copy.
  const copy = row.querySelector('.sidebar-item-copy');
  let copied = '';
  win.navigator.clipboard = {
    writeText: t => {
      copied = t;
      return Promise.resolve();
    },
  };
  copy.click();
  return Promise.resolve().then(() => {
    assert.strictEqual(
      copied,
      'refactor the parser',
      `${label}: the copy button still copies the task text`,
    );
    assert.ok(
      copy.classList.contains('copied'),
      `${label}: the copy button flashes its success state`,
    );
    assert.strictEqual(
      px(win.getComputedStyle(copy.querySelector('svg')).width),
      NEW_ICON_PX,
      `${label}: the check mark stays ${NEW_ICON_PX}px`,
    );
    win.close();
    console.log(`  ok - ${label}: icon size survives every state swap`);
  });
}

// ---------------------------------------------------------------------------
// 4. Rows without a task id only show the collapse toggle - still on its
//    own, full-width line.
// ---------------------------------------------------------------------------

function testRowWithoutTaskIdKeepsOwnLine(remote) {
  const label = remote ? 'webapp' : 'extension';
  const {win} = makeWebview(remote);
  loadHistory(win, [
    {
      id: 'chat-2',
      title: 'old imported chat',
      preview: 'old imported chat',
      has_events: false,
      timestamp: 1700000000,
    },
  ]);
  const row = historyRow(win);
  const actions = row.querySelector('.sidebar-item-actions');
  assert.ok(actions, `${label}: action strip rendered without a task id`);
  assert.strictEqual(
    row.querySelector('.sidebar-item-favorite'),
    null,
    `${label}: no favourite button without a task id`,
  );
  assert.strictEqual(
    row.querySelector('.sidebar-item-copy'),
    null,
    `${label}: no copy button without a task id`,
  );
  assert.strictEqual(
    win.getComputedStyle(actions).flexBasis,
    '100%',
    `${label}: the lone collapse toggle still gets its own line`,
  );
  const toggle = actions.querySelector('.sidebar-item-collapse');
  assert.strictEqual(
    px(win.getComputedStyle(toggle.querySelector('svg')).width),
    NEW_ICON_PX,
    `${label}: the lone collapse toggle is 50% bigger too`,
  );
  win.close();
  console.log(`  ok - ${label}: task-id-less rows keep the new layout`);
}

// ---------------------------------------------------------------------------
// 5. The enlarged strip must not break the existing row behaviour.
// ---------------------------------------------------------------------------

function testBehaviourUnchanged(remote) {
  const label = remote ? 'webapp' : 'extension';
  const {win, posted} = makeWebview(remote);
  loadHistory(win, [makeSession()]);
  const row = historyRow(win);

  const toggle = row.querySelector('.sidebar-item-collapse');
  assert.ok(
    row.classList.contains('collapsed'),
    `${label}: the task panel is still collapsed by default`,
  );
  toggle.click();
  assert.ok(
    !row.classList.contains('collapsed'),
    `${label}: the bigger chevron still expands the panel`,
  );
  assert.ok(
    !posted.some(m => m.type === 'resumeSession'),
    `${label}: button clicks must not bubble into the row click handler`,
  );

  row.click();
  assert.ok(
    posted.some(m => m.type === 'resumeSession'),
    `${label}: clicking the row body still opens the chat`,
  );
  win.close();
  console.log(`  ok - ${label}: row behaviour is unchanged`);
}

// ---------------------------------------------------------------------------
// 6. Only the *history* task panels change - the frequent-tasks and
//    inject lists keep their compact inline buttons.
// ---------------------------------------------------------------------------

function testFrequentListUnaffected() {
  const {win} = makeWebview(false);
  win.dispatchEvent(
    new win.MessageEvent('message', {
      data: {
        type: 'frequentTasks',
        tasks: [{task: 'run the tests', count: 3}],
      },
    }),
  );
  const item = win.document.querySelector('#frequent-list .sidebar-item');
  assert.ok(item, 'frequent-task row rendered');
  assert.ok(
    !item.classList.contains('running-item'),
    'frequent-task rows are not history task panels',
  );
  const copy = item.querySelector('.sidebar-item-copy');
  assert.ok(copy, 'frequent-task row has a copy button');
  const svg = copy.querySelector('svg');
  // No stylesheet rule may resize these icons: jsdom leaves the
  // computed width empty because nothing in the cascade targets them,
  // so the icon keeps the intrinsic 12px of its markup attributes.
  assert.strictEqual(
    px(win.getComputedStyle(svg).width),
    0,
    'no CSS rule may resize the compact frequent-task icons',
  );
  assert.strictEqual(
    px(svg.getAttribute('width')),
    OLD_ICON_PX,
    `frequent-task buttons keep the compact ${OLD_ICON_PX}px icon`,
  );
  const btnStyle = win.getComputedStyle(copy);
  assert.strictEqual(
    px(btnStyle.minWidth),
    0,
    'frequent-task buttons keep their compact intrinsic box',
  );
  assert.strictEqual(
    px(btnStyle.paddingLeft),
    0,
    'frequent-task buttons keep the compact "2px 0" padding',
  );
  win.close();
  console.log('  ok - frequent-task rows keep their compact buttons');
}

async function main() {
  [false, true].forEach(remote => {
    testActionsOnSeparateLine(remote);
    testActionsFiftyPercentBigger(remote);
    testRowWithoutTaskIdKeepsOwnLine(remote);
    testBehaviourUnchanged(remote);
  });
  await testIconSizeSurvivesStateChanges(false);
  await testIconSizeSurvivesStateChanges(true);
  testFrequentListUnaffected();
  console.log('All historyTaskActionsRow tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
