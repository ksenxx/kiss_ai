// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The fixed task panel at the top of a chat webview must use the SAME
// background and foreground colors as the thinking panel (.think) on
// every surface (sidebar webview, editor-tab webview, remote webapp
// and shared chat pages all inline media/main.css), plus a thick cyan
// border. Fresh installs must default to editor-tabs mode
// (kissSorcar.editorTabsMode default true in package.json).

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const MEDIA = path.join(__dirname, '..', 'media');
const CSS = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');

/**
 * Return the concatenated declaration bodies of every top-level CSS
 * rule whose selector list contains *selector* exactly (comments
 * stripped so commented-out declarations never match assertions).
 */
function ruleBody(selector) {
  const css = CSS.replace(/\/\*[\s\S]*?\*\//g, '');
  const blockRe = /([^{}]+)\{([^{}]*)\}/g;
  let match;
  const bodies = [];
  while ((match = blockRe.exec(css)) !== null) {
    const selectors = match[1].split(',').map(s => s.trim());
    if (selectors.includes(selector)) bodies.push(match[2]);
  }
  return bodies.join('\n');
}

function decl(body, prop) {
  const re = new RegExp(
    '(?:^|[;\\s])' + prop.replace(/[-\\]/g, '\\$&') + '\\s*:\\s*([^;]+)',
  );
  const m = re.exec(body);
  return m ? m[1].trim() : null;
}

function testTaskPanelBackgroundMatchesThinkingPanel() {
  const think = ruleBody('.think');
  const panel = ruleBody('#task-panel');
  assert.ok(think, 'main.css must style .think');
  assert.ok(panel, 'main.css must style #task-panel');

  const thinkBg = decl(think, 'background');
  assert.ok(thinkBg, '.think must declare a background');

  // #task-panel paints var(--panel-bg); that variable must hold the
  // exact background value the thinking panel uses.
  const panelBgVar = decl(panel, '--panel-bg');
  const panelBg = decl(panel, 'background');
  assert.strictEqual(
    panelBg,
    'var(--panel-bg)',
    '#task-panel must paint its background from --panel-bg',
  );
  assert.strictEqual(
    panelBgVar,
    thinkBg,
    'BUG: #task-panel --panel-bg must equal the .think background ' +
      `("${thinkBg}") so the task panel matches the thinking panel, ` +
      `got "${panelBgVar}"`,
  );
  console.log('  ok - #task-panel background equals .think background');
}

function testTaskPanelForegroundMatchesThinkingPanel() {
  const thinkCnt = ruleBody('.think .cnt');
  const panel = ruleBody('#task-panel');
  const thinkFg = decl(thinkCnt, 'color');
  assert.strictEqual(
    thinkFg,
    'var(--dim)',
    '.think .cnt (thinking panel text) must be colored var(--dim)',
  );
  assert.strictEqual(
    decl(panel, 'color'),
    thinkFg,
    'BUG: #task-panel text color must equal the thinking panel text ' +
      `color ("${thinkFg}")`,
  );
  // Secondary content (controls, info rows, scrollbar) must stay
  // readable: the standard --dim, not a faded translucent variant.
  assert.strictEqual(
    decl(panel, '--panel-dim'),
    'var(--dim)',
    'BUG: --panel-dim must be the readable var(--dim), not a ' +
      'translucent low-contrast mix',
  );
  console.log('  ok - #task-panel foreground equals .think text color');
}

function testTaskPanelHasThickCyanBorder() {
  const panel = ruleBody('#task-panel');
  const border = decl(panel, 'border');
  assert.ok(border, '#task-panel must declare a border');
  const m = /^(\d+(?:\.\d+)?)px\s+solid\s+var\(\s*--cyan\s*\)$/.exec(border);
  assert.ok(
    m,
    'BUG: #task-panel border must be "<N>px solid var(--cyan)", got ' +
      `"${border}"`,
  );
  assert.ok(
    parseFloat(m[1]) >= 3,
    `BUG: the cyan border must be thick (>= 3px), got ${m[1]}px`,
  );
  console.log('  ok - #task-panel has a thick cyan border');
}

function testInvertedPaletteIsGone() {
  const panel = ruleBody('#task-panel');
  assert.notStrictEqual(
    decl(panel, '--panel-bg'),
    'var(--fg)',
    'BUG: the old inverted palette (--panel-bg: var(--fg)) must be gone',
  );
  assert.ok(
    !/rgb\(\s*255\s+200\s+0/.test(panel),
    'BUG: the old yellow border color must be gone from #task-panel',
  );
  const tooltip = ruleBody('#custom-tooltip.task-panel-tooltip');
  assert.notStrictEqual(
    decl(tooltip, 'background'),
    'var(--fg)',
    'BUG: the task-panel tooltip must no longer use the inverted ' +
      'palette (background: var(--fg))',
  );
  assert.strictEqual(
    decl(tooltip, 'color'),
    'var(--dim)',
    'the task-panel tooltip text must match the thinking-panel ' +
      'foreground (var(--dim))',
  );
  console.log('  ok - inverted task-panel palette fully removed');
}

function testEditorTabsModeDefaultsOn() {
  const pkg = JSON.parse(
    fs.readFileSync(path.join(__dirname, '..', 'package.json'), 'utf8'),
  );
  const prop =
    pkg.contributes.configuration.properties['kissSorcar.editorTabsMode'];
  assert.ok(prop, 'kissSorcar.editorTabsMode setting must exist');
  assert.strictEqual(
    prop.default,
    true,
    'BUG: kissSorcar.editorTabsMode must default to true so a fresh ' +
      'install of KISS Sorcar starts in editor-tabs mode',
  );
  console.log('  ok - editorTabsMode defaults to true (fresh installs)');
}

function main() {
  console.log('taskPanelThinkStyle.test.js');
  testTaskPanelBackgroundMatchesThinkingPanel();
  testTaskPanelForegroundMatchesThinkingPanel();
  testTaskPanelHasThickCyanBorder();
  testInvertedPaletteIsGone();
  testEditorTabsModeDefaultsOn();
  console.log('all task-panel style tests passed');
}

main();
