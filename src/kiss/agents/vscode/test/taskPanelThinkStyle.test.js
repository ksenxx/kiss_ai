// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The task panel that opens every task of the chat thread is the one
// panel the eye should find first: it paints the accent tint under a
// 1px accent hairline, on every surface (sidebar webview, editor-tab
// webview, remote webapp and shared chat pages all inline
// media/main.css).
// Fresh installs must default to editor-tabs mode
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

function testTaskPanelIsTheAccentTintedPanel() {
  const think = ruleBody('.think');
  const panel = ruleBody('.ev.task-panel');
  assert.ok(think, 'main.css must style .think');
  assert.ok(panel, 'main.css must style .ev.task-panel');

  // The transcript panels share the neutral --panel-tint; the task
  // panel alone paints the accent tint under an accent hairline.
  assert.strictEqual(
    decl(think, 'background'),
    'var(--panel-tint)',
    '.think must paint the neutral --panel-tint',
  );
  assert.strictEqual(
    decl(panel, 'background'),
    'var(--accent-tint)',
    'BUG: .ev.task-panel background must be the accent tint',
  );
  assert.strictEqual(
    decl(panel, 'border'),
    '1px solid var(--accent-line)',
    'BUG: .ev.task-panel border must be the accent hairline',
  );
  assert.strictEqual(
    decl(panel, 'border-radius'),
    decl(think, 'border-radius'),
    'the task panel is rounded like the other transcript panels',
  );
  console.log('  ok - .ev.task-panel paints the accent tint and hairline');
}

function testTaskPanelTextMatchesTheTranscript() {
  const text = ruleBody('.task-panel-text');
  assert.ok(text, 'main.css must style .task-panel-text');
  assert.strictEqual(
    decl(text, 'white-space'),
    'pre-wrap',
    'the task text keeps its line breaks',
  );
  assert.strictEqual(
    decl(text, 'color'),
    'var(--fg)',
    'the task text is the transcript foreground',
  );
  // The old fixed panel's inverted palette and its tooltip are gone.
  assert.strictEqual(ruleBody('#task-panel'), '', 'no fixed #task-panel');
  assert.strictEqual(
    ruleBody('#custom-tooltip.task-panel-tooltip'),
    '',
    'no task-panel tooltip variant',
  );
  console.log('  ok - .task-panel-text reads like the transcript');
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
  testTaskPanelIsTheAccentTintedPanel();
  testTaskPanelTextMatchesTheTranscript();
  testEditorTabsModeDefaultsOn();
  console.log('all task-panel style tests passed');
}

main();
