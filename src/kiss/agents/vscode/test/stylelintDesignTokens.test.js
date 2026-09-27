// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the `kiss/design-tokens` Stylelint rule
// (scripts/stylelint-design-tokens.mjs): each case lints real CSS through
// Stylelint's Node API with the plugin loaded, and the last case lints the
// shipped stylesheets with the repo's own .stylelintrc.json.

/* global require, __dirname, console, process */

'use strict';

const assert = require('assert');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const PLUGIN = path.join(ROOT, 'scripts', 'stylelint-design-tokens.mjs');

let stylelint;

async function lint(code, options) {
  const ruleValue = options === undefined ? true : [true, options];
  const res = await stylelint.lint({
    code,
    config: {plugins: [PLUGIN], rules: {'kiss/design-tokens': ruleValue}},
  });
  return res.results[0];
}

async function words(code, options) {
  const result = await lint(code, options);
  return result.warnings.map(w => w.text);
}

async function testHexColours() {
  const tokens = {tokenSelectors: [':root', 'body.remote-chat']};
  assert.deepStrictEqual(await words(':root { --bg: #1f1f1f; }', tokens), []);
  assert.deepStrictEqual(await words('body.remote-chat { --bg: #fff; }', tokens), []);
  const bad = await words('.a { color: #d29922; }', tokens);
  assert.strictEqual(bad.length, 1);
  assert.match(bad[0], /Hex colour "#d29922" outside the token block/);
  const fallback = await words('.a { color: var(--yellow, #abc); }', tokens);
  assert.match(fallback[0], /"#abc"/);
  // A custom property in a component rule is not the token block.
  assert.strictEqual((await words('.a { --x: #fff; }', tokens)).length, 1);
  // A plain property inside the token block is not a token.
  assert.strictEqual((await words(':root { color: #fff; }', tokens)).length, 1);
  // A custom property whose parent is an at-rule, not a style rule.
  assert.strictEqual((await words('@page { --x: #fff; }', tokens)).length, 1);
  // url() and strings are not colours.
  assert.deepStrictEqual(
    await words('.a { background: url(#frag) , url("a.svg#abc"); content: "#fff"; }', tokens),
    [],
  );
  console.log('  ok - hex colours only inside token-block custom properties');
}

async function testRadii() {
  assert.match((await words('.a { border-radius: 6px; }'))[0], /Raw radius "6px"/);
  assert.match(
    (await words('.a { border-top-left-radius: 4PX; }'))[0],
    /Raw radius "4PX"/,
  );
  assert.deepStrictEqual(
    await words('.a { border-radius: 0; } .b { border-radius: 50%; } .c { border-radius: var(--radius-md); }'),
    [],
  );
  // Only border radii and --radius-* tokens are radii.
  assert.deepStrictEqual(
    await words('.a { --blur-radius: 6px; filter: blur(var(--blur-radius)); }'),
    [],
  );
  console.log('  ok - px radii rejected, 0 / % / tokens allowed');
}

async function testZIndex() {
  assert.match((await words('.a { z-index: 10; }'))[0], /Numeric z-index "10"/);
  assert.match((await words('.a { z-index: 0; }'))[0], /Numeric z-index "0"/);
  assert.match(
    (await words('.a { z-index: calc(var(--z-modal) + 1); }'))[0],
    /Numeric z-index "1"/,
  );
  assert.deepStrictEqual(
    await words('.a { z-index: var(--z-modal); } .b { z-index: auto; }'),
    [],
  );
  console.log('  ok - numeric z-index rejected, tokens and auto allowed');
}

async function testFontSize() {
  assert.match((await words('.a { font-size: 12px; }'))[0], /Raw font size "12px"/);
  assert.match((await words('.a { font-size: 1.15em; }'))[0], /Raw font size "1.15em"/);
  assert.match((await words('.a { font-size: var(--fs-sm, 11px); }'))[0], /"11px"/);
  assert.deepStrictEqual(
    await words('.a { font-size: var(--fs-sm); } .b { font-size: inherit; } .c { font-size: 0; }'),
    [],
  );
  // Other properties may use raw lengths.
  assert.deepStrictEqual(await words('.a { padding: 4px; width: 12px; }'), []);
  // The font shorthand: the size is checked, the line height after "/" is not.
  assert.match((await words('.a { font: 12px sans-serif; }'))[0], /Raw font size "12px"/);
  assert.deepStrictEqual(
    await words('.a { font: oblique 10deg var(--fs-sm)/1.4em var(--vscode-font-family); }'),
    [],
  );
  console.log('  ok - unit font sizes rejected, --fs-* tokens allowed');
}

async function testTokenRedefinitionsOutsideTokenBlock() {
  const bad = await words(
    '.a { --radius-md: 6px; --z-dialog: 300; --fs-base: 12px; --space-2: 8px; }',
  );
  assert.deepStrictEqual(
    bad.map(t => t.split(' outside')[0]),
    ['Raw radius "6px"', 'Numeric z-index "300"', 'Raw font size "12px"'],
  );
  assert.deepStrictEqual(await words(':root { --radius-md: 6px; --z-dialog: 300; }'), []);
  console.log('  ok - component-level token redefinitions rejected');
}

async function testDefaultTokenSelectorIsRoot() {
  assert.deepStrictEqual(await words(':root { --x: #fff; }'), []);
  assert.strictEqual((await words('body.remote-chat { --x: #fff; }')).length, 1);
  console.log('  ok - default token block is :root');
}

async function testInvalidOptionIsReported() {
  const result = await lint('.a { color: #fff; }', {tokenSelectors: [1]});
  assert.strictEqual(result.warnings.length, 0);
  assert.strictEqual(result.invalidOptionWarnings.length, 1);
  console.log('  ok - invalid tokenSelectors reported, rule skipped');
}

async function testShippedStylesheetsPass() {
  const res = await stylelint.lint({
    files: ['media/**/*.css'],
    cwd: ROOT,
    ignorePattern: ['media/**/*.min.css'],
  });
  const problems = res.results.flatMap(r =>
    r.warnings.map(w => `${path.relative(ROOT, r.source)}:${w.line} ${w.text}`),
  );
  assert.deepStrictEqual(problems, []);
  const main = res.results.find(r => r.source.endsWith(path.join('media', 'main.css')));
  assert.ok(main, 'main.css was linted');
  console.log('  ok - media/*.css pass the repo Stylelint config');
}

async function runTests() {
  stylelint = (await import('stylelint')).default;
  await testHexColours();
  await testRadii();
  await testZIndex();
  await testFontSize();
  await testTokenRedefinitionsOutsideTokenBlock();
  await testDefaultTokenSelectorIsRoot();
  await testInvalidOptionIsReported();
  await testShippedStylesheetsPass();
}

runTests().then(
  () => {
    console.log('\n8 passed, 0 failed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
