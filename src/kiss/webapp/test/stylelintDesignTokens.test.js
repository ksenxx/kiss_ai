// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the `kiss/design-tokens` Stylelint rule
// (scripts/stylelint-design-tokens.mjs): each case lints real CSS through
// Stylelint's Node API with the plugin loaded.  The shipped stylesheets
// are linted with the repo's .stylelintrc.json by `npm run lint:css`.

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

// `line:column-endColumn` of every warning, in source order.
async function ranges(code) {
  const result = await lint(code);
  return result.warnings.map(w => `${w.line}:${w.column}-${w.endColumn}`);
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
    await words('.a { font: oblique 10deg 700 var(--fs-sm)/1.4em var(--theme-font-family); }'),
    [],
  );
  // Every length unit is a font size, not only the common ones.
  for (const size of ['1in', '2.54cm', '25.4mm', '4Q', '4dvw', '1lh', '2cqw']) {
    assert.match((await words(`.a { font-size: ${size}; }`))[0], /Raw font size/, size);
  }
  assert.match((await words('.a { font: 1in/1.5 sans-serif; }'))[0], /Raw font size "1in"/);
  assert.match((await words('.a { --fs-base: 1in; }'))[0], /Raw font size "1in"/);
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
  // Custom property names are case-sensitive: `--FS-gap` is not a token of
  // the `--fs-*` scale (A30); ordinary properties are case-insensitive.
  assert.deepStrictEqual(
    await words('.a { --FS-gap: 20px; --Z-count: 42; --RADIUS-blur: 6px; }'),
    [],
  );
  assert.match((await words('.a { FONT-SIZE: 12px; }'))[0], /Raw font size "12px"/);
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

// A22-1 (astra): a `/` inside a math function is a division, not the
// line-height separator of the `font` shorthand; it must not hide the raw
// sizes that follow it.
async function testDivisionInsideFunctionsKeepsFontSizeChecks() {
  assert.match(
    (await words('.a { font-size: min(var(--fs-base) / 2, 12px); }'))[0],
    /Raw font size "12px"/,
  );
  assert.deepStrictEqual(
    (await words('.a { font-size: clamp(var(--fs-base) / 2, 12px, 20px); }')).map(
      t => t.split(' outside')[0],
    ),
    ['Raw font size "12px"', 'Raw font size "20px"'],
  );
  assert.match(
    (await words('.a { font: oblique min(10deg / 2, 20deg) 12px sans-serif; }'))[0],
    /Raw font size "12px"/,
  );
  // The top-level `/` of the shorthand still starts the line height, even
  // when the line height itself divides.
  assert.deepStrictEqual(
    await words('.a { font: var(--fs-sm)/calc(2em / 2) sans-serif; }'),
    [],
  );
  // So does the `/` of a shorthand written as a var() fallback (A30): the
  // fallback is the whole shorthand, its line height no font size, while
  // a `/` inside a math function of the fallback's size stays a division.
  assert.deepStrictEqual(
    await words('.a { font: var(--font, var(--fs-sm)/1.4em sans-serif); }'),
    [],
  );
  assert.deepStrictEqual(
    await words('.a { font: var(--font, var(--fs-sm)/calc(2em / 2) serif); }'),
    [],
  );
  assert.match(
    (await words('.a { font: var(--font, min(12px / 2, 10px) serif); }'))[0],
    /Raw font size "12px"/,
  );
  // W10/F3: a var() nested in a math function is arithmetic, not shorthand
  // text: the `/` of its fallback must not start a line height and hide the
  // sizes after it; `font-size` has no line height at all.
  assert.match(
    (await words('.a { font-size: calc(var(--half, var(--fs-base) / 2) + 12px); }'))[0],
    /Raw font size "12px"/,
  );
  assert.match(
    (await words('.a { font-size: min(var(--half, var(--fs-base) / 2), 12px); }'))[0],
    /Raw font size "12px"/,
  );
  assert.match(
    (await words('.a { font: calc(var(--half, var(--fs-base) / 2) + 12px) serif; }'))[0],
    /Raw font size "12px"/,
  );
  assert.match(
    (await words('.a { font-size: var(--size, 10px / 2) 12px; }'))[1],
    /Raw font size "12px"/,
  );
  console.log('  ok - division inside functions does not hide font sizes');
}

// A22-2 (astra): the option validator accepts a single selector string as
// well as an array; a string must match selectors exactly, not as a
// substring (`':root'.includes('root')`).
async function testScalarTokenSelectorMatchesExactly() {
  const scalar = {tokenSelectors: ':root'};
  const result = await lint('root { --bg: #fff; }', scalar);
  assert.strictEqual(result.invalidOptionWarnings.length, 0);
  assert.strictEqual(result.warnings.length, 1);
  assert.match(result.warnings[0].text, /Hex colour "#fff"/);
  assert.deepStrictEqual(await words(':root { --bg: #fff; }', scalar), []);
  console.log('  ok - a scalar tokenSelectors string matches exactly');
}

// A22-2 (fable): each warning is placed at its own value word, not at the
// first occurrence of the same text in `prop: value`.
async function testWarningRangesPointAtTheValueWord() {
  assert.deepStrictEqual(
    await ranges('.a { border-radius: 6px 6px 0 0; }'),
    ['1:21-24', '1:25-28'],
  );
  assert.deepStrictEqual(await ranges('.a { --z-1: 1; }'), ['1:13-14']);
  assert.deepStrictEqual(await ranges('.a { border-radius : 6px; }'), ['1:22-25']);
  assert.deepStrictEqual(await ranges('.a { border-radius: /* x */ 6px; }'), ['1:29-32']);
  assert.deepStrictEqual(
    await ranges('.a {\n  font-size: var(--fs-sm,\n    11px);\n}'),
    ['3:5-9'],
  );
  console.log('  ok - warning ranges point at the value word');
}

const TESTS = [
  testHexColours,
  testRadii,
  testZIndex,
  testFontSize,
  testTokenRedefinitionsOutsideTokenBlock,
  testDefaultTokenSelectorIsRoot,
  testInvalidOptionIsReported,
  testDivisionInsideFunctionsKeepsFontSizeChecks,
  testScalarTokenSelectorMatchesExactly,
  testWarningRangesPointAtTheValueWord,
];

async function runTests() {
  stylelint = (await import('stylelint')).default;
  for (const t of TESTS) await t();
}

runTests().then(
  () => {
    console.log(`\n${TESTS.length} passed, 0 failed`);
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
