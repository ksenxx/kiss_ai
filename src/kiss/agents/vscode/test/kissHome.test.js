// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The extension host's state directory is the brand's: `$KISS_HOME` when
// set, else `~/<home_dir>` from media/brand.json (`.kiss` for stock KISS
// Sorcar, `.s10s` for a white-label build), the twin of
// kiss.core.config.kiss_home().  Exercises kissHome.js from src/ (as the
// tests require it) and from out/ (as the extension does), brand.ts's
// `homeDir`/`{{HOME_DIR}}`, and UpdateChecker's cache path, against real
// brand.json copies: the module reads the file next to it, so a branded
// copy of the compiled output lives in a temp dir.

/* global require, process, console, __dirname */

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {spawnSync} = require('child_process');

const SRC = path.join(__dirname, '..', 'src');
const OUT = path.join(__dirname, '..', 'out');
const MEDIA = path.join(__dirname, '..', 'media');

function runNode(code, env) {
  const res = spawnSync(process.execPath, ['-e', code], {
    env: {...process.env, ...env, USERPROFILE: env.HOME},
    encoding: 'utf-8',
  });
  if (res.status !== 0) {
    throw new Error(`child failed: ${res.stderr || res.stdout}`);
  }
  return res.stdout.trim();
}

const {brandHomeDirName, kissHomeDir, DEFAULT_HOME_DIR_NAME} = require(
  path.join(SRC, 'kissHome.js'),
);

// brandHomeDirName(file): a plain name is taken, anything else is .kiss.
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-home-'));
const brandFile = path.join(tmp, 'brand.json');
const cases = [
  [{home_dir: '.s10s'}, '.s10s'],
  [{home_dir: 'state'}, 'state'],
  [{}, '.kiss'],
  [{home_dir: ''}, '.kiss'],
  [{home_dir: '.'}, '.kiss'],
  [{home_dir: '..'}, '.kiss'],
  [{home_dir: 'a/b'}, '.kiss'],
  [{home_dir: 'a\\b'}, '.kiss'],
  [{home_dir: 7}, '.kiss'],
];
for (const [json, expected] of cases) {
  fs.writeFileSync(brandFile, JSON.stringify(json));
  assert.strictEqual(brandHomeDirName(brandFile), expected, JSON.stringify(json));
}
fs.writeFileSync(brandFile, '{not json');
assert.strictEqual(brandHomeDirName(brandFile), '.kiss');
assert.strictEqual(brandHomeDirName(path.join(tmp, 'missing.json')), '.kiss');
assert.strictEqual(DEFAULT_HOME_DIR_NAME, '.kiss');
console.log('  ok - brandHomeDirName validates home_dir');

// The checkout's own brand.json drives the default; $KISS_HOME wins.
const stockName = JSON.parse(fs.readFileSync(path.join(MEDIA, 'brand.json'), 'utf-8')).home_dir;
assert.strictEqual(
  runNode(`console.log(require(${JSON.stringify(path.join(SRC, 'kissHome.js'))}).kissHomeDir())`, {
    HOME: tmp,
    KISS_HOME: '',
  }),
  path.join(tmp, stockName),
);
assert.strictEqual(
  runNode(`console.log(require(${JSON.stringify(path.join(SRC, 'kissHome.js'))}).kissHomeDir())`, {
    HOME: tmp,
    KISS_HOME: path.join(tmp, 'elsewhere'),
  }),
  path.join(tmp, 'elsewhere'),
);
assert.strictEqual(typeof kissHomeDir(), 'string');
console.log('  ok - kissHomeDir: $KISS_HOME, else ~/<brand home_dir>');

// A branded copy of the compiled extension (out/ next to media/brand.json
// saying .s10s): every consumer resolves ~/.s10s and renders {{HOME_DIR}}.
const branded = path.join(tmp, 'ext');
fs.mkdirSync(path.join(branded, 'media'), {recursive: true});
fs.cpSync(OUT, path.join(branded, 'out'), {recursive: true});
fs.writeFileSync(
  path.join(branded, 'media', 'brand.json'),
  JSON.stringify({product_name: 'Seamless Loop', short_name: 's10s', home_dir: '.s10s'}),
);
const probe = `
  const out = ${JSON.stringify(path.join(branded, 'out'))};
  const path = require('path');
  const brand = require(path.join(out, 'brand.js'));
  const assets = require(path.join(out, 'userAssets.js'));
  const checker = require(path.join(out, 'UpdateChecker.js'));
  console.log(brand.BRAND.homeDir);
  console.log(assets.kissHomeDir());
  console.log(assets.sorcarEndpointPath());
  console.log(checker.kissHomeDir ? 'exported' : 'internal');
`;
const lines = runNode(probe, {HOME: tmp, KISS_HOME: '', KISS_SORCAR_LOCAL: ''}).split('\n');
assert.deepStrictEqual(lines, [
  '.s10s',
  path.join(tmp, '.s10s'),
  path.join(tmp, '.s10s', 'sorcar-local.json'),
  'internal',
]);
console.log('  ok - a branded build resolves ~/<home_dir> everywhere');

fs.rmSync(tmp, {recursive: true, force: true});
console.log('kissHome.test.js: all passed');
