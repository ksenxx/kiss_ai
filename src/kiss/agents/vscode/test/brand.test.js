// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/**
 * Product branding: `src/brand.ts` (runtime strings) and
 * `scripts/apply-brand.js` (manifest rewrite) both read `media/brand.json`.
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {execFileSync} = require('child_process');

const EXT_ROOT = path.resolve(__dirname, '..');
const {BRAND, BRAND_FILE, PRODUCT_NAME, SHORT_NAME, loadBrand} = require('../out/brand.js');
const {brandManifest, brandText} = require('../scripts/apply-brand.js');

const CHECKOUT_BRAND = JSON.parse(fs.readFileSync(path.join(EXT_ROOT, 'media', 'brand.json'), 'utf-8'));
const STOCK_BRAND = {
  product_name: 'KISS Sorcar',
  short_name: 'KISS',
  tagline: 'Your AI assistant. Ask me anything!',
  extension_description: 'The open-source AI coding agent.',
};
// A stock-shaped manifest fixture: the pure function is tested against this
// so the assertions hold whichever brand the checkout's package.json carries.
const manifest = {
  name: 'kiss-sorcar',
  displayName: 'KISS Sorcar',
  description: STOCK_BRAND.extension_description,
  icon: 'media/thumbnail.jpeg',
  capabilities: {untrustedWorkspaces: {supported: true, description: 'KISS Sorcar runs a local Python backend.'}},
  contributes: {
    viewsContainers: {
      activitybar: [{id: 'kissSorcarContainer', title: 'KISS Sorcar', icon: 'media/kiss-icon.svg'}],
      secondarySidebar: [{id: 'kissSorcarSecondary', title: 'KISS Sorcar', icon: 'media/kiss-icon.svg'}],
    },
    commands: [
      {command: 'kissSorcar.openPanel', title: 'KISS: Open Chat'},
      {command: 'kissSorcar.showHistory', title: 'KISS: Show Chat History', icon: 'media/kiss-icon.png'},
    ],
    configuration: {
      title: 'KISS Sorcar',
      properties: {
        'kissSorcar.kissProjectPath': {type: 'string', description: 'Path to KISS project root'},
        'kissSorcar.editorTabsMode': {type: 'boolean', description: 'Open each KISS Sorcar chat as an editor tab'},
      },
    },
  },
};
const S10S = {
  product_name: 'Seamless Loop',
  short_name: 's10s',
  tagline: 'SeamlessLabs assistant',
  extension_description: 'The SeamlessLabs AI assistant.',
};

function withTempDir(fn) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-brand-'));
  try {
    return fn(dir);
  } finally {
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

// --- src/brand.ts -----------------------------------------------------------

// The module-level constants come from media/brand.json (whatever brand the
// checkout carries: stock KISS Sorcar or a re-branded distribution).
assert.strictEqual(path.resolve(BRAND_FILE), path.join(EXT_ROOT, 'media', 'brand.json'));
assert.deepStrictEqual(BRAND, {
  productName: CHECKOUT_BRAND.product_name,
  shortName: CHECKOUT_BRAND.short_name,
  tagline: CHECKOUT_BRAND.tagline,
});
assert.strictEqual(PRODUCT_NAME, BRAND.productName);
assert.strictEqual(SHORT_NAME, BRAND.shortName);

// A custom file overrides only the keys it names; junk values fall back.
withTempDir(dir => {
  const file = path.join(dir, 'brand.json');
  fs.writeFileSync(file, JSON.stringify({product_name: 'Seamless Loop', short_name: '', tagline: 7}));
  assert.deepStrictEqual(loadBrand(file), {
    productName: 'Seamless Loop',
    shortName: 'KISS',
    tagline: STOCK_BRAND.tagline,
  });
  fs.writeFileSync(file, '{oops');
  assert.deepStrictEqual(loadBrand(file), BRAND, 'malformed JSON falls back to stock');
  fs.writeFileSync(file, '[1,2]');
  assert.deepStrictEqual(loadBrand(file), BRAND, 'non-object JSON falls back to stock');
  assert.deepStrictEqual(loadBrand(path.join(dir, 'missing.json')), BRAND, 'missing file falls back');
});

// --- scripts/apply-brand.js --------------------------------------------------

// Stock brand: the manifest is unchanged, field for field.
assert.deepStrictEqual(brandManifest(manifest, STOCK_BRAND), manifest);
assert.strictEqual(brandText('KISS: Open Chat', STOCK_BRAND), 'KISS: Open Chat');
assert.strictEqual(brandText('KISS: Open Chat', S10S), 's10s: Open Chat');
assert.strictEqual(brandText('Welcome to KISS Sorcar', S10S), 'Welcome to Seamless Loop');

// The checkout's real manifest is consistent with its real brand.json: applying
// the brand it already carries changes nothing.
const realManifest = JSON.parse(fs.readFileSync(path.join(EXT_ROOT, 'package.json'), 'utf-8'));
assert.deepStrictEqual(brandManifest(realManifest, CHECKOUT_BRAND), realManifest);

// Custom brand: display strings change, identifiers do not.
const branded = brandManifest(manifest, S10S);
assert.strictEqual(branded.displayName, 'Seamless Loop');
assert.strictEqual(branded.description, S10S.extension_description);
assert.strictEqual(branded.name, manifest.name, 'extension id is untouched');
assert.strictEqual(branded.icon, manifest.icon, 'icon path is untouched (the file is swapped instead)');
assert.strictEqual(
  branded.capabilities.untrustedWorkspaces.description,
  'Seamless Loop runs a local Python backend.',
);
for (const containers of Object.values(branded.contributes.viewsContainers)) {
  for (const c of containers) assert.strictEqual(c.title, 'Seamless Loop');
}
assert.deepStrictEqual(
  branded.contributes.commands.map(c => c.command),
  manifest.contributes.commands.map(c => c.command),
  'command ids are untouched',
);
for (const c of branded.contributes.commands) {
  assert.ok(c.title.startsWith('s10s: '), c.title);
  assert.ok(!/KISS|Sorcar/.test(c.title), c.title);
}
assert.strictEqual(branded.contributes.configuration.title, 'Seamless Loop');
assert.deepStrictEqual(
  Object.keys(branded.contributes.configuration.properties),
  Object.keys(manifest.contributes.configuration.properties),
  'setting keys are untouched',
);
assert.strictEqual(
  branded.contributes.configuration.properties['kissSorcar.editorTabsMode'].description,
  'Open each Seamless Loop chat as an editor tab',
);
assert.strictEqual(
  branded.contributes.configuration.properties['kissSorcar.kissProjectPath'].description,
  'Path to KISS project root',
  'the KISS project-root wording is not a product name',
);
assert.ok(
  !JSON.stringify(branded).includes('KISS Sorcar'),
  'no stock product name survives a custom brand',
);
// Second application is a no-op.
assert.deepStrictEqual(brandManifest(branded, S10S), branded);
// Missing keys keep the stock values.
const partial = brandManifest(manifest, {product_name: 'Seamless Loop'});
assert.strictEqual(partial.contributes.commands[0].title, 'KISS: Open Chat');
assert.strictEqual(partial.description, manifest.description);

// CLI: rewrites the manifest on disk only when something changed.
withTempDir(dir => {
  const brandFile = path.join(dir, 'brand.json');
  const manifestFile = path.join(dir, 'package.json');
  fs.writeFileSync(brandFile, JSON.stringify(S10S));
  fs.writeFileSync(manifestFile, JSON.stringify(manifest, null, 2) + '\n');
  const script = path.join(EXT_ROOT, 'scripts', 'apply-brand.js');
  const out1 = execFileSync(process.execPath, [script, brandFile, manifestFile], {encoding: 'utf-8'});
  assert.ok(out1.includes('Branded'), out1);
  const written = JSON.parse(fs.readFileSync(manifestFile, 'utf-8'));
  assert.strictEqual(written.displayName, 'Seamless Loop');
  const before = fs.statSync(manifestFile).mtimeMs;
  const out2 = execFileSync(process.execPath, [script, brandFile, manifestFile], {encoding: 'utf-8'});
  assert.strictEqual(out2, '', 'second run prints nothing');
  assert.strictEqual(fs.statSync(manifestFile).mtimeMs, before, 'second run does not rewrite');
  // Default arguments point at the real manifest and brand, which are
  // already consistent: a no-op on disk as well.
  const realText = fs.readFileSync(path.join(EXT_ROOT, 'package.json'), 'utf-8');
  assert.strictEqual(execFileSync(process.execPath, [script], {cwd: EXT_ROOT, encoding: 'utf-8'}), '');
  assert.strictEqual(fs.readFileSync(path.join(EXT_ROOT, 'package.json'), 'utf-8'), realText);
});

console.log('brand.test.js: all assertions passed');
