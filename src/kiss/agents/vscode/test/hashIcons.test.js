// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/**
 * Content-hashed icon names in the VSIX (scripts/hash-icons.js,
 * scripts/package-vsix.js) and their runtime counterpart `mediaIconPath()`
 * in src/brand.ts.
 *
 * The VS Code server caches every extension file on the client for a year
 * under a URL that is only the file path inside `<publisher>.<name>-<version>/`,
 * so a rebuilt extension with the same version and a different icon kept
 * showing the old icon.  The build must therefore point the packaged
 * manifest at content-hashed copies while leaving the tracked manifest
 * untouched.
 */

const assert = require('assert');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {execFileSync, spawn, spawnSync} = require('child_process');

const EXT_ROOT = path.resolve(__dirname, '..');
const PACKAGE_VSIX = path.join(EXT_ROOT, 'scripts', 'package-vsix.js');
const {
  applyHashedIcons,
  hashIconManifest,
  hashedIconIndex,
  HASHED_DIR,
  INDEX_FILE,
} = require('../scripts/hash-icons.js');
const {mediaIconPath} = require('../out/brand.js');

const SVG = '<svg xmlns="http://www.w3.org/2000/svg"><text>KS</text></svg>\n';
const PNG = Buffer.from('89504e470d0a1a0a', 'hex');
const md5 = data =>
  crypto.createHash('md5').update(data).digest('hex').slice(0, 8);
const PACK_ARGS = [
  PACKAGE_VSIX,
  '--allow-missing-repository',
  '-o',
  'demo.vsix',
];

/**
 * A minimal extension vsce accepts, with icons referenced like the real
 * manifest and the repository's own .vscodeignore.
 */
function makeExtension(manifest, files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hash-icons-'));
  fs.mkdirSync(path.join(root, 'media', 'sub'), {recursive: true});
  for (const [rel, data] of Object.entries(files))
    fs.writeFileSync(path.join(root, rel), data);
  fs.writeFileSync(path.join(root, 'LICENSE'), 'MIT\n');
  fs.writeFileSync(
    path.join(root, 'extension.js'),
    'exports.activate = () => {};\n',
  );
  fs.copyFileSync(
    path.join(EXT_ROOT, '.vscodeignore'),
    path.join(root, '.vscodeignore'),
  );
  fs.writeFileSync(path.join(root, 'package.json'), manifest);
  return root;
}

const STOCK_MANIFEST = {
  name: 'demo',
  displayName: 'Demo',
  version: '1.0.0',
  publisher: 'demo',
  engines: {vscode: '^1.98.0'},
  main: './extension.js',
  icon: 'media/thumb.png',
  contributes: {
    viewsContainers: {
      activitybar: [{id: 'c', title: 'Demo', icon: 'media/kiss-icon.svg'}],
      secondarySidebar: [
        {id: 's', title: 'Demo', icon: 'media/kiss-icon.svg', order: -100},
      ],
    },
    commands: [
      {
        command: 'demo.open',
        title: 'Open',
        icon: {light: 'media/kiss-icon.svg', dark: 'media/kiss-icon.svg'},
      },
      {command: 'demo.missing', title: 'Missing', icon: 'media/missing.svg'},
      {command: 'demo.nested', title: 'Nested', icon: 'media/sub/deep.svg'},
      {command: 'demo.other', title: 'Other', icon: 'other/kiss-icon.svg'},
      {
        command: 'demo.text',
        title: 'media/kiss-icon.svg is not a file here',
        icon: '$(sparkle)',
      },
    ],
  },
};
const FILES = {
  'media/kiss-icon.svg': SVG,
  'media/thumb.png': PNG,
  'media/spinner.svg': SVG.replace('KS', 'ring'),
  'media/sub/deep.svg': SVG,
  'media/notes.txt': 'not an image\n',
};
const SVG_TARGET = `media/hashed/kiss-icon-${md5(SVG)}.svg`;
const PNG_TARGET = `media/hashed/thumb-${md5(PNG)}.png`;
const SPINNER_TARGET = `media/hashed/spinner-${md5(FILES['media/spinner.svg'])}.svg`;

function listVsix(vsix) {
  const script =
    'import sys,zipfile; print("\\n".join(zipfile.ZipFile(sys.argv[1]).namelist()))';
  return execFileSync('python3', ['-c', script, vsix], {encoding: 'utf-8'})
    .trim()
    .split('\n');
}

function readVsixEntry(vsix, name) {
  const script =
    'import sys,zipfile; sys.stdout.write(zipfile.ZipFile(sys.argv[1]).read(sys.argv[2]).decode())';
  return execFileSync('python3', ['-c', script, vsix, name], {
    encoding: 'utf-8',
  });
}

function readManifest(root) {
  return fs.readFileSync(path.join(root, 'package.json'), 'utf-8');
}

function testIndex() {
  const root = makeExtension(JSON.stringify(STOCK_MANIFEST), FILES);
  // Every image directly under media/, nothing else: no nested dir, no text file.
  assert.deepStrictEqual(hashedIconIndex(root), {
    'media/kiss-icon.svg': SVG_TARGET,
    'media/spinner.svg': SPINNER_TARGET,
    'media/thumb.png': PNG_TARGET,
  });
  assert.deepStrictEqual(
    hashedIconIndex(path.join(root, 'nowhere')),
    {},
    'no media dir: empty index',
  );
  fs.rmSync(root, {recursive: true, force: true});
}

function testPureRewrite() {
  const root = makeExtension(JSON.stringify(STOCK_MANIFEST), FILES);
  const index = hashedIconIndex(root);
  const manifest = hashIconManifest(STOCK_MANIFEST, index);
  assert.strictEqual(manifest.icon, PNG_TARGET);
  assert.strictEqual(
    manifest.contributes.viewsContainers.activitybar[0].icon,
    SVG_TARGET,
  );
  assert.strictEqual(
    manifest.contributes.viewsContainers.secondarySidebar[0].icon,
    SVG_TARGET,
  );
  assert.deepStrictEqual(manifest.contributes.commands[0].icon, {
    light: SVG_TARGET,
    dark: SVG_TARGET,
  });
  // Missing, nested, non-media, codicon and free-text strings are left alone.
  assert.strictEqual(
    manifest.contributes.commands[1].icon,
    'media/missing.svg',
  );
  assert.strictEqual(
    manifest.contributes.commands[2].icon,
    'media/sub/deep.svg',
  );
  assert.strictEqual(
    manifest.contributes.commands[3].icon,
    'other/kiss-icon.svg',
  );
  assert.strictEqual(manifest.contributes.commands[4].icon, '$(sparkle)');
  assert.strictEqual(
    manifest.contributes.commands[4].title,
    'media/kiss-icon.svg is not a file here',
  );
  assert.strictEqual(
    manifest.contributes.viewsContainers.secondarySidebar[0].order,
    -100,
  );
  assert.strictEqual(manifest.version, '1.0.0');
  // Idempotent: a manifest left hashed by a killed build re-resolves to the
  // current hash of the plain file, whether the old hash is current or stale.
  assert.strictEqual(hashIconManifest(manifest, index).icon, PNG_TARGET);
  const stale = hashIconManifest(
    {
      icon: 'media/hashed/thumb-00000000.png',
      other: 'media/hashed/gone-00000000.png',
    },
    index,
  );
  assert.strictEqual(stale.icon, PNG_TARGET);
  assert.strictEqual(
    stale.other,
    'media/hashed/gone-00000000.png',
    'a hashed ref without a plain file stays',
  );
  // The input manifest is not mutated.
  assert.strictEqual(STOCK_MANIFEST.icon, 'media/thumb.png');
  fs.rmSync(root, {recursive: true, force: true});
}

function testApplyAndRestore() {
  const original =
    '{\n\t"name": "demo",\n\t"version": "1.0.0",\n\t"icon": "media/thumb.png"\n}\n';
  const root = makeExtension(original, FILES);
  // Leftovers from an earlier build are cleared, not shipped again.
  fs.mkdirSync(path.join(root, HASHED_DIR), {recursive: true});
  fs.writeFileSync(
    path.join(root, HASHED_DIR, 'kiss-icon-00000000.svg'),
    'old brand\n',
  );
  const restore = applyHashedIcons(root);
  assert.strictEqual(JSON.parse(readManifest(root)).icon, PNG_TARGET);
  assert.ok(
    fs.readFileSync(path.join(root, PNG_TARGET)).equals(PNG),
    'hashed copy has the icon bytes',
  );
  assert.ok(
    fs.existsSync(path.join(root, SVG_TARGET)),
    'unreferenced icons are hashed for runtime use',
  );
  assert.ok(
    !fs.existsSync(path.join(root, HASHED_DIR, 'kiss-icon-00000000.svg')),
    'stale copy removed',
  );
  assert.deepStrictEqual(
    JSON.parse(fs.readFileSync(path.join(root, INDEX_FILE), 'utf-8')),
    hashedIconIndex(root),
  );
  assert.strictEqual(
    fs.readdirSync(path.join(root, HASHED_DIR)).length,
    4,
    '3 icons + index.json',
  );
  restore();
  assert.strictEqual(readManifest(root), original);
  assert.ok(
    !fs.existsSync(path.join(root, HASHED_DIR)),
    'restore removes media/hashed/',
  );
  fs.rmSync(root, {recursive: true, force: true});
}

function testApplyWithoutIcons() {
  const original = '{"name": "demo", "version": "1.0.0"}\n';
  const root = makeExtension(original, {});
  const restore = applyHashedIcons(root);
  assert.strictEqual(
    readManifest(root),
    '{\n  "name": "demo",\n  "version": "1.0.0"\n}\n',
  );
  assert.deepStrictEqual(
    JSON.parse(fs.readFileSync(path.join(root, INDEX_FILE), 'utf-8')),
    {},
  );
  restore();
  assert.strictEqual(readManifest(root), original);
  fs.rmSync(root, {recursive: true, force: true});
}

function testPackagedVsix() {
  const original = JSON.stringify(STOCK_MANIFEST, null, 2) + '\n';
  const root = makeExtension(original, FILES);
  const res = spawnSync(process.execPath, PACK_ARGS, {
    cwd: root,
    encoding: 'utf-8',
  });
  assert.strictEqual(
    res.status,
    0,
    `package-vsix.js failed:\n${res.stdout}\n${res.stderr}`,
  );
  assert.match(res.stdout, /Packaged: demo\.vsix/);
  // The tracked manifest is byte-identical and the scratch dir is gone.
  assert.strictEqual(readManifest(root), original);
  assert.ok(!fs.existsSync(path.join(root, HASHED_DIR)));
  const vsix = path.join(root, 'demo.vsix');
  const names = listVsix(vsix);
  for (const rel of [
    SVG_TARGET,
    PNG_TARGET,
    SPINNER_TARGET,
    'media/hashed/index.json',
    'media/kiss-icon.svg',
  ]) {
    assert.ok(
      names.includes(`extension/${rel}`),
      `VSIX ships ${rel}: ${names.join(', ')}`,
    );
  }
  const packaged = JSON.parse(readVsixEntry(vsix, 'extension/package.json'));
  assert.strictEqual(
    packaged.contributes.viewsContainers.activitybar[0].icon,
    SVG_TARGET,
  );
  assert.strictEqual(packaged.contributes.commands[0].icon.dark, SVG_TARGET);
  assert.strictEqual(packaged.icon, PNG_TARGET);
  assert.strictEqual(
    JSON.parse(readVsixEntry(vsix, 'extension/media/hashed/index.json'))[
      'media/spinner.svg'
    ],
    SPINNER_TARGET,
  );
  // A second icon version of the same extension version gets a different URL.
  const svg2 = SVG.replace('KS', 'SL');
  fs.writeFileSync(path.join(root, 'media', 'kiss-icon.svg'), svg2);
  const res2 = spawnSync(
    process.execPath,
    [...PACK_ARGS.slice(0, -1), 'demo2.vsix'],
    {cwd: root, encoding: 'utf-8'},
  );
  assert.strictEqual(res2.status, 0, res2.stdout + res2.stderr);
  const packaged2 = JSON.parse(
    readVsixEntry(path.join(root, 'demo2.vsix'), 'extension/package.json'),
  );
  assert.strictEqual(
    packaged2.contributes.viewsContainers.activitybar[0].icon,
    `media/hashed/kiss-icon-${md5(svg2)}.svg`,
  );
  assert.notStrictEqual(
    packaged2.contributes.viewsContainers.activitybar[0].icon,
    SVG_TARGET,
  );
  fs.rmSync(root, {recursive: true, force: true});
}

function testRetryAfterKilledBuild() {
  // A build killed with SIGKILL leaves the hashed manifest behind; the
  // next build must still pick up a changed icon.
  const root = makeExtension(
    JSON.stringify(STOCK_MANIFEST, null, 2) + '\n',
    FILES,
  );
  applyHashedIcons(root); // restore function dropped: simulates the kill
  assert.ok(readManifest(root).includes(SVG_TARGET));
  const svg2 = SVG.replace('KS', 'NEW');
  fs.writeFileSync(path.join(root, 'media', 'kiss-icon.svg'), svg2);
  const res = spawnSync(process.execPath, PACK_ARGS, {
    cwd: root,
    encoding: 'utf-8',
  });
  assert.strictEqual(res.status, 0, res.stdout + res.stderr);
  const packaged = JSON.parse(
    readVsixEntry(path.join(root, 'demo.vsix'), 'extension/package.json'),
  );
  assert.strictEqual(
    packaged.contributes.viewsContainers.activitybar[0].icon,
    `media/hashed/kiss-icon-${md5(svg2)}.svg`,
  );
  assert.strictEqual(
    readVsixEntry(
      path.join(root, 'demo.vsix'),
      `extension/media/hashed/kiss-icon-${md5(svg2)}.svg`,
    ),
    svg2,
  );
  // The checkout ends up with plain names again, never with references
  // into the deleted media/hashed/.
  assert.strictEqual(
    readManifest(root),
    JSON.stringify(STOCK_MANIFEST, null, 2) + '\n',
  );
  assert.ok(!fs.existsSync(path.join(root, HASHED_DIR)));
  fs.rmSync(root, {recursive: true, force: true});
}

function testFailedPackagingRestoresManifest() {
  // A non-semver version: vsce rejects the manifest after hash-icons rewrote it.
  const original =
    JSON.stringify({...STOCK_MANIFEST, version: 'not.a.version'}, null, 2) +
    '\n';
  const root = makeExtension(original, FILES);
  const res = spawnSync(process.execPath, PACK_ARGS, {
    cwd: root,
    encoding: 'utf-8',
  });
  assert.strictEqual(res.status, 1, `expected vsce to fail:\n${res.stdout}`);
  assert.match(res.stderr, /version/i);
  assert.strictEqual(readManifest(root), original);
  assert.ok(!fs.existsSync(path.join(root, HASHED_DIR)));
  assert.ok(!fs.existsSync(path.join(root, 'demo.vsix')));
  fs.rmSync(root, {recursive: true, force: true});
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function testInterruptedPackagingRestoresManifest() {
  const original = JSON.stringify(STOCK_MANIFEST, null, 2) + '\n';
  // 48 MB of random bytes keep deflate busy for well over a second.
  const root = makeExtension(original, {
    ...FILES,
    'media/big.bin': crypto.randomBytes(48 * 1024 * 1024),
  });
  const child = spawn(process.execPath, PACK_ARGS, {
    cwd: root,
    stdio: 'ignore',
  });
  let exitCode;
  const exited = new Promise(resolve =>
    child.on('exit', code => {
      exitCode = code;
      resolve(code);
    }),
  );
  const deadline = Date.now() + 20_000;
  while (
    exitCode === undefined &&
    !readManifest(root).includes('media/hashed/') &&
    Date.now() < deadline
  ) {
    await sleep(2);
  }
  assert.strictEqual(
    exitCode,
    undefined,
    'packaging finished before the test could interrupt it',
  );
  assert.ok(
    readManifest(root).includes('media/hashed/'),
    'manifest was rewritten for vsce',
  );
  child.kill('SIGTERM');
  const code = await Promise.race([
    exited,
    sleep(10_000).then(() => 'timeout'),
  ]);
  if (code === 'timeout') child.kill('SIGKILL');
  assert.strictEqual(code, 1, 'the signal handler exits with status 1');
  assert.strictEqual(readManifest(root), original);
  assert.ok(!fs.existsSync(path.join(root, HASHED_DIR)));
  fs.rmSync(root, {recursive: true, force: true});
}

function testMediaIconPath() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'media-icon-'));
  // Running from source: no index, plain paths.
  assert.strictEqual(
    mediaIconPath(root, 'kiss-icon.svg'),
    'media/kiss-icon.svg',
  );
  fs.mkdirSync(path.join(root, HASHED_DIR), {recursive: true});
  const indexFile = path.join(root, INDEX_FILE);
  fs.writeFileSync(
    indexFile,
    JSON.stringify({
      'media/kiss-icon.svg': 'media/hashed/kiss-icon-deadbeef.svg',
      'media/x.svg': '',
    }),
  );
  assert.strictEqual(
    mediaIconPath(root, 'kiss-icon.svg'),
    'media/hashed/kiss-icon-deadbeef.svg',
  );
  assert.strictEqual(
    mediaIconPath(root, 'spinner-running.svg'),
    'media/spinner-running.svg',
    'unlisted icon',
  );
  assert.strictEqual(
    mediaIconPath(root, 'x.svg'),
    'media/x.svg',
    'empty mapping',
  );
  fs.writeFileSync(indexFile, '{not json');
  assert.strictEqual(
    mediaIconPath(root, 'kiss-icon.svg'),
    'media/kiss-icon.svg',
    'corrupt index',
  );
  // The checkout itself carries no index once a build has finished.
  assert.ok(
    !fs.existsSync(path.join(EXT_ROOT, INDEX_FILE)),
    'a finished build leaves no media/hashed/ in the checkout',
  );
  assert.strictEqual(
    mediaIconPath(EXT_ROOT, 'kiss-icon.svg'),
    'media/kiss-icon.svg',
  );
  fs.rmSync(root, {recursive: true, force: true});
}

async function main() {
  testIndex();
  testPureRewrite();
  testApplyAndRestore();
  testApplyWithoutIcons();
  testMediaIconPath();
  testPackagedVsix();
  testRetryAfterKilledBuild();
  testFailedPackagingRestoresManifest();
  await testInterruptedPackagingRestoresManifest();
  console.log('hashIcons.test.js: all tests passed');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
