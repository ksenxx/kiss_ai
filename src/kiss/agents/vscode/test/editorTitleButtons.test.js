// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The editor-title (top-right of every editor window) buttons in
// editor-tabs mode:
//  - a settings gear (kissSorcar.openSettings) sits left of the KS
//    button and opens the settings UI
//  - the KS button (kissSorcar.showHistory) uses the colorful brand
//    PNG, not the monochrome gray SVG
// These are declarative package.json contributions VS Code reads
// directly, so the manifest is the runtime behavior being verified.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const manifest = JSON.parse(
  fs.readFileSync(path.join(root, 'package.json'), 'utf8'),
);

const editorTitle = manifest.contributes.menus['editor/title'];
assert.ok(Array.isArray(editorTitle), 'editor/title menu contributions exist');

// --- settings button ---------------------------------------------------
const settings = editorTitle.find(
  e => e.command === 'kissSorcar.openSettings',
);
assert.ok(settings, 'openSettings contributed to editor/title');
assert.strictEqual(
  settings.when,
  'config.kissSorcar.editorTabsMode',
  'settings button shows only in editor-tabs mode',
);

// --- KS button ---------------------------------------------------------
const ks = editorTitle.find(e => e.command === 'kissSorcar.showHistory');
assert.ok(ks, 'showHistory (KS button) contributed to editor/title');
assert.strictEqual(ks.when, 'config.kissSorcar.editorTabsMode');

// The gear sorts before the KS button (navigation@-2 < navigation@-1).
const settingsOrder = Number(settings.group.split('@')[1]);
const ksOrder = Number(ks.group.split('@')[1]);
assert.ok(
  settings.group.startsWith('navigation@') &&
    ks.group.startsWith('navigation@') &&
    settingsOrder < ksOrder,
  `gear (${settings.group}) must sort left of KS (${ks.group})`,
);

// Both commands are declared with icons.
const commands = new Map(manifest.contributes.commands.map(c => [c.command, c]));
const gearCmd = commands.get('kissSorcar.openSettings');
assert.strictEqual(gearCmd.icon, '$(settings-gear)', 'gear uses codicon');

const ksCmd = commands.get('kissSorcar.showHistory');
assert.strictEqual(
  ksCmd.icon,
  'media/kiss-icon.png',
  'KS button uses the colorful PNG brand icon',
);

// The icon asset really exists and is a truecolor(+alpha) PNG — i.e.
// colorful, unlike the single-gray SVG used elsewhere.
const png = fs.readFileSync(path.join(root, 'media', 'kiss-icon.png'));
assert.strictEqual(
  png.subarray(0, 8).toString('hex'),
  '89504e470d0a1a0a',
  'kiss-icon.png is a PNG',
);
// IHDR color type byte: 2 = truecolor, 6 = truecolor with alpha.
const colorType = png[25];
assert.ok(
  colorType === 2 || colorType === 6,
  `kiss-icon.png must be truecolor (got color type ${colorType})`,
);

console.log('editorTitleButtons.test.js: all assertions passed');
