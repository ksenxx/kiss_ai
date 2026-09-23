#!/usr/bin/env node
// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/**
 * Rewrite the user-visible strings of the extension manifest from
 * `media/brand.json`.
 *
 * `package.json` is static: VS Code reads the display name, the
 * activity-bar titles and the command titles straight from it, so unlike
 * the runtime code (which reads brand.json through `src/brand.ts` and
 * `kiss.core.brand`) the manifest has to be rewritten on disk.  `copy-kiss.sh`
 * runs this before every VSIX build, right after syncing the version.
 *
 * The checked-in manifest carries the stock KISS Sorcar strings.  This
 * script maps those stock tokens to the brand: "KISS Sorcar" becomes
 * `product_name`, the "KISS: " command prefix becomes `short_name` + ": ",
 * and `description` becomes `extension_description`.  With the stock
 * brand.json every field maps to itself, so the manifest is left
 * byte-for-byte unchanged; with a custom brand a second run is a no-op
 * because no stock token remains.  Command ids, view ids and setting keys
 * are never touched.
 *
 * Usage: node scripts/apply-brand.js [path/to/brand.json] [path/to/package.json]
 */

const fs = require('fs');
const path = require('path');

const EXT_ROOT = path.resolve(__dirname, '..');
const STOCK = {product_name: 'KISS Sorcar', short_name: 'KISS'};

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf-8'));
}

/** Replace the stock product tokens inside one display string. */
function brandText(text, brand) {
  return text
    .replace(/KISS Sorcar/g, brand.product_name)
    .replace(/^KISS: /, `${brand.short_name}: `);
}

/**
 * Return a copy of the manifest with its display strings re-branded.
 *
 * @param {object} manifest Parsed package.json.
 * @param {object} brand Parsed brand.json (missing keys keep the stock values).
 */
function brandManifest(manifest, brand) {
  const b = {
    product_name: brand.product_name || STOCK.product_name,
    short_name: brand.short_name || STOCK.short_name,
    extension_description: brand.extension_description || manifest.description,
  };
  const out = JSON.parse(JSON.stringify(manifest));
  out.displayName = brandText(out.displayName, b);
  out.description = b.extension_description;
  const untrusted = out.capabilities && out.capabilities.untrustedWorkspaces;
  if (untrusted && untrusted.description) {
    untrusted.description = brandText(untrusted.description, b);
  }
  const contributes = out.contributes || {};
  for (const containers of Object.values(contributes.viewsContainers || {})) {
    for (const container of containers) container.title = brandText(container.title, b);
  }
  for (const command of contributes.commands || []) {
    command.title = brandText(command.title, b);
  }
  const configuration = contributes.configuration;
  if (configuration) {
    if (configuration.title) configuration.title = brandText(configuration.title, b);
    for (const prop of Object.values(configuration.properties || {})) {
      if (prop.description) prop.description = brandText(prop.description, b);
    }
  }
  return out;
}

function main(argv) {
  const brandFile = argv[0] || path.join(EXT_ROOT, 'media', 'brand.json');
  const manifestFile = argv[1] || path.join(EXT_ROOT, 'package.json');
  const branded = brandManifest(readJson(manifestFile), readJson(brandFile));
  const text = JSON.stringify(branded, null, 2) + '\n';
  if (fs.readFileSync(manifestFile, 'utf-8') !== text) {
    fs.writeFileSync(manifestFile, text);
    console.log(`Branded ${manifestFile} as "${branded.displayName}"`);
  }
}

module.exports = {brandManifest, brandText};

if (require.main === module) main(process.argv.slice(2));
