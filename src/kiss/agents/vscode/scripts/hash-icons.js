// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/**
 * Give the extension's icons content-hashed file names for the duration of
 * a VSIX build.
 *
 * The VS Code server serves everything below its extensions directory with
 * `Cache-Control: public, max-age=31536000`, and the resource URL is just
 * the file path inside `<publisher>.<name>-<version>/`.  A rebuilt extension
 * that keeps its version but changes an icon (a brand overlay applied by
 * install.sh, `--revert`, a developer editing the icon) therefore keeps
 * showing the old icon in the activity bar, the editor tabs and the
 * Extensions view for up to a year.  Content-hashed names give a changed
 * icon a new URL, so the client fetches it again.
 *
 * `applyHashedIcons(root)` copies every image directly under `media/` to
 * `media/hashed/<name>-<md5[:8]>.<ext>`, writes the plain-to-hashed map to
 * `media/hashed/index.json` (read at runtime by `mediaIconPath` in
 * src/brand.ts), rewrites `package.json` to point at the copies and returns
 * a function that restores the original manifest and deletes `media/hashed/`.
 * `scripts/package-vsix.js` wraps `vsce pack` in it; the tracked manifest
 * keeps the plain names, so nothing in git changes and the extension still
 * runs from source.  `media/hashed/` is git-ignored.
 */

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const IMAGE_EXT = /\.(svg|png|jpe?g|gif|webp)$/i;
/** A hashed reference left in the manifest by a killed build, e.g. `media/hashed/kiss-icon-1a2b3c4d.svg`. */
const HASHED_REF = /^media\/hashed\/(.+)-[0-9a-f]{8}(\.[^./]+)$/i;
const HASHED_DIR = path.join('media', 'hashed');
const INDEX_FILE = path.join(HASHED_DIR, 'index.json');

/**
 * Map every image directly under `<root>/media` from its plain
 * extension-relative path to its content-hashed path.
 *
 * @param {string} root Extension root.
 * @returns {Record<string, string>} e.g. `{"media/kiss-icon.svg": "media/hashed/kiss-icon-63131785.svg"}`.
 */
function hashedIconIndex(root) {
  const index = {};
  const mediaDir = path.join(root, 'media');
  if (!fs.existsSync(mediaDir)) return index;
  for (const name of fs.readdirSync(mediaDir).sort()) {
    const file = path.join(mediaDir, name);
    if (!IMAGE_EXT.test(name) || !fs.statSync(file).isFile()) continue;
    const digest = crypto
      .createHash('md5')
      .update(fs.readFileSync(file))
      .digest('hex');
    const ext = path.extname(name);
    const base = name.slice(0, -ext.length);
    index[`media/${name}`] = `media/hashed/${base}-${digest.slice(0, 8)}${ext}`;
  }
  return index;
}

/** Plain extension-relative name behind a hashed reference, or `value` itself. */
function plainIconRef(value) {
  const m = HASHED_REF.exec(value);
  return m ? `media/${m[1]}${m[2]}` : value;
}

/** Copy of `node` with `fn` applied to every string value. */
function mapStrings(node, fn) {
  if (typeof node === 'string') return fn(node);
  if (Array.isArray(node)) return node.map(child => mapStrings(child, fn));
  if (node && typeof node === 'object') {
    const out = {};
    for (const [key, child] of Object.entries(node)) {
      out[key] = mapStrings(child, fn);
    }
    return out;
  }
  return node;
}

/**
 * Return a copy of `manifest` whose icon references point at the hashed
 * copies listed in `index`.  A stale hashed reference (from a build that was
 * killed before it could restore the manifest) is resolved through its plain
 * name, so the rewrite is idempotent.  Other strings are left untouched.
 *
 * @param {object} manifest Parsed package.json.
 * @param {Record<string, string>} index Result of `hashedIconIndex`.
 */
function hashIconManifest(manifest, index) {
  return mapStrings(manifest, value => index[plainIconRef(value)] || value);
}

/**
 * Create `<root>/media/hashed/` (copies + index.json) and rewrite
 * `<root>/package.json` to reference the hashed icons.  Returns a function
 * that writes the plain manifest back and removes `media/hashed/`.
 *
 * @param {string} root Extension root.
 */
function applyHashedIcons(root) {
  const manifestFile = path.join(root, 'package.json');
  const hashedDir = path.join(root, HASHED_DIR);
  const original = fs.readFileSync(manifestFile, 'utf-8');
  const manifest = JSON.parse(original);
  // Byte-for-byte restore normally; a manifest a killed build left hashed is
  // restored with plain names, never with references into the deleted dir.
  const plain = mapStrings(manifest, plainIconRef);
  const restoreText =
    JSON.stringify(plain) === JSON.stringify(manifest)
      ? original
      : JSON.stringify(plain, null, 2) + '\n';
  fs.rmSync(hashedDir, {recursive: true, force: true});
  const index = hashedIconIndex(root);
  fs.mkdirSync(hashedDir, {recursive: true});
  for (const [plainRef, target] of Object.entries(index)) {
    fs.copyFileSync(path.join(root, plainRef), path.join(root, target));
  }
  fs.writeFileSync(
    path.join(root, INDEX_FILE),
    JSON.stringify(index, null, 2) + '\n',
  );
  const hashed = hashIconManifest(manifest, index);
  fs.writeFileSync(manifestFile, JSON.stringify(hashed, null, 2) + '\n');
  return function restoreManifest() {
    fs.writeFileSync(manifestFile, restoreText);
    fs.rmSync(hashedDir, {recursive: true, force: true});
  };
}

module.exports = {
  applyHashedIcons,
  hashIconManifest,
  hashedIconIndex,
  HASHED_DIR,
  INDEX_FILE,
};
