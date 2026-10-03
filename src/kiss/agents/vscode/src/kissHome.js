// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The extension host's state directory, the twin of
// `kiss.core.config.kiss_home()`: `$KISS_HOME` when set, else
// `~/<home_dir>` with `home_dir` from `media/brand.json` (`.kiss` for stock
// KISS Sorcar; a white-label brand names its own directory so it never
// shares state with a stock install).  Plain CJS because UpdateChecker.js,
// which the tests require() straight from src/, needs it too.

const fs = require('fs');
const os = require('os');
const path = require('path');

const DEFAULT_HOME_DIR_NAME = '.kiss';

/** `media/brand.json`, resolved from `out/` and from `src/` alike. */
const BRAND_FILE = path.join(__dirname, '..', 'media', 'brand.json');

/**
 * Return the brand's state directory name under `$HOME` from `file`
 * (`home_dir` in brand.json), or `.kiss` when the file is missing, is not
 * JSON, or names anything but a single path component.
 */
function brandHomeDirName(file = BRAND_FILE) {
  let name;
  try {
    name = JSON.parse(fs.readFileSync(file, 'utf-8')).home_dir;
  } catch {
    return DEFAULT_HOME_DIR_NAME;
  }
  const isDirName =
    typeof name === 'string' &&
    name !== '' &&
    name !== '.' &&
    name !== '..' &&
    !name.includes('/') &&
    !name.includes('\\');
  return isDirName ? name : DEFAULT_HOME_DIR_NAME;
}

const HOME_DIR_NAME = brandHomeDirName();

/** `$KISS_HOME`, else `~/<home_dir>` of the brand (`~/.kiss` for stock KISS). */
function kissHomeDir() {
  return process.env.KISS_HOME || path.join(os.homedir(), HOME_DIR_NAME);
}

module.exports = {
  BRAND_FILE,
  DEFAULT_HOME_DIR_NAME,
  HOME_DIR_NAME,
  brandHomeDirName,
  kissHomeDir,
};
