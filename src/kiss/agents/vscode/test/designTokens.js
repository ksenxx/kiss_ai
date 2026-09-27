// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// jsdom does not resolve var(): a declaration such as
// `padding: var(--space-0-5) 0` or `color: var(--status-ok)` comes back
// from getComputedStyle unresolved (or empty for shorthands).  Suites
// that read computed font sizes, spacing, radii, z-index or status colours from
// media/main.css load the stylesheet through inlineDesignTokens(), which
// substitutes the design-token values defined in main.css's :root, so
// the assertions keep checking the concrete values a browser applies.

/* global require, module */

'use strict';

const fs = require('fs');
const path = require('path');

const MAIN_CSS = path.join(__dirname, '..', 'media', 'main.css');

// Names of the design tokens (see the "Design tokens" block in main.css).
const TOKEN_NAME =
  /^--(fs|space|radius|shadow|scrim|z|dur|ease|status|favorite|on-accent|paper|ink|attention|panel-tint|panel-tint-solid|panel-line|accent-tint|accent-tint-solid|accent-line)\b/;

/**
 * Return {name: value} for every design token in main.css's first :root.
 */
function designTokens() {
  const css = fs.readFileSync(MAIN_CSS, 'utf8');
  const root = css
    .match(/:root\s*\{([^}]*)\}/)[1]
    .replace(/\/\*[\s\S]*?\*\//g, '');
  const tokens = {};
  for (const m of root.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) {
    if (TOKEN_NAME.test(m[1])) tokens[m[1]] = m[2].trim();
  }
  return tokens;
}

/**
 * Replace every `var(--token)` in `css` with the token's value, repeating
 * until tokens defined in terms of other tokens are fully expanded.
 */
function inlineDesignTokens(css) {
  const tokens = designTokens();
  let prev;
  do {
    prev = css;
    css = css.replace(/var\((--[\w-]+)\)/g, (m, name) =>
      name in tokens ? tokens[name] : m,
    );
  } while (css !== prev);
  return css;
}

module.exports = {inlineDesignTokens};
