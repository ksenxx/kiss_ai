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

const MEDIA = path.join(__dirname, '..', 'media');
const MAIN_CSS = path.join(MEDIA, 'main.css');

// Names of the design tokens (see the "Design tokens" block in main.css).
const TOKEN_NAME =
  /^--(font|fs|space|radius|shadow|scrim|z|dur|ease|status|favorite|on-accent|paper|ink|attention|panel-tint|panel-tint-solid|panel-line|accent-tint|accent-tint-solid|accent-line)\b/;

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

/**
 * Append media/`name` to `win`'s <head> as a <style> with its design
 * tokens inlined, so computed styles carry the concrete values.
 */
function installTokenisedCss(win, name) {
  const style = win.document.createElement('style');
  style.textContent = inlineDesignTokens(
    fs.readFileSync(path.join(MEDIA, name), 'utf8'),
  );
  win.document.head.appendChild(style);
  return style;
}

module.exports = {inlineDesignTokens, installTokenisedCss};
