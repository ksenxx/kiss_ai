// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The settings dialog is lifted off the page with a 1px theme hairline
// plus the --shadow-lg elevation token, in the base stylesheet (main.css);
// the remote page's palette (remote-codex.css) picks the hairline colour.
// The hairline colour comes from a theme variable (never a hard-coded
// colour) so each palette picks its own shade, and no later rule may
// repaint it.
//
// This suite is a deliberate cssRule exception to the rule that a test
// checks what the user sees: it reads the stylesheet text.  Every value
// it pins is built from custom properties, and jsdom's getComputedStyle
// does not resolve var(): `border: 1px solid var(--x)` computes to the
// initial `medium` black border and `box-shadow: var(--s)` comes back
// verbatim (measured with jsdom 29.1.1).  Only a real browser can
// measure the painted border, so this file pins the declarations the
// cascade applies instead.

const assert = require('assert');
const fs = require('fs');
const path = require('path');

// The last rule for a selector is the one the cascade applies, so a
// later same-selector override cannot hide behind an earlier match.
const {lastCssRule: cssRule} = require('./designTokens');

const MEDIA = path.join(__dirname, '..', 'media');
const MAIN_CSS = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
const REMOTE_CSS = fs.readFileSync(
  path.join(MEDIA, 'remote-codex.css'),
  'utf8',
);

function testMainCssUsesHairlineAndElevation() {
  const rule = cssRule(MAIN_CSS, '#settings-panel');
  assert.ok(
    /border:\s*1px solid var\(--settings-border, var\(--border\)\)/.test(rule),
    `the settings panel needs the 1px theme hairline — ` +
      `got: ${rule.trim()}`,
  );
  assert.ok(
    /box-shadow:\s*var\(--shadow-lg\)/.test(rule),
    `the settings panel needs the --shadow-lg elevation — got: ${rule.trim()}`,
  );
  // The palette is the first :root block; a later one inside
  // @media (prefers-reduced-motion) only zeroes the motion tokens.
  const root = MAIN_CSS.match(/:root\s*\{([^}]*)\}/)[1];
  assert.ok(
    /--border:\s*var\(--theme-panel-border\b/.test(root),
    '--border must be a theme colour derived from the theme palette',
  );
  console.log('PASS main.css gives the settings panel a hairline and elevation');
}

function testRemoteSurfaceUsesHairlineAndElevation() {
  // The remote page keeps main.css's hairline and elevation: its
  // palette, re-derived on body.remote-chat so the light theme's
  // variable overrides flow through, points the --settings-border alias
  // at the widget border (the computed border is measured in
  // src/kiss/tests/webapp/test_remote_stylesheet_cascade_sweep1030.py).
  const palette = cssRule(REMOTE_CSS, 'body.remote-chat');
  assert.ok(
    /--settings-border:\s*var\(--border-widget\)/.test(palette),
    'the remote palette must alias --settings-border to the widget border',
  );
  assert.ok(
    /--border-widget:\s*var\(--theme-widget-border\b/.test(palette),
    'the remote palette must derive --border-widget from --theme-widget-border',
  );
  console.log('PASS remote-codex.css keeps the hairline on remote');
}

function testNoLaterRuleOverridesTheBorder() {
  // No rule targeting the panel in either stylesheet may repaint any
  // part of its border (border, border-color, or a per-side shorthand)
  // away from the theme hairline.  Grouped selector lists are checked
  // per-selector, so a rule shared with #settings-panel-close is still
  // scanned when the list also names the panel itself.
  for (const [name, css] of [
    ['main.css', MAIN_CSS],
    ['remote-codex.css', REMOTE_CSS],
  ]) {
    const stripped = css.replace(/\/\*[\s\S]*?\*\//g, '');
    const re = /([^{}]+)\{([^{}]*)\}/g;
    let m;
    while ((m = re.exec(stripped)) !== null) {
      const targetsPanel = m[1]
        .split(',')
        .some((sel) => /#settings-panel(?![\w-])/.test(sel));
      if (!targetsPanel) continue;
      const sel = m[1].trim();
      const borderRe = /(border(?:-(?:left|right|top|bottom))?(?:-color)?)\s*:\s*([^;]+)/g;
      let d;
      while ((d = borderRe.exec(m[2])) !== null) {
        const value = d[2].trim();
        assert.ok(
          /var\(--settings-border/.test(value) || /^(none|0)$/.test(value),
          `${name} rule "${sel}" must not repaint the settings-panel ` +
            `border away from the theme hairline — got: ${d[1]}: ${value}`,
        );
      }
    }
  }
  console.log('PASS no later CSS rule overrides the hairline');
}

testMainCssUsesHairlineAndElevation();
testRemoteSurfaceUsesHairlineAndElevation();
testNoLaterRuleOverridesTheBorder();
console.log('All settingsPanelBorder tests passed');
