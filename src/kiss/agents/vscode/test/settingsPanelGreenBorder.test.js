// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The settings panel carries a green theme-color border on every
// surface: the VS Code sidebar and editor-panel webviews (styled by
// main.css) and the remote webapp in dark and light themes (restyled
// by remote-codex.css).  The green must come from the theme variable
// --green, never a hard-coded color, so each surface's palette (VS
// Code theme, remote dark, remote light) picks its own shade.

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const MEDIA = path.join(__dirname, '..', 'media');
const MAIN_CSS = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
const REMOTE_CSS = fs.readFileSync(
  path.join(MEDIA, 'remote-codex.css'),
  'utf8',
);

function cssRule(css, selector) {
  const source = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(source + String.raw`\s*(?:,[^{]*)?\{([^}]*)\}`, 'g');
  let body = null;
  let m;
  while ((m = re.exec(css)) !== null) body = m[1];
  assert.ok(body !== null, `CSS rule for ${selector} missing`);
  return body;
}

function testVsCodeSurfacesUseGreenThemeBorder() {
  const rule = cssRule(MAIN_CSS, '#settings-panel');
  assert.ok(
    /border:\s*2px solid var\(--green\)/.test(rule),
    `the sidebar/editor settings panel needs a green theme border — ` +
      `got: ${rule.trim()}`,
  );
  const root = cssRule(MAIN_CSS, ':root');
  assert.ok(
    /--green:\s*var\(--vscode-terminal-ansiGreen\)/.test(root),
    '--green must be a theme color derived from the VS Code palette',
  );
  console.log('PASS main.css gives the settings panel a green theme border');
}

function testRemoteSurfaceUsesGreenThemeBorder() {
  // The remote stylesheet may not use decorative tokens directly
  // (test_no_decorative_color_tokens_in_codex_css), so the border goes
  // through the --settings-border palette alias of --green.
  const rule = cssRule(REMOTE_CSS, 'body.remote-chat #settings-panel');
  assert.ok(
    /border:\s*2px solid var\(--settings-border\)/.test(rule),
    `the remote settings panel needs the green theme border — ` +
      `got: ${rule.trim()}`,
  );
  const palette = cssRule(REMOTE_CSS, 'body.remote-chat');
  assert.ok(
    /--settings-border:\s*var\(--green\)/.test(palette),
    'the remote palette must alias --settings-border to the theme green',
  );
  // The light theme re-points --green at body level so the border
  // re-themes with the sun/moon toggle.
  const light = cssRule(REMOTE_CSS, 'body.remote-chat.light-theme');
  assert.ok(
    /--green:\s*#107c10/.test(light),
    'the remote light theme must override --green for the border',
  );
  console.log('PASS remote-codex.css keeps the green border on remote');
}

function testNoLaterRuleOverridesTheGreenBorder() {
  // No rule targeting the panel in either stylesheet may repaint any
  // part of its border (border, border-color, or a per-side shorthand)
  // away from the theme green.  Grouped selector lists are checked
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
          /var\(--green\)|var\(--settings-border\)/.test(value) ||
            /^(none|0)$/.test(value),
          `${name} rule "${sel}" must not repaint the settings-panel ` +
            `border away from the theme green — got: ${d[1]}: ${value}`,
        );
      }
    }
  }
  console.log('PASS no later CSS rule overrides the green border');
}

testVsCodeSurfacesUseGreenThemeBorder();
testRemoteSurfaceUsesGreenThemeBorder();
testNoLaterRuleOverridesTheGreenBorder();
console.log('All settingsPanelGreenBorder tests passed');
