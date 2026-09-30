// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import {findKissProject} from './kissPaths';
import {ensureUserAssetFromDefault, kissHomeDir} from './userAssets';
import {readVersionPy} from './UpdateChecker';
import {BRAND, renderBrand} from './brand';

export const MY_INJECTION_DEFAULT_BODY =
  'Write end-to-end 100% coverage tests for the feature first.' +
  '  Then implement the feature.';

export const DEFAULT_MY_INJECTION =
  '## Trick\n\n' + MY_INJECTION_DEFAULT_BODY + '\n';

export function getVersion(): string {
  const kissRoot = findKissProject();
  if (!kissRoot) return '';
  return (
    readVersionPy(path.join(kissRoot, 'src', 'kiss', 'core', '_version.py')) ||
    ''
  );
}

function unescapeMarkdown(s: string): string {
  return s.replace(/\\([\\`*_{}[\]()#+\-.!<>|~"'$%&,/:;=?@^])/g, '$1');
}

function readMarkdownSections(markdownFile: string, heading: string): string[] {
  let text: string;
  try {
    text = fs.readFileSync(markdownFile, 'utf-8');
  } catch {
    return [];
  }
  const items: string[] = [];
  const sections = text.split(/^##\s+/m);
  for (let i = 1; i < sections.length; i++) {
    const section = sections[i];
    const newline = section.indexOf('\n');
    if (newline < 0) continue;
    const title = section.slice(0, newline).trim();
    if (title !== heading) continue;
    const body = unescapeMarkdown(section.slice(newline + 1).trim());
    if (body) items.push(body);
  }
  return items;
}

/**
 * The Inject promptlet list plus how many leading entries the user owns.
 *
 * `tricks` is ~/.kiss/MY_INJECTION.md's `## Trick` sections followed by
 * the bundled INJECTIONS.md ones; `userCount` is the length of the first
 * part, the rows the panel shows a delete button on.  Same shape as the
 * daemon's `tricksData` event.
 */
export function getTricksData(): {tricks: string[]; userCount: number} {
  const items: string[] = [];

  const myInjectionPath = ensureUserAssetFromDefault(
    'MY_INJECTION.md',
    DEFAULT_MY_INJECTION,
  );
  if (myInjectionPath !== null) {
    items.push(...readMarkdownSections(myInjectionPath, 'Trick'));
  }
  const userCount = items.length;

  const bundledOverride = process.env.KISS_INJECTIONS_PATH;
  let bundledPath: string | null = bundledOverride || null;
  if (!bundledPath) {
    const kissRoot = findKissProject();
    if (kissRoot) {
      bundledPath = path.join(kissRoot, 'src', 'kiss', 'INJECTIONS.md');
    }
  }
  if (bundledPath) {
    items.push(...readMarkdownSections(bundledPath, 'Trick'));
  }

  return {tricks: items, userCount};
}

/** The Inject promptlet list alone (see `getTricksData`). */
export function getTricks(): string[] {
  return getTricksData().tricks;
}

function parseTipSections(text: string): string[] {
  const tips: string[] = [];
  const sections = text.split(/^# Tip.*$/m);
  for (let i = 1; i < sections.length; i++) {
    const body = sections[i].trim();
    if (body) tips.push(body);
  }
  return tips;
}

export function getTips(): string[] {
  let tipsPath: string | null = process.env.KISS_TIPS_PATH || null;
  if (!tipsPath) {
    const kissRoot = findKissProject();
    if (kissRoot) tipsPath = path.join(kissRoot, 'src', 'kiss', 'TIPS.md');
  }
  if (!tipsPath) return [];
  let text: string;
  try {
    text = fs.readFileSync(tipsPath, 'utf-8');
  } catch {
    return [];
  }
  return parseTipSections(renderBrand(text));
}

/** The persisted "don't show tips again" flag, under `$KISS_HOME`. */
function tipsOptOutPath(): string {
  return path.join(kissHomeDir(), 'TIPS_DISABLED');
}

/** Whether the user opted out of the tips window ("Don't show again"). */
export function tipsDisabled(): boolean {
  return fs.existsSync(tipsOptOutPath());
}

/**
 * Persist the user's "don't show tips again" choice — the host side of
 * the webview's `{type: 'tipsOptOut'}` message (tips.js).  Idempotent;
 * an unwritable `$KISS_HOME` is ignored (the in-session tips window is
 * already closed, the choice simply is not remembered).
 */
export function recordTipsOptOut(): void {
  try {
    fs.mkdirSync(kissHomeDir(), {recursive: true});
    fs.writeFileSync(tipsOptOutPath(), new Date().toISOString() + '\n');
  } catch {
    // Nothing to do: see the docstring.
  }
}

/** Forget a persisted "don't show tips again" choice (tips.js unticks the box). */
export function clearTipsOptOut(): void {
  fs.rmSync(tipsOptOutPath(), {force: true});
}

/** `$KISS_HOME/TIPS_SHOWN-<version>`: the popup was opened for `version`. */
function tipsShownMarker(version: string): string {
  const safe = version.replace(/[^A-Za-z0-9.]/g, '_') || 'unknown';
  return path.join(kissHomeDir(), 'TIPS_SHOWN-' + safe);
}

/**
 * Claim the tips popup for the running version.  The popup opens once
 * per `$KISS_HOME` for every version: on the first run and again after
 * each update, so the user sees what changed.  A persisted opt-out
 * (recordTipsOptOut) keeps it closed for good.
 *
 * @returns true for the single caller that may open the popup.
 */
export function claimTipsPopup(): boolean {
  if (tipsDisabled()) return false;
  const marker = tipsShownMarker(getVersion());
  // audit0903-coverage:start
  try {
    fs.mkdirSync(path.dirname(marker), {recursive: true});
    // 'wx' claims the marker atomically.  An existsSync-then-write check
    // raced: two windows activating at once both passed the check before
    // either wrote, and the popup opened in both.  With 'wx' exactly one
    // writer wins; every other caller (a concurrent window, a later run,
    // or an unwritable ~/.kiss) lands in the catch and stays quiet.
    fs.writeFileSync(marker, new Date().toISOString() + '\n', {flag: 'wx'});
  } catch {
    return false;
  }
  // audit0903-coverage:end
  // The pre-2026.10 unversioned `TIPS_SHOWN` is spent: retire it.  The
  // claims of other versions stay, so two installations of different
  // versions sharing one home (KISS_PROJECT_PATH override, a bundled
  // copy) cannot erase each other's claim and reopen the popup.
  fs.rmSync(path.join(path.dirname(marker), 'TIPS_SHOWN'), {force: true});
  return true;
}

export function getNonce(): string {
  return crypto
    .randomBytes(24)
    .toString('base64')
    .replace(/[^A-Za-z0-9]/g, '')
    .slice(0, 32);
}

/**
 * Content hash of the packaged media asset *name*, the `?v=` value that
 * busts a webview's cache when the file changes under the same path.
 */
export function mediaAssetVersion(
  extensionUri: vscode.Uri,
  name: string,
): string {
  const file = vscode.Uri.joinPath(extensionUri, 'media', name).fsPath;
  const bytes = fs.readFileSync(file);
  return crypto.createHash('sha256').update(bytes).digest('hex').slice(0, 16);
}

/** Escape a string for interpolation into an HTML text position. */
function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Initial state of a chat hosted in an editor tab (editor-tabs mode).
 *
 * Travels into the webview as `data-kiss-*` attributes on `<body>`
 * (via the existing BODY_CLASS_ATTR substitution, so the daemon's
 * remote web app — which builds from the same chat.html — needs no new
 * placeholder): main.js adopts `tabId` as its single root chat tab's
 * id and, when `resumeChatId`/`resumeTaskId` are present, resumes that
 * history entry into the tab right after `ready`.
 */
export interface EditorTabInit {
  tabId: string;
  title?: string;
  resumeChatId?: string;
  resumeTaskId?: string;
  /**
   * Composer draft carried over from the panel that opened this one
   * (its + button / Cmd+T posts `openChatPanel` with the draft), so
   * the new chat's textarea starts out with the same text — parity
   * with the sidebar webview, whose createNewTab copies the draft
   * into the new internal tab.
   */
  pendingText?: string;
  /**
   * The tab is already in the daemon's registry (a panel materialized
   * from a `tabs_state` entry on mode switch-on), so the webview may
   * treat its disappearance from the first snapshot it sees as a close
   * by another client.
   */
  inRegistry?: boolean;
}

/**
 * The `<body>` attribute string for a chat webview hosted in an editor
 * tab: the `editor-tab-mode` class plus the tab's initial state as
 * `data-kiss-*` attributes (see EditorTabInit).
 *
 * @param init The panel's initial tab state.
 * @returns An attribute string starting with a space, ready to splice
 *     into `<body {{BODY_CLASS_ATTR}}>`.
 */
export function editorTabBodyAttrs(init: EditorTabInit): string {
  const attrs = [' class="editor-tab-mode"'];
  attrs.push(` data-kiss-tab-id="${escapeHtml(init.tabId)}"`);
  if (init.title) {
    attrs.push(` data-kiss-tab-title="${escapeHtml(init.title)}"`);
  }
  if (init.resumeChatId) {
    attrs.push(` data-kiss-resume-chat-id="${escapeHtml(init.resumeChatId)}"`);
  }
  if (init.resumeTaskId) {
    attrs.push(` data-kiss-resume-task-id="${escapeHtml(init.resumeTaskId)}"`);
  }
  if (init.pendingText) {
    attrs.push(` data-kiss-pending-text="${escapeHtml(init.pendingText)}"`);
  }
  if (init.inRegistry) {
    attrs.push(' data-kiss-in-registry="1"');
  }
  return attrs.join('');
}

/**
 * Root chat tab id of the primary-sidebar history panel's webview.
 *
 * The id is fixed (not random) so a reloaded window's history panel is
 * the same client as before; the tab itself never runs a task, never
 * binds to a chat and is never announced to the daemon's registry.
 */
export const HISTORY_PANEL_TAB_ID = 'history-panel';

/**
 * The `<body>` attribute string for the PRIMARY-sidebar history panel
 * (editor-tabs mode). The webview reuses the editor-tab chat surface —
 * so every history click already travels to the host as an
 * `openChatPanel` message — but `history-panel-mode` (main.js /
 * main.css) shows only the history sidebar, permanently open.
 *
 * @returns An attribute string ready for `<body {{BODY_CLASS_ATTR}}>`.
 */
export function historyPanelBodyAttrs(): string {
  return (
    ' class="editor-tab-mode history-panel-mode"' +
    ` data-kiss-tab-id="${HISTORY_PANEL_TAB_ID}"`
  );
}

/**
 * Root chat tab id of the secondary-sidebar Task Info panel's webview.
 *
 * Fixed like {@link HISTORY_PANEL_TAB_ID} and for the same reasons:
 * the tab never runs a task, never binds to a chat and is never
 * announced to the daemon's registry.
 */
export const META_PANEL_TAB_ID = 'meta-panel';

/**
 * The `<body>` attribute string for the SECONDARY-sidebar Task Info
 * panel (editor-tabs mode). The webview reuses the chat surface but
 * `meta-panel-mode` (main.js / main.css) shows only the task-info
 * panel (#meta-panel) — the remote webapp's rightmost desktop panel —
 * which renders the `metaState` relays of the active chat editor tab.
 *
 * @returns An attribute string ready for `<body {{BODY_CLASS_ATTR}}>`.
 */
export function metaPanelBodyAttrs(): string {
  return (
    ' class="editor-tab-mode meta-panel-mode"' +
    ` data-kiss-tab-id="${META_PANEL_TAB_ID}"`
  );
}

/**
 * Public URL of the browser wake-word model archive.
 *
 * Documented twin of ``VOICE_MODEL_URL`` in
 * ``kiss/server/web_server.py`` (which proxies the same archive to the
 * remote webapp as ``/voice-model.tar.gz``). The webview's in-page
 * voice pipeline fetches it directly — inside its blob Worker — because
 * a webview cannot reach the daemon's HTTPS port, and the browser's
 * HTTP cache keeps repeat downloads cheap.
 */
export const VOICE_MODEL_URL =
  'https://ccoreilly.github.io/vosk-browser/models/' +
  'vosk-model-small-en-us-0.15.tar.gz';

/** Origin of {@link VOICE_MODEL_URL}, for the webview CSP connect-src. */
function voiceModelOrigin(): string {
  return new URL(VOICE_MODEL_URL).origin;
}

export function buildChatHtml(
  webview: vscode.Webview,
  extensionUri: vscode.Uri,
  selectedModel: string,
  bodyAttrs?: string,
): string {
  const nonce = getNonce();
  const version = getVersion();
  const tricksData = getTricksData();
  const tricksJson = JSON.stringify(tricksData.tricks).replace(/<\//g, '<\\/');
  const tips = getTips();
  const tipsJson = JSON.stringify({
    tips,
    show: tips.length > 0 && claimTipsPopup(),
    version,
  }).replace(/<\//g, '<\\/');
  const mod = process.platform === 'darwin' ? '⌘' : 'Ctrl+';

  const tplPath = vscode.Uri.joinPath(
    extensionUri,
    'media',
    'chat.html',
  ).fsPath;
  const tpl = fs.readFileSync(tplPath, 'utf-8');

  const u = (name: string): string => {
    const uri = webview.asWebviewUri(
      vscode.Uri.joinPath(extensionUri, 'media', name),
    );
    const sep = uri.toString().includes('?') ? '&' : '?';
    return uri.toString() + sep + 'v=' + mediaAssetVersion(extensionUri, name);
  };

  /* eslint-disable quotes */
  // 'wasm-unsafe-eval', `worker-src blob:` and the model-origin
  // connect-src exist for the in-page voice pipeline (voice.js browser
  // fallback): vosk.js spawns its recognizer as a blob Worker that
  // fetches the wake-word model archive and runs a Kaldi WASM build.
  const csp =
    `<meta http-equiv="Content-Security-Policy" content="default-src 'none';` +
    ` style-src ${webview.cspSource} 'unsafe-inline';` +
    ` script-src 'nonce-${nonce}' 'wasm-unsafe-eval';` +
    ` worker-src blob:;` +
    ` connect-src ${voiceModelOrigin()};` +
    ` img-src ${webview.cspSource} data: https:;` +
    ` font-src ${webview.cspSource};` +
    ` media-src data: ${webview.cspSource};` +
    ` form-action 'none'; frame-src 'none'; object-src 'none'; base-uri 'none';">`;

  const placeholder = `Ask anything... (@ for files, ${mod}T new chat)`;

  const subs: Record<string, string> = {
    VIEWPORT: 'width=device-width, initial-scale=1.0',
    CSP_META: csp,
    // The webview loads its assets from the extension's own files:
    // the remote page's asset-load reload guard has nothing to do here.
    HEAD_SCRIPT: '',
    STYLE_HREF: u('main.css'),
    BRAND_STYLE_HREF: u('brand.css'),
    WELCOME_LOGO_SRC: u('welcome-logo.png'),
    WELCOME_LOGO_DARK_SRC: u('welcome-logo-dark.png'),
    // The dark sheet is the initial one; main.js (followVscodeTheme)
    // swaps in the light sheet whenever the editor theme is light.
    HLJS_CSS_HREF: u('highlight-vscode-dark.css'),
    HEAD_STYLE: '',
    BODY_CLASS_ATTR: bodyAttrs || '',
    PRODUCT_NAME: escapeHtml(BRAND.productName),
    TAGLINE: escapeHtml(BRAND.tagline),
    BRAND_JSON: JSON.stringify({
      productName: BRAND.productName,
      shortName: BRAND.shortName,
    }).replace(/<\//g, '<\\/'),
    INPUT_PLACEHOLDER: placeholder,
    ENTERKEYHINT: '',
    // The model name can come from user settings or the daemon; escape it
    // so a crafted value cannot inject markup into the privileged webview.
    MODEL_NAME: escapeHtml(selectedModel),
    VERSION_SUFFIX: version ? ' ' + version : '',
    AUTH_MODAL: '',
    NONCE_ATTR: ` nonce="${nonce}"`,
    HLJS_SRC: u('highlight.min.js'),
    MARKED_SRC: u('marked.min.js'),
    API_SRC: u('api.js'),
    PANEL_COPY_SRC: u('panelCopy.js'),
    CTX_MENU_SRC: u('contentContextMenu.js'),
    TREE_MENU_SRC: u('treeContextMenu.js'),
    BROWSER_TAB_SRC: u('browserTab.js'),
    PDF_VIEW_SRC: u('pdfView.js'),
    MAIN_SRC: u('main.js'),
    SHIM_SCRIPT:
      `<script nonce="${nonce}">window.__HLJS_THEME_CSS__ = ` +
      JSON.stringify({
        dark: u('highlight-vscode-dark.css'),
        light: u('highlight-vscode-light.css'),
      }).replace(/<\//g, '<\\/') +
      ';</script>',
    TRICKS_JSON: tricksJson,
    MY_TRICKS_COUNT: String(tricksData.userCount),
    TIPS_JSON: tipsJson,
    TIPS_SRC: u('tips.js'),
    VOICE_SRC: u('voice.js'),
    // voskSrc/modelUrl/nonce power the in-page capture fallback: when
    // the machine hosting this extension has no microphone, voice.js
    // records with the BROWSER's mic (embedder permitting) exactly like
    // the remote webapp, instead of erroring. The nonce lets voice.js
    // inject the vosk.js script tag under this page's CSP.
    VOICE_CONFIG: JSON.stringify({
      mode: 'webview',
      ackAudioUrl: u('working-on-it.mp3'),
      voskSrc: u('vosk.js'),
      modelUrl: VOICE_MODEL_URL,
      nonce,
    }),
  };

  return substituteTemplate(tpl, subs);
}

/**
 * Placeholders whose values are whole attribute strings that carry their
 * own leading space (or are empty): `' class="remote-chat"'`,
 * `' nonce="…"'`, `' enterkeyhint="send"'`.
 *
 * The template writes them after a separating space
 * (`<body {{BODY_CLASS_ATTR}}>`, `<script {{NONCE_ATTR}} src=…>`) so
 * htmlhint can parse the tags; {@link substituteTemplate} drops that
 * template space for these keys, so the rendered markup is exactly
 * `<body class="remote-chat">` / `<body>` / `<script src=…>`.
 */
const ATTR_STRING_KEYS: ReadonlySet<string> = new Set([
  'BODY_CLASS_ATTR',
  'ENTERKEYHINT',
  'NONCE_ATTR',
]);

/**
 * Fill every `{{KEY}}` of `tpl` from `subs`, leaving unknown keys
 * untouched. Mirrors `_build_html` in `kiss/server/web_server.py`, which
 * renders the same `media/chat.html` for the remote web app.
 */
export function substituteTemplate(
  tpl: string,
  subs: Record<string, string>,
): string {
  return tpl.replace(
    /( ?)\{\{([A-Z_]+)\}\}/g,
    (match, space: string, key: string) => {
      if (!Object.prototype.hasOwnProperty.call(subs, key)) return match;
      return ATTR_STRING_KEYS.has(key) ? subs[key] : space + subs[key];
    },
  );
}
