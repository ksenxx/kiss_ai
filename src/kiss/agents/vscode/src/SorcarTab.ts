// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * Shared utilities for Sorcar webview HTML rendering.
 * Used by SorcarSidebarView.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import {findKissProject} from './kissPaths';
import {ensureUserAssetFromDefault, kissHomeDir} from './userAssets';

/** Default ``## Trick`` body auto-seeded into ``~/.kiss/MY_INJECTION.md``. */
export const MY_INJECTION_DEFAULT_BODY =
  'Write end-to-end 100% coverage tests for the feature first.' +
  '  Then implement the feature.';

/** Full default file content for ``~/.kiss/MY_INJECTION.md``. */
export const DEFAULT_MY_INJECTION =
  '## Trick\n\n' + MY_INJECTION_DEFAULT_BODY + '\n';

/** Read the KISS project version from ``_version.py`` on disk. */
export function getVersion(): string {
  try {
    const kissRoot = findKissProject();
    if (kissRoot) {
      const versionFile = path.join(kissRoot, 'src', 'kiss', '_version.py');
      const content = fs.readFileSync(versionFile, 'utf-8');
      const match = content.match(/__version__\s*=\s*["']([^"']+)["']/);
      if (match) return match[1];
    }
  } catch {
    /* ignore */
  }
  return '';
}

/**
 * Strip CommonMark backslash escapes so a chip/dropdown entry displays
 * the text the author typed.  ``mdformat`` automatically inserts ``\``
 * before any character that could otherwise be parsed as Markdown
 * syntax (e.g. ``<<x>>`` becomes ``\<<x>>``).  Both INJECTIONS.md and
 * SAMPLE_TASKS.md are passed through mdformat by ``uv run check``, so
 * the loaders must reverse those escapes before the strings reach the
 * webview (which only HTML-escapes them).  Only the ASCII-punctuation
 * set that CommonMark §2.4 explicitly allows to be escaped is
 * unescaped; any other ``\X`` sequence is preserved.
 */
function unescapeMarkdown(s: string): string {
  return s.replace(/\\([\\`*_{}[\]()#+\-.!<>|~"'$%&,/:;=?@^])/g, '$1');
}

/**
 * Read ``markdownFile`` and return the body of every ``## <heading>``
 * section whose heading is ``heading``.  Bodies are trimmed and
 * backslash-unescaped (so mdformat-produced ``\<<x>>`` reverts to
 * ``<<x>>``).  Empty bodies are skipped.  Returns ``[]`` if the file
 * is missing or unreadable.
 */
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
 * Build the "Inject instruction" trick list from two Markdown files.
 *
 * Order matters — user-curated tricks come first, bundled tricks
 * second:
 *
 *   1. ``~/.kiss/MY_INJECTION.md`` — purely user-curated tricks.
 *      Auto-created on first read with the seed content
 *      ``## Trick\n\nWrite end-to-end 100% coverage tests for the
 *      feature first.  Then implement the feature.\n`` so the file is
 *      always present and editable.  Never overwritten once it
 *      exists; user edits survive across upgrades and daemon
 *      restarts.
 *
 *   2. ``<kissRoot>/src/kiss/INJECTIONS.md`` — the bundled tricks
 *      shipped with the extension.  Read **directly from the package
 *      copy**; never copied into ``~/.kiss/``.  This way every
 *      extension upgrade automatically delivers the latest bundled
 *      tricks without clobbering the user's curated list.
 *
 * Each ``## Trick`` section yields one string entry, with the body
 * trimmed and mdformat backslash escapes reverted.  Returns ``[]``
 * when no section is found in either file so the Tricks button still
 * renders, just with an empty list.
 *
 * The bundled file path honours the ``KISS_INJECTIONS_PATH``
 * environment variable (used by the test suite to pin a known set of
 * bundled tricks), matching the Python helper
 * :func:`kiss.agents.vscode.tricks._bundled_injections_path`.
 */
export function getTricks(): string[] {
  const items: string[] = [];

  // (1) User-curated tricks from ~/.kiss/MY_INJECTION.md.
  const myInjectionPath = ensureUserAssetFromDefault(
    'MY_INJECTION.md',
    DEFAULT_MY_INJECTION,
  );
  if (myInjectionPath !== null) {
    items.push(...readMarkdownSections(myInjectionPath, 'Trick'));
  }

  // (2) Bundled tricks read directly from the package.  No
  //     ``~/.kiss/INJECTIONS.md`` is ever created — the bundled file
  //     is the only source of these entries.
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

  return items;
}

/**
 * Parse the bundled ``TIPS.md`` text into a list of markdown tips.
 *
 * Every line starting with ``# Tip`` begins a new tip; the tip body is
 * the markdown text up to the next such line (or EOF), trimmed.  Text
 * before the first ``# Tip`` line and tips with empty bodies are
 * skipped.  Bodies are NOT backslash-unescaped — they are rendered as
 * markdown by ``window.marked`` in the webview, which handles escapes.
 */
function parseTipSections(text: string): string[] {
  const tips: string[] = [];
  const sections = text.split(/^# Tip.*$/m);
  for (let i = 1; i < sections.length; i++) {
    const body = sections[i].trim();
    if (body) tips.push(body);
  }
  return tips;
}

/**
 * Read the bundled ``src/kiss/TIPS.md`` and return one markdown string
 * per tip (see :func:`parseTipSections`).
 *
 * The file path honours the ``KISS_TIPS_PATH`` environment variable
 * (used by the test suite to pin deterministic tips), matching the
 * Python helper ``kiss.agents.vscode.tips.read_tips``.  Returns ``[]``
 * when the file is missing or unreadable so the chat webview still
 * renders without a tips window.
 */
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
  return parseTipSections(text);
}

/**
 * Return ``true`` exactly once per installation — on the first call
 * after a fresh install — and persist that fact.
 *
 * The persistent marker is ``~/.kiss/TIPS_SHOWN`` (honouring
 * ``KISS_HOME``); ``~/.kiss/`` survives extension upgrades, so the
 * tips window only auto-opens after a genuinely fresh installation.
 * When the marker cannot be written (read-only FS, missing HOME) the
 * function returns ``false`` so the tips window can never re-appear
 * on every reload.
 */
export function consumeTipsFirstRun(): boolean {
  const marker = path.join(kissHomeDir(), 'TIPS_SHOWN');
  try {
    if (fs.existsSync(marker)) return false;
    fs.mkdirSync(path.dirname(marker), {recursive: true});
    fs.writeFileSync(marker, new Date().toISOString() + '\n');
    return true;
  } catch {
    return false;
  }
}

/**
 * Re-arm the Tips window after an extension rebuild/reinstall.
 *
 * ``install.sh``, ``scripts/build-extension.sh`` and
 * ``scripts/release.sh`` all write ``~/.kiss/.extension-updated`` as
 * their final step (right after restarting the kiss-web daemon), which
 * makes every VS Code window reload.  When that marker is present at
 * activation time the extension has just been (re)installed, so the
 * ``TIPS_SHOWN`` marker is removed here; the next chat webview render
 * then auto-opens the Tips window exactly once again (via
 * ``consumeTipsFirstRun()``).  Without this reset the Tips window
 * would only ever auto-open once per machine — never after a rebuild.
 *
 * Called from ``activate()`` before the chat webview is created.
 * Errors are swallowed: a read-only filesystem must never break
 * activation.
 */
export function resetTipsOnExtensionUpdate(): void {
  const home = kissHomeDir();
  try {
    if (fs.existsSync(path.join(home, '.extension-updated'))) {
      fs.rmSync(path.join(home, 'TIPS_SHOWN'), {force: true});
    }
  } catch {
    // ignore — never break activation over the tips marker
  }
}

/**
 * Build the welcome-screen chip list from two Markdown files.
 *
 * Order matters — user-curated chips come first, bundled chips
 * second:
 *
 *   1. ``~/.kiss/MY_TASK_TEMPLATES.md`` — purely user-curated tasks.
 *      Auto-created on first read with the seed content
 *      ``## Task\n\nHi!\n`` so the file is always present and
 *      editable.  Never overwritten once it exists; user edits
 *      survive across upgrades and daemon restarts.
 *
 *   2. ``<extensionRoot>/kiss_project/src/kiss/SAMPLE_TASKS.md`` (or,
 *      in dev checkouts, ``<extensionRoot>/../../SAMPLE_TASKS.md``)
 *      — the bundled sample tasks shipped with the extension.  Read
 *      **directly from the package copy**; never copied into
 *      ``~/.kiss/``.  This way every extension upgrade automatically
 *      delivers the latest bundled chips without clobbering the
 *      user's curated list.
 *
 * Each ``## Task`` section yields one ``{text}`` entry, with the body
 * trimmed and mdformat backslash escapes reverted.  Returns ``[]``
 * when neither file contributes any sections so the welcome screen
 * still renders without chips.
 */
export function readSampleTasks(extensionRoot: string): Array<{text: string}> {
  const items: Array<{text: string}> = [];

  // (1) User-curated chips from ~/.kiss/MY_TASK_TEMPLATES.md.
  const myTasksPath = ensureUserAssetFromDefault(
    'MY_TASK_TEMPLATES.md',
    '## Task\n\nHi!\n',
  );
  if (myTasksPath !== null) {
    for (const text of readMarkdownSections(myTasksPath, 'Task')) {
      items.push({text});
    }
  }

  // (2) Bundled chips read directly from the package copy.  No
  //     ``~/.kiss/SAMPLE_TASKS.md`` is ever created here — the
  //     bundled file is the only source of these chips.
  const packagePath = path.join(
    extensionRoot,
    'kiss_project',
    'src',
    'kiss',
    'SAMPLE_TASKS.md',
  );
  const sourcePath = path.join(extensionRoot, '..', '..', 'SAMPLE_TASKS.md');
  const bundledPath = fs.existsSync(packagePath) ? packagePath : sourcePath;
  for (const text of readMarkdownSections(bundledPath, 'Task')) {
    items.push({text});
  }

  return items;
}

/**
 * Generate a cryptographically random nonce string for Content Security
 * Policy.  Uses Node's CSPRNG (``crypto.randomBytes``) — never
 * ``Math.random``, which is predictable.
 */
export function getNonce(): string {
  // 24 random bytes -> 32 base64url chars; restrict to [A-Za-z0-9] for
  // CSP-safe nonce values.
  return crypto
    .randomBytes(24)
    .toString('base64')
    .replace(/[^A-Za-z0-9]/g, '')
    .slice(0, 32);
}

/** Return a short content hash for a packaged media asset. */
function mediaAssetVersion(extensionUri: vscode.Uri, name: string): string {
  const file = vscode.Uri.joinPath(extensionUri, 'media', name).fsPath;
  const bytes = fs.readFileSync(file);
  return crypto.createHash('sha256').update(bytes).digest('hex').slice(0, 16);
}

/**
 * Build the full chat HTML for a Sorcar webview.
 *
 * The HTML body is loaded from ``media/chat.html`` — the same file the
 * remote web server uses — so the extension and the remote webapp can
 * never drift in markup or script ordering.  Only mode-specific values
 * (CSP, nonces, webview URIs, input placeholder with platform modifier
 * keys) are substituted in here.
 */
export function buildChatHtml(
  webview: vscode.Webview,
  extensionUri: vscode.Uri,
  selectedModel: string,
): string {
  const nonce = getNonce();
  const version = getVersion();
  // ``</`` must never appear raw inside an inline <script> block — a
  // user trick body containing ``</script>`` would otherwise terminate
  // the ``window.__TRICKS__`` script per the HTML spec and break the
  // whole chat webview.  Same escape as ``tipsJson`` below; ``<\/`` is
  // a valid JSON string escape so the payload round-trips unchanged.
  const tricksJson = JSON.stringify(getTricks()).replace(/<\//g, '<\\/');
  const tips = getTips();
  // ``</`` must never appear raw inside the inline <script> block —
  // a tip body containing ``</script>`` would otherwise terminate it.
  // Only consume the first-run marker when there is at least one tip to
  // show; if a malformed package ever ships without TIPS.md, a later
  // fixed package can still show the tips once.
  const tipsJson = JSON.stringify({
    tips,
    show: tips.length > 0 && consumeTipsFirstRun(),
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
  const csp =
    `<meta http-equiv="Content-Security-Policy" content="default-src 'none';` +
    ` style-src ${webview.cspSource} 'unsafe-inline';` +
    ` script-src 'nonce-${nonce}';` +
    ` img-src ${webview.cspSource} data: https:;` +
    ` font-src ${webview.cspSource};` +
    // media-src data: lets the webview play the GPT-synthesized talk
    // audio (a base64 MP3 data: URI shipped inside the 'talk' event by
    // speech_synthesis.py); without it the talk would stay silent
    // (the robotic Web Speech fallback is gone).  The webview resource
    // origin is also allowed so voice.js can play the bundled
    // "Working on it" ack clip (media/working-on-it.mp3).
    ` media-src data: ${webview.cspSource};` +
    ` form-action 'none'; frame-src 'none'; object-src 'none'; base-uri 'none';">`;

  const placeholder =
    `Ask anything... (@ for files,` +
    ` ${mod}D toggle between editor and chat,` +
    ` ${mod}T new chat,` +
    ` ${mod}E run selected text as task,` +
    ` ${mod}L copy text to chat)`;

  const subs: Record<string, string> = {
    VIEWPORT: 'width=device-width, initial-scale=1.0',
    CSP_META: csp,
    STYLE_HREF: u('main.css'),
    HLJS_CSS_HREF: u('highlight-github-dark.min.css'),
    HEAD_STYLE: '',
    BODY_CLASS_ATTR: '',
    INPUT_PLACEHOLDER: placeholder,
    ENTERKEYHINT: '',
    MODEL_NAME: selectedModel,
    VERSION_SUFFIX: version ? ' ' + version : '',
    AUTH_MODAL: '',
    NONCE_ATTR: ` nonce="${nonce}"`,
    HLJS_SRC: u('highlight.min.js'),
    MARKED_SRC: u('marked.min.js'),
    PANEL_COPY_SRC: u('panelCopy.js'),
    MAIN_SRC: u('main.js'),
    DEMO_SRC: u('demo.js'),
    SHIM_SCRIPT: '',
    TRICKS_JSON: tricksJson,
    TIPS_JSON: tipsJson,
    TIPS_SRC: u('tips.js'),
    // Webviews cannot capture the microphone (VS Code denies
    // getUserMedia), so voice.js runs in "webview" mode: the extension
    // host owns the local wake-word listener and forwards wake events.
    VOICE_SRC: u('voice.js'),
    // ackAudioUrl: GPT-synthesized "Working on it." clip voice.js
    // plays after submitting a voice-dictated task.
    VOICE_CONFIG: JSON.stringify({
      mode: 'webview',
      ackAudioUrl: u('working-on-it.mp3'),
    }),
  };

  // Single-pass substitution: each ``{{KEY}}`` in the TEMPLATE is
  // replaced exactly once, so a placeholder-looking string inside a
  // substituted VALUE (e.g. a user trick containing ``{{TIPS_JSON}}``)
  // is never rewritten by a later key's pass.  Unknown placeholders
  // are left untouched.
  return tpl.replace(/\{\{([A-Z_]+)\}\}/g, (match, key: string) =>
    Object.prototype.hasOwnProperty.call(subs, key) ? subs[key] : match,
  );
}
