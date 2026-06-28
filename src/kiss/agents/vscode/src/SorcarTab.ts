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
import {ensureUserAssetFromDefault} from './userAssets';

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
  const tricksJson = JSON.stringify(getTricks());
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
  };

  let html = tpl;
  for (const [key, value] of Object.entries(subs)) {
    html = html.split(`{{${key}}}`).join(value);
  }
  return html;
}
