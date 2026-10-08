// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';
import * as path from 'path';
import {escapeHtml, getNonce, mediaAssetVersion} from './SorcarTab';

/**
 * The jsDelivr origin pdfView.js loads pdf.js from; the panel's CSP
 * admits it for scripts (the library module) and connections (the
 * worker script, fetched and started from a blob: URL).
 */
const PDFJS_CDN_ORIGIN = 'https://cdn.jsdelivr.net';

/**
 * The message a PDF preview panel posts when its Download link is
 * clicked: a webview cannot save files itself, so the extension host
 * (SorcarSidebarView._openPdfPreviewTab) offers a save dialog and copies
 * the file there.
 */
export const PDF_DOWNLOAD_MESSAGE = 'download';

/**
 * The HTML of a PDF preview panel for *filePath*, rendered by *webview*:
 * VS Code has no PDF editor of its own (a `vscode.open` on a PDF shows a
 * "binary file" notice), so the panel draws the pages with the same
 * pdf.js viewer (media/pdfView.js) the remote web app uses.  The page
 * fetches the file through its webview resource URI, so the file's
 * directory must be among the webview's localResourceRoots.
 */
export function buildPdfPreviewHtml(
  webview: vscode.Webview,
  extensionUri: vscode.Uri,
  filePath: string,
): string {
  const nonce = getNonce();
  const media = (name: string): string =>
    webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, 'media', name)) +
    '?v=' +
    mediaAssetVersion(extensionUri, name);
  const pdfUri = webview.asWebviewUri(vscode.Uri.file(filePath)).toString();
  const name = path.basename(filePath);
  const csp =
    `default-src 'none'; style-src ${webview.cspSource};` +
    // wasm-unsafe-eval lets the worker compile pdf.js's JPEG 2000 / JBIG2
    // decoders; without it they fall back to their slower JS builds.
    ` script-src 'nonce-${nonce}' 'wasm-unsafe-eval' ${PDFJS_CDN_ORIGIN};` +
    ` worker-src blob:; connect-src ${webview.cspSource} ${PDFJS_CDN_ORIGIN};` +
    ` img-src ${webview.cspSource} blob: data:; font-src ${webview.cspSource};`;
  const config = JSON.stringify({
    pdfUri,
    name,
    downloadMessage: PDF_DOWNLOAD_MESSAGE,
  }).replace(/</g, '\\u003c');
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <title>${escapeHtml(name)}</title>
  <link href="${media('main.css')}" rel="stylesheet">
  <link href="${media('brand.css')}" rel="stylesheet">
</head>
<body class="pdf-preview-body">
  <div id="pdf-holder" class="content-binary-holder"></div>
  <script nonce="${nonce}" src="${media('pdfView.js')}"></script>
  <script nonce="${nonce}">
    (function () {
      const config = ${config};
      const vscodeApi = acquireVsCodeApi();
      const holder = document.getElementById('pdf-holder');
      fetch(config.pdfUri)
        .then(res => {
          if (!res.ok) throw new Error('HTTP ' + res.status);
          return res.arrayBuffer();
        })
        .then(buf => {
          window.mountPdfViewer(holder, new Uint8Array(buf), {
            name: config.name,
            onDownload: () => vscodeApi.postMessage({type: config.downloadMessage}),
          });
        })
        .catch(err => {
          const note = document.createElement('div');
          note.className = 'content-binary-note';
          note.textContent = 'Cannot read ' + config.name + ': ' + String(err);
          holder.appendChild(note);
        });
    })();
  </script>
</body>
</html>`;
}
