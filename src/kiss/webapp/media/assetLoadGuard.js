/* global sessionStorage */
// Reloads the remote webapp when one of its own page assets fails to
// load.  A page script that never arrives leaves the app half-booted
// (with api.js missing, main.js throws at ``createSorcarApi`` and the
// loading overlay covers #app for good while the WebSocket
// authenticates happily); a stylesheet that never arrives leaves it
// unstyled for its whole lifetime (without remote-codex.css the
// docked history panel is main.css's 90vw drawer, open on top of the
// chat).  Chromium aborts every in-flight request with
// ERR_NETWORK_CHANGED when the network path changes during the load
// (a Wi-Fi/cellular hand-over on a phone; on a busy CI host, the
// veth links of other jobs' containers), so this is a transient a
// reload fixes.  Load errors do not bubble, but a capture-phase
// listener on window still sees them.  The reload timestamps in
// sessionStorage stop a reload loop when an asset is really broken:
// up to three reloads per 30 s, then the failure stays visible.  (One
// change can outlast a single reload: in the full-suite run of
// 2026-10-11 the reloaded page's remote-codex.css was aborted too and
// a one-reload budget left the chat under the drawer.)  The reload is
// immediate, from inside the error event: a reload deferred to a
// timer lets the broken document go on booting (main.js runs, the
// WebSocket opens) with requests in flight that the reload then
// aborts, after which Chromium has failed the reloaded page's own
// assets with ERR_TOO_MANY_RETRIES.
//
// _build_html inlines this file as the first thing in <head> (the
// HEAD_SCRIPT placeholder of media/chat.html), before the stylesheet
// links: a stylesheet's error event is dispatched asynchronously once
// its fetch fails, so a listener installed by a later script could
// miss it, and a <script src> that itself fails to load could not
// guard anything.
//
// Only the page's own assets count: same-origin ``<script src>`` and
// ``<link rel=stylesheet>`` that fail while the document is still
// parsing.  Assets main.js adds later on demand (the Monaco editor
// from its CDN, the voice model, the swapped highlight theme sheet)
// have their own fallbacks and must not restart the app.
(function () {
  // A JSON list of the Date.now() stamps of this tab's recent reloads.
  const _RELOADS = 'sorcar-asset-reloads';
  const _WINDOW_MS = 30000;
  const _MAX_RELOADS = 3;
  // Set once this document reloads: the other assets of the same
  // document that fail too (the document lives on until the reload
  // commits) must not each take a stamp of the budget.
  let _reloading = false;
  // The URL of the failed asset ('' for anything else: an inline
  // script, an image, the window's own error events).  src / href /
  // rel read as '' when the attribute is missing.
  function _assetUrl(target) {
    const tag = target.tagName;
    if (tag === 'SCRIPT') return target.src;
    if (tag === 'LINK' && /\bstylesheet\b/.test(target.rel)) return target.href;
    return '';
  }
  function _reloadOnAssetLoadError(ev) {
    const url = _assetUrl(ev.target);
    if (!url) return;
    if (_reloading) return;
    if (document.readyState !== 'loading') return;
    if (url.indexOf(window.location.origin + '/') !== 0) return;
    try {
      const now = Date.now();
      const reloads = (
        JSON.parse(sessionStorage.getItem(_RELOADS)) || []
      ).filter(stamp => {
        return now - stamp < _WINDOW_MS;
      });
      if (reloads.length >= _MAX_RELOADS) return;
      reloads.push(now);
      sessionStorage.setItem(_RELOADS, JSON.stringify(reloads));
    } catch (_e) {
      // No storage means no loop guard: leave the failure visible.
      return;
    }
    _reloading = true;
    window.location.reload();
  }
  window.addEventListener('error', _reloadOnAssetLoadError, true);
})();
