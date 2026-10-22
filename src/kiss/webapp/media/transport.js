/* global sessionStorage, WebSocket, PasswordCredential, prompt */
// WebSocket transport of the webapp: provides acquireTransport(), the
// postMessage / getState / setState object media/main.js and
// media/api.js talk to the daemon through.  Every frame sent through
// it is a command of the server API catalog defined in
// src/kiss/server/sorcar.py (dispatched by
// kiss.server.sorcar.ServerApi.dispatch); the pre-app ``auth``
// handshake frames sent below are serviced by
// kiss.server.sorcar.ServerApi.authenticate before the daemon starts
// dispatching this connection's commands.  Server frames reach the
// app as window ``message`` events.
(function () {
  let _state = null;
  try {
    _state = JSON.parse(sessionStorage.getItem('sorcar-state'));
  } catch (_e) {}
  let _ws = null;
  let _pending = [];
  let _authenticated = false;
  // True once an authenticated session has lost its socket (a server
  // restart, a network blip, the phone's browser sleeping the tab).
  // The page is NOT reloaded when the socket comes back: main.js keeps
  // every tab, transcript and draft in memory, and on the
  // ``daemonStatus`` ``connected: true`` that follows the re-auth it
  // sends ``ready`` again, which makes the server push what changed
  // meanwhile (tab registry, transcript replays, tasks, settings).  A
  // reload would repaint the whole app from a blank page on every
  // blip — visible as flicker on a phone that drops the connection
  // each time the browser is backgrounded.  This flag only picks the
  // wording of the status surface ("Reconnecting" vs "starting").
  let _hadAuthThenClosed = false;
  // Reconnect backoff attempt count — reset to 0 after a successful
  // ``auth_ok`` so a fresh disconnect tries again almost immediately.
  let _reconnectAttempt = 0;
  // Pending reconnect timer id, used so visibilitychange / pageshow /
  // online wake-ups can short-circuit the scheduled delay.
  let _reconnectTimer = null;

  /**
   * Replace the overlay text so the user sees an accurate status.
   *
   * On a brand-new tab the message is "KISS Sorcar Server is starting
   * ..." because the server may legitimately not be up yet.  Once we
   * have proven the server is reachable (a previous ``auth_ok`` came
   * through, then the socket later closed) every subsequent display of
   * the overlay represents a RECONNECT, not a cold start — say so.
   */
  function _updateLoadingMsg(reconnecting) {
    const msg = document.getElementById('kiss-server-loading-msg');
    if (!msg) return;
    _setOverlayAction(null, null);
    msg.textContent = reconnecting
      ? 'Reconnecting to ' + _productName() + ' Server ...'
      : _productName() + ' Server is starting ...';
  }

  function _productName() {
    return (window.__BRAND__ && window.__BRAND__.productName) || 'KISS Sorcar';
  }

  // The overlay's single action button ("Retry now" / "Enter
  // password").  Created on demand next to the message node — the
  // overlay markup itself lives in media/chat.html — and removed
  // whenever the overlay text is not about a choice the user can make.
  let _overlayAction = null;

  /**
   * Show one button under the overlay message, or remove it.
   *
   * ``label`` is the verb on the button and ``handler`` runs on
   * click; passing ``null`` for either removes the button.  A second
   * call replaces the previous label/handler, so the button never
   * accumulates listeners.
   */
  function _setOverlayAction(label, handler) {
    const msg = document.getElementById('kiss-server-loading-msg');
    if (!msg || !msg.parentNode) return;
    if (!label || !handler) {
      if (_overlayAction && _overlayAction.parentNode) {
        _overlayAction.parentNode.removeChild(_overlayAction);
      }
      _overlayAction = null;
      return;
    }
    if (!_overlayAction) {
      _overlayAction = document.createElement('button');
      _overlayAction.type = 'button';
      _overlayAction.id = 'kiss-server-loading-action';
      _overlayAction.className = 'auth-modal-btn auth-modal-ok';
      _overlayAction.style.marginTop = '12px';
    }
    _overlayAction.textContent = label;
    _overlayAction.onclick = handler;
    if (_overlayAction.parentNode !== msg.parentNode) {
      msg.parentNode.insertBefore(_overlayAction, msg.nextSibling);
    }
  }

  // Wall-clock start of the current outage (0 while connected).  Set
  // by the first ``onclose`` after a connect attempt fails, cleared by
  // ``auth_ok``.  Drives the overlay escalation below.
  let _outageSince = 0;
  // Once-a-second overlay refresh while disconnected: after
  // OVERLAY_ESCALATE_MS of failed reconnects the message gains the
  // elapsed time, a plain hint and a "Retry now" button, and during
  // an auth lockout it counts the wait down.  Cleared on ``auth_ok``.
  let _overlayTick = null;
  const OVERLAY_ESCALATE_MS = 8000;

  function _startOverlayTick() {
    if (_overlayTick) return;
    _overlayTick = setInterval(_refreshOverlay, 1000);
  }

  function _stopOverlayTick() {
    if (!_overlayTick) return;
    clearInterval(_overlayTick);
    _overlayTick = null;
  }

  /**
   * Re-render the overlay text for the current disconnected state.
   *
   * Lockout wins (the server told us exactly how long to wait); then
   * an outage past OVERLAY_ESCALATE_MS gets the escalated wording
   * with the elapsed seconds and a "Retry now" button, so the user is
   * never left with a bare spinner and no idea whether waiting helps.
   * The tick only runs between ``onclose`` / ``auth_locked`` (which set
   * ``_outageSince`` / ``_lockedUntil``) and the next ``auth_ok`` /
   * ``auth_required`` (which clear them and stop it).  A lockout that
   * has run out with no server answer hands over to the outage (or
   * cancelled-dialog) wording via ``_endLockout`` instead of counting
   * "0 s" forever.
   */
  function _refreshOverlay() {
    if (_lockedUntil && Date.now() < _lockedUntil) {
      _showLockedMsg(Math.ceil((_lockedUntil - Date.now()) / 1000));
      return;
    }
    if (_lockedUntil) _endLockout();
    if (!_outageSince) return;
    const elapsed = Date.now() - _outageSince;
    if (elapsed < OVERLAY_ESCALATE_MS) return;
    const msg = document.getElementById('kiss-server-loading-msg');
    if (!msg) return;
    msg.textContent =
      'Still trying to reach the ' +
      _productName() +
      ' server (' +
      Math.floor(elapsed / 1000) +
      ' s). ' +
      'Check that it is running on this machine.';
    _setOverlayAction('Retry now', _reconnectNowIfNeeded);
  }

  /**
   * The lockout deadline passed without a server answer clearing it.
   *
   * Only ``auth_ok`` / ``auth_required`` reset ``_lockedUntil``, so an
   * expired lockout with the socket still down means the reconnect
   * scheduled for its end is failing (server gone, device offline).
   * Stop the countdown and hand the overlay to the state that now
   * applies: the cancelled-dialog explanation ("A password is
   * needed" + "Enter password"), or the outage wording, whose clock
   * starts at the lockout's end so "Still trying ..." + "Retry now"
   * follows OVERLAY_ESCALATE_MS later.  An open socket is about to
   * be answered by the server, so only the countdown ends.
   */
  function _endLockout() {
    const until = _lockedUntil;
    _lockedUntil = 0;
    if (_ws && _ws.readyState === WebSocket.OPEN) return;
    if (_promptDeclined) {
      _stopOverlayTick();
      _showPasswordNeeded();
      return;
    }
    if (_outageSince) return;
    _outageSince = until;
    _updateLoadingMsg(_hadAuthThenClosed || _offlineShell);
  }

  // Non-zero while the server has told us (via an ``auth_locked``
  // frame) that this source IP is rate-limited after too many failed
  // logins.  Holds the retry delay in milliseconds; consumed by the
  // ``onclose`` handler so the follow-up reconnect waits out the
  // lockout instead of hammering the server on the fast backoff, and
  // so the overlay keeps showing the lockout explanation rather than
  // the generic "starting ..." label.
  let _lockedRetryMs = 0;
  // Wall-clock end of the lockout (0 when not locked) for the countdown.
  let _lockedUntil = 0;
  // True after a pre-auth ``error`` frame was written onto the overlay
  // (e.g. "remote access is turned off"); the close that follows must
  // not replace that explanation with the generic "starting" label.
  let _preAuthErrorShown = false;

  /**
   * Replace the overlay text with the auth-lockout explanation.
   *
   * Shown when the server answers the handshake with ``auth_locked``:
   * without this the user would stare at a promptless "KISS Sorcar
   * Server is starting ..." spinner with no hint that the remote
   * password rate-limit is what is keeping the password modal away.
   * The lock is per source network (the tunnel collapses every
   * visitor onto one IP), so the wording does not blame the reader,
   * and the seconds count down once a second via the overlay tick.
   */
  function _showLockedMsg(secs) {
    const msg = document.getElementById('kiss-server-loading-msg');
    if (!msg) return;
    _setOverlayAction(null, null);
    msg.textContent =
      'Too many wrong passwords from this network. ' +
      'You can try again in ' +
      secs +
      ' s.';
  }

  // A page the service worker answered from its cache because the
  // server was unreachable carries the ``kiss-offline-shell`` meta
  // (see media/sw.js).  Its code may be older than what the server
  // now runs, so the first successful handshake reloads it — the one
  // reload the shim still performs, and only on a page that was never
  // live.  The sessionStorage flag survives that reload and stops a
  // loop when the page fetch keeps timing out (slow link) while the
  // WebSocket still comes up: the second cached load is kept.  A page
  // the server itself served clears the flag.
  //
  // The worker only holds a copy because the server was reachable
  // earlier, so a cached page is a reconnect from the user's point of
  // view and its overlay says so from the start (and keeps saying so
  // while the connection attempts fail).
  const _OFFLINE_RELOADED_FLAG = 'sorcar-offline-reloaded';
  const _offlineShell = !!document.querySelector(
    'meta[name="kiss-offline-shell"]',
  );
  let _reloadOnFirstAuth = false;
  if (_offlineShell) _updateLoadingMsg(true);
  // The commands flushed on a reconnected socket, kept until the
  // server's ``pong`` confirms it has taken them (see ``auth_ok``).
  let _inflight = [];
  try {
    if (_offlineShell) {
      _reloadOnFirstAuth =
        sessionStorage.getItem(_OFFLINE_RELOADED_FLAG) !== '1';
    } else {
      sessionStorage.removeItem(_OFFLINE_RELOADED_FLAG);
    }
  } catch (_e) {}

  // Half-open sockets: when the network path dies silently (Wi-Fi to
  // cellular hand-over, a NAT entry expiring, the laptop lid closing)
  // the browser keeps reporting the socket OPEN and no ``onclose``
  // ever fires, so the app would sit "connected" while every frame
  // is lost.  The server sends a ``heartbeat`` frame to every client
  // right after each of its own keep-alive pings (every 15 s, see
  // ``RemoteAccessServer._ping_one_ws``); a socket that has been
  // silent for ``_STALE_AFTER_MS`` is therefore dead and is dropped
  // here so the regular reconnect path (banner, backoff, resync on
  // re-auth) takes over.  Any frame counts as life, so a slow server
  // command cannot cause a false alarm.  The check runs from
  // ``onopen``, not from ``auth_ok``: the server heartbeats every
  // connection whether or not it has authenticated, and a socket whose
  // auth answer never comes (the path died right after the upgrade,
  // or the answer was lost) would otherwise sit OPEN for ever with
  // nothing to retry it — ``_reconnectNowIfNeeded`` takes OPEN as
  // healthy, and a cold load has no outage clock for "Retry now".
  const _STALE_CHECK_MS = 15000;
  const _STALE_AFTER_MS = 45000;
  let _lastFrameAt = 0;
  let _staleTimer = null;
  // The reconnect backoff is forgiven only once a session has stayed
  // up this long.  Forgiving it on ``auth_ok`` let a daemon that
  // accepts the socket and then stalls or drops it during the replay
  // be retried every half second for as long as it kept failing.
  const _STABLE_AFTER_MS = 30000;
  let _stableTimer = null;

  function _stopStaleCheck() {
    if (_staleTimer !== null) {
      try {
        clearTimeout(_staleTimer);
      } catch (_e) {}
      _staleTimer = null;
    }
    if (_stableTimer !== null) {
      try {
        clearTimeout(_stableTimer);
      } catch (_e) {}
      _stableTimer = null;
    }
  }

  function _checkStale() {
    _staleTimer = null;
    if (!_ws || _ws.readyState !== WebSocket.OPEN) return;
    if (Date.now() - _lastFrameAt >= _STALE_AFTER_MS) {
      _dropSocket(_ws);
      _onSocketClosed();
      return;
    }
    _staleTimer = setTimeout(_checkStale, _STALE_CHECK_MS);
  }

  // Neutralise *ws* so none of its late events reach the shim, then
  // close it.  Shared by connect() (replacing a dead socket on
  // wake-up) and the stale check.
  function _dropSocket(ws) {
    try {
      ws.onopen = null;
      ws.onmessage = null;
      ws.onclose = null;
      ws.onerror = null;
    } catch (_e) {}
    try {
      ws.close();
    } catch (_e) {}
  }

  // Offline app shell: the service worker served at ``/sw.js`` (see
  // ``_build_service_worker``) caches this page and its ``/media``
  // assets, so the app still opens — and stays on screen — when the
  // connection is slow, flaky or gone; a page load is served
  // network-first.  Best effort: browsers refuse
  // a worker fetched over a self-signed certificate (the LAN URL),
  // and the app must keep working without one.
  if (
    typeof navigator !== 'undefined' &&
    navigator.serviceWorker &&
    typeof navigator.serviceWorker.register === 'function'
  ) {
    try {
      navigator.serviceWorker
        .register('/sw.js', {updateViaCache: 'none'})
        .catch(() => {});
    } catch (_e) {}
  }

  // Deliver a server frame (or a synthesised ``daemonStatus`` post) to
  // the app as a window ``message`` event.  This shim script runs at
  // the TOP of the body script list, so the WebSocket regularly
  // authenticates while the HTML parser is still fetching
  // ``media/main.js`` — whose ``message`` listener therefore is not
  // registered yet.  A MessageEvent dispatched in that gap is silently
  // lost; the observed symptom was the one-shot
  // ``daemonStatus connected:true`` falling into it, leaving the
  // "KISS Sorcar Server is starting ..." overlay covering ``#app``
  // forever.  Queue events while the document is still parsing and
  // flush once DOMContentLoaded fires: every body script (main.js
  // included) has run by then, so the listener exists and ordering is
  // preserved.  The queue is decided ONCE here, by whether the parser
  // is still running: the browser sets readyState 'interactive' and
  // then fires DOMContentLoaded from a LATER task (HTML "the end", step
  // 4), so a frame arriving in that gap must keep queueing behind the
  // frames already waiting, not jump ahead of them.  A copy of this
  // script evaluated in a finished document (the jsdom tests) never
  // sees a DOMContentLoaded, so it starts with no queue at all.
  let _preParseQueue = document.readyState === 'loading' ? [] : null;
  document.addEventListener('DOMContentLoaded', () => {
    const q = _preParseQueue;
    _preParseQueue = null;
    if (!q) return;
    for (let i = 0; i < q.length; i++) {
      window.dispatchEvent(new MessageEvent('message', {data: q[i]}));
    }
  });
  function _dispatchToApp(data) {
    if (_preParseQueue !== null) {
      _preParseQueue.push(data);
      return;
    }
    window.dispatchEvent(new MessageEvent('message', {data: data}));
  }

  function _scheduleReconnect() {
    if (_reconnectTimer !== null) return;
    // Exponential backoff with full jitter: 0.5-1s, 1-2s, 2-4s, ...
    // capped at 30-60s.  The first attempt is quick so a blip heals
    // at once (visibilitychange / online reconnect immediately
    // anyway, see _reconnectNowIfNeeded); the cap is high so a daemon
    // that is busy or stalled is not hammered every few seconds by
    // every open browser, and the jitter keeps the browsers that one
    // outage disconnected together from reconnecting in one burst.
    // The standalone app's popup window (kiss-app on macOS) is capped
    // at 5-10 s instead: one window on loopback is no load, and the
    // daemon takes the popup for closed while it is not connected
    // (a start handed to the running browser opens a second window
    // then; a windowless browser is quit), so it must be back soon
    // after a daemon restart.
    const cap = window.name === 'kiss-app-window' ? 5000 : 30000;
    const base = Math.min(cap, 500 * Math.pow(2, _reconnectAttempt));
    const delay = base + Math.floor(Math.random() * base);
    _reconnectAttempt++;
    _reconnectTimer = setTimeout(() => {
      _reconnectTimer = null;
      connect();
    }, delay);
  }

  function _reconnectNowIfNeeded() {
    // Called from visibilitychange / pageshow / online handlers so a
    // user who left Safari for another app does not wait the full
    // backoff after returning, and by the overlay's "Retry now"
    // button (the escalated text and the button stay, the tick keeps
    // the seconds current; auth_ok or auth_required clears them).  We
    // treat CONNECTING as "in flight, don't disturb": connect() closes
    // the current socket first, so restarting a slow handshake on
    // every click would keep it from ever finishing.  CLOSED / CLOSING
    // / null all warrant an immediate attempt.
    if (
      _ws &&
      (_ws.readyState === WebSocket.OPEN ||
        _ws.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }
    if (_reconnectTimer !== null) {
      try {
        clearTimeout(_reconnectTimer);
      } catch (_e) {}
      _reconnectTimer = null;
    }
    _reconnectAttempt = 0;
    connect();
  }

  // Custom auth modal — replaces the browser-native prompt(), which is
  // rendered tall with wasted space below its buttons on most desktop
  // browsers.  Falls back to prompt() when the modal nodes are not in
  // the DOM (e.g. unit tests that load the shim in isolation).
  //
  // The dialog stays open until the server accepts the password
  // (``auth_ok`` calls ``_hideAuthModal``) or the user cancels: a
  // wrong password is reported INSIDE the dialog, with the typed
  // text kept and selected, instead of the dialog vanishing and a
  // generic spinner taking its place.
  //
  // ``_authPrompt`` is the one pending prompt promise.  Every
  // ``auth_required`` (and every silent reconnect while the dialog is
  // open — the server drops an idle unauthenticated socket after
  // 60 s) goes through ``_promptForPassword``, which returns early
  // while a prompt is pending: no duplicate listeners, no duplicate
  // ``auth`` frames, and the value the user is typing is never
  // cleared under them.
  let _authPrompt = null;
  // Pending interactive submission, never populated from cached credentials.
  let _authPassword = '';
  // The password the current socket's ``auth`` frame carried.  The
  // server checks the password on EVERY frame of a password-
  // authenticated client, so ``_sendFrame`` repeats it in each one.
  // A browser on the daemon's own machine (http://127.0.0.1:8787) is
  // admitted without a password and keeps this empty: its frames go
  // out bare.
  let _sessionPassword = '';

  /** Send the JSON text *data* of one frame with the session password added. */
  function _sendFrame(data) {
    if (!_sessionPassword) {
      _ws.send(data);
      return;
    }
    const pwd = JSON.stringify({password: _sessionPassword}).slice(1, -1);
    _ws.send('{' + pwd + (data.length > 2 ? ',' : '') + data.slice(1));
  }
  // Detaches the pending prompt's listeners without resolving it;
  // set while a prompt is pending, used by ``_hideAuthModal`` so a
  // dialog hidden from outside (``auth_ok`` won on a reconnect with
  // a password stored by another tab) never leaves a stale prompt
  // that would make the next ``_promptForPassword`` a no-op.
  let _authPromptCancel = null;

  // True after the user cancelled the password dialog: the overlay then
  // says a password is needed and offers "Enter password", and the
  // ``auth_required`` of every quiet reconnect underneath (the server
  // drops idle unauthenticated sockets after 60 s) must NOT pop the
  // dialog back up.  Cleared by that button and by ``auth_ok``.
  let _promptDeclined = false;

  function _showAuthModal() {
    const pending = new Promise(_runAuthModal);
    // Only a dialog that is waiting for the user is "pending".  The
    // prompt() fallback resolves inside the executor, so remembering
    // its promise would make every later ``_promptForPassword`` a
    // no-op and leave the revealed app without a way to log in
    // (SECURITY: the frontend gate must always re-prompt or re-gate).
    if (_authPromptCancel) _authPrompt = pending;
    return pending;
  }

  function _runAuthModal(resolve) {
    const modal = document.getElementById('auth-modal');
    const input = document.getElementById('auth-modal-input');
    const okBtn = document.getElementById('auth-modal-ok');
    const cnclBtn = document.getElementById('auth-modal-cancel');
    if (!modal || !input || !okBtn || !cnclBtn) {
      resolve(prompt('Enter remote access password:'));
      return;
    }
    const wasOpen = modal.style.display === 'flex';
    if (!wasOpen) {
      input.value = '';
      _setAuthError('');
      modal.style.display = 'flex';
    }
    setTimeout(() => {
      try {
        input.focus();
        if (wasOpen) input.select();
      } catch (_e) {}
    }, 0);

    function cleanup() {
      okBtn.removeEventListener('click', onOk);
      cnclBtn.removeEventListener('click', onCancel);
      input.removeEventListener('keydown', onKey);
      _authPrompt = null;
      _authPromptCancel = null;
    }
    function onOk() {
      const v = input.value;
      cleanup();
      resolve(v);
    }
    function onCancel() {
      cleanup();
      _hideAuthModal();
      resolve(null);
    }
    function onKey(e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        onOk();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        onCancel();
      }
    }
    okBtn.addEventListener('click', onOk);
    cnclBtn.addEventListener('click', onCancel);
    input.addEventListener('keydown', onKey);
    _authPromptCancel = cleanup;
  }

  function _hideAuthModal() {
    _authPassword = '';
    if (_authPromptCancel) _authPromptCancel();
    const input = document.getElementById('auth-modal-input');
    if (input) input.value = '';
    const modal = document.getElementById('auth-modal');
    if (modal) modal.style.display = 'none';
    _setAuthError('');
  }

  function _authModalOpen() {
    const modal = document.getElementById('auth-modal');
    return !!(modal && modal.style.display === 'flex');
  }

  /** Write ``text`` into the dialog's ``role="alert"`` line ('' clears). */
  function _setAuthError(text) {
    const err = document.getElementById('auth-modal-error');
    if (err) err.textContent = text;
  }

  /**
   * Ask for the password (once) and send it on the current socket.
   *
   * Idempotent: while a prompt is pending this is a no-op, so a
   * repeated ``auth_required`` (reconnect with the dialog open) or a
   * wrong-password ``error`` re-uses the open dialog.  On Cancel the
   * app is re-gated behind the overlay, which then explains that a
   * password is needed and offers "Enter password" to re-open the
   * dialog — never a spinner that pretends the server is starting.
   */
  function _promptForPassword() {
    if (_authPrompt) return;
    _showAuthModal().then(_onPasswordEntered);
  }

  /**
   * Gate the app behind the overlay after a cancelled password dialog.
   *
   * The overlay explains why (never a spinner that pretends the server
   * is starting) and offers "Enter password" to re-open the dialog.
   */
  function _showPasswordNeeded() {
    _promptDeclined = true;
    _dispatchToApp({type: 'daemonStatus', connected: false});
    const msg = document.getElementById('kiss-server-loading-msg');
    if (msg) msg.textContent = 'A password is needed to use this server.';
    _setOverlayAction('Enter password', _reopenPasswordPrompt);
  }

  function _onPasswordEntered(pwd) {
    if (pwd === null) {
      _showPasswordNeeded();
      return;
    }
    _authPassword = pwd;
    try {
      localStorage.setItem('sorcar-remote-pwd', pwd);
    } catch (_e) {}
    if (_ws && _ws.readyState === WebSocket.OPEN) {
      _sessionPassword = pwd;
      _ws.send(JSON.stringify({type: 'auth', password: pwd}));
    } else {
      // The server dropped the idle socket while the user typed; the
      // stored password goes out in ``onopen`` of the reconnect.
      _reconnectNowIfNeeded();
    }
  }

  /** "Enter password" on the overlay after a Cancel: show the dialog again. */
  function _reopenPasswordPrompt() {
    _promptDeclined = false;
    _setOverlayAction(null, null);
    _updateLoadingMsg(_hadAuthThenClosed);
    _dispatchToApp({type: 'daemonStatus', connected: true});
    _promptForPassword();
    if (!_ws || _ws.readyState !== WebSocket.OPEN) _reconnectNowIfNeeded();
  }

  window.acquireTransport = function () {
    return {
      postMessage: function (msg) {
        const data = JSON.stringify(msg);
        if (_ws && _ws.readyState === WebSocket.OPEN && _authenticated) {
          _sendFrame(data);
        } else {
          _pending.push(data);
        }
      },
      getState: function () {
        return _state;
      },
      setState: function (s) {
        _state = s;
        try {
          sessionStorage.setItem('sorcar-state', JSON.stringify(s));
        } catch (_e) {}
      },
    };
  };

  // A connection that went away while its ``pong`` was still awaited
  // proves nothing about the batch flushed on it: the next connection
  // sends the batch again, ahead of whatever has been queued since.
  function _requeueUnconfirmed() {
    if (_inflight.length === 0) return;
    _pending = _inflight.concat(_pending);
    _inflight = [];
  }

  // Shared by the socket's ``onclose`` and the stale-socket check
  // (``_checkStale``), which drops a half-open socket that will never
  // fire ``onclose`` on its own.
  function _onSocketClosed() {
    // Latch "we had a real session and then lost it": the status
    // surface below says "Reconnecting" rather than "starting", and
    // main.js keeps the app on screen under a banner.  Only a socket
    // that had completed its auth handshake counts — a fresh page
    // that never authenticated has nothing on screen to keep.
    if (_authenticated) _hadAuthThenClosed = true;
    _authenticated = false;
    _requeueUnconfirmed();
    _stopStaleCheck();
    if (_lockedRetryMs > 0) {
      // This close follows an ``auth_locked`` frame: the server is
      // rate-limiting this IP after too many failed logins.  Keep
      // the lockout explanation on the overlay (do NOT overwrite it
      // with the generic label below) and hold off reconnecting
      // until the server-provided lockout expiry — the fast backoff
      // would only harvest more silent refusals.  The wake-up
      // listeners may still reconnect earlier; the server then just
      // re-sends ``auth_locked`` with a fresher ``retry_after``.
      const lockedDelay = _lockedRetryMs;
      _lockedRetryMs = 0;
      _dispatchToApp({type: 'daemonStatus', connected: false});
      try {
        clearTimeout(_reconnectTimer);
      } catch (_e) {}
      _reconnectTimer = setTimeout(() => {
        _reconnectTimer = null;
        connect();
      }, lockedDelay);
      return;
    }
    if (_authPrompt || _authModalOpen()) {
      // The server drops an idle unauthenticated socket after 60 s.
      // The user is still typing the password: keep the dialog (and
      // the typed text) exactly as it is, say nothing on the overlay,
      // and reconnect underneath; the reconnect's ``auth_required``
      // is a no-op on the open dialog and the Unlock sends on the
      // new socket.
      _scheduleReconnect();
      return;
    }
    if (_preAuthErrorShown || _promptDeclined) {
      // The overlay already explains why the server refused us (or
      // that a password is needed); a spinner label would hide that.
      // Keep retrying quietly.
      _preAuthErrorShown = false;
      _scheduleReconnect();
      return;
    }
    if (!_outageSince) _outageSince = Date.now();
    _startOverlayTick();
    // Switch the overlay text BEFORE re-revealing it: once this page
    // has had a successful handshake (or came from the worker's cache,
    // which only exists because the server was reachable before) every
    // overlay appearance is a reconnect from the user's perspective.
    // Past OVERLAY_ESCALATE_MS the tick owns the text (elapsed time,
    // hint, "Retry now"); do not flip it back to the bare label.
    if (Date.now() - _outageSince < OVERLAY_ESCALATE_MS) {
      _updateLoadingMsg(_hadAuthThenClosed || _offlineShell);
    }
    // Tell the app the socket is down.  Symmetric to the ``auth_ok``
    // dispatch above.  ``reconnecting: true`` — the
    // app was authenticated and on screen in THIS page — makes
    // ``main.js`` keep ``#app`` visible under a slim "Reconnecting
    // ..." banner instead of covering it with the full-screen
    // overlay: a flaky link must not blank the app every few
    // seconds.  A page that never authenticated has nothing to show
    // and keeps the full overlay.  The banner also tells the user
    // that sending is on hold (``main.js`` holds prompts back while
    // the daemon is down); on the next ``auth_ok`` main.js sends
    // ``ready`` again and the server resyncs everything in place.
    _dispatchToApp({
      type: 'daemonStatus',
      connected: false,
      reconnecting: _hadAuthThenClosed,
    });
    _scheduleReconnect();
  }

  function connect() {
    // Neutralise the previous socket BEFORE we install a fresh one.
    // On iOS Safari the OS may kill the underlying WebSocket while
    // the tab is backgrounded; when JS resumes, the wake-up listeners
    // (visibilitychange / focus / pageshow) frequently fire BEFORE
    // the queued ``onclose`` of the dead socket.  If we don't clear
    // the old handlers, that late ``onclose`` will run against the
    // module-level ``_ws`` we just replaced -- it would call
    // ``_scheduleReconnect()`` (overwriting the in-flight new socket
    // after the backoff fires) and any late ``onopen``/``onmessage``
    // on the old socket would ``_ws.send(...)`` on the new one.
    // Nulling the handlers and closing the old socket here makes the
    // replacement atomic from the rest of the shim's perspective.
    if (_ws) {
      // Nulling ``onclose`` below also discards the latch it would
      // have taken: when the wake-up listeners win the race against
      // the dead socket's queued ``onclose`` (the common mobile Safari
      // case), an authenticated session is being replaced right here,
      // so record the loss now and tell the app, exactly as the
      // ``onclose`` would have: without the ``connected: false`` the
      // app never learns the session dropped, so it would not send
      // ``ready`` again on the replacement's ``auth_ok`` and miss the
      // resync.  (No reconnect is scheduled here: this IS the
      // reconnect.)
      if (_authenticated) {
        _hadAuthThenClosed = true;
        _updateLoadingMsg(true);
        _dispatchToApp({
          type: 'daemonStatus',
          connected: false,
          reconnecting: true,
        });
      }
      _requeueUnconfirmed();
      _dropSocket(_ws);
    }
    _stopStaleCheck();
    // Plain ``ws://`` on a plain ``http://`` page (the daemon's own
    // machine, http://127.0.0.1:PORT); ``wss://`` everywhere else.
    // The popup window of the macOS standalone app (kiss-app, named
    // by media/appWindow.html) marks its connection: the daemon quits
    // a browser whose popup closed (kiss.server.standalone_app).
    const scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
    const query = window.name === 'kiss-app-window' ? '?appWindow=1' : '';
    _ws = new WebSocket(scheme + location.host + '/ws' + query);
    _authenticated = false;

    _ws.onopen = function () {
      let pwd = '';
      try {
        pwd = localStorage.getItem('sorcar-remote-pwd') || '';
      } catch (_e) {}
      if (_authPassword !== pwd) _authPassword = '';
      _sessionPassword = pwd;
      _ws.send(JSON.stringify({type: 'auth', password: pwd}));
      _lastFrameAt = Date.now();
      _staleTimer = setTimeout(_checkStale, _STALE_CHECK_MS);
    };

    _ws.onmessage = function (event) {
      const msg = JSON.parse(event.data);
      _lastFrameAt = Date.now();
      if (msg.type === 'heartbeat') return;
      if (msg.type === 'auth_ok') {
        if (_reloadOnFirstAuth) {
          // A page the service worker served from its offline cache
          // may run code older than the server's: reload it once, now
          // that the server is reachable.  Nothing posted so far
          // matters — the overlay kept the app off screen, so the
          // queue holds only the boot commands the fresh page repeats.
          _reloadOnFirstAuth = false;
          try {
            sessionStorage.setItem(_OFFLINE_RELOADED_FLAG, '1');
          } catch (_e) {}
          try {
            window.location.reload();
          } catch (_e) {}
          return;
        }
        // Recover from a server restart / network blip in place: the
        // page keeps its state and main.js, on the ``daemonStatus``
        // dispatched below, sends ``ready`` again so the server pushes
        // what changed while the socket was down.  No reload — that
        // would repaint the whole app on every blip.
        _hadAuthThenClosed = false;
        _authenticated = true;
        _stableTimer = setTimeout(() => {
          _stableTimer = null;
          _reconnectAttempt = 0;
        }, _STABLE_AFTER_MS);
        _outageSince = 0;
        _lockedUntil = 0;
        _promptDeclined = false;
        _stopOverlayTick();
        _setOverlayAction(null, null);
        // WebSocket auth is invisible to the browser's login heuristics.
        // Complete an interactive login explicitly; otherwise Terminal's
        // later xterm fetch looks like its successful form submission and
        // opens Save Password there. Never re-offer on silent reconnects.
        if (
          _authPassword &&
          window.PasswordCredential &&
          navigator.credentials
        ) {
          try {
            navigator.credentials
              .store(
                new PasswordCredential({
                  id: 'remote',
                  password: _authPassword,
                }),
              )
              .catch(() => {});
          } catch (_e) {
            /* Password saving must not block authentication. */
          }
        }
        _hideAuthModal();
        // Everything the page posted while the connection was down
        // (a settings save, a model change, a closed tab, ...) goes
        // out now, on the new connection, in the order it was posted.
        // The server handles a connection's commands one at a time, so
        // its ``pong`` proves it has taken every command sent before
        // the ``ping``; until then the batch stays in ``_inflight`` and
        // a connection that dies first re-sends it on the next one
        // (at-least-once: a settings save may be applied twice, never
        // silently lost).  main.js holds prompts back while the daemon
        // is down, so a ``runTask`` is never in such a batch.
        const batch = _pending;
        _pending = [];
        for (let i = 0; i < batch.length; i++) _sendFrame(batch[i]);
        if (batch.length > 0) {
          _inflight = batch;
          _sendFrame(JSON.stringify({type: 'ping'}));
        }
        // Hide the "KISS Sorcar Server is starting ..." overlay now
        // that the WebSocket is authenticated: the ``daemonStatus``
        // post is synthesised here (the daemon is this WSS server) as
        // the window ``message`` event ``media/main.js`` listens for.
        // Without this the overlay covers ``#app`` forever and the
        // user only ever sees "KISS Sorcar Server is starting ...".
        // On a reconnect main.js answers this with a fresh ``ready``
        // (its ``daemonWasDown`` path), which is what resyncs the
        // page: it goes out after the batch above, so a queued change
        // is applied before the server replays state to this client.
        _dispatchToApp({type: 'daemonStatus', connected: true});
        return;
      }
      if (msg.type === 'pong') {
        // The server has taken the whole batch flushed on this
        // connection; nothing is owed any more.
        _inflight = [];
        return;
      }
      if (msg.type === 'auth_required') {
        _authPassword = '';
        // Stored password (if any) was rejected; drop it so a refresh
        // re-prompts instead of silently retrying the bad value.
        try {
          localStorage.removeItem('sorcar-remote-pwd');
        } catch (_e) {}
        // Reveal ``#app`` so the auth modal (which lives INSIDE #app
        // in the chat.html template — see the ``AUTH_MODAL`` template
        // placeholder substituted by ``_build_html``) is no longer
        // hidden by its display:none parent.  Without this
        // dispatch a password-protected webapp shows the loading
        // overlay forever and the user can never enter their
        // password.  Symmetric to the auth_ok dispatch above — both
        // states prove the server is reachable.
        //
        // SECURITY — ``_onPasswordEntered`` re-gates the app
        // (``connected: false``) when the prompt is cancelled, so the
        // reveal never exposes the unauthenticated webapp.
        // The server answered, so any outage is over: stop the
        // overlay escalation / lockout countdown.
        _outageSince = 0;
        _lockedUntil = 0;
        _stopOverlayTick();
        if (_promptDeclined) {
          // The user closed the dialog earlier; the overlay still says
          // a password is needed and how to enter it.  Do not pop the
          // dialog back up on every reconnect.
          _showPasswordNeeded();
          return;
        }
        _setOverlayAction(null, null);
        _dispatchToApp({type: 'daemonStatus', connected: true});
        _promptForPassword();
        return;
      }
      if (
        msg.type === 'error' &&
        msg.code === 'auth_failed' &&
        _authenticated
      ) {
        // The password was changed (Settings > Remote password on the
        // daemon's computer, or install.sh on a fresh install) after
        // this socket authenticated: the server refused a frame and is
        // closing the socket.  Forget the stale password so the
        // reconnect probes with the uncounted empty one and gets the
        // password dialog, not a counted wrong guess.
        try {
          localStorage.removeItem('sorcar-remote-pwd');
        } catch (_e) {}
        _authPassword = '';
        _sessionPassword = '';
        return;
      }
      if (
        msg.type === 'remotePasswordChanged' &&
        _sessionPassword &&
        typeof msg.password === 'string'
      ) {
        // This page saved a new remote password (Settings > Remote
        // password) and is the one client told it back.  Adopt it
        // before the page sends another frame: the server checks the
        // password on every frame, so the old one would be refused
        // and this session shut out.  A page admitted without a
        // password (loopback) keeps sending bare frames.  An EMPTY
        // password turns remote access off, this page included: it
        // is signed out at its next frame, so forget the old
        // credential now (its grace ended with the save) and let the
        // reconnect's handshake report that remote access is off.
        _sessionPassword = msg.password;
        if (!msg.password) _authPassword = '';
        try {
          if (msg.password) {
            localStorage.setItem('sorcar-remote-pwd', msg.password);
          } else {
            localStorage.removeItem('sorcar-remote-pwd');
          }
        } catch (_e) {}
      }
      if (msg.type === 'error' && !_authenticated) {
        // Pre-auth errors are the server's last word before it closes
        // the socket.  A wrong password (``code: 'auth_failed'``) is
        // reported inside the still-open dialog — the typed text is
        // kept and selected so the user can retype at once — and the
        // stored copy is dropped so the reconnect's probe is the
        // uncounted empty one, not a second wrong guess.  Any other
        // pre-auth error (e.g. remote access disabled) replaces the
        // overlay spinner text so the user is told why.  The server
        // answered, so any outage is over: stop the overlay escalation
        // / lockout countdown, else the next tick overwrites the
        // explanation with "Still trying to reach ..." and a stale
        // "Retry now" button.
        _outageSince = 0;
        _lockedUntil = 0;
        _stopOverlayTick();
        _setOverlayAction(null, null);
        const text = (msg.text && String(msg.text)) || 'Something went wrong.';
        if (msg.code === 'auth_failed') {
          try {
            localStorage.removeItem('sorcar-remote-pwd');
          } catch (_e) {}
          if (_authModalOpen()) {
            _setAuthError(text);
            _promptForPassword();
            return;
          }
        }
        const over = document.getElementById('kiss-server-loading-msg');
        if (over) over.textContent = text;
        _preAuthErrorShown = true;
        _dispatchToApp({type: 'daemonStatus', connected: false});
        return;
      }
      if (msg.type === 'auth_locked') {
        // The server refused the handshake because this source IP is
        // rate-limited after too many failed logins (behind the
        // cloudflared tunnel EVERY visitor shares one loopback IP, so
        // this can be someone else's guesses).  The server closes the
        // socket right after this frame.  Explain the wait on the
        // overlay and remember the retry delay so ``onclose`` waits
        // out the lockout instead of reconnecting on the fast backoff
        // — the eventual reconnect gets ``auth_required`` again and
        // the password modal finally appears.
        let secs = Math.ceil(Number(msg.retry_after));
        if (!(secs > 0)) secs = 60;
        _lockedRetryMs = secs * 1000;
        _lockedUntil = Date.now() + _lockedRetryMs;
        _showLockedMsg(secs);
        _startOverlayTick();
        // Re-gate the app while we wait (idempotent when the loading
        // overlay is already up, e.g. on a fresh page load).
        _dispatchToApp({type: 'daemonStatus', connected: false});
        return;
      }
      // SECURITY — never forward server data frames to the app before
      // the connection is authenticated.  ``auth_ok`` / ``auth_required``
      // are handled above; any other frame that arrives while
      // ``_authenticated`` is false must be dropped so an unauthenticated
      // client can never act on backend data (defense in depth — the
      // server does not send data pre-auth, but a bug or a hostile proxy
      // must not be able to bypass the remote-password gate this way).
      if (!_authenticated) return;
      _dispatchToApp(msg);
    };

    _ws.onclose = _onSocketClosed;

    _ws.onerror = function () {};
  }

  // Wake-up listeners — mobile Safari pauses JS in backgrounded tabs,
  // so a scheduled ``setTimeout(connect, ...)`` may not fire until the
  // user returns.  These events fire AS SOON AS the user comes back,
  // triggering an immediate reconnect instead of waiting for the
  // backoff timer.  Without them the user would stare at the loading
  // overlay for the remainder of the (paused) backoff after every
  // app-switch round-trip.
  if (
    typeof document !== 'undefined' &&
    typeof document.addEventListener === 'function'
  ) {
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        _reconnectNowIfNeeded();
      }
    });
  }
  if (
    typeof window !== 'undefined' &&
    typeof window.addEventListener === 'function'
  ) {
    // ``pageshow`` covers Safari's bfcache restore, which does not
    // fire ``visibilitychange``.
    window.addEventListener('pageshow', () => {
      _reconnectNowIfNeeded();
    });
    window.addEventListener('online', () => {
      _reconnectNowIfNeeded();
    });
    // ``focus`` is the universal fallback for older mobile browsers
    // that ignore visibilitychange/pageshow under certain conditions.
    window.addEventListener('focus', () => {
      _reconnectNowIfNeeded();
    });
  }

  connect();
})();
