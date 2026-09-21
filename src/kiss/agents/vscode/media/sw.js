// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

/* global self, caches, Headers, Response, __KISS_SW_SHELL__ */

// Service worker of the KISS Sorcar remote webapp.  It keeps the app
// shell — the page and the /media assets it loads — available when the
// connection to the server is slow, flaky or gone, so the app opens
// and stays on screen instead of the browser's "no internet" page.
//
// kiss.server.web_server serves this file at /sw.js after replacing
// __KISS_SW_SHELL__ with {"version": <hash>, "urls": [...]}: "/" plus
// every cache-busted "/media/<name>?v=<hash>" URL the rendered page
// references.  The manifest is part of the script, so any asset change
// yields a byte-different worker, a fresh install with a new cache
// name, and the old cache is dropped on activate.
//
// Strategies:
//   * /media/* (content-hashed URLs): cache first, network fallback.
//   * navigations to "/" (the app page): network first; when the
//     network fails, answers with a 5xx (the tunnel is down) or stays
//     silent for NAV_TIMEOUT_MS, the cached page is served instead
//     (tagged with a <meta name="kiss-offline-shell"> so it reloads
//     itself once the server answers again) and the network answer, if
//     it arrives later, refreshes the cache.
//   * everything else (/ws, /api/*, /trajectories, the voice model) is
//     not intercepted.
(function () {
  'use strict';

  const SHELL = __KISS_SW_SHELL__;
  const CACHE_PREFIX = 'kiss-shell-';
  const CACHE_NAME = CACHE_PREFIX + SHELL.version;
  const APP_PAGE = '/';
  const NAV_TIMEOUT_MS = 4000;

  function putInCache(key, response) {
    return caches.open(CACHE_NAME).then(cache => cache.put(key, response));
  }

  function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms, null));
  }

  function isStaleShellCache(name) {
    return name.indexOf(CACHE_PREFIX) === 0 && name !== CACHE_NAME;
  }

  // The cached page is served with this tag in its <head> so the page's
  // WebSocket shim (kiss.server.web_server._WS_SHIM_JS) knows it did not
  // come from the server and reloads once the server is reachable again.
  const OFFLINE_META = '<meta name="kiss-offline-shell" content="1">';

  function markOffline(cached) {
    const headers = new Headers(cached.headers);
    headers.delete('content-length');
    return cached.text().then(html => {
      const marked = html.replace(/<head[^>]*>/i, tag => tag + OFFLINE_META);
      return new Response(marked, {status: 200, headers});
    });
  }

  // Network first for the app page.  `network` settles with the server's
  // response; the race below gives up on it after NAV_TIMEOUT_MS (or on
  // error / 5xx) and serves the cached page.  The worker is kept alive
  // until the network answers so the cached page is refreshed even when
  // the timeout already answered the navigation.
  function networkFirstPage(event) {
    let refresh = null;
    const network = fetch(event.request).then(response => {
      if (response.ok) refresh = putInCache(APP_PAGE, response.clone());
      return response;
    });
    event.waitUntil(network.then(() => refresh).catch(() => null));
    const usable = network
      .then(response => (response.status < 500 ? response : null))
      .catch(() => null);
    return Promise.race([usable, delay(NAV_TIMEOUT_MS)]).then(response => {
      if (response) return response;
      return caches
        .match(APP_PAGE)
        .then(cached => (cached ? markOffline(cached) : network));
    });
  }

  // Cache first for the content-hashed /media assets.  A URL that is not
  // in the precache (a brand-new asset) is fetched once and stored.
  function cacheFirstMedia(event) {
    return caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request).then(response => {
        if (response.ok) {
          event.waitUntil(putInCache(event.request, response.clone()));
        }
        return response;
      });
    });
  }

  self.addEventListener('install', event => {
    event.waitUntil(
      caches
        .open(CACHE_NAME)
        .then(cache => cache.addAll(SHELL.urls))
        .then(() => self.skipWaiting()),
    );
  });

  self.addEventListener('activate', event => {
    event.waitUntil(
      caches
        .keys()
        .then(names =>
          Promise.all(
            names.filter(isStaleShellCache).map(name => caches.delete(name)),
          ),
        )
        .then(() => self.clients.claim()),
    );
  });

  self.addEventListener('fetch', event => {
    const request = event.request;
    if (request.method !== 'GET') return;
    const url = new URL(request.url);
    if (url.origin !== self.location.origin) return;
    if (request.mode === 'navigate') {
      if (url.pathname === APP_PAGE) event.respondWith(networkFirstPage(event));
      return;
    }
    if (url.pathname.indexOf('/media/') === 0) {
      event.respondWith(cacheFirstMedia(event));
    }
  });
})();
