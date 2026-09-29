// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The in-app PDF viewer, shared by the remote web app's content tabs
// (main.js renderBinaryContent) and the VS Code extension's PDF preview
// panel (src/pdfPreview.ts).
//
// The browser's own PDF plugin cannot be relied on: iOS Safari renders a
// PDF inside an <iframe> as a static picture of its first page (nothing
// scrolls), Android Chrome has no inline PDF plugin at all, and a VS Code
// webview never shows a PDF in a frame.  So the pages are drawn by
// pdf.js (pdfjs-dist, fetched from the same jsDelivr CDN that already
// serves the Monaco editor) onto <canvas> elements stacked in a plain
// scrollable box.  Every page box gets its final size up front, so the
// scrollbar is right before a single page is drawn; a page is rendered
// when it comes within a screen of the viewport, re-rendered when the
// zoom changes, and gives its bitmap back once it scrolls a screen away.
//
// Zoom: the initial scale fits the first page's width to the box ("fit
// width", kept across resizes until the user zooms); the toolbar's -/+
// step by a quarter, the percentage button returns to fit width; a
// two-finger pinch and Ctrl/Cmd + wheel zoom around the gesture's point.
//
// The toolbar also shows "Page N of M" for the page under the middle of
// the view (kept current while scrolling and zooming); N is a field, and
// typing a page number into it and pressing Enter jumps to that page.
// When the host provides a way to save the file there is a Download
// link as well.
//
// mountPdfViewer(holder, bytes, opts) is the only entry point; it returns
// {dispose} so the host can stop pending renders and free the document.

/* global Worker */

(function () {
  'use strict';

  const PDFJS_ROOT = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@6.3.289/';
  const PDFJS_BUILD = PDFJS_ROOT + 'legacy/build/';
  const ZOOM_STEP = 1.25;
  const MIN_SCALE = 0.2;
  const MAX_SCALE = 8;
  // Canvas backing stores above this many pixels are refused or blank on
  // iOS, so the device-pixel oversampling shrinks for huge pages.
  const MAX_CANVAS_PIXELS = 16 * 1024 * 1024;

  let pdfjsPromise = null;

  /**
   * Load pdf.js once: the library module, plus the worker script as a
   * blob: URL.  Every viewer starts its own worker from that URL (a
   * VS Code webview can only spawn workers from blob:/data: URLs, and a
   * cross-origin worker URL is refused everywhere): pdf.js tears the
   * worker down with the document, so two documents must not share
   * one.  A failed load is forgotten so the next viewer retries.
   */
  function loadPdfJs() {
    if (pdfjsPromise) return pdfjsPromise;
    pdfjsPromise = Promise.all([
      import(PDFJS_BUILD + 'pdf.min.mjs'),
      fetch(PDFJS_BUILD + 'pdf.worker.min.mjs').then(res => {
        if (!res.ok) throw new Error('pdf.js worker HTTP ' + res.status);
        return res.blob();
      }),
    ])
      .then(([lib, workerBlob]) => ({
        lib,
        workerUrl: URL.createObjectURL(
          new Blob([workerBlob], {type: 'text/javascript'}),
        ),
      }))
      .catch(err => {
        pdfjsPromise = null;
        throw err;
      });
    return pdfjsPromise;
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function clamp(value, lo, hi) {
    return Math.min(hi, Math.max(lo, value));
  }

  function touchDistance(touches) {
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.hypot(dx, dy);
  }

  /**
   * Mount a viewer for the PDF *bytes* (a Uint8Array; pdf.js takes the
   * buffer over, so pass a copy if the bytes are needed afterwards) into
   * *holder*.  opts.name labels the document.  The toolbar offers a
   * Download link when opts.downloadUrl (a URL of the same bytes, saved
   * under opts.name) or opts.onDownload (a callback: a VS Code webview
   * cannot download, so the panel asks the extension host to save a
   * copy) is given; the URL is also offered if the PDF cannot be shown.
   * Returns {dispose} which cancels pending renders and frees the
   * document; disposing twice is harmless.
   */
  function mountPdfViewer(holder, bytes, opts) {
    opts = opts || {};
    const name = opts.name || 'file.pdf';
    const root = el('div', 'pdf-viewer');
    const toolbar = el('div', 'pdf-toolbar');
    const zoomOut = el('button', 'pdf-zoom-out', '\u2212');
    zoomOut.type = 'button';
    zoomOut.title = 'Zoom out';
    const zoomLevel = el('button', 'pdf-zoom-level', '');
    zoomLevel.type = 'button';
    zoomLevel.title = 'Fit width';
    const zoomIn = el('button', 'pdf-zoom-in', '+');
    zoomIn.type = 'button';
    zoomIn.title = 'Zoom in';
    // "Loading…", then "Page [N] of M": N is the page under the view's
    // middle, in a field that jumps to the page typed into it on Enter.
    const status = el('span', 'pdf-status', 'Loading\u2026');
    const pageInput = el('input', 'pdf-page-input');
    pageInput.type = 'text';
    pageInput.inputMode = 'numeric';
    pageInput.enterKeyHint = 'go';
    pageInput.autocomplete = 'off';
    pageInput.title = 'Type a page number and press Enter';
    pageInput.setAttribute('aria-label', 'Page number');
    toolbar.append(zoomOut, zoomLevel, zoomIn, status);
    if (opts.downloadUrl || opts.onDownload) {
      const download = el('a', 'pdf-download', 'Download');
      download.title = 'Download ' + name;
      if (opts.downloadUrl) {
        download.href = opts.downloadUrl;
        download.download = name;
      } else {
        download.href = '#';
        download.addEventListener('click', ev => {
          ev.preventDefault();
          opts.onDownload();
        });
      }
      toolbar.appendChild(download);
    }
    const scroller = el('div', 'pdf-scroller');
    const pagesBox = el('div', 'pdf-pages');
    scroller.appendChild(pagesBox);
    root.append(toolbar, scroller);
    holder.appendChild(root);

    const state = {
      worker: null, // this viewer's Worker
      pdfWorker: null, // pdf.js's PDFWorker over it
      loadingTask: null,
      // {page, base, box, canvas, visible, renderTask, renderScale,
      //  renderedScale}: renderScale is the scale of the draw in
      // flight, renderedScale that of the canvas on show.
      pages: [],
      scale: 1,
      fitWidth: true,
      disposed: false,
      observer: null,
      resizeObserver: null,
      pinch: null, // {startDistance, startScale, focusX, focusY}
      wheelTimer: 0,
      scrollFrame: 0, // requestAnimationFrame id of a pending indicator update
    };

    function fail(err) {
      status.textContent = '';
      const note = el(
        'div',
        'content-binary-note',
        'Cannot display ' + name + ': ' + String(err),
      );
      scroller.replaceChildren(note);
      if (opts.downloadUrl) {
        const link = el('a', 'content-binary-note', 'Download ' + name);
        link.href = opts.downloadUrl;
        link.download = name;
        scroller.appendChild(link);
      }
    }

    /**
     * The 1-based number of the page under the vertical middle of the
     * view: the first page whose bottom edge lies below it (the next
     * page when the middle falls in a gap; the last page past the end).
     * A binary search over the page boxes keeps a long document cheap.
     */
    function currentPage() {
      const middle = scroller.scrollTop + scroller.clientHeight / 2;
      let lo = 0;
      let hi = state.pages.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        const entry = state.pages[mid];
        if (boxOffset(entry).top + entry.box.offsetHeight <= middle) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      return lo + 1;
    }

    /**
     * Show the page under the middle of the view in the page field,
     * unless the user is in the field (a number being typed must not be
     * overwritten by a scroll; the field catches up when it is left).
     */
    function updatePageIndicator() {
      state.scrollFrame = 0;
      if (state.disposed || !state.pages.length) return;
      if (document.activeElement === pageInput) return;
      pageInput.value = String(currentPage());
    }

    /** Refresh the indicator once per frame however often it scrolls. */
    function onScroll() {
      if (state.scrollFrame) return;
      state.scrollFrame = requestAnimationFrame(updatePageIndicator);
    }

    /**
     * Scroll so that page *number* (1-based; out-of-range numbers go to
     * the first or last page) starts at the top of the view, with the
     * same margin above it as the first page has at scroll offset 0.
     */
    function goToPage(number) {
      const entry = state.pages[clamp(number, 1, state.pages.length) - 1];
      const padding = parseFloat(window.getComputedStyle(pagesBox).paddingTop);
      scroller.scrollTop = boxOffset(entry).top - padding;
    }

    /** Select the whole number on focus so typing replaces it. */
    function onPageInputFocus() {
      pageInput.select();
    }

    /**
     * Enter jumps to the typed page (a non-number is ignored), Escape
     * abandons the edit; either leaves the field, which puts the page
     * under the middle of the view back into it.
     */
    function onPageInputKey(ev) {
      if (ev.key !== 'Enter' && ev.key !== 'Escape') return;
      ev.preventDefault();
      const number = parseInt(pageInput.value, 10);
      if (ev.key === 'Enter' && state.pages.length && !Number.isNaN(number)) {
        goToPage(number);
      }
      pageInput.blur();
    }

    /** The scale at which the first page's width fills the scroller. */
    function fitWidthScale() {
      const first = state.pages[0];
      const styles = window.getComputedStyle(pagesBox);
      const padding =
        parseFloat(styles.paddingLeft) + parseFloat(styles.paddingRight);
      const avail = scroller.clientWidth - padding;
      if (!first || avail <= 0) return 1;
      return clamp(avail / first.base.width, MIN_SCALE, MAX_SCALE);
    }

    /** Size every page box for state.scale (no drawing). */
    function layout() {
      for (const entry of state.pages) {
        entry.box.style.width =
          Math.floor(entry.base.width * state.scale) + 'px';
        entry.box.style.height =
          Math.floor(entry.base.height * state.scale) + 'px';
      }
      zoomLevel.textContent = Math.round(state.scale * 100) + '%';
    }

    function cancelRender(entry) {
      if (!entry.renderTask) return;
      entry.renderTask.cancel();
      entry.renderTask = null;
    }

    function dropCanvas(entry) {
      cancelRender(entry);
      if (entry.canvas) {
        entry.canvas.remove();
        entry.canvas = null;
      }
      entry.renderedScale = 0;
    }

    /**
     * Draw *entry*'s page at the current scale unless a canvas or a
     * draw in flight already has it; a draw in flight at another scale
     * is cancelled first, so it cannot land after a newer zoom.
     */
    function renderPage(entry) {
      if (state.disposed) return;
      const scale = state.scale;
      if (entry.renderTask && entry.renderScale === scale) return;
      cancelRender(entry);
      if (entry.renderedScale === scale) return;
      const viewport = entry.page.getViewport({scale});
      const dpr = window.devicePixelRatio || 1;
      const outputScale = Math.min(
        dpr,
        Math.sqrt(MAX_CANVAS_PIXELS / (viewport.width * viewport.height)),
      );
      const canvas = document.createElement('canvas');
      canvas.width = Math.floor(viewport.width * outputScale);
      canvas.height = Math.floor(viewport.height * outputScale);
      const task = entry.page.render({
        canvasContext: canvas.getContext('2d'),
        viewport,
        transform:
          outputScale === 1 ? null : [outputScale, 0, 0, outputScale, 0, 0],
      });
      entry.renderTask = task;
      entry.renderScale = scale;
      task.promise.then(
        () => {
          if (state.disposed || entry.renderTask !== task) return;
          entry.renderTask = null;
          entry.renderedScale = scale;
          if (entry.canvas) entry.canvas.remove();
          entry.canvas = canvas;
          entry.box.appendChild(canvas);
        },
        () => {
          // Cancelled by a newer render or by dispose: nothing to show.
          if (entry.renderTask === task) entry.renderTask = null;
        },
      );
    }

    /** Re-draw the pages currently within reach of the viewport. */
    function renderVisible() {
      for (const entry of state.pages) {
        if (entry.visible) renderPage(entry);
      }
    }

    /** A page box's top-left corner in scroller content coordinates. */
    function boxOffset(entry) {
      const rect = entry.box.getBoundingClientRect();
      const base = scroller.getBoundingClientRect();
      return {
        left: rect.left - base.left + scroller.scrollLeft,
        top: rect.top - base.top + scroller.scrollTop,
      };
    }

    /**
     * Zoom to *scale*, keeping the document point under (focusX, focusY)
     * (scroller-relative pixels) in place; *fitWidth* records whether the
     * scale still tracks the box width.  The point is anchored on the
     * page under it (the gaps and padding between pages do not scale,
     * so scaling the raw scroll offset would drift).  *draw* false only
     * re-sizes the boxes (a pinch in progress stretches the old
     * canvases; the final draw comes when the fingers lift).
     */
    function setScale(scale, fitWidth, focusX, focusY, draw) {
      scale = clamp(scale, MIN_SCALE, MAX_SCALE);
      const fx = focusX === undefined ? scroller.clientWidth / 2 : focusX;
      const fy = focusY === undefined ? scroller.clientHeight / 2 : focusY;
      const x = scroller.scrollLeft + fx;
      const y = scroller.scrollTop + fy;
      let anchor = null;
      for (const entry of state.pages) {
        const offset = boxOffset(entry);
        anchor = {
          entry,
          dx: (x - offset.left) / state.scale,
          dy: (y - offset.top) / state.scale,
        };
        if (y < offset.top + entry.box.offsetHeight) break;
      }
      state.scale = scale;
      state.fitWidth = fitWidth;
      layout();
      if (anchor) {
        const offset = boxOffset(anchor.entry);
        scroller.scrollLeft = offset.left + anchor.dx * scale - fx;
        scroller.scrollTop = offset.top + anchor.dy * scale - fy;
      }
      if (draw !== false) renderVisible();
      // The page under the middle can change without a scroll event
      // (the boxes grew or shrank around a clamped scroll offset).
      updatePageIndicator();
    }

    function onZoomOut() {
      setScale(state.scale / ZOOM_STEP, false);
    }

    function onZoomIn() {
      setScale(state.scale * ZOOM_STEP, false);
    }

    function onFitWidth() {
      setScale(fitWidthScale(), true);
    }

    function onWheel(ev) {
      if (!ev.ctrlKey && !ev.metaKey) return;
      ev.preventDefault();
      const rect = scroller.getBoundingClientRect();
      const factor = Math.exp(-ev.deltaY * 0.01);
      setScale(
        state.scale * factor,
        false,
        ev.clientX - rect.left,
        ev.clientY - rect.top,
        false,
      );
      // A trackpad pinch arrives as a burst of wheel events: draw once
      // the burst is over instead of on every tick.
      clearTimeout(state.wheelTimer);
      state.wheelTimer = setTimeout(renderVisible, 120);
    }

    function onTouchStart(ev) {
      if (ev.touches.length !== 2) return;
      const rect = scroller.getBoundingClientRect();
      state.pinch = {
        startDistance: touchDistance(ev.touches),
        startScale: state.scale,
        focusX: (ev.touches[0].clientX + ev.touches[1].clientX) / 2 - rect.left,
        focusY: (ev.touches[0].clientY + ev.touches[1].clientY) / 2 - rect.top,
      };
    }

    function onTouchMove(ev) {
      if (!state.pinch || ev.touches.length !== 2) return;
      // Keep the browser from panning (or zooming the page) with the two
      // fingers that are zooming the document.
      ev.preventDefault();
      const pinch = state.pinch;
      const scale =
        (pinch.startScale * touchDistance(ev.touches)) / pinch.startDistance;
      setScale(scale, false, pinch.focusX, pinch.focusY, false);
    }

    function onTouchEnd(ev) {
      if (!state.pinch || ev.touches.length >= 2) return;
      state.pinch = null;
      renderVisible();
    }

    function onIntersect(entries) {
      for (const item of entries) {
        const entry = state.pages[Number(item.target.dataset.pageIndex)];
        if (!entry) continue;
        entry.visible = item.isIntersecting;
        // A page a screen or more away gives its canvas back: a long
        // document would otherwise hold hundreds of bitmaps.
        if (entry.visible) renderPage(entry);
        else dropCanvas(entry);
      }
    }

    function onResize() {
      if (state.fitWidth && state.pages.length) onFitWidth();
      // Otherwise the pages stay put, but the view's middle moved with
      // its height (no scroll event for that).
      else updatePageIndicator();
    }

    function show(pdfjs) {
      if (state.disposed) return undefined;
      state.worker = new Worker(pdfjs.workerUrl, {type: 'module'});
      state.pdfWorker = pdfjs.lib.PDFWorker.create({port: state.worker});
      state.loadingTask = pdfjs.lib.getDocument({
        data: bytes,
        worker: state.pdfWorker,
        // Decoders and data pdf.js loads on demand: CJK character maps,
        // the 14 standard fonts, JPEG 2000 / JBIG2 decoders, ICC
        // profiles.  Without them such content renders blank.
        cMapUrl: PDFJS_ROOT + 'cmaps/',
        cMapPacked: true,
        standardFontDataUrl: PDFJS_ROOT + 'standard_fonts/',
        wasmUrl: PDFJS_ROOT + 'wasm/',
        iccUrl: PDFJS_ROOT + 'iccs/',
      });
      return state.loadingTask.promise
        .then(doc => {
          if (state.disposed) return;
          const gets = [];
          for (let i = 1; i <= doc.numPages; i++) gets.push(doc.getPage(i));
          return Promise.all(gets);
        })
        .then(pages => {
          if (!pages || state.disposed) return;
          state.observer = new IntersectionObserver(onIntersect, {
            root: scroller,
            rootMargin: '100% 0px',
          });
          pages.forEach((page, index) => {
            const box = el('div', 'pdf-page');
            box.dataset.pageIndex = String(index);
            pagesBox.appendChild(box);
            state.pages.push({
              page,
              base: page.getViewport({scale: 1}),
              box,
              canvas: null,
              visible: false,
              renderTask: null,
              renderScale: 0,
              renderedScale: 0,
            });
            state.observer.observe(box);
          });
          state.scale = fitWidthScale();
          layout();
          // Room for the largest page number plus the field's padding.
          pageInput.style.width = String(pages.length).length + 2 + 'ch';
          status.replaceChildren('Page ', pageInput, ' of ' + pages.length);
          updatePageIndicator();
          state.resizeObserver = new ResizeObserver(onResize);
          state.resizeObserver.observe(scroller);
        });
    }

    zoomOut.addEventListener('click', onZoomOut);
    zoomIn.addEventListener('click', onZoomIn);
    zoomLevel.addEventListener('click', onFitWidth);
    pageInput.addEventListener('focus', onPageInputFocus);
    pageInput.addEventListener('keydown', onPageInputKey);
    pageInput.addEventListener('blur', updatePageIndicator);
    scroller.addEventListener('scroll', onScroll, {passive: true});
    scroller.addEventListener('wheel', onWheel, {passive: false});
    scroller.addEventListener('touchstart', onTouchStart, {passive: true});
    scroller.addEventListener('touchmove', onTouchMove, {passive: false});
    scroller.addEventListener('touchend', onTouchEnd);
    scroller.addEventListener('touchcancel', onTouchEnd);

    loadPdfJs()
      .then(show)
      .catch(err => {
        if (!state.disposed) fail(err);
      });

    function dispose() {
      if (state.disposed) return;
      state.disposed = true;
      clearTimeout(state.wheelTimer);
      cancelAnimationFrame(state.scrollFrame);
      if (state.observer) state.observer.disconnect();
      if (state.resizeObserver) state.resizeObserver.disconnect();
      for (const entry of state.pages) cancelRender(entry);
      // Free the document on both sides of the port, then drop pdf.js's
      // handle on the port and stop the viewer's own worker.
      const worker = state.worker;
      const pdfWorker = state.pdfWorker;
      if (state.loadingTask) {
        state.loadingTask
          .destroy()
          .catch(() => {})
          .then(() => {
            pdfWorker.destroy();
            worker.terminate();
          });
      }
      root.remove();
    }

    return {dispose};
  }

  window.mountPdfViewer = mountPdfViewer;
})();
