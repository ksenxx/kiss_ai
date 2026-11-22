// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/* global screen */
// Tells the stylesheet how round the corners of an iPhone's screen are,
// so the composer's bottom corners can follow them (the
// `.display-corners` rules of remote-codex.css).
//
// A page opened from an iPhone's Home Screen icon has the whole glass
// to itself, and the composer sits in its two bottom corners.  A box
// looks right there only when its corners are CONCENTRIC with the
// display's: one gap to the left, right and bottom edge, and a radius
// of the display's radius minus that gap.  No web API tells the
// display's radius (CSSWG issue 6259, open since 2021), so it comes
// from the table below, keyed on what the phone does report: its
// screen size in CSS px and its pixel ratio.
//
// Only a Home Screen web app on a listed iPhone is touched
// (`navigator.standalone`, which iOS alone defines): in a Safari tab
// the browser's toolbar lies between the page and the corners.  While
// the phone is held upright the page asks for the whole screen
// (`viewport-fit=cover`), without which iOS reports no bottom inset to
// keep the home indicator clear of, and <html> carries the class
// `display-corners`.  Held sideways the composer is nowhere near a
// corner: both are dropped and the ordinary layout applies.
//
// _build_html inlines this file in <head> (the HEAD_SCRIPT placeholder
// of media/chat.html), so the first paint already has the final shape.
(function () {
  'use strict';

  // [width, height, pixel ratio, corner radius]: the screen held
  // upright, in CSS px.  Apple publishes no radius; these are the
  // values iOS itself uses (UIScreen's private _displayCornerRadius)
  // as listed per model by github.com/markbattistella/BezelKit
  // (SupportedDeviceList.md, read 2026-11-21).  Phones that share a
  // row share a radius, except 375x812, where the minis have 44 and
  // the X, XS and 11 Pro 39: the larger one is used, which only widens
  // the gap in the bend (the smaller one would bring the box's corner
  // closer to the glass's).  A phone that is not listed keeps the
  // ordinary layout.
  const DISPLAYS = [
    [402, 874, 3, 62], // 16 Pro, 17, 17 Pro, 18 Pro
    [440, 956, 3, 62], // 16 Pro Max, 17 Pro Max, 18 Pro Max
    [420, 912, 3, 62], // Air
    [393, 852, 3, 55], // 14 Pro, 15, 15 Pro, 16
    [430, 932, 3, 55], // 14 Pro Max, 15 Plus, 15 Pro Max, 16 Plus
    [428, 926, 3, 53.33], // 12 Pro Max, 13 Pro Max, 14 Plus
    [390, 844, 3, 47.33], // 12, 12 Pro, 13, 13 Pro, 14, 16e, 17e
    [375, 812, 3, 44], // 12 mini, 13 mini; X, XS, 11 Pro
    [414, 896, 3, 39], // XS Max, 11 Pro Max
    [414, 896, 2, 41.5], // XR, 11
  ];

  if (!window.navigator.standalone) return;

  const width = Math.min(screen.width, screen.height);
  const height = Math.max(screen.width, screen.height);
  const display = DISPLAYS.find(
    d => d[0] === width && d[1] === height && d[2] === window.devicePixelRatio,
  );
  if (!display) return;

  const root = document.documentElement;
  const viewport = document.querySelector('meta[name="viewport"]');
  const ordinary = viewport.content;
  const upright = window.matchMedia('(orientation: portrait)');
  root.style.setProperty('--display-radius', display[3] + 'px');

  /** Take the whole screen and follow its corners while held upright. */
  function followCorners() {
    viewport.content = upright.matches
      ? ordinary + ',viewport-fit=cover'
      : ordinary;
    root.classList.toggle('display-corners', upright.matches);
  }

  followCorners();
  upright.addEventListener('change', followCorners);
})();
