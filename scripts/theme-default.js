// Seed light mode as default for first-time visitors.
// MyST book-theme uses 'myst:theme' in localStorage; if nothing is stored
// it falls back to prefers-color-scheme which may be dark on user systems.
// This script sets light as the explicit default without touching the DOM,
// so React hydration is never affected.
(function () {
  try {
    if (!localStorage.getItem('myst:theme')) {
      localStorage.setItem('myst:theme', 'light');
    }
  } catch (e) {}
})();
