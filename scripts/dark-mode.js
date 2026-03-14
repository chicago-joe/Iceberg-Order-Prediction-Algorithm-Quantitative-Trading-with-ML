// Force dark mode on initial load to activate Tailwind dark: variants
// Runs before first paint to prevent flash of white
(function () {
  document.documentElement.classList.add('dark');
  // Persist preference
  try { localStorage.setItem('theme', 'dark'); } catch (e) {}
})();
