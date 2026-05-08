/* Shared theme toggle — used by every page on the site. */
(() => {
  const sunIcon  = `<path d="M12 4V2m0 20v-2m8-8h2M2 12h2m13.66-5.66l1.41-1.41M4.93 19.07l1.41-1.41m0-11.31L4.93 4.93m14.14 14.14l-1.41-1.41M12 7a5 5 0 100 10 5 5 0 000-10z" stroke="currentColor" stroke-width="1.6" fill="none" stroke-linecap="round" />`;
  const moonIcon = `<path d="M21 12.79A9 9 0 1111.21 3a7 7 0 009.79 9.79z" />`;
  function applyTheme(t) {
    document.body.dataset.theme = t;
    const icon = document.getElementById('theme-icon');
    if (icon) icon.innerHTML = (t === 'dark') ? sunIcon : moonIcon;
    try { localStorage.setItem('theme', t); } catch (_) {}
  }
  const btn = document.getElementById('theme-toggle');
  if (btn) {
    btn.addEventListener('click', () => {
      const cur = document.body.dataset.theme || 'light';
      applyTheme(cur === 'dark' ? 'light' : 'dark');
    });
  }
  let saved = null;
  try { saved = localStorage.getItem('theme'); } catch (_) {}
  if (!saved) saved = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  applyTheme(saved);
})();
