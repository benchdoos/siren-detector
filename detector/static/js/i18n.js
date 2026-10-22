/**
 * Lightweight i18n module for Siren Detector UI.
 * Loads JSON locale files, applies translations via data-i18n attributes,
 * and exposes t() for JS string lookups.
 */
const I18n = (function() {
  const SUPPORTED = ['en', 'ru'];
  const DEFAULT = 'en';
  let _locale = DEFAULT;
  let _strings = {};
  let _fallback = {};

  function detectLocale() {
    const saved = localStorage.getItem('lang');
    if (saved && SUPPORTED.includes(saved)) return saved;
    const nav = (navigator.language || '').slice(0, 2);
    return SUPPORTED.includes(nav) ? nav : DEFAULT;
  }

  async function load(locale) {
    const base = '/static/i18n/';
    try {
      const [main, fb] = await Promise.all([
        fetch(base + locale + '.json').then(r => r.json()),
        locale !== DEFAULT ? fetch(base + DEFAULT + '.json').then(r => r.json()) : Promise.resolve({}),
      ]);
      _strings = main;
      _fallback = fb;
      _locale = locale;
      localStorage.setItem('lang', locale);
    } catch (e) {
      console.error('i18n: failed to load locale', locale, e);
    }
  }

  /** Look up a translation key. Returns the key itself if not found. */
  function t(key, params) {
    let str = _strings[key] || _fallback[key] || key;
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        str = str.replace(new RegExp('\\{' + k + '\\}', 'g'), v);
      }
    }
    return str;
  }

  /** Apply translations to all elements with data-i18n attribute. */
  function apply() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.dataset.i18n;
      const translated = t(key);
      if (translated === key) return;
      // Handle different element types
      if (el.tagName === 'INPUT' && el.type !== 'checkbox') {
        if (el.placeholder) el.placeholder = translated;
      } else if (el.tagName === 'OPTION') {
        el.textContent = translated;
      } else if (el.children.length > 0) {
        // Preserve child elements (e.g. <label> with <span class="s-help">)
        el.childNodes[0].textContent = translated + ' ';
      } else {
        el.textContent = translated;
      }
    });
    // data-i18n-html for elements with HTML content (tooltips)
    document.querySelectorAll('[data-i18n-html]').forEach(el => {
      const key = el.dataset.i18nHtml;
      const translated = t(key);
      if (translated !== key) el.innerHTML = translated;
    });
    // data-i18n-placeholder for inputs
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      el.placeholder = t(el.dataset.i18nPlaceholder);
    });
    // data-i18n-title for tooltips
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
      el.title = t(el.dataset.i18nTitle);
    });
    // Update html lang
    document.documentElement.lang = _locale;
  }

  /** Switch locale and re-apply. */
  async function setLocale(locale) {
    if (!SUPPORTED.includes(locale)) return;
    await load(locale);
    apply();
    // Dispatch event so pages can re-render dynamic content
    window.dispatchEvent(new CustomEvent('i18n:changed', { detail: { locale: _locale } }));
  }

  function getLocale() { return _locale; }
  function getSupported() { return [...SUPPORTED]; }

  /** Initialize: load locale and apply. Call after DOM ready. */
  async function init() {
    _locale = detectLocale();
    await load(_locale);
    apply();
  }

  return { t, apply, init, setLocale, getLocale, getSupported };
})();

// Shorthand
const t = I18n.t;
