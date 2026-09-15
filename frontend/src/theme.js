// Theme persistence: 'dark' (default, Dimension) or 'light' (#fafafa/#ffffff).
// The attribute lives on <html> so every page — including auth screens outside
// the Layout shell — picks it up.

const KEY = 'ng-theme'

export function getTheme() {
  return localStorage.getItem(KEY) === 'light' ? 'light' : 'dark'
}

export function applyTheme(theme) {
  document.documentElement.dataset.theme = theme
  localStorage.setItem(KEY, theme)
}

export function initTheme() {
  applyTheme(getTheme())
}

export function toggleTheme() {
  const next = getTheme() === 'dark' ? 'light' : 'dark'
  applyTheme(next)
  return next
}
