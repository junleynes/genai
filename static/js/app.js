/**
 * WanForge client – auth, theme, nav, helpers
 */
const TOKEN_KEY = 'wanforge_token';
const USER_KEY = 'wanforge_user';

const API = {
  async request(path, opts = {}) {
    const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
    const token = localStorage.getItem(TOKEN_KEY);
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const res = await fetch(path, { ...opts, headers });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(data.detail || data.message || res.statusText);
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  },
  get: (p) => API.request(p),
  post: (p, body) => API.request(p, { method: 'POST', body: JSON.stringify(body) }),
  put: (p, body) => API.request(p, { method: 'PUT', body: JSON.stringify(body) }),
  patch: (p, body) => API.request(p, { method: 'PATCH', body: JSON.stringify(body) }),
  del: (p) => API.request(p, { method: 'DELETE' }),
};

function getUser() {
  try { return JSON.parse(localStorage.getItem(USER_KEY) || 'null'); } catch { return null; }
}
function setAuth(token, user) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}
function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}
function isLoggedIn() { return !!localStorage.getItem(TOKEN_KEY); }
function isAdmin() { const u = getUser(); return u && u.role === 'admin'; }

// Theme
function initTheme() {
  const stored = localStorage.getItem('theme');
  const preferDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const dark = stored === 'dark' || (!stored && preferDark);
  document.documentElement.classList.toggle('dark', dark);
}
function toggleTheme() {
  const isDark = document.documentElement.classList.toggle('dark');
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

// Nav
function renderNav() {
  const links = document.getElementById('nav-links');
  const authArea = document.getElementById('auth-area');
  if (!links || !authArea) return;

  const user = getUser();
  if (user) {
    links.innerHTML = `
      <a href="/generate" class="px-3 py-2 rounded-lg text-sm font-medium hover:bg-slate-200/70 dark:hover:bg-slate-800 transition">Generate</a>
      <a href="/jobs" class="px-3 py-2 rounded-lg text-sm font-medium hover:bg-slate-200/70 dark:hover:bg-slate-800 transition">My Jobs</a>
      ${user.role === 'admin' ? `<a href="/admin" class="px-3 py-2 rounded-lg text-sm font-medium hover:bg-slate-200/70 dark:hover:bg-slate-800 transition">Admin</a>` : ''}
    `;
    authArea.innerHTML = `
      <span class="hidden sm:inline text-sm text-slate-600 dark:text-slate-300">${escapeHtml(user.name)}</span>
      <span class="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-800 text-slate-600 dark:text-slate-400">${user.role}</span>
      <button onclick="logout()" class="text-sm px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 transition">Logout</button>
    `;
  } else {
    links.innerHTML = '';
    authArea.innerHTML = `
      <a href="/login" class="text-sm px-3 py-1.5 rounded-lg hover:bg-slate-200/70 dark:hover:bg-slate-800 transition">Log in</a>
      <a href="/register" class="text-sm px-3 py-1.5 rounded-lg btn-brand shadow-sm">Sign up</a>
    `;
  }
}

function logout() {
  clearAuth();
  window.location.href = '/';
}

function requireAuth(redirect = '/login') {
  if (!isLoggedIn()) {
    window.location.href = redirect + '?next=' + encodeURIComponent(location.pathname);
    return false;
  }
  return true;
}

function requireAdmin() {
  if (!requireAuth()) return false;
  if (!isAdmin()) {
    window.location.href = '/';
    return false;
  }
  return true;
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function toast(msg, type = 'info') {
  let el = document.getElementById('toast');
  if (!el) {
    el = document.createElement('div');
    el.id = 'toast';
    el.className = 'fixed bottom-6 right-6 z-[100] max-w-sm px-4 py-3 rounded-xl shadow-xl text-sm font-medium transition-all duration-300 translate-y-4 opacity-0';
    document.body.appendChild(el);
  }
  const colors = {
    info: 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900',
    success: 'bg-emerald-600 text-white',
    error: 'bg-red-600 text-white',
  };
  el.className = `fixed bottom-6 right-6 z-[100] max-w-sm px-4 py-3 rounded-xl shadow-xl text-sm font-medium transition-all duration-300 ${colors[type] || colors.info}`;
  el.textContent = msg;
  requestAnimationFrame(() => {
    el.classList.remove('translate-y-4', 'opacity-0');
  });
  clearTimeout(el._t);
  el._t = setTimeout(() => {
    el.classList.add('translate-y-4', 'opacity-0');
  }, 3200);
}

function statusBadge(status) {
  const map = {
    queued: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300',
    processing: 'bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300',
    completed: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300',
    failed: 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300',
  };
  return `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${map[status] || map.queued}">${status}</span>`;
}

function typeLabel(t) {
  const map = { t2v: 'Text→Video', i2v: 'Image→Video', t2i: 'Text→Image', i2i: 'Image→Image', ia2v: 'Image+Audio→Video', v2v: 'Video→Video' };
  return map[t] || t;
}

// Init
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  document.getElementById('theme-toggle')?.addEventListener('click', toggleTheme);
  renderNav();
});
