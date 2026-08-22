/**
 * Opensource Generative AI client – auth, theme, sidebar, helpers
 */
const TOKEN_KEY = 'genai_token';
const USER_KEY = 'genai_user';

const API = {
  async request(path, opts = {}) {
    const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
    const token = localStorage.getItem(TOKEN_KEY);
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const res = await fetch(path, { ...opts, headers });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(
        typeof data.detail === 'string'
          ? data.detail
          : (data.detail?.[0]?.msg || data.message || res.statusText)
      );
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  },
  get: (p) => API.request(p),
  post: (p, body) => API.request(p, { method: 'POST', body: JSON.stringify(body || {}) }),
  put: (p, body) => API.request(p, { method: 'PUT', body: JSON.stringify(body || {}) }),
  patch: (p, body) => API.request(p, { method: 'PATCH', body: JSON.stringify(body || {}) }),
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

function navLink(href, label, icon) {
  const active = location.pathname === href || (href !== '/' && location.pathname.startsWith(href));
  return `<a href="${href}" class="sidebar-link flex items-center gap-2.5 px-3 py-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition ${active ? 'active' : ''}">
    <span class="opacity-70">${icon || '•'}</span><span>${label}</span>
  </a>`;
}

function renderNav() {
  const side = document.getElementById('sidebar-nav');
  const sideAuth = document.getElementById('sidebar-auth');
  const authArea = document.getElementById('auth-area');
  const user = getUser();
  const path = location.pathname;

  const icons = {
    home: '🏠',
    gen: '✨',
    jobs: '📋',
    admin: '⚙️',
    brand: '🎨',
    server: '🖥️',
    users: '👥',
  };

  if (side) {
    let html = navLink('/', 'Home', icons.home);
    if (user) {
      html += navLink('/generate', 'Generate', icons.gen);
      html += navLink('/jobs', 'My Jobs', icons.jobs);
      if (user.role === 'admin') {
        html += `<div class="pt-3 pb-1 px-3 text-[10px] uppercase tracking-wider text-slate-400 font-semibold">Admin</div>`;
        html += navLink('/admin', 'Dashboard', icons.admin);
        html += navLink('/admin/branding', 'Branding', icons.brand);
        html += navLink('/admin/server', 'Server & Queue', icons.server);
        html += navLink('/admin/users', 'Users', icons.users);
      }
    } else {
      html += navLink('/login', 'Log in', '→');
      html += navLink('/register', 'Sign up', '+');
    }
    side.innerHTML = html;
  }

  if (sideAuth) {
    if (user) {
      sideAuth.innerHTML = `
        <div class="px-3 py-2 rounded-lg bg-slate-50 dark:bg-slate-800/80">
          <p class="font-medium truncate text-sm">${escapeHtml(user.name || user.email)}</p>
          <p class="text-[11px] text-slate-500 truncate">${escapeHtml(user.email || '')} · ${user.role}</p>
        </div>
        <button type="button" onclick="logout()" class="w-full mt-2 text-sm px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800">Log out</button>
      `;
    } else {
      sideAuth.innerHTML = `
        <a href="/login" class="block text-center text-sm px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 mb-2">Log in</a>
        <a href="/register" class="block text-center text-sm px-3 py-2 rounded-lg btn-brand">Sign up</a>
      `;
    }
  }

  if (authArea) {
    authArea.innerHTML = '';
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
  d.textContent = s == null ? '' : String(s);
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
    cancelled: 'bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300',
    canceled: 'bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300',
  };
  return `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${map[status] || map.queued}">${status}</span>`;
}

function typeLabel(t) {
  const map = { t2v: 'Text→Video', i2v: 'Image→Video', t2i: 'Text→Image', i2i: 'Image→Image', ia2v: 'Image+Audio→Video', v2v: 'Video→Video' };
  return map[t] || t;
}

function isLandingPage() {
  const p = (location.pathname || '/').replace(/\/$/, '') || '/';
  return p === '/';
}

function applyLayoutMode() {
  const landing = isLandingPage();
  document.body.classList.toggle('is-landing', landing);
  document.documentElement.classList.remove('landing-pending');
  const main = document.getElementById('main-column');
  if (main) {
    if (landing) {
      main.classList.remove('md:pl-64');
    } else {
      main.classList.add('md:pl-64');
    }
  }
  const sidebar = document.getElementById('sidebar');
  if (landing) {
    sidebar?.classList.add('hidden');
    sidebar?.classList.remove('flex');
  } else if (window.matchMedia('(min-width: 768px)').matches) {
    sidebar?.classList.remove('hidden');
    sidebar?.classList.add('flex');
  }
}

function initSidebarMobile() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebar-overlay');
  const openBtn = document.getElementById('mobile-menu-btn');
  const open = () => {
    if (isLandingPage()) return;
    sidebar?.classList.remove('hidden');
    sidebar?.classList.add('flex');
    overlay?.classList.remove('hidden');
  };
  const close = () => {
    if (window.matchMedia('(min-width: 768px)').matches && !isLandingPage()) return;
    sidebar?.classList.add('hidden');
    sidebar?.classList.remove('flex');
    overlay?.classList.add('hidden');
  };
  openBtn?.addEventListener('click', open);
  overlay?.addEventListener('click', close);
}

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  applyLayoutMode();
  document.getElementById('theme-toggle')?.addEventListener('click', toggleTheme);
  renderNav();
  initSidebarMobile();
});
