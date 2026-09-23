/**
 * Opensource Generative AI client – auth, theme, top nav, helpers
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
  // Multipart: must NOT set Content-Type — the browser adds the boundary.
  async postForm(path, formData) {
    const headers = {};
    const token = localStorage.getItem(TOKEN_KEY);
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const res = await fetch(path, { method: 'POST', headers, body: formData });
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

function logout() {
  clearAuth();
  window.location.href = '/';
}

function requireAuth() {
  if (!isLoggedIn()) {
    // Login only exists as a popup on the landing page.
    // Keep the query string too, so e.g. a prompt typed on the landing page
    // (/generate?type=t2v&prompt=…) survives the login round-trip.
    window.location.href = '/?login=1&next=' + encodeURIComponent(location.pathname + location.search);
    return false;
  }
  return true;
}

function openLoginModal(next) {
  const modal = document.getElementById('login-modal');
  if (!modal) return;
  if (next) modal.dataset.next = next;
  modal.classList.remove('hidden');
  modal.classList.add('flex');
  document.body.classList.add('overflow-hidden');
  modal.querySelector('input[name="email"]')?.focus();
}

function closeLoginModal() {
  const modal = document.getElementById('login-modal');
  if (!modal) return;
  modal.classList.add('hidden');
  modal.classList.remove('flex');
  document.body.classList.remove('overflow-hidden');
}

function openRegisterModal(next) {
  const modal = document.getElementById('register-modal');
  if (!modal) return;
  if (next) modal.dataset.next = next;
  modal.classList.remove('hidden');
  modal.classList.add('flex');
  document.body.classList.add('overflow-hidden');
  modal.querySelector('input[name="name"]')?.focus();
}

function closeRegisterModal() {
  const modal = document.getElementById('register-modal');
  if (!modal) return;
  modal.classList.add('hidden');
  modal.classList.remove('flex');
  document.body.classList.remove('overflow-hidden');
}

function switchAuthModal(target) {
  const next = document.getElementById('login-modal')?.dataset.next
    || document.getElementById('register-modal')?.dataset.next;
  closeLoginModal();
  closeRegisterModal();
  if (target === 'register') openRegisterModal(next);
  else openLoginModal(next);
}

function initLoginModal() {
  const modal = document.getElementById('login-modal');
  const form = document.getElementById('login-modal-form');
  if (!modal || !form) return;

  document.getElementById('login-modal-close')?.addEventListener('click', closeLoginModal);
  document.getElementById('login-modal-backdrop')?.addEventListener('click', closeLoginModal);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !modal.classList.contains('hidden')) closeLoginModal();
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    try {
      const data = await API.post('/api/auth/login', {
        email: fd.get('email'),
        password: fd.get('password'),
      });
      setAuth(data.token, data.user);
      toast('Logged in', 'success');
      const next = modal.dataset.next || '/generate';
      window.location.href = next;
    } catch (err) {
      toast(err.message || 'Login failed', 'error');
    }
  });

  // Auto-open when linked here from a page that required auth, e.g. /?login=1&next=/jobs
  const params = new URLSearchParams(location.search);
  if (params.get('login') === '1') {
    openLoginModal(params.get('next') || '');
  }
}

function initRegisterModal() {
  const modal = document.getElementById('register-modal');
  const form = document.getElementById('register-modal-form');
  if (!modal || !form) return;

  document.getElementById('register-modal-close')?.addEventListener('click', closeRegisterModal);
  document.getElementById('register-modal-backdrop')?.addEventListener('click', closeRegisterModal);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !modal.classList.contains('hidden')) closeRegisterModal();
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    try {
      const data = await API.post('/api/auth/register', {
        name: fd.get('name'),
        email: fd.get('email'),
        password: fd.get('password'),
      });
      setAuth(data.token, data.user);
      toast('Account created', 'success');
      const next = modal.dataset.next || '/generate';
      window.location.href = next;
    } catch (err) {
      toast(err.message || 'Registration failed', 'error');
    }
  });

  const params = new URLSearchParams(location.search);
  if (params.get('register') === '1') {
    openRegisterModal(params.get('next') || '');
  }
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
    el.setAttribute('role', 'status');
    el.setAttribute('aria-live', 'polite');
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

function renderStudioNav() {
  const tabs = document.getElementById('studio-nav-tabs');
  const auth = document.getElementById('studio-nav-auth');
  if (!tabs || !auth) return;
  const user = getUser();
  const path = (location.pathname || '/').replace(/\/$/, '') || '/';
  const items = [['/', 'Explore'], ['/generate', 'Create']];
  if (user) {
    items.push(['/jobs', 'Jobs'], ['/library', 'Library']);
    if (user.role === 'admin') items.push(['/admin', 'Admin']);
  }
  tabs.innerHTML = items.map(([href, label]) => {
    const on = href === '/' ? path === '/' : path.startsWith(href);
    return `<a href="${href}" class="hf-tab${on ? ' active' : ''}"${on ? ' aria-current="page"' : ''}>${label}</a>`;
  }).join('');
  // On narrow screens the strip scrolls; bring the current tab into view.
  tabs.querySelector('.hf-tab.active')?.scrollIntoView({ inline: 'center', block: 'nearest' });

  if (user) {
    const name = user.name || user.email || '';
    const initial = (name.trim()[0] || '?').toUpperCase();
    auth.innerHTML = `
      <a href="/generate" class="hf-btn hf-btn-accent hf-hide-sm">Create</a>
      <div class="hf-user">
        <button type="button" class="hf-avatar" aria-haspopup="true" aria-expanded="false"
          title="${escapeHtml(name)}">${escapeHtml(initial)}</button>
        <div class="hf-menu" role="menu" hidden>
          <div class="hf-menu-head">
            <p class="hf-menu-name">${escapeHtml(name)}</p>
            <p class="hf-menu-sub">${escapeHtml(user.email || '')} · ${escapeHtml(user.role || '')}</p>
          </div>
          <a role="menuitem" href="/library">Library</a>
          <a role="menuitem" href="/jobs">My jobs</a>
          <button role="menuitem" type="button" onclick="logout()">Log out</button>
        </div>
      </div>`;
    const btn = auth.querySelector('.hf-avatar');
    const menu = auth.querySelector('.hf-menu');
    const close = () => { menu.hidden = true; btn.setAttribute('aria-expanded', 'false'); };
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      menu.hidden = !menu.hidden;
      btn.setAttribute('aria-expanded', String(!menu.hidden));
    });
    document.addEventListener('click', (e) => { if (!auth.contains(e.target)) close(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
  } else {
    auth.innerHTML = `
      <button type="button" class="hf-btn hf-btn-ghost" onclick="openLoginModal()">Log in</button>
      <button type="button" class="hf-btn hf-btn-accent" onclick="openRegisterModal()">Sign up</button>`;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  renderStudioNav();
  initLoginModal();
  initRegisterModal();
});
