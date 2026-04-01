/**
 * Agentic RAG — Main application module.
 * Entry point: tab routing, API client, health check, init.
 */

import { initChat } from './chat.js';
import { initIngest } from './ingest.js';

/* ═══════════════════════════════════════════
   API CLIENT
   ═══════════════════════════════════════════ */

class ApiClient {
  async query(text, sessionId, strategy = 'hybrid', limit = 10) {
    const res = await fetch('/api/v1/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: text,
        session_id: sessionId,
        strategy,
        limit,
      }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Query failed (${res.status})`);
    }
    return res.json();
  }

  async upload(file) {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch('/api/v1/upload', {
      method: 'POST',
      body: form,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Upload failed (${res.status})`);
    }
    return res.json();
  }

  async health() {
    const res = await fetch('/health', { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error(`Health check failed (${res.status})`);
    return res.json();
  }
}


/* ═══════════════════════════════════════════
   UTILITIES
   ═══════════════════════════════════════════ */

function generateId() {
  return crypto.randomUUID();
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function escapeHtml(text) {
  const el = document.createElement('span');
  el.textContent = text;
  return el.innerHTML;
}

function renderMarkdown(text) {
  if (typeof marked !== 'undefined') {
    return marked.parse(text, { breaks: true });
  }
  return escapeHtml(text).replace(/\n/g, '<br>');
}


/* ═══════════════════════════════════════════
   TAB ROUTING
   ═══════════════════════════════════════════ */

function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  const views = document.querySelectorAll('.view');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      tabs.forEach(t => t.classList.remove('active'));
      views.forEach(v => v.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(`view-${target}`).classList.add('active');
    });
  });
}


/* ═══════════════════════════════════════════
   HEALTH CHECK
   ═══════════════════════════════════════════ */

function startHealthCheck(api) {
  const dot = document.getElementById('health-dot');
  const text = document.getElementById('health-text');

  async function check() {
    try {
      await api.health();
      dot.className = 'health-dot healthy';
      text.textContent = 'Connected';
    } catch {
      dot.className = 'health-dot unhealthy';
      text.textContent = 'Disconnected';
    }
  }

  check();
  setInterval(check, 30_000);
}


/* ═══════════════════════════════════════════
   INIT
   ═══════════════════════════════════════════ */

const api = new ApiClient();

initTabs();
startHealthCheck(api);
initChat(api, { generateId, escapeHtml, renderMarkdown });
initIngest(api, { formatFileSize });
