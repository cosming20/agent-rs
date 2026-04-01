/**
 * Agentic RAG — Chat module.
 * Handles message sending, rendering, session management, and citations.
 */

let api, utils;
let sessionId = null;
let busy = false;

export function initChat(apiClient, shared) {
  api = apiClient;
  utils = shared;

  sessionId = sessionStorage.getItem('rag_session_id');
  if (!sessionId) {
    sessionId = utils.generateId();
    sessionStorage.setItem('rag_session_id', sessionId);
  }
  updateSessionBadge();

  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 180) + 'px';
  });

  sendBtn.addEventListener('click', sendMessage);

  document.getElementById('new-session').addEventListener('click', newConversation);

  // Suggestion chips
  document.getElementById('suggestions')?.addEventListener('click', e => {
    const chip = e.target.closest('.suggestion-chip');
    if (!chip) return;
    input.value = chip.textContent;
    sendMessage();
  });

  // Event delegation: citation toggles
  document.getElementById('chat-messages').addEventListener('click', e => {
    const toggle = e.target.closest('.citations-toggle');
    if (toggle) {
      toggle.closest('.message-citations').classList.toggle('expanded');
    }
  });
}


/* ═══════ SEND ═══════ */

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text || busy) return;

  busy = true;
  setSendLoading(true);
  hideWelcome();
  appendUserMessage(text);

  input.value = '';
  input.style.height = 'auto';

  showTyping();

  try {
    const res = await api.query(text, sessionId);
    hideTyping();
    appendAssistantMessage(res);
  } catch (err) {
    hideTyping();
    appendErrorMessage(err.message);
  } finally {
    busy = false;
    setSendLoading(false);
    input.focus();
  }
}


/* ═══════ RENDER MESSAGES ═══════ */

function appendUserMessage(text) {
  const el = msgEl('message--user');
  el.innerHTML = `<div class="message-bubble"><p>${utils.escapeHtml(text)}</p></div>`;
  appendAndScroll(el);
}

function appendAssistantMessage(res) {
  const el = msgEl('message--assistant');

  const confLevel = res.confidence >= 0.8 ? 'high' : res.confidence >= 0.5 ? 'medium' : 'low';
  const confPct = Math.round(res.confidence * 100);

  let citationsHtml = '';
  if (res.citations?.length) {
    const items = res.citations.map(c => {
      // Detect rank-based fake scores (1/n pattern) vs real similarity
      const isRankBased = isRankScore(c.score);
      const scoreDisplay = isRankBased
        ? `#${c.index}`
        : `${(c.score * 100).toFixed(0)}%`;
      return `
      <div class="citation">
        <div class="citation-header">
          <span class="citation-index">[${c.index}]</span>
          <span class="citation-source">${utils.escapeHtml(c.source)}</span>
          <span class="citation-score${isRankBased ? ' rank-based' : ''}">${scoreDisplay}</span>
        </div>
        <p class="citation-text">${utils.escapeHtml(c.content_snippet)}</p>
      </div>`;
    }).join('');

    citationsHtml = `
      <div class="message-citations">
        <button class="citations-toggle">
          <span>${res.citations.length} citation${res.citations.length !== 1 ? 's' : ''}</span>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>
        </button>
        <div class="citations-list">${items}</div>
      </div>`;
  }

  const groundedIcon = res.is_grounded
    ? '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>'
    : '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>';
  const groundedClass = res.is_grounded ? 'grounded' : 'ungrounded';
  const groundedLabel = res.is_grounded ? 'Grounded' : 'Ungrounded';

  el.innerHTML = `
    <div class="message-bubble">
      <div class="message-answer">${utils.renderMarkdown(res.answer)}</div>
      ${citationsHtml}
      <div class="message-meta">
        <span class="meta-confidence ${confLevel}">
          <span class="confidence-dot"></span>
          ${confPct}% confidence
        </span>
        <span class="meta-grounded ${groundedClass}">${groundedIcon} ${groundedLabel}</span>
        <span class="meta-strategy">${utils.escapeHtml(res.strategy)}</span>
      </div>
    </div>`;

  appendAndScroll(el);
}

function appendErrorMessage(message) {
  const el = msgEl('message--error');
  el.innerHTML = `
    <div class="message-bubble">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>
      </svg>
      ${utils.escapeHtml(message)}
    </div>`;
  appendAndScroll(el);
}


/* ═══════ TYPING INDICATOR ═══════ */

function showTyping() {
  const el = msgEl('message--typing');
  el.id = 'typing-indicator';
  el.innerHTML = `
    <div class="message-bubble">
      <div class="typing-indicator">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    </div>`;
  appendAndScroll(el);
}

function hideTyping() {
  document.getElementById('typing-indicator')?.remove();
}


/* ═══════ SESSION ═══════ */

function newConversation() {
  sessionId = utils.generateId();
  sessionStorage.setItem('rag_session_id', sessionId);
  updateSessionBadge();

  const container = document.getElementById('chat-messages');
  container.innerHTML = '';
  showWelcome(container);
}

function showWelcome(container) {
  container.innerHTML = `
    <div class="chat-welcome" id="chat-welcome">
      <div class="welcome-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>
          <path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/>
        </svg>
      </div>
      <h2>Ask anything about your knowledge base</h2>
      <p>Queries are processed through hybrid retrieval with citations and grounding verification.</p>
      <div class="suggestions" id="suggestions">
        <button class="suggestion-chip">What are the main topics in my knowledge base?</button>
        <button class="suggestion-chip">Summarize the most recent document</button>
        <button class="suggestion-chip">What entities are connected in the knowledge graph?</button>
      </div>
    </div>`;
}


/* ═══════ HELPERS ═══════ */

function hideWelcome() {
  const w = document.getElementById('chat-welcome');
  if (w) w.remove();
}

function updateSessionBadge() {
  const badge = document.getElementById('session-badge');
  if (badge && sessionId) badge.textContent = `Session: ${sessionId.slice(0, 8)}`;
}

function setSendLoading(on) {
  const btn = document.getElementById('chat-send');
  btn.classList.toggle('loading', on);
  btn.disabled = on;
}

function msgEl(className) {
  const el = document.createElement('div');
  el.className = `message ${className}`;
  return el;
}

function appendAndScroll(el) {
  const container = document.getElementById('chat-messages');
  container.appendChild(el);
  container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
}

function isRankScore(score) {
  // Detect rank-based 1/(n+1) fake scores: 1.0, 0.5, 0.333, 0.25, 0.2, 0.166...
  // Real cosine similarities are unlikely to be exact reciprocals of small integers
  if (score <= 0 || score > 1) return true;
  for (let n = 1; n <= 20; n++) {
    if (Math.abs(score - 1 / n) < 0.001) return true;
  }
  return false;
}
