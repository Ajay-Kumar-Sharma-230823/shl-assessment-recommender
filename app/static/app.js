'use strict';

/* ── Sample Queries ─────────────────────────────────── */
const SAMPLES = {
  java: {
    role: 'Java Developer', level: 'Senior',
    focus: 'Cognitive ability, coding, and personality',
    remote: true, fast: false,
    prompt: 'Senior Java developer for a fintech company. Need coding assessment, cognitive ability reasoning, and personality evaluation. Remote testing is required.'
  },
  sales: {
    role: 'Sales Associate', level: 'Graduate',
    focus: 'Sales judgment, motivation, and communication',
    remote: true, fast: false,
    prompt: 'Graduate sales hiring program for a national intake. Need sales potential, motivation, communication, and situational judgment assessments. Remote testing preferred.'
  },
  support: {
    role: 'Customer Support Representative', level: 'Mid-level',
    focus: 'Customer service, language, and situational judgment',
    remote: true, fast: true,
    prompt: 'Customer support representative for a high-volume contact center. Need customer service judgment, language communication, and a short screening under 45 minutes.'
  }
};

/* ── State ──────────────────────────────────────────── */
const state = {
  messages: [],
  busy: false,
  turns: 0
};

/* ── DOM Refs ────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const el = {
  healthStatus:     $('healthStatus'),
  roleInput:        $('roleInput'),
  levelSelect:      $('levelSelect'),
  focusSelect:      $('focusSelect'),
  remoteCheck:      $('remoteCheck'),
  fastCheck:        $('fastCheck'),
  buildBriefBtn:    $('buildBriefBtn'),
  promptInput:      $('promptInput'),
  chatForm:         $('chatForm'),
  messageStream:    $('messageStream'),
  recommendationList: $('recommendationList'),
  emptyResults:     $('emptyResults'),
  resetBtn:         $('resetBtn'),
  sendBtn:          $('sendBtn'),
  sendLabel:        $('sendLabel'),
  turnBadge:        $('turnBadge'),
  chatSub:          $('chatSub'),
  roleSignal:       $('roleSignal'),
  focusSignal:      $('focusSignal'),
  constraintSignal: $('constraintSignal'),
  contextScore:     $('contextScore'),
  scoreRingFill:    $('scoreRingFill')
};

/* ── Utilities ───────────────────────────────────────── */
function esc(v) {
  return String(v)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ── Health Check ────────────────────────────────────── */
async function checkHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    const ok = res.ok && data.status === 'ok';
    el.healthStatus.classList.toggle('is-ok', ok);
    el.healthStatus.classList.toggle('is-down', !ok);
    el.healthStatus.querySelector('.status-label').textContent = ok ? 'API Ready' : 'API Offline';
  } catch {
    el.healthStatus.classList.add('is-down');
    el.healthStatus.classList.remove('is-ok');
    el.healthStatus.querySelector('.status-label').textContent = 'API Offline';
  }
}

/* ── Turn Badge ─────────────────────────────────────── */
function updateTurnBadge() {
  el.turnBadge.textContent = `${state.turns} / 8 turns`;
  el.turnBadge.style.background = state.turns >= 7 ? '#fef2f2' : '';
  el.turnBadge.style.borderColor = state.turns >= 7 ? '#fecaca' : '';
  el.turnBadge.style.color      = state.turns >= 7 ? '#dc2626' : '';
}

/* ── Score Ring ─────────────────────────────────────── */
function updateScore(score) {
  const pct = Math.max(0, Math.min(100, score));
  const circumference = 314;
  const offset = circumference - (pct / 100) * circumference;
  el.scoreRingFill.style.strokeDashoffset = offset;
  el.contextScore.textContent = `${pct}%`;

  // Color shift: low=teal, high=green
  const color = pct >= 70 ? '#10b981' : pct >= 40 ? '#087f8c' : '#94a3b8';
  el.scoreRingFill.style.stroke = color;
}

/* ── Signal Panel ────────────────────────────────────── */
function analyzeSignal(text) {
  const low = text.toLowerCase();
  const role = el.roleInput.value.trim() || '—';

  const focus = [];
  if (/coding|java|python|javascript|programming/.test(low)) focus.push('Coding');
  if (/cognitive|reasoning|numerical|verbal|aptitude/.test(low)) focus.push('Cognitive');
  if (/personality|opq|behaviour|behavior/.test(low)) focus.push('Personality');
  if (/sales|motivation/.test(low)) focus.push('Sales');
  if (/language|communication|grammar/.test(low)) focus.push('Language');
  if (/situational|judgment|sjt/.test(low)) focus.push('SJT');

  const constraints = [];
  if (/remote|online/.test(low)) constraints.push('Remote');
  if (/45|short|under/.test(low)) constraints.push('≤45 min');
  if (/graduate|entry|junior/.test(low)) constraints.push('Early career');

  const score = Math.min(96, 30 + focus.length * 14 + constraints.length * 9 + (role.length > 4 ? 15 : 0) + (state.turns > 0 ? 10 : 0));

  el.roleSignal.textContent = role;
  el.focusSignal.textContent = focus.length ? focus.join(' + ') : '—';
  el.constraintSignal.textContent = constraints.length ? constraints.join(', ') : 'None';
  updateScore(score);
}

/* ── Messages ─────────────────────────────────────────── */
function appendMessage(role, content, type = '') {
  // Remove welcome card on first message
  const welcome = el.messageStream.querySelector('.welcome-card');
  if (welcome) welcome.remove();

  const article = document.createElement('article');
  article.className = `msg ${role} ${type}`.trim();
  article.innerHTML = `
    <div class="msg-role">${role === 'user' ? 'You' : type === 'thinking' ? '⏳ Advisor' : '✦ Advisor'}</div>
    <div class="msg-body">${esc(content)}</div>
  `;
  el.messageStream.appendChild(article);
  el.messageStream.scrollTop = el.messageStream.scrollHeight;
  return article;
}

function appendThinking() {
  const article = document.createElement('article');
  article.className = 'msg assistant thinking';
  article.innerHTML = `
    <div class="msg-role">⏳ Advisor</div>
    <div class="typing-dots"><span></span><span></span><span></span></div>
  `;
  el.messageStream.appendChild(article);
  el.messageStream.scrollTop = el.messageStream.scrollHeight;
  return article;
}

/* ── Recommendations ─────────────────────────────────── */
function renderRecommendations(recs) {
  el.recommendationList.innerHTML = '';
  el.emptyResults.hidden = recs.length > 0;

  recs.forEach((rec, i) => {
    const card = document.createElement('div');
    card.className = 'rec-card';
    card.innerHTML = `
      <div class="rec-rank">#${i + 1}</div>
      <div class="rec-name">${esc(rec.name)}</div>
      <div class="rec-tags">
        <span class="rec-tag">${esc(rec.test_type || 'Assessment')}</span>
        <span class="rec-tag validated">✓ Catalog</span>
      </div>
      <a class="rec-link" href="${esc(rec.url)}" target="_blank" rel="noreferrer">
        View on SHL
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
      </a>
    `;
    el.recommendationList.appendChild(card);
  });
}

/* ── Reset ───────────────────────────────────────────── */
function resetConversation() {
  state.messages = [];
  state.turns = 0;
  el.messageStream.innerHTML = `
    <div class="welcome-card">
      <div class="welcome-icon">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
      </div>
      <h3>Hello! I'm your SHL Assessment Advisor</h3>
      <p>Describe the role, seniority level, and assessment needs. I'll return catalog-grounded SHL recommendations.</p>
      <div class="welcome-chips">
        <span class="w-chip">🧠 Cognitive</span>
        <span class="w-chip">💻 Technical</span>
        <span class="w-chip">🤝 Personality</span>
        <span class="w-chip">🎯 Situational</span>
      </div>
    </div>`;
  renderRecommendations([]);
  updateTurnBadge();
  updateScore(0);
  el.roleSignal.textContent = '—';
  el.focusSignal.textContent = '—';
  el.constraintSignal.textContent = 'None';
  el.chatSub.textContent = 'Ready to recommend assessments';
}

/* ── Build Brief ─────────────────────────────────────── */
function buildBriefPrompt() {
  const constraints = [];
  if (el.remoteCheck.checked) constraints.push('Remote testing required');
  if (el.fastCheck.checked) constraints.push('Under 45 minutes preferred');
  const brief = `${el.levelSelect.value} ${el.roleInput.value}. Need ${el.focusSelect.value.toLowerCase()} assessments. ${constraints.join('. ')} Recommend the best SHL catalog options.`;
  el.promptInput.value = brief.replace(/\s+/g, ' ').trim();
  analyzeSignal(el.promptInput.value);
}

/* ── Submit ──────────────────────────────────────────── */
async function submitPrompt(prompt) {
  prompt = prompt.trim();
  if (!prompt || state.busy) return;
  if (state.turns >= 8) {
    appendMessage('assistant', 'We have reached the 8-turn limit. Please reset the conversation to start fresh.');
    return;
  }

  state.busy = true;
  state.turns++;
  el.sendBtn.disabled = true;
  el.sendLabel.textContent = 'Sending…';
  el.chatSub.textContent = 'Searching SHL catalog…';

  state.messages.push({ role: 'user', content: prompt });
  appendMessage('user', prompt);
  const thinking = appendThinking();
  updateTurnBadge();
  analyzeSignal(prompt);
  el.promptInput.value = '';

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: state.messages })
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    // Store assistant reply for stateless history
    state.messages.push({
      role: 'assistant',
      content: JSON.stringify({ reply: data.reply, recommendations: data.recommendations || [], end_of_conversation: !!data.end_of_conversation })
    });

    thinking.remove();
    appendMessage('assistant', data.reply || 'No reply generated.');
    renderRecommendations(data.recommendations || []);

    const recCount = (data.recommendations || []).length;
    el.chatSub.textContent = recCount
      ? `Found ${recCount} assessment${recCount > 1 ? 's' : ''}`
      : 'Gathering more context…';

    if (data.end_of_conversation) {
      el.chatSub.textContent = '✓ Recommendation complete';
    }

  } catch (err) {
    state.messages.pop();
    state.turns--;
    thinking.remove();
    appendMessage('assistant', `⚠️ Error: ${err.message}. Please try again.`);
    el.chatSub.textContent = 'Error — please retry';
  } finally {
    state.busy = false;
    el.sendBtn.disabled = false;
    el.sendLabel.textContent = 'Send';
    updateTurnBadge();
  }
}

/* ── Event Listeners ─────────────────────────────────── */
// Sample chips
document.querySelectorAll('[data-sample]').forEach(btn => {
  btn.addEventListener('click', () => {
    const s = SAMPLES[btn.dataset.sample];
    if (!s) return;
    resetConversation();
    el.roleInput.value   = s.role;
    el.levelSelect.value = s.level;
    el.focusSelect.value = s.focus;
    el.remoteCheck.checked = s.remote;
    el.fastCheck.checked   = s.fast;
    el.promptInput.value   = s.prompt;
    analyzeSignal(s.prompt);
    submitPrompt(s.prompt);
  });
});

el.buildBriefBtn.addEventListener('click', () => {
  buildBriefPrompt();
  submitPrompt(el.promptInput.value);
});

el.chatForm.addEventListener('submit', e => {
  e.preventDefault();
  submitPrompt(el.promptInput.value);
});

el.promptInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitPrompt(el.promptInput.value);
  }
});

el.resetBtn.addEventListener('click', resetConversation);

el.promptInput.addEventListener('input', () => analyzeSignal(el.promptInput.value));
el.roleInput.addEventListener('input', () => analyzeSignal(el.promptInput.value || el.roleInput.value));
el.focusSelect.addEventListener('change', buildBriefPrompt);
el.levelSelect.addEventListener('change', buildBriefPrompt);
el.remoteCheck.addEventListener('change', buildBriefPrompt);
el.fastCheck.addEventListener('change', buildBriefPrompt);

/* ── Init ────────────────────────────────────────────── */
checkHealth();
buildBriefPrompt();
updateScore(0);
