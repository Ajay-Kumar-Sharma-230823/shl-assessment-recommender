'use strict';

/* ── Sample prompts ─────────────────────────────────── */
const SAMPLES = {
  java: {
    role:'Java Developer', level:'Senior',
    focus:'Cognitive ability, coding, and personality',
    remote:true, fast:false,
    prompt:'Senior Java developer for a fintech company. Need coding assessment, cognitive ability reasoning, and personality evaluation. Remote testing required.'
  },
  sales: {
    role:'Sales Associate', level:'Graduate',
    focus:'Sales judgment, motivation, and communication',
    remote:true, fast:false,
    prompt:'Graduate sales hiring program for a national intake. Need sales potential, motivation, communication, and situational judgment assessments. Remote testing preferred.'
  },
  support: {
    role:'Customer Support Representative', level:'Mid-level',
    focus:'Customer service, language, and situational judgment',
    remote:true, fast:true,
    prompt:'Customer support representative for a high-volume contact center. Need customer service judgment, language communication, and screening under 45 minutes.'
  }
};

/* ── State ──────────────────────────────────────────── */
const state = { messages:[], busy:false, turns:0 };

/* ── DOM ────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const el = {
  healthStatus:  $('healthStatus'),
  roleInput:     $('roleInput'),
  levelSelect:   $('levelSelect'),
  focusSelect:   $('focusSelect'),
  remoteCheck:   $('remoteCheck'),
  fastCheck:     $('fastCheck'),
  buildBriefBtn: $('buildBriefBtn'),
  promptInput:   $('promptInput'),
  chatForm:      $('chatForm'),
  messages:      $('messageStream'),
  recList:       $('recList'),
  emptyState:    $('emptyState'),
  resetBtn:      $('resetBtn'),
  sendBtn:       $('sendBtn'),
  sendLabel:     $('sendLabel'),
  turnPill:      $('turnPill'),
  chatStatus:    $('chatStatus'),
  sigRole:       $('sigRole'),
  sigFocus:      $('sigFocus'),
  sigConstraint: $('sigConstraint'),
  ringPct:       $('ringPct'),
  ringProgress:  $('ringProgress')
};

/* ── Utils ──────────────────────────────────────────── */
function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

/* ── Health ─────────────────────────────────────────── */
async function checkHealth(){
  try{
    const r = await fetch('/health');
    const d = await r.json();
    const ok = r.ok && d.status==='ok';
    el.healthStatus.className = 'health-pill ' + (ok?'ok':'down');
    el.healthStatus.querySelector('.health-label').textContent = ok?'API Ready':'API Offline';
  } catch {
    el.healthStatus.className = 'health-pill down';
    el.healthStatus.querySelector('.health-label').textContent = 'API Offline';
  }
}

/* ── Ring ───────────────────────────────────────────── */
function setScore(pct){
  pct = Math.max(0, Math.min(100, pct));
  const c = 301.6;
  el.ringProgress.style.strokeDashoffset = c - (pct/100)*c;
  el.ringPct.textContent = pct + '%';
  el.ringProgress.style.stroke = pct>=70?'#2ec4b6': pct>=40?'#4361ee':'#8b91a8';
}

/* ── Turn pill ──────────────────────────────────────── */
function updateTurns(){
  el.turnPill.textContent = `${state.turns} / 8 turns`;
  el.turnPill.classList.toggle('warning', state.turns >= 7);
}

/* ── Signal analyze ─────────────────────────────────── */
function analyzeSignal(text){
  const low = (text||'').toLowerCase();
  const role = el.roleInput.value.trim() || '—';
  const focus=[], cons=[];
  if(/coding|java|python|javascript|programming/.test(low)) focus.push('Coding');
  if(/cognitive|reasoning|numerical|verbal|aptitude/.test(low)) focus.push('Cognitive');
  if(/personality|opq|behaviour|behavior/.test(low)) focus.push('Personality');
  if(/sales|motivation/.test(low)) focus.push('Sales');
  if(/language|communication/.test(low)) focus.push('Language');
  if(/situational|judgment|sjt/.test(low)) focus.push('SJT');
  if(/remote|online/.test(low)) cons.push('Remote');
  if(/45|short|under/.test(low)) cons.push('≤45 min');
  el.sigRole.textContent = role;
  el.sigFocus.textContent = focus.length ? focus.join(' + ') : '—';
  el.sigConstraint.textContent = cons.length ? cons.join(', ') : 'None';
  const score = Math.min(96, 28 + focus.length*15 + cons.length*10 + (role.length>3?14:0) + state.turns*5);
  setScore(score);
}

/* ── Messages ───────────────────────────────────────── */
function clearWelcome(){ const w=el.messages.querySelector('.welcome'); if(w) w.remove(); }

function addMsg(role, text, cls=''){
  clearWelcome();
  const a = document.createElement('article');
  a.className = `msg ${role} ${cls}`.trim();
  a.innerHTML = `<div class="msg-who">${role==='user'?'You':'✦ Advisor'}</div><div class="msg-body">${esc(text)}</div>`;
  el.messages.appendChild(a);
  el.messages.scrollTop = el.messages.scrollHeight;
  return a;
}

function addThinking(text='Searching the SHL catalog…'){
  clearWelcome();
  const a = document.createElement('article');
  a.className = 'msg thinking';
  a.innerHTML = `<div class="msg-who">⏳ Advisor</div><div class="msg-body" style="font-size:12px;color:var(--txt3)">${esc(text)}</div><div class="dots" style="margin-top:6px"><span></span><span></span><span></span></div>`;
  el.messages.appendChild(a);
  el.messages.scrollTop = el.messages.scrollHeight;
  return a;
}

/* ── Recommendations ─────────────────────────────────── */
function renderRecs(recs){
  el.recList.innerHTML = '';
  el.emptyState.hidden = recs.length > 0;
  recs.forEach((r,i)=>{
    const d = document.createElement('div');
    d.className = 'rec-item';
    d.innerHTML = `
      <div class="rec-num">#${i+1}</div>
      <div class="rec-name">${esc(r.name)}</div>
      <div class="rec-tags">
        <span class="rec-tag type">${esc(r.test_type||'Assessment')}</span>
        <span class="rec-tag ok">✓ Catalog</span>
      </div>
      <a class="rec-link" href="${esc(r.url)}" target="_blank" rel="noreferrer">
        View on SHL
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
      </a>`;
    el.recList.appendChild(d);
  });
}

/* ── Reset ──────────────────────────────────────────── */
function reset(){
  state.messages=[]; state.turns=0;
  el.messages.innerHTML = `
    <div class="welcome">
      <div class="welcome-glow"></div>
      <div class="welcome-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg></div>
      <h3 class="welcome-title">Hello! I'm your SHL Assessment Advisor</h3>
      <p class="welcome-desc">Describe the role, seniority, and assessment needs. I'll find catalog-grounded SHL recommendations.</p>
      <div class="welcome-tags"><span>🧠 Cognitive</span><span>💻 Technical</span><span>🤝 Personality</span><span>🎯 Situational</span></div>
    </div>`;
  renderRecs([]);
  updateTurns(); setScore(0);
  el.sigRole.textContent='—'; el.sigFocus.textContent='—'; el.sigConstraint.textContent='None';
  el.chatStatus.textContent='Ready to recommend assessments';
}

/* ── Build Brief ─────────────────────────────────────── */
function buildBrief(){
  const cons=[];
  if(el.remoteCheck.checked) cons.push('Remote testing required');
  if(el.fastCheck.checked)   cons.push('Under 45 minutes preferred');
  const t=`${el.levelSelect.value} ${el.roleInput.value}. Need ${el.focusSelect.value.toLowerCase()} assessments. ${cons.join('. ')} Recommend the best SHL catalog options.`;
  el.promptInput.value = t.replace(/\s+/g,' ').trim();
  analyzeSignal(el.promptInput.value);
}

/* ── Submit ──────────────────────────────────────────── */
async function submit(prompt){
  prompt = (prompt||'').trim();
  if(!prompt||state.busy) return;
  if(state.turns>=8){
    addMsg('bot','We have reached the 8-turn conversation limit. Please reset to start a new conversation.');
    return;
  }
  state.busy=true; state.turns++;
  el.sendBtn.disabled=true; el.sendLabel.textContent='Sending…';
  el.chatStatus.textContent='Searching SHL catalog…';

  state.messages.push({role:'user',content:prompt});
  addMsg('user',prompt);

  // Show a helpful message if it's the first turn (cold start)
  const thinkingText = state.turns === 1
    ? 'Loading SHL catalog… (first request may take up to 2 min on free server)'
    : 'Searching the SHL catalog…';
  const thinking = addThinking(thinkingText);
  updateTurns(); analyzeSignal(prompt);
  el.promptInput.value='';

  try{
    // 150-second timeout — accounts for Render cold start + model download
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 150000);

    const res = await fetch('/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({messages:state.messages}),
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    const data = await res.json();
    if(!res.ok) throw new Error(data.detail||`HTTP ${res.status}`);

    state.messages.push({
      role:'assistant',
      content:JSON.stringify({reply:data.reply,recommendations:data.recommendations||[],end_of_conversation:!!data.end_of_conversation})
    });
    thinking.remove();
    addMsg('bot', data.reply||'No reply generated.');
    renderRecs(data.recommendations||[]);
    const n=(data.recommendations||[]).length;
    el.chatStatus.textContent = data.end_of_conversation
      ? '✓ Recommendation complete'
      : n ? `Found ${n} assessment${n>1?'s':''}`
          : 'Gathering more context…';
  } catch(e){
    state.messages.pop(); state.turns--;
    thinking.remove();
    const isTimeout = e.name === 'AbortError';
    const msg = isTimeout
      ? '⏱️ The server is still waking up (cold start). Please click Send again — it should work now!'
      : `⚠️ ${e.message}. Please try again.`;
    addMsg('bot', msg);
    el.chatStatus.textContent = isTimeout ? 'Cold start — retry now!' : 'Error — please retry';
  } finally{
    state.busy=false;
    el.sendBtn.disabled=false; el.sendLabel.textContent='Send';
    updateTurns();
  }
}

/* ── Events ──────────────────────────────────────────── */
document.querySelectorAll('[data-sample]').forEach(b=>{
  b.addEventListener('click',()=>{
    const s=SAMPLES[b.dataset.sample]; if(!s) return;
    reset();
    el.roleInput.value=s.role; el.levelSelect.value=s.level;
    el.focusSelect.value=s.focus;
    el.remoteCheck.checked=s.remote; el.fastCheck.checked=s.fast;
    el.promptInput.value=s.prompt;
    analyzeSignal(s.prompt);
    submit(s.prompt);
  });
});

el.buildBriefBtn.addEventListener('click',()=>{ buildBrief(); submit(el.promptInput.value); });
el.chatForm.addEventListener('submit',e=>{ e.preventDefault(); submit(el.promptInput.value); });
el.promptInput.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); submit(el.promptInput.value); }});
el.resetBtn.addEventListener('click',reset);
el.promptInput.addEventListener('input',()=>analyzeSignal(el.promptInput.value));
el.roleInput.addEventListener('input',()=>analyzeSignal(el.promptInput.value||el.roleInput.value));
el.focusSelect.addEventListener('change',buildBrief);
el.levelSelect.addEventListener('change',buildBrief);
el.remoteCheck.addEventListener('change',buildBrief);
el.fastCheck.addEventListener('change',buildBrief);

/* ── Init ────────────────────────────────────────────── */
checkHealth();
buildBrief();
updateTurns();
setScore(0);
