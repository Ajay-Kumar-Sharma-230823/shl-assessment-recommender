const samples = {
  java: {
    role: "Java developer",
    level: "Senior",
    focus: "Cognitive ability, coding, and personality",
    prompt: "Senior Java developer. Need coding, cognitive ability, and personality assessments. Remote testing is required and the hiring team wants a shortlist of SHL catalog options."
  },
  sales: {
    role: "Sales associate",
    level: "Graduate",
    focus: "Sales judgment, motivation, and communication",
    prompt: "Graduate sales hiring program for a national intake. Need sales potential, motivation, communication, and situational judgment assessments. Remote testing preferred."
  },
  support: {
    role: "Customer support representative",
    level: "Mid-level",
    focus: "Customer service, language, and situational judgment",
    prompt: "Customer support representative hiring for a high-volume contact center. Need customer service judgment, language communication, and a short screening experience under 45 minutes."
  }
};

const state = {
  messages: [],
  busy: false
};

const el = {
  healthStatus: document.querySelector("#healthStatus"),
  roleInput: document.querySelector("#roleInput"),
  levelSelect: document.querySelector("#levelSelect"),
  focusSelect: document.querySelector("#focusSelect"),
  remoteCheck: document.querySelector("#remoteCheck"),
  fastCheck: document.querySelector("#fastCheck"),
  buildBriefBtn: document.querySelector("#buildBriefBtn"),
  promptInput: document.querySelector("#promptInput"),
  chatForm: document.querySelector("#chatForm"),
  messageStream: document.querySelector("#messageStream"),
  recommendationList: document.querySelector("#recommendationList"),
  emptyResults: document.querySelector("#emptyResults"),
  resetBtn: document.querySelector("#resetBtn"),
  sendBtn: document.querySelector("#sendBtn"),
  turnCounter: document.querySelector("#turnCounter"),
  roleSignal: document.querySelector("#roleSignal"),
  focusSignal: document.querySelector("#focusSignal"),
  constraintSignal: document.querySelector("#constraintSignal"),
  contextScore: document.querySelector("#contextScore"),
  contextMeter: document.querySelector("#contextMeter")
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setHealth(ok) {
  el.healthStatus.classList.toggle("is-ok", ok);
  el.healthStatus.classList.toggle("is-down", !ok);
  el.healthStatus.querySelector("span:last-child").textContent = ok ? "API Ready" : "API Offline";
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    setHealth(response.ok && data.status === "ok");
  } catch (error) {
    setHealth(false);
  }
}

function updateTurnCounter() {
  const turns = state.messages.filter((message) => message.role === "user").length;
  el.turnCounter.textContent = `${turns} ${turns === 1 ? "turn" : "turns"}`;
}

function renderMessage(role, content, variant = "") {
  const article = document.createElement("article");
  article.className = `message ${role === "user" ? "user-message" : "assistant-message"} ${variant}`.trim();
  article.innerHTML = `
    <span class="message-role">${role === "user" ? "You" : "Advisor"}</span>
    <p>${escapeHtml(content)}</p>
  `;
  el.messageStream.appendChild(article);
  el.messageStream.scrollTop = el.messageStream.scrollHeight;
  return article;
}

function renderRecommendations(recommendations) {
  el.recommendationList.innerHTML = "";
  el.emptyResults.hidden = recommendations.length > 0;

  recommendations.forEach((rec, index) => {
    const row = document.createElement("article");
    row.className = "recommendation-row";
    row.innerHTML = `
      <h3>${index + 1}. ${escapeHtml(rec.name)}</h3>
      <div class="rec-meta">
        <span class="rec-chip">${escapeHtml(rec.test_type || "Assessment")}</span>
        <span class="rec-chip">Catalog validated</span>
      </div>
      <a class="rec-link" href="${escapeHtml(rec.url)}" target="_blank" rel="noreferrer">Open SHL product</a>
    `;
    el.recommendationList.appendChild(row);
  });
}

function resetConversation() {
  state.messages = [];
  el.messageStream.innerHTML = "";
  renderMessage("assistant", "Share the role, seniority, assessment focus, and constraints. I will return catalog-grounded SHL recommendations.");
  renderRecommendations([]);
  updateTurnCounter();
}

function summarizeSignal(text) {
  const lowered = text.toLowerCase();
  const role = el.roleInput.value.trim() || "Hiring role";
  const focus = [];
  const constraints = [];

  if (lowered.includes("coding") || lowered.includes("java") || lowered.includes("python")) focus.push("Coding");
  if (lowered.includes("cognitive")) focus.push("Cognitive");
  if (lowered.includes("personality") || lowered.includes("opq")) focus.push("Personality");
  if (lowered.includes("sales")) focus.push("Sales");
  if (lowered.includes("language") || lowered.includes("communication")) focus.push("Communication");
  if (lowered.includes("judgment") || lowered.includes("situational")) focus.push("Judgment");

  if (lowered.includes("remote")) constraints.push("Remote testing");
  if (lowered.includes("45") || lowered.includes("short") || lowered.includes("under")) constraints.push("Time boxed");
  if (lowered.includes("graduate")) constraints.push("Early career");

  const score = Math.min(96, 42 + focus.length * 12 + constraints.length * 9 + (role.length > 4 ? 12 : 0));

  el.roleSignal.textContent = role;
  el.focusSignal.textContent = focus.length ? focus.join(" + ") : "Assessment mix pending";
  el.constraintSignal.textContent = constraints.length ? constraints.join(", ") : "No constraint";
  el.contextScore.textContent = `${score}%`;
  el.contextMeter.style.width = `${score}%`;
}

function buildBrief(options = {}) {
  const constraints = [];
  if (el.remoteCheck.checked) constraints.push("Remote testing required");
  if (el.fastCheck.checked) constraints.push("Under 45 minutes preferred");

  const brief = `${el.levelSelect.value} ${el.roleInput.value}. Need ${el.focusSelect.value.toLowerCase()} assessments. ${constraints.join(". ")}. Recommend the best SHL catalog options.`;
  el.promptInput.value = brief.replace(/\s+/g, " ").trim();
  summarizeSignal(el.promptInput.value);
  if (options.focus) {
    el.promptInput.focus();
  }
}

async function submitPrompt(prompt) {
  if (!prompt || state.busy) return;

  state.busy = true;
  el.sendBtn.disabled = true;
  el.sendBtn.textContent = "Sending";

  state.messages.push({ role: "user", content: prompt });
  renderMessage("user", prompt);
  const loadingMessage = renderMessage("assistant", "Searching the SHL catalog and matching assessments...", "system-message");
  updateTurnCounter();
  summarizeSignal(prompt);
  el.promptInput.value = "";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: state.messages })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Request failed");
    }

    const assistantContent = JSON.stringify({
      reply: data.reply,
      recommendations: data.recommendations || [],
      end_of_conversation: Boolean(data.end_of_conversation)
    });

    state.messages.push({ role: "assistant", content: assistantContent });
    loadingMessage.remove();
    renderMessage("assistant", data.reply || "I could not generate a reply.");
    renderRecommendations(data.recommendations || []);
  } catch (error) {
    state.messages.pop();
    loadingMessage.querySelector("p").textContent = `The advisor could not complete this request: ${error.message}`;
  } finally {
    state.busy = false;
    el.sendBtn.disabled = false;
    el.sendBtn.textContent = "Send";
    updateTurnCounter();
  }
}

document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => {
    const sample = samples[button.dataset.sample];
    resetConversation();
    el.roleInput.value = sample.role;
    el.levelSelect.value = sample.level;
    el.focusSelect.value = sample.focus;
    el.remoteCheck.checked = true;
    el.fastCheck.checked = button.dataset.sample === "support";
    el.promptInput.value = sample.prompt;
    summarizeSignal(sample.prompt);
    submitPrompt(sample.prompt);
  });
});

el.buildBriefBtn.addEventListener("click", () => {
  buildBrief();
  submitPrompt(el.promptInput.value.trim());
});

el.chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitPrompt(el.promptInput.value.trim());
});

el.resetBtn.addEventListener("click", () => {
  resetConversation();
});

el.promptInput.addEventListener("input", () => summarizeSignal(el.promptInput.value));
el.roleInput.addEventListener("input", () => summarizeSignal(el.promptInput.value || el.roleInput.value));
el.focusSelect.addEventListener("change", buildBrief);
el.levelSelect.addEventListener("change", buildBrief);
el.remoteCheck.addEventListener("change", buildBrief);
el.fastCheck.addEventListener("change", buildBrief);

checkHealth();
buildBrief();
