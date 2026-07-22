/* Poker IA — logique du dashboard. Parle à server.py via /api/*. */
"use strict";

const RANKS = "23456789TJQKA";
const SUITS = ["♠", "♥", "♦", "♣"];
const RED = new Set([1, 2]);
const STREETS = ["préflop", "flop", "turn", "river"];
const FOLD = 0, CHECK_CALL = 1, RAISE_HALF = 2, RAISE_POT = 3, ALL_IN = 4;

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const bb = (chips) => {
  const v = chips / 2;
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
};

let busy = false;
let currentState = null;
let trainingData = [];

/* ------------------------------------------------------------------ API */

async function api(path, body) {
  const opts = body !== undefined
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : undefined;
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Erreur inconnue");
  return data;
}

/* ---------------------------------------------------------------- cartes */

function cardEl(code) {
  const el = document.createElement("div");
  if (code === null) { el.className = "card slot"; return el; }
  if (code === "back") { el.className = "card back"; return el; }
  const rank = RANKS[Math.floor(code / 4)];
  const suit = SUITS[code % 4];
  el.className = "card" + (RED.has(code % 4) ? " red" : "");
  el.innerHTML = `<div class="corner">${rank}<br>${suit}</div><div class="pip">${suit}</div>`;
  return el;
}

function renderCards(container, codes) {
  container.replaceChildren(...codes.map(cardEl));
}

function renderBoard(codes) {
  const slots = [...codes];
  while (slots.length < 5) slots.push(null);
  renderCards($("board"), slots);
}

/* ----------------------------------------------------------------- table */

function renderTable(state) {
  const r = state.result;
  renderCards($("cards-you"), state.your_cards);
  if (r && r.showdown && r.ai_cards) {
    renderCards($("cards-ai"), r.ai_cards);
    renderBoard(r.full_board);
  } else {
    renderCards($("cards-ai"), state.your_cards.length ? ["back", "back"] : []);
    renderBoard(state.board);
  }

  $("cat-you").textContent = r && r.categories ? r.categories.toi : "";
  $("cat-ai").textContent = r && r.categories ? r.categories.ia : "";

  $("stack-you").textContent = `${bb(state.stacks[0])} BB`;
  $("stack-ai").textContent = `${bb(state.stacks[1])} BB`;
  $("dealer-you").hidden = state.button !== 0;
  $("dealer-ai").hidden = state.button !== 1;

  $("pot").hidden = state.pot === 0;
  $("pot").textContent = `Pot ${bb(state.pot)} BB`;

  for (const [id, amount] of [["bet-you", state.bets[0]], ["bet-ai", state.bets[1]]]) {
    const el = $(id);
    el.hidden = state.terminal || amount === 0;
    el.textContent = `${bb(amount)} BB`;
  }
}

/* --------------------------------------------------------------- actions */

function actionButton(label, sub, cls, onClick) {
  const btn = document.createElement("button");
  btn.className = "btn " + cls;
  btn.innerHTML = sub ? `${label}<small>${sub}</small>` : label;
  btn.addEventListener("click", onClick);
  return btn;
}

function renderActions(state) {
  const box = $("actions");
  box.replaceChildren();
  if (state.terminal) {
    box.append(actionButton("Nouvelle main", "touche N", "btn-new", newHand));
    return;
  }
  if (!state.your_turn) return;
  const p = state.raise_preview;
  const defs = {
    [FOLD]: ["Se coucher", "", "btn-fold"],
    [CHECK_CALL]: state.to_call > 0
      ? [`Suivre ${bb(p.call)} BB`, "", "btn-call"]
      : ["Check", "", "btn-call"],
    [RAISE_HALF]: ["Relance ½ pot", `→ ${bb(p.half)} BB`, ""],
    [RAISE_POT]: ["Relance pot", `→ ${bb(p.pot)} BB`, ""],
    [ALL_IN]: ["Tapis", `${bb(p.allin)} BB`, "btn-allin"],
  };
  for (const a of state.legal_actions) {
    const [label, sub, cls] = defs[a];
    box.append(actionButton(label, sub, cls, () => playAction(a)));
  }
}

function setStatus(text) { $("status").textContent = text; }

function statusFor(state) {
  if (state.terminal) return "Main terminée — clique sur « Nouvelle main ».";
  if (state.your_turn) {
    return state.to_call > 0
      ? `À toi de jouer — ${bb(state.to_call)} BB à suivre (${STREETS[state.street]}).`
      : `À toi de jouer — parole (${STREETS[state.street]}).`;
  }
  return "L'IA réfléchit…";
}

/* ------------------------------------------------------------------- log */

function log(html, cls = "") {
  const box = $("log");
  const line = document.createElement("div");
  line.className = "log-line " + cls;
  line.innerHTML = html;
  box.append(line);
  box.scrollTop = box.scrollHeight;
}

const ACTION_LABELS = ["fold", "check/call", "relance ½ pot", "relance pot", "all-in"];

function describeAction(ev) {
  const who = ev.who === "toi" ? "Toi" : "IA";
  return `<strong>${who}</strong> : ${ACTION_LABELS[ev.action]} <span style="color:var(--muted)">(pot ${bb(ev.pot)} BB)</span>`;
}

let aiBubbleTimer = null;
function showAiBubble(text) {
  const el = $("ai-bubble");
  if (aiBubbleTimer) clearTimeout(aiBubbleTimer);
  el.textContent = text;
  el.hidden = false;
  aiBubbleTimer = setTimeout(() => { el.hidden = true; aiBubbleTimer = null; }, 950);
}

/* Applique un instantané de table (jetons, mises, pot, board) — sert à animer
   la main étape par étape sans divulguer l'état final à l'avance. */
function applySnapshot(snap) {
  if (!snap) return;
  $("stack-you").textContent = `${bb(snap.stacks[0])} BB`;
  $("stack-ai").textContent = `${bb(snap.stacks[1])} BB`;
  $("pot").hidden = !snap.pot;
  $("pot").textContent = `Pot ${bb(snap.pot)} BB`;
  for (const [id, amt] of [["bet-you", snap.bets[0]], ["bet-ai", snap.bets[1]]]) {
    const el = $(id);
    el.hidden = !amt;
    el.textContent = `${bb(amt)} BB`;
  }
  if (snap.board) renderBoard(snap.board);
}

/* --------------------------------------------------- déroulé d'une action */

async function processState(state, isNewHand) {
  currentState = state;
  $("banner").hidden = true;
  renderActions({ terminal: false, your_turn: false });

  if (isNewHand) {
    $("cat-you").textContent = "";
    $("cat-ai").textContent = "";
    renderCards($("cards-you"), state.your_cards);
    renderCards($("cards-ai"), ["back", "back"]);
    $("dealer-you").hidden = state.button !== 0;
    $("dealer-ai").hidden = state.button !== 1;
    log(`Main n°${state.hand_id} — tu es ${state.button === 0 ? "au bouton" : "en grosse blind"}`, "hand-sep");
    applySnapshot(state.start);  // blinds seules, avant toute action
  }

  for (const ev of state.events) {
    if (ev.who === "table") {
      await sleep(480);
      applySnapshot(ev);
      log(`— ${STREETS[ev.street]} : ${ev.board.map(c => RANKS[c >> 2] + SUITS[c % 4]).join(" ")}`, "street");
    } else if (ev.who === "ia") {
      setStatus("L'IA réfléchit…");
      await sleep(650);
      showAiBubble(["Je passe.", "Check / call.", "Relance ½ pot !", "Relance pot !", "TAPIS !"][ev.action]);
      applySnapshot(ev);
      log(describeAction(ev), "ai");
    } else {
      applySnapshot(ev);
      log(describeAction(ev), "you");
    }
  }

  renderTable(state);
  renderActions(state);
  setStatus(statusFor(state));
  renderSession(state.session);

  if (state.terminal && state.result) showResult(state.result);
}

function showResult(r) {
  const title = { toi: `Tu gagnes ${bb(r.pot)} BB 🎉`, ia: `L'IA gagne ${bb(r.pot)} BB`, partage: "Partage du pot" }[r.winner];
  let sub = r.payoff_bb >= 0 ? `+${r.payoff_bb} BB pour toi` : `${r.payoff_bb} BB pour toi`;
  if (r.showdown && r.categories) sub = `${r.categories.toi} contre ${r.categories.ia} · ` + sub;
  else if (r.winner === "toi") sub = "L'IA s'est couchée · " + sub;
  else if (r.winner === "ia") sub = "Tu t'es couché · " + sub;
  $("banner-title").textContent = title;
  $("banner-sub").textContent = sub;
  $("banner").hidden = false;
  log(`<strong>${title}</strong> (${sub})`);
}

async function newHand() {
  if (busy) return;
  busy = true;  // armé AVANT le fetch : bloque tout double-clic pendant la latence
  try {
    await processState(await api("/api/new-hand", {}), true);
  } catch (e) {
    setStatus(e.message);
  } finally {
    busy = false;
  }
}

async function playAction(a) {
  if (busy || !currentState || !currentState.your_turn) return;
  if (!currentState.legal_actions.includes(a)) return;
  busy = true;  // armé AVANT le fetch : une 2e action ne peut pas se glisser
  try {
    await processState(await api("/api/action", { action: a }), false);
  } catch (e) {
    setStatus(e.message);
  } finally {
    busy = false;
  }
}

/* ------------------------------------------------------------ statistiques */

function renderSession(s) {
  const total = s.total_bb;
  const tile = $("tile-total");
  tile.textContent = `${total > 0 ? "+" : ""}${total} BB`;
  tile.className = "tile-value " + (total > 0 ? "up" : total < 0 ? "down" : "");
  const bb100 = $("tile-bb100");
  bb100.textContent = `${s.bb100 > 0 ? "+" : ""}${s.bb100}`;
  bb100.className = "tile-value " + (s.bb100 > 0 ? "up" : s.bb100 < 0 ? "down" : "");
  $("tile-hands").textContent = s.hands;
  $("tile-pot").textContent = `${bb(s.biggest_pot)} BB`;
  const sd = s.showdowns;
  $("showdown-stats").textContent =
    `Abattages : ${sd.gagnes} gagnés · ${sd.perdus} perdus · ${sd.partages} partagés`;

  $("empty-session").hidden = s.history.length > 0;
  let cumul = 0;
  const points = s.history.map((v, i) => { cumul += v; return [i + 1, cumul]; });
  drawChart($("chart-session"), $("tt-session"), {
    series: [{ name: "Cumul BB", color: getCss("--series-1"), points }],
    xLabel: "main",
    yFormat: (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)} BB`,
  });
}

function renderTraining() {
  if (!trainingData.length) { $("training-box").hidden = true; return; }
  const mk = (key) => trainingData.map((r) => [r.hands, r[key]]);
  const series = [
    { name: "vs bot à règles", color: getCss("--series-1"), points: mk("vs_regles") },
    { name: "vs bot suiveur", color: getCss("--series-2"), points: mk("vs_call") },
    { name: "vs bot aléatoire", color: getCss("--series-3"), points: mk("vs_aleatoire") },
  ];
  $("legend-training").replaceChildren(...series.map((s) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    item.innerHTML = `<span class="legend-swatch" style="background:${s.color}"></span>${s.name}`;
    return item;
  }));
  drawChart($("chart-training"), $("tt-training"), {
    series,
    xLabel: "mains d'entraînement",
    yFormat: (v) => `${v > 0 ? "+" : ""}${Math.round(v)} bb/100`,
    xFormat: (v) => `${Math.round(v / 1000)}k`,
  });
  const table = $("table-training");
  table.innerHTML = "<tr><th>Mains</th><th>vs règles</th><th>vs suiveur</th><th>vs aléatoire</th></tr>" +
    trainingData.map((r) =>
      `<tr><td>${r.hands}</td><td>${r.vs_regles}</td><td>${r.vs_call}</td><td>${r.vs_aleatoire}</td></tr>`).join("");
}

function getCss(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* ------------------------------------------------------- graphique canvas */

const chartConfigs = new Map();

function drawChart(canvas, tooltip, cfg, hoverX) {
  chartConfigs.set(canvas.id, { canvas, tooltip, cfg });
  const dpr = window.devicePixelRatio || 1;
  if (!canvas.dataset.h) canvas.dataset.h = canvas.getAttribute("height");
  const h = +canvas.dataset.h;
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const all = cfg.series.flatMap((s) => s.points);
  if (!all.length) { tooltip.hidden = true; return; }
  const pad = { l: 44, r: 12, t: 10, b: 22 };
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  let xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(0, ...ys), yMax = Math.max(0, ...ys);
  if (xMin === xMax) { xMin -= 1; xMax += 1; }
  const ySpan = (yMax - yMin) || 1;
  yMin -= ySpan * 0.08; yMax += ySpan * 0.08;

  const X = (v) => pad.l + ((v - xMin) / (xMax - xMin)) * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * (h - pad.t - pad.b);

  const muted = getCss("--muted"), grid = getCss("--grid");
  ctx.font = "10.5px system-ui, sans-serif";
  ctx.fillStyle = muted;

  // grille + graduations Y
  const ticks = niceTicks(yMin, yMax, 4);
  for (const t of ticks) {
    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, Y(t)); ctx.lineTo(w - pad.r, Y(t)); ctx.stroke();
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(formatTick(t), pad.l - 6, Y(t));
  }
  // ligne du zéro
  if (yMin < 0 && yMax > 0) {
    ctx.strokeStyle = getCss("--baseline");
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(pad.l, Y(0)); ctx.lineTo(w - pad.r, Y(0)); ctx.stroke();
    ctx.setLineDash([]);
  }
  // graduations X (min, milieu, max)
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const xFmt = cfg.xFormat || ((v) => Math.round(v));
  for (const t of [xMin, (xMin + xMax) / 2, xMax]) {
    ctx.fillText(xFmt(t), X(t), h - pad.b + 6);
  }

  // séries
  for (const s of cfg.series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    s.points.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
    ctx.stroke();
    if (s.points.length === 1) {
      const [x, y] = s.points[0];
      ctx.fillStyle = s.color;
      ctx.beginPath(); ctx.arc(X(x), Y(y), 3.5, 0, 7); ctx.fill();
      ctx.fillStyle = muted;
    }
  }

  // survol : réticule + points
  if (hoverX !== undefined) {
    const nearest = nearestX(cfg.series, xMin + ((hoverX - pad.l) / (w - pad.l - pad.r)) * (xMax - xMin));
    if (nearest !== null) {
      ctx.strokeStyle = muted;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(X(nearest), pad.t); ctx.lineTo(X(nearest), h - pad.b); ctx.stroke();
      ctx.setLineDash([]);
      const rows = [];
      for (const s of cfg.series) {
        const pt = s.points.find((p) => p[0] === nearest);
        if (!pt) continue;
        ctx.fillStyle = s.color;
        ctx.beginPath(); ctx.arc(X(pt[0]), Y(pt[1]), 4, 0, 7); ctx.fill();
        rows.push(`<span style="color:${s.color}">●</span> ${s.name} : ${cfg.yFormat(pt[1])}`);
      }
      tooltip.innerHTML = `<strong>${cfg.xLabel} ${xFmt(nearest)}</strong><br>` + rows.join("<br>");
      tooltip.hidden = false;
      const left = X(nearest) > w / 2 ? X(nearest) - tooltip.offsetWidth - 12 : X(nearest) + 12;
      tooltip.style.left = `${left}px`;
      tooltip.style.top = "8px";
      return;
    }
  }
  tooltip.hidden = true;
}

function nearestX(series, target) {
  let best = null, dist = Infinity;
  for (const s of series) for (const [x] of s.points) {
    const d = Math.abs(x - target);
    if (d < dist) { dist = d; best = x; }
  }
  return best;
}

function niceTicks(min, max, count) {
  const span = max - min;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => span / s <= count + 1) || mag * 10;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max; t += step) ticks.push(t);
  return ticks;
}

function formatTick(v) {
  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(Math.abs(v) >= 10000 ? 0 : 1)}k`;
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

function attachHover(canvasId) {
  const canvas = $(canvasId);
  canvas.addEventListener("mousemove", (e) => {
    const entry = chartConfigs.get(canvasId);
    if (!entry) return;
    const rect = canvas.getBoundingClientRect();
    drawChart(entry.canvas, entry.tooltip, entry.cfg, e.clientX - rect.left);
  });
  canvas.addEventListener("mouseleave", () => {
    const entry = chartConfigs.get(canvasId);
    if (entry) drawChart(entry.canvas, entry.tooltip, entry.cfg);
  });
}

/* ------------------------------------------------------------------ init */

document.addEventListener("keydown", (e) => {
  if (e.repeat) return;  // ignore l'auto-répétition d'une touche maintenue
  if (e.target.tagName === "INPUT" || e.ctrlKey || e.metaKey) return;
  const key = e.key.toLowerCase();
  if (key === "n" && currentState && currentState.terminal) return void newHand();
  const map = { f: FOLD, c: CHECK_CALL, r: RAISE_HALF, p: RAISE_POT, a: ALL_IN };
  if (key in map) playAction(map[key]);
});

$("btn-reset").addEventListener("click", async () => {
  if (busy) return;
  if (!confirm("Remettre les statistiques de session à zéro ?")) return;
  busy = true;
  try {
    const state = await api("/api/reset-session", {});
    currentState = state;
    $("log").replaceChildren();
    log("La partie n'a pas encore commencé.", "muted");
    renderCards($("cards-you"), []);
    renderCards($("cards-ai"), []);
    $("cat-you").textContent = "";
    $("cat-ai").textContent = "";
    $("banner").hidden = true;
    applySnapshot({ stacks: state.stacks, bets: state.bets, pot: state.pot, board: [] });
    renderActions({ terminal: true });
    setStatus("Session remise à zéro — lance une nouvelle main.");
    renderSession(state.session);
  } catch (e) {
    setStatus(`Échec de la remise à zéro : ${e.message}`);
  } finally {
    busy = false;
  }
});

window.addEventListener("resize", () => {
  for (const { canvas, tooltip, cfg } of chartConfigs.values()) drawChart(canvas, tooltip, cfg);
});

(async function init() {
  attachHover("chart-session");
  attachHover("chart-training");
  try {
    const [state, stats] = await Promise.all([api("/api/state"), api("/api/stats")]);
    trainingData = stats.training;
    renderTraining();
    currentState = state;
    renderSession(state.session);
    if (state.your_cards.length && !state.terminal) {
      renderTable(state);
      renderActions(state);
      setStatus(statusFor(state));
    } else {
      renderBoard([]);
      renderCards($("cards-ai"), []);
      renderActions({ terminal: true });
      setStatus("Prêt à jouer — lance la première main !");
    }
  } catch (e) {
    setStatus(`Impossible de joindre le serveur : ${e.message}`);
  }
})();
