/* Page « Apprentissage en direct » : courbe auto-actualisée + niveau estimé. */
"use strict";

const $ = (id) => document.getElementById(id);
const REFRESH_MS = 2500;
const TOTAL_DEFAULT = 300000; // affichage du % ; corrigé si l'ETA dit autre chose

/* ---------------------------------------------------- niveau « humain » */
/* Barème volontairement prudent. Phase d'exploration : d'après epsilon.
   Ensuite : d'après la dernière mesure réelle contre le bot à règles. */
const STAGES = [
  { emoji: "🥚", label: "Découverte",        hours: "0 – 1 h",        short: "Hasard",
    blurb: "L'IA clique au hasard : elle découvre à peine que les cartes ont une valeur." },
  { emoji: "🐣", label: "Premières leçons",  hours: "1 – 5 h",        short: "Débuts",
    blurb: "Encore beaucoup de coups aléatoires, mais les premières associations « bonne main → miser » se forment." },
  { emoji: "📖", label: "Débutant",          hours: "5 – 15 h",       short: "Débutant",
    blurb: "Comme un humain après quelques soirées : trop de mains jouées, des calls curieux, mais l'idée est là." },
  { emoji: "🎓", label: "Novice appliqué",   hours: "15 – 40 h",      short: "Novice",
    blurb: "Le niveau d'un joueur qui a lu un article ou deux : les mains poubelles partent au muck, les monstres relancent." },
  { emoji: "♟️", label: "Amateur",           hours: "40 – 100 h",     short: "Amateur",
    blurb: "Niveau partie entre amis sérieuse : cote du pot approximative, agression à bon escient de temps en temps." },
  { emoji: "🃏", label: "Amateur solide",    hours: "100 – 250 h",    short: "Solide",
    blurb: "Tient tête au bot à règles : sélection de mains correcte, value bets, quelques folds disciplinés." },
  { emoji: "🏆", label: "Bon joueur de club", hours: "250 – 600 h",   short: "Club",
    blurb: "Bat nettement le bot à règles : pression en position, tailles de mise cohérentes, lecture du board." },
  { emoji: "🦈", label: "Régulier sérieux",  hours: "600 – 1 500 h",  short: "Régulier",
    blurb: "Écrase ses adversaires d'entraînement. Attention : ça reste très loin d'un semi-pro humain — mais ça pique." },
];

function levelIndex(eps, vsRule) {
  // Tant que l'exploration domine, l'IA joue essentiellement au hasard,
  // quel que soit son score mesuré.
  if (eps > 0.75) return 0;
  if (eps > 0.45) return 1;
  if (vsRule === null) return 2;   // pas encore de mesure réelle
  if (vsRule < -300) return 2;
  if (vsRule < -120) return 3;
  if (vsRule < -30) return 4;
  if (vsRule < 40) return 5;
  if (vsRule < 120) return 6;
  return 7;
}

function renderLevel(progress, training, cfrMetrics) {
  const last = progress.length ? progress[progress.length - 1] : null;
  const lastM = training.length ? training[training.length - 1] : null;
  const lastCfr = cfrMetrics && cfrMetrics.length ? cfrMetrics[cfrMetrics.length - 1] : null;
  if (!last && !lastCfr) return;
  // Le CFR (sans exploration) prime dès qu'il a une mesure ; sinon le DQN.
  const idx = lastCfr ? levelIndex(0.0, lastCfr.vs_regles)
                      : levelIndex(last.eps, lastM ? lastM.vs_regles : null);
  if (!last) {
    renderLadderOnly(idx);
    return;
  }
  const st = STAGES[idx];
  $("lv-emoji").textContent = st.emoji;
  $("lv-label").textContent = st.label;
  $("lv-hours").textContent = `≈ ${st.hours} d'expérience humaine`;
  $("lv-blurb").textContent = st.blurb;
  const volH = Math.round(last.hands / 70); // ~70 mains/h en heads-up en ligne
  $("lv-volume").textContent =
    `Volume brut : ${last.hands.toLocaleString("fr-FR")} mains vues — un humain jouerait ${volH.toLocaleString("fr-FR")} h ` +
    `pour en voir autant (l'IA apprend moins par main qu'un humain attentif, d'où l'écart avec le niveau estimé).`;
  const ladder = $("lv-ladder");
  ladder.replaceChildren(...STAGES.map((s, i) => {
    const d = document.createElement("div");
    d.className = "tr-step" + (i < idx ? " done" : i === idx ? " now" : "");
    d.textContent = `${s.emoji} ${s.short}`;
    return d;
  }));
}

/* ------------------------------------------------------------- tuiles */

function renderLadderOnly(idx) {
  const st = STAGES[idx];
  $("lv-emoji").textContent = st.emoji;
  $("lv-label").textContent = st.label;
  $("lv-hours").textContent = `≈ ${st.hours} d'expérience humaine`;
  $("lv-blurb").textContent = st.blurb;
  $("lv-ladder").replaceChildren(...STAGES.map((s, i) => {
    const d = document.createElement("div");
    d.className = "tr-step" + (i < idx ? " done" : i === idx ? " now" : "");
    d.textContent = `${s.emoji} ${s.short}`;
    return d;
  }));
}

function fmtEta(min) {
  if (min == null || !isFinite(min)) return "—";
  const h = Math.floor(min / 60), m = Math.round(min % 60);
  return h > 0 ? `${h} h ${String(m).padStart(2, "0")}` : `${m} min`;
}

/* ------------------------------------------------------------- bloc CFR */

function renderCfr(cfr, cfrMetrics) {
  const section = $("cfr-section");
  if (!cfr) { section.hidden = true; return; }
  section.hidden = false;
  $("c-iters").textContent = cfr.iters.toLocaleString("fr-FR");
  // Objectif déduit du ticker : itérations faites + (vitesse × temps restant)
  const target = cfr.iters + cfr.speed * cfr.eta_min * 60;
  const pct = target > 0 ? Math.min(100, cfr.iters / target * 100) : 100;
  $("c-iters-pct").textContent = `${pct.toFixed(1)} % de l'objectif`;
  $("c-infosets").textContent = cfr.infosets.toLocaleString("fr-FR");
  $("c-speed").textContent = cfr.speed.toFixed(0);
  $("c-eta").textContent = fmtEta(cfr.eta_min);
  const last = cfrMetrics.length ? cfrMetrics[cfrMetrics.length - 1] : null;
  for (const [id, val] of [["c-rule", last ? last.vs_regles : null],
                           ["c-dqn", last ? last.vs_dqn : null]]) {
    const el = $(id);
    el.textContent = (val == null || !isFinite(val)) ? "—" : `${val > 0 ? "+" : ""}${val.toFixed(0)}`;
    el.style.color = (val == null || !isFinite(val)) ? "" : (val >= 0 ? "var(--good)" : "var(--bad)");
  }
  drawCfrChart(cfrMetrics);
}

function drawCfrChart(metrics) {
  const canvas = $("cfr-chart");
  const wrap = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const W = wrap.clientWidth, H = wrap.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  $("cfr-empty").hidden = metrics.length > 0;
  if (!metrics.length) return;

  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const P = { l: 58, r: 14, t: 10, b: 24 };
  const iw = W - P.l - P.r, ih = H - P.t - P.b;
  const series = [
    { color: css("--series-3"), pts: metrics.map(m => [m.iters, m.vs_regles]) },
    { color: css("--series-2"), pts: metrics.filter(m => isFinite(m.vs_dqn)).map(m => [m.iters, m.vs_dqn]) },
  ];
  const xMax = Math.max(metrics[metrics.length - 1].iters, 1);
  const ys = series.flatMap(s => s.pts.map(p => p[1])).concat([0]);
  let yMin = Math.min(...ys), yMax2 = Math.max(...ys);
  const pad = Math.max((yMax2 - yMin) * 0.15, 20);
  yMin -= pad; yMax2 += pad;
  const X = (v) => P.l + (v / xMax) * iw;
  const Y = (v) => P.t + (1 - (v - yMin) / (yMax2 - yMin)) * ih;

  ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  ctx.fillStyle = css("--muted"); ctx.font = "10.5px system-ui"; ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const v = yMin + (i / 4) * (yMax2 - yMin);
    ctx.beginPath(); ctx.moveTo(P.l, Y(v)); ctx.lineTo(W - P.r, Y(v)); ctx.stroke();
    ctx.fillText(Math.round(v).toLocaleString("fr-FR"), P.l - 7, Y(v) + 3.5);
  }
  ctx.strokeStyle = css("--baseline"); ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(P.l, Y(0)); ctx.lineTo(W - P.r, Y(0)); ctx.stroke();
  ctx.textAlign = "center";
  ctx.fillText("itérations →", W / 2, H - 6);

  for (const s of series) {
    if (!s.pts.length) continue;
    ctx.strokeStyle = s.color; ctx.lineWidth = 2;
    ctx.beginPath();
    s.pts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(X(x), Y(y)) : ctx.lineTo(X(x), Y(y)));
    ctx.stroke();
    ctx.fillStyle = s.color;
    s.pts.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(X(x), Y(y), 3, 0, Math.PI * 2); ctx.fill(); });
  }
}

function renderTiles(progress, training) {
  const last = progress.length ? progress[progress.length - 1] : null;
  const lastM = training.length ? training[training.length - 1] : null;
  if (last) {
    const total = last.eta_min === 0 && last.hands > 0 ? last.hands : TOTAL_DEFAULT;
    $("t-hands").textContent = last.hands.toLocaleString("fr-FR");
    $("t-hands-pct").textContent = `${Math.min(100, (last.hands / total) * 100).toFixed(1)} % de l'objectif`;
    $("t-speed").textContent = last.speed.toFixed(0);
    $("t-eta").textContent = fmtEta(last.eta_min);
    $("t-eps").textContent = `${(last.eps * 100).toFixed(0)} %`;
    $("t-loss").textContent = last.loss == null ? "—" : last.loss.toFixed(4);
  }
  $("t-rule").textContent = lastM ? `${lastM.vs_regles > 0 ? "+" : ""}${lastM.vs_regles.toFixed(0)}` : "—";
  $("t-rule").style.color = lastM ? (lastM.vs_regles >= 0 ? "var(--good)" : "var(--bad)") : "";
}

/* ------------------------------------------------------------- courbe */

function movingAvg(values, k) {
  const out = new Array(values.length);
  let sum = 0;
  const q = [];
  for (let i = 0; i < values.length; i++) {
    q.push(values[i]); sum += values[i];
    if (q.length > k) sum -= q.shift();
    out[i] = sum / q.length;
  }
  return out;
}

function drawChart(progress, training) {
  const canvas = $("tr-chart");
  const wrap = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const W = wrap.clientWidth, H = wrap.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  $("tr-empty").hidden = progress.length > 0;
  if (!progress.length) return;

  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const P = { l: 58, r: 14, t: 12, b: 30 };
  const iw = W - P.l - P.r, ih = H - P.t - P.b;

  const xs = progress.map(p => p.hands);
  const raw = progress.map(p => p.bb100_recent);
  const smooth = movingAvg(raw, 25);
  const xMax = Math.max(xs[xs.length - 1], training.length ? training[training.length - 1].hands : 0, 1000);

  const ys = smooth.concat(training.map(t => t.vs_regles), [0]);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  const pad = Math.max((yMax - yMin) * 0.12, 20);
  yMin -= pad; yMax += pad;

  const X = (h) => P.l + (h / xMax) * iw;
  const Y = (v) => P.t + (1 - (v - yMin) / (yMax - yMin)) * ih;

  // grille + axes
  ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  ctx.fillStyle = css("--muted"); ctx.font = "10.5px system-ui"; ctx.textAlign = "right";
  const ySteps = 5;
  for (let i = 0; i <= ySteps; i++) {
    const v = yMin + (i / ySteps) * (yMax - yMin);
    const y = Y(v);
    ctx.beginPath(); ctx.moveTo(P.l, y); ctx.lineTo(W - P.r, y); ctx.stroke();
    ctx.fillText(Math.round(v).toLocaleString("fr-FR"), P.l - 7, y + 3.5);
  }
  ctx.textAlign = "center";
  const xTick = xMax > 150000 ? 50000 : xMax > 40000 ? 20000 : xMax > 8000 ? 5000 : 1000;
  for (let h = 0; h <= xMax; h += xTick) {
    ctx.fillText(h >= 1000 ? `${h / 1000}k` : String(h), X(h), H - P.b + 16);
  }
  // ligne zéro
  ctx.strokeStyle = css("--baseline"); ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(P.l, Y(0)); ctx.lineTo(W - P.r, Y(0)); ctx.stroke();

  // série brute (fantôme) + lissée
  const drawLine = (vals, color, width, alpha) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.globalAlpha = alpha;
    ctx.beginPath();
    vals.forEach((v, i) => {
      const x = X(xs[i]), y = Math.max(P.t, Math.min(H - P.b, Y(v)));
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke(); ctx.globalAlpha = 1;
  };
  drawLine(raw, css("--series-1"), 1, 0.22);
  drawLine(smooth, css("--series-1"), 2.2, 1);

  // mesures réelles vs bot à règles
  if (training.length) {
    ctx.strokeStyle = css("--series-2"); ctx.lineWidth = 2;
    ctx.beginPath();
    training.forEach((t, i) => {
      const x = X(t.hands), y = Y(t.vs_regles);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = css("--series-2");
    training.forEach((t) => {
      ctx.beginPath(); ctx.arc(X(t.hands), Y(t.vs_regles), 3.5, 0, Math.PI * 2); ctx.fill();
    });
  }
}

/* ------------------------------------------------------------ refresh */

let lastOk = 0;

async function refresh() {
  try {
    const r = await fetch("/api/progress");
    if (!r.ok) throw new Error(r.status);
    const data = await r.json();
    lastOk = Date.now();
    $("tr-dot").classList.remove("stale");
    $("tr-updated").textContent = `en direct — maj à ${new Date().toLocaleTimeString("fr-FR")}`;
    renderTiles(data.progress, data.training);
    renderLevel(data.progress, data.training, data.cfr_metrics || []);
    renderCfr(data.cfr || null, data.cfr_metrics || []);
    drawChart(data.progress, data.training);
  } catch (e) {
    $("tr-dot").classList.add("stale");
    $("tr-updated").textContent = lastOk
      ? `connexion perdue (dernière maj ${new Date(lastOk).toLocaleTimeString("fr-FR")})`
      : "serveur injoignable — lance python server.py";
  }
}

refresh();
setInterval(refresh, REFRESH_MS);
window.addEventListener("resize", () => refresh());
