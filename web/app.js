const chat = document.getElementById("chat");
const tickerEl = document.getElementById("ticker");
const daysEl = document.getElementById("days");
const modelEl = document.getElementById("model");
const msgEl = document.getElementById("msg");

const chartEmpty = document.getElementById("chartEmpty");
const chartSub = document.getElementById("chartSub");
const statusText = document.getElementById("statusText");
const kpiRow = document.getElementById("kpiRow");

const plotDiv = document.getElementById("plotlyChart");

function addMessage(who, text, extraNode = null) {
  const div = document.createElement("div");
  div.className = "msg";

  const head = document.createElement("div");
  head.innerHTML = `<b>${escapeHtml(who)}:</b> ${escapeHtml(text).replace(/\n/g, "<br>")}`;
  div.appendChild(head);

  if (extraNode) div.appendChild(extraNode);

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function setStatus(text) {
  if (statusText) statusText.textContent = text;
}

function escapeHtml(str) {
  return (str || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function clearKPIs(){
  kpiRow.innerHTML = "";
}

function renderKPIsSingle(metrics) {
  clearKPIs();
  if (!metrics) return;

  const items = [
    {k: "cagr", label: "CAGR"},
    {k: "sharpe", label: "Sharpe"},
    {k: "vol_annual", label: "Volatility (ann)"},
    {k: "max_drawdown", label: "Max Drawdown"},
    {k: "risk", label: "Risk"}
  ];

  items.forEach(it => {
    const card = document.createElement("div");
    card.className = "kpiCard";

    const lbl = document.createElement("div");
    lbl.className = "kpiLabel";
    lbl.textContent = it.label;

    const val = document.createElement("div");
    val.className = "kpiValue";

    let v = metrics[it.k];
    if (v === null || v === undefined || Number.isNaN(v)) v = "n/a";
    else if (it.k === "risk") v = String(v).toUpperCase();
    else if (["cagr","vol_annual","max_drawdown"].includes(it.k)) v = (v * 100).toFixed(2) + "%";
    else v = typeof v === "number" ? v.toFixed(2) : v;

    val.textContent = v;

    card.appendChild(lbl);
    card.appendChild(val);
    kpiRow.appendChild(card);
  });
}

function renderKPIsCompare(bestOf) {
  clearKPIs();
  if (!bestOf) return;

  const items = [
    {k: "best_cagr", label: "Best CAGR"},
    {k: "best_sharpe", label: "Best Sharpe"},
    {k: "lowest_vol", label: "Lowest Vol"},
    {k: "least_drawdown", label: "Least Drawdown"},
  ];

  items.forEach(it => {
    const card = document.createElement("div");
    card.className = "kpiCard";

    const lbl = document.createElement("div");
    lbl.className = "kpiLabel";
    lbl.textContent = it.label;

    const val = document.createElement("div");
    val.className = "kpiValue";
    val.textContent = bestOf[it.k] || "n/a";

    card.appendChild(lbl);
    card.appendChild(val);
    kpiRow.appendChild(card);
  });
}

/* ---------- Plotly rendering ---------- */
function renderPlotlyFigure(fig, subtitle = "") {
  if (!fig) return;

  chartEmpty.style.display = "none";
  if (chartSub) chartSub.textContent = subtitle || "Hover, zoom, toggle";

  const layout = fig.layout || {};
  layout.font = layout.font || {family: "Inter, system-ui, sans-serif", color: "rgba(234,240,255,0.90)"};
  layout.xaxis = layout.xaxis || {};
  layout.yaxis = layout.yaxis || {};
  layout.xaxis.zeroline = false;
  layout.yaxis.zeroline = false;

  const config = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d"]
  };

  Plotly.react(plotDiv, fig.data || [], layout, config);
}

/* ---------- Compare heatmap table ---------- */
function clamp01(x){ return Math.max(0, Math.min(1, x)); }
function isNum(x){ return typeof x === "number" && Number.isFinite(x); }

function makeHeatColor(score){
  const a = 0.10 + 0.28 * score;
  const r = 0.10 + 0.28 * (1-score);
  if (score >= 0.5){
    return `rgba(34,197,94,${a.toFixed(3)})`;
  }
  return `rgba(239,68,68,${r.toFixed(3)})`;
}

function fmtPct(x){ return !isNum(x) ? "n/a" : (x*100).toFixed(2) + "%"; }
function fmtNum(x){ return !isNum(x) ? "n/a" : x.toFixed(2); }

function buildCompareTable(rows){
  const metrics = [
    {key:"cagr", label:"CAGR", fmt: fmtPct, higherBetter:true},
    {key:"sharpe", label:"Sharpe", fmt: fmtNum, higherBetter:true},
    {key:"vol_annual", label:"Vol (ann)", fmt: fmtPct, higherBetter:false},
    {key:"max_drawdown", label:"Max DD", fmt: fmtPct, higherBetter:true},
    {key:"risk", label:"Risk", fmt: (x)=> String(x||"").toUpperCase(), noHeat:true},
  ];

  const stats = {};
  metrics.forEach(m => {
    if (m.noHeat) return;
    const vals = rows.map(r => r[m.key]).filter(isNum);
    if (!vals.length) return;
    stats[m.key] = {min: Math.min(...vals), max: Math.max(...vals)};
  });

  const wrap = document.createElement("div");
  wrap.className = "compareWrap";

  const table = document.createElement("table");
  table.className = "compareTable";

  const thead = document.createElement("thead");
  const trh = document.createElement("tr");

  const th0 = document.createElement("th");
  th0.textContent = "Ticker";
  trh.appendChild(th0);

  metrics.forEach(m => {
    const th = document.createElement("th");
    th.textContent = m.label;
    trh.appendChild(th);
  });
  thead.appendChild(trh);

  const tbody = document.createElement("tbody");

  rows.forEach(r => {
    const tr = document.createElement("tr");

    const tdTicker = document.createElement("td");
    tdTicker.textContent = r.ticker;
    tdTicker.className = "compareTicker";
    tr.appendChild(tdTicker);

    metrics.forEach(m => {
      const td = document.createElement("td");
      td.className = "heatCell";

      const v = r[m.key];
      td.textContent = m.fmt(v);

      if (!m.noHeat && isNum(v) && stats[m.key]) {
        const {min, max} = stats[m.key];
        const denom = (max - min) === 0 ? 1 : (max - min);
        let score = (v - min) / denom;
        score = clamp01(score);
        if (!m.higherBetter) score = 1 - score;
        td.style.backgroundColor = makeHeatColor(score);
      }

      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  wrap.appendChild(table);
  return wrap;
}

/* ---------- API ---------- */
async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload || {})
  });

  const ct = res.headers.get("content-type") || "";
  if (!ct.includes("application/json")) {
    const txt = await res.text();
    throw new Error(`Server error (non-JSON): ${txt.slice(0, 200)}...`);
  }

  const data = await res.json();
  if (!res.ok) throw new Error(data.response || "Request failed");
  return data;
}

async function handleLoad() {
  try {
    setStatus("Loading…");
    const data = await postJSON("/api/load", {});
    addMessage("Assistant", data.response);
    setStatus("Loaded");
  } catch (e) {
    addMessage("Assistant", `Error: ${e.message}`);
    setStatus("Ready");
  }
}

async function handleSummary() {
  const t = (tickerEl.value || "").trim();
  try {
    setStatus("Working…");
    const data = await postJSON("/api/summary", {ticker: t});
    addMessage("Assistant", data.response);
    if (data.plotly) renderPlotlyFigure(data.plotly, `${t.toUpperCase()} Close Price`);
    renderKPIsSingle(data.metrics);
    setStatus("Ready");
  } catch (e) {
    addMessage("Assistant", `Error: ${e.message}`);
    setStatus("Ready");
  }
}

async function handleRisk() {
  const t = (tickerEl.value || "").trim();
  try {
    setStatus("Working…");
    const data = await postJSON("/api/risk", {ticker: t});
    addMessage("Assistant", data.response);
    if (data.plotly) renderPlotlyFigure(data.plotly, `${t.toUpperCase()} Risk View`);
    renderKPIsSingle(data.metrics);
    setStatus("Ready");
  } catch (e) {
    addMessage("Assistant", `Error: ${e.message}`);
    setStatus("Ready");
  }
}

async function handleForecast() {
  const t = (tickerEl.value || "").trim();
  const days = parseInt(daysEl.value || "30", 10);
  const model = (modelEl.value || "arima").trim();
  try {
    setStatus("Working…");
    const data = await postJSON("/api/forecast", {ticker: t, days, model});
    addMessage("Assistant", data.response);
    if (data.plotly) renderPlotlyFigure(data.plotly, `${t.toUpperCase()} Forecast`);
    renderKPIsSingle(data.metrics);
    setStatus("Ready");
  } catch (e) {
    addMessage("Assistant", `Error: ${e.message}`);
    setStatus("Ready");
  }
}

async function handleCompare() {
  const s = (tickerEl.value || "").trim();
  try {
    setStatus("Working…");
    const data = await postJSON("/api/compare", {tickers: s});

    const tableNode = buildCompareTable(data.compare || []);
    addMessage("Assistant", data.response, tableNode);

    if (data.plotly) renderPlotlyFigure(data.plotly, `Compare: ${(data.tickers || []).join(" vs ")}`);
    renderKPIsCompare(data.kpis_compare);

    setStatus("Ready");
  } catch (e) {
    addMessage("Assistant", `Error: ${e.message}`);
    setStatus("Ready");
  }
}

async function handleSend() {
  const msg = (msgEl.value || "").trim();
  if (!msg) return;
  addMessage("You", msg);
  msgEl.value = "";
  addMessage("Assistant", "Use the buttons (Summary/Risk/Forecast/Compare) for interactive charts.");
}

document.getElementById("btnLoad").addEventListener("click", handleLoad);
document.getElementById("btnSummary").addEventListener("click", handleSummary);
document.getElementById("btnRisk").addEventListener("click", handleRisk);
document.getElementById("btnForecast").addEventListener("click", handleForecast);
document.getElementById("btnCompare").addEventListener("click", handleCompare);
document.getElementById("btnSend").addEventListener("click", handleSend);

addMessage("Assistant", "Welcome! Press Load, then use Summary / Risk / Forecast / Compare. Forecast uses ARIMA/Prophet with confidence bands.");
chartEmpty.style.display = "block";
clearKPIs();
