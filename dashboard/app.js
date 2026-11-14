const INR = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});
const INR2 = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 2,
});
const PCT = (x) => `${(Number(x) * 100).toFixed(2)}%`;
const rng = (a) => (Array.isArray(a) && a.length === 2 ? `${a[0]}–${a[1]}` : "—");

function $(id) {
  return document.getElementById(id);
}

function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    el.hidden = true;
  }, 2800);
}

function setLog(obj) {
  $("log").textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
}

async function fetchSnapshot() {
  try {
    const r = await fetch("/api/snapshot?fast=1", { cache: "no-store" });
    if (r.ok) return await r.json();
  } catch (_) {}
  const r2 = await fetch("/data/snapshot.json", { cache: "no-store" });
  if (!r2.ok) throw new Error("Could not load desk snapshot");
  return await r2.json();
}

function renderPerf(p) {
  p = p || {};
  $("perfLive").textContent = p.live ?? p.suggestions_live ?? "—";
  $("perfOk").textContent = p.successful ?? p.suggestions_hit_target ?? "—";
  $("perfFail").textContent = p.failed ?? p.suggestions_hit_sl ?? "—";
  $("perfWR").textContent =
    p.suggestion_win_rate != null ? PCT(p.suggestion_win_rate) : "—";
  $("perfPnl").textContent = INR2.format(p.journal_realized_pnl || 0);
  $("perfExp").textContent = p.suggestions_expired ?? "—";
}

function renderJournal(rows) {
  const body = $("journalBody");
  body.innerHTML = "";
  rows = rows || [];
  $("journalEmpty").hidden = rows.length > 0;
  for (const j of rows.slice(0, 20)) {
    const tr = document.createElement("tr");
    const pnl = j.pnl == null ? "—" : INR2.format(j.pnl);
    const cls = j.pnl == null ? "" : j.pnl >= 0 ? "up" : "down";
    const actions =
      j.status === "open"
        ? `<button type="button" class="btn ghost btn-close" data-id="${j.id}">Close</button>
           <button type="button" class="btn ghost btn-edit" data-id="${j.id}" data-entry="${j.entry_price}">Edit entry</button>`
        : "";
    tr.innerHTML = `
      <td>${j.stock}</td>
      <td>${j.qty}</td>
      <td>${INR2.format(j.entry_price)}</td>
      <td>${j.exit_price != null ? INR2.format(j.exit_price) : "—"}</td>
      <td class="${cls}">${pnl}</td>
      <td>${j.status}</td>
      <td>${actions}</td>`;
    body.appendChild(tr);
  }
  body.querySelectorAll(".btn-close").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const exit_price = Number(prompt("Exit / sell price?", ""));
      if (!exit_price) return;
      const r = await fetch("/api/journal/close", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: btn.dataset.id, exit_price }),
      });
      const data = await r.json();
      toast(data.ok ? `Closed · PnL ${data.closed.pnl}` : "Close failed");
      load();
    });
  });
  body.querySelectorAll(".btn-edit").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const entry_price = Number(prompt("Update entry price", btn.dataset.entry));
      if (!entry_price) return;
      await fetch("/api/journal/update-entry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: btn.dataset.id, entry_price }),
      });
      toast("Entry updated");
      load();
    });
  });
}

function renderOpportunities(snap) {
  const oppBody = $("oppBody");
  const cards = $("oppCards");
  oppBody.innerHTML = "";
  cards.innerHTML = "";
  const scan = snap.scan || {};
  $("scanMeta").textContent = scan.scanned
    ? `Scanned ${scan.scanned} · ${scan.regime || ""} · news+zones gated`
    : "Live suggestions";
  const list = snap.opportunities || [];
  for (const o of list.slice(0, 20)) {
    const tr = document.createElement("tr");
    const grade = o.quality_grade || "C";
    const exp = o.expires_at
      ? new Date(o.expires_at).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })
      : "—";
    tr.innerHTML = `
      <td><span class="grade grade-${grade}">${grade}</span></td>
      <td>${o.stock}</td>
      <td class="signal-${o.signal}">${o.signal}</td>
      <td>${rng(o.entry_range) || o.buy || "—"}</td>
      <td>${rng(o.exit_range) || o.sell || "—"}</td>
      <td>${rng(o.sl_range) || o.stop || "—"}</td>
      <td>${o.rr ?? "—"}</td>
      <td>${exp}</td>
      <td class="why-cell">${o.why_chosen || o.why || "—"}</td>`;
    oppBody.appendChild(tr);
  }
  for (const o of (snap.actionable || list.filter((x) => x.signal === "BUY")).slice(0, 6)) {
    const div = document.createElement("div");
    div.className = "card-opp";
    div.innerHTML = `
      <h3>${o.stock} <span class="grade grade-${o.quality_grade || "C"}">${o.quality_grade || "C"}</span></h3>
      <p>${o.strategy || ""} · expires ${o.expires_in_hours ?? "?"}h</p>
      <dl>
        <dt>Entry rng</dt><dd>${rng(o.entry_range)}</dd>
        <dt>Exit rng</dt><dd>${rng(o.exit_range)}</dd>
        <dt>SL rng</dt><dd>${rng(o.sl_range)}</dd>
        <dt>RR</dt><dd>${o.rr ?? "—"}</dd>
      </dl>
      <p class="why-cell">${o.why_chosen || ""}</p>
      <button type="button" class="btn primary btn-mark" data-stock="${o.stock}" data-buy="${o.buy}" data-stop="${o.stop}" data-sell="${o.sell}" data-sid="${o.suggestion_id || ""}">Mark my entry</button>`;
    cards.appendChild(div);
  }
  cards.querySelectorAll(".btn-mark").forEach((btn) => {
    btn.addEventListener("click", () => {
      const f = $("journalForm");
      f.stock.value = btn.dataset.stock;
      f.entry_price.value = btn.dataset.buy;
      f.stop.value = btn.dataset.stop;
      f.target.value = btn.dataset.sell;
      f.suggestion_id.value = btn.dataset.sid || "";
      f.qty.focus();
      toast("Fill qty and submit to journal");
    });
  });
}

function renderHistory(rows) {
  const body = $("histBody");
  body.innerHTML = "";
  for (const o of (rows || []).slice(0, 30)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${o.created_at ? new Date(o.created_at).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" }) : "—"}</td>
      <td>${o.stock}</td>
      <td>${o.status || "—"}</td>
      <td>${o.buy ?? "—"}</td>
      <td>${o.sell ?? "—"}</td>
      <td>${o.stop ?? "—"}</td>
      <td class="why-cell">${o.why_chosen || "—"}</td>`;
    body.appendChild(tr);
  }
}

function render(snap) {
  $("tagline").textContent = snap.tagline || "Adaptive Indian equities";
  $("clock").textContent = new Date(snap.generated_at || Date.now()).toLocaleString("en-IN", {
    timeZone: "Asia/Kolkata",
  });
  const open = !!snap.market_open;
  $("marketPill").textContent = open ? "NSE OPEN" : "NSE CLOSED";
  $("marketPill").className = `pill ${open ? "live" : "closed"}`;

  const reg = snap.regime || {};
  $("regimeEyebrow").textContent = `Regime · conf ${(reg.confidence ?? 0).toFixed(2)}`;
  $("regimeTitle").textContent = reg.regime || "—";
  $("regimeNotes").textContent = reg.notes || "";
  $("vix").textContent = reg.india_vix != null ? Number(reg.india_vix).toFixed(2) : "—";
  $("niftyLast").textContent = reg.nifty_last != null ? Number(reg.nifty_last).toFixed(0) : "—";
  $("breadth").textContent = reg.breadth != null ? Number(reg.breadth).toFixed(2) : "—";
  $("confidence").textContent = reg.confidence != null ? Number(reg.confidence).toFixed(2) : "—";
  $("universe").textContent = snap.universe_size ?? "—";

  const acct = snap.account || {};
  $("equityMtm").textContent = INR.format(acct.equity_mtm ?? 0);
  const ret = Number(acct.mtm_return || 0);
  $("mtmReturn").textContent = PCT(ret);
  $("mtmReturn").className = `stat-value ${ret >= 0 ? "up" : "down"}`;
  $("cash").textContent = INR.format(acct.cash ?? 0);
  $("openPos").textContent = `${acct.open_positions ?? 0} / 3`;

  const research = snap.research || {};
  const promoted = $("promotedList");
  promoted.innerHTML = "";
  for (const id of research.promoted || []) {
    const li = document.createElement("li");
    const sharpe = id.includes("adaptive")
      ? research.adaptive_oos_sharpe
      : research.dual_momentum_oos_sharpe;
    li.innerHTML = `<span>${id}</span><span>OOS ${Number(sharpe).toFixed(2)}</span>`;
    promoted.appendChild(li);
  }
  $("researchNote").textContent = research.note || "";

  const broker = snap.broker || {};
  $("zStatus").textContent = broker.message || "Manual journal";
  if (broker.kite_web) $("kiteLink").href = broker.kite_web;

  renderPerf(snap.performance);
  renderJournal(snap.journal);

  const posBody = $("posBody");
  posBody.innerHTML = "";
  const positions = snap.positions || [];
  $("posEmpty").hidden = positions.length > 0;
  for (const p of positions) {
    const tr = document.createElement("tr");
    const cls = p.unrealized_pnl >= 0 ? "up" : "down";
    tr.innerHTML = `
      <td>${p.symbol}</td>
      <td>${p.qty}</td>
      <td>${INR2.format(p.buy ?? p.entry)}</td>
      <td>${INR2.format(p.sell ?? p.t1)}</td>
      <td>${INR2.format(p.last)}</td>
      <td class="${cls}">${INR2.format(p.unrealized_pnl)} (${PCT(p.unrealized_pct)})</td>
      <td>${INR2.format(p.stop)}</td>
      <td>${p.rr ?? "—"}</td>`;
    posBody.appendChild(tr);
  }

  renderOpportunities(snap);
  renderHistory(snap.suggestion_history);

  const grid = $("metricsGrid");
  grid.innerHTML = "";
  for (const m of (snap.metrics || []).slice().reverse().slice(0, 8)) {
    const div = document.createElement("div");
    div.className = "metric";
    div.innerHTML = `
      <div class="k">${m.oos ? "OOS" : "IS"} · ${m.strategy_id}</div>
      <div class="v">Sharpe ${Number(m.sharpe).toFixed(2)}</div>
      <div class="s">ret ${PCT(m.total_return)} · dd ${PCT(m.max_drawdown)}</div>`;
    grid.appendChild(div);
  }
  $("generatedAt").textContent = `Snapshot ${snap.generated_at || ""}`;
}

async function load() {
  try {
    const snap = await fetchSnapshot();
    render(snap);
    setLog({
      status: "loaded",
      perf: snap.performance,
      actionable: (snap.actionable || []).length,
      history: (snap.suggestion_history || []).length,
    });
  } catch (err) {
    setLog(String(err));
    toast("Failed to load desk data");
  }
}

async function runSession() {
  const btn = $("btnSession");
  btn.disabled = true;
  try {
    const r = await fetch("/api/session?mode=auto", { method: "POST" });
    const data = await r.json();
    if (data.snapshot) render(data.snapshot);
    setLog(data.session || data);
    toast("Session complete");
  } catch (err) {
    toast("Session API unavailable");
    setLog(String(err));
  } finally {
    btn.disabled = false;
  }
}

async function refreshLive() {
  const btn = $("btnRefresh");
  btn.disabled = true;
  try {
    let snap;
    try {
      const r = await fetch("/api/refresh", { method: "POST" });
      if (r.ok) snap = await r.json();
    } catch (_) {}
    if (!snap) snap = await fetchSnapshot();
    render(snap);
    toast("Desk refreshed");
  } catch (err) {
    toast("Refresh failed");
    setLog(String(err));
  } finally {
    btn.disabled = false;
  }
}

async function analyzeNow() {
  const btn = $("btnAnalyze");
  btn.disabled = true;
  toast("Running news + zones analysis…");
  try {
    const r = await fetch("/api/analyze?limit=50", { method: "POST" });
    const data = await r.json();
    if (data.snapshot) render(data.snapshot);
    setLog(data.feed && data.feed.counts ? data.feed : data);
    toast("Analysis updated");
  } catch (err) {
    toast("Analyze unavailable");
    setLog(String(err));
  } finally {
    btn.disabled = false;
  }
}

$("journalForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  const body = {
    stock: f.stock.value,
    qty: Number(f.qty.value),
    entry_price: Number(f.entry_price.value),
    stop: f.stop.value ? Number(f.stop.value) : null,
    target: f.target.value ? Number(f.target.value) : null,
    suggestion_id: f.suggestion_id.value || null,
  };
  const r = await fetch("/api/journal/open", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  toast(data.ok ? "Entry marked" : "Failed");
  f.reset();
  load();
});

$("btnSession").addEventListener("click", runSession);
$("btnRefresh").addEventListener("click", refreshLive);
$("btnAnalyze").addEventListener("click", analyzeNow);
load();
setInterval(load, 90000);
