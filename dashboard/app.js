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
  // Prefer live API; fall back to static snapshot for hosted/static mode
  try {
    const r = await fetch("/api/snapshot?fast=0", { cache: "no-store" });
    if (r.ok) return await r.json();
  } catch (_) {
    /* static host */
  }
  const r2 = await fetch("/data/snapshot.json", { cache: "no-store" });
  if (!r2.ok) throw new Error("Could not load desk snapshot");
  return await r2.json();
}

function render(snap) {
  const liveApi = snap._fromApi !== false;
  $("tagline").textContent = snap.tagline || "Adaptive Indian equities — paper ₹1L";
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
  const items = (research.promoted || []).map((id) => {
    const sharpe =
      id.includes("adaptive") ? research.adaptive_oos_sharpe : research.dual_momentum_oos_sharpe;
    return { id, sharpe };
  });
  if (!items.length) {
    promoted.innerHTML = "<li><span>None yet</span><span>—</span></li>";
  } else {
    for (const it of items) {
      const li = document.createElement("li");
      li.innerHTML = `<span>${it.id}</span><span>OOS ${Number(it.sharpe).toFixed(2)}</span>`;
      promoted.appendChild(li);
    }
  }
  $("researchNote").textContent = research.note || "";

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
      <td>${INR2.format(p.entry)}</td>
      <td>${INR2.format(p.last)}</td>
      <td class="${cls}">${INR2.format(p.unrealized_pnl)} (${PCT(p.unrealized_pct)})</td>
      <td>${INR2.format(p.stop)}</td>
      <td>${p.strategy}</td>`;
    posBody.appendChild(tr);
  }

  const oppBody = $("oppBody");
  oppBody.innerHTML = "";
  const scan = snap.scan || {};
  $("scanMeta").textContent = scan.scanned
    ? `Scanned ${scan.scanned} · ${scan.regime || ""}`
    : "Scanner";
  for (const o of snap.opportunities || []) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${o.stock}</td>
      <td class="signal-${o.signal}">${o.signal}</td>
      <td>${o.score}</td>
      <td>${o.strategy}</td>
      <td>${o.rr ?? "—"}</td>`;
    oppBody.appendChild(tr);
  }

  const grid = $("metricsGrid");
  grid.innerHTML = "";
  const metrics = snap.metrics || [];
  const show = metrics.slice().reverse().slice(0, 8);
  if (!show.length) {
    grid.innerHTML = `<div class="metric"><div class="k">Walk-forward</div><div class="v">${Number(
      research.adaptive_oos_sharpe || 0
    ).toFixed(2)}</div><div class="s">Adaptive OOS Sharpe · stability ${Number(
      research.adaptive_oos_stability || 0
    ).toFixed(2)}</div></div>`;
  } else {
    for (const m of show) {
      const div = document.createElement("div");
      div.className = "metric";
      div.innerHTML = `
        <div class="k">${m.oos ? "OOS" : "IS"} · ${m.strategy_id}</div>
        <div class="v">Sharpe ${Number(m.sharpe).toFixed(2)}</div>
        <div class="s">ret ${PCT(m.total_return)} · dd ${PCT(m.max_drawdown)} · n=${m.trades}<br/>${m.version}</div>`;
      grid.appendChild(div);
    }
  }

  $("generatedAt").textContent = `Snapshot ${snap.generated_at || ""} · ${
    liveApi ? "live engine" : "static host"
  }`;
}

async function load() {
  try {
    const snap = await fetchSnapshot();
    render(snap);
    setLog({ status: "loaded", at: snap.generated_at, positions: (snap.positions || []).length });
  } catch (err) {
    setLog(String(err));
    toast("Failed to load desk data");
  }
}

async function runSession() {
  const btn = $("btnSession");
  btn.disabled = true;
  toast("Running paper session…");
  try {
    const r = await fetch("/api/session?mode=auto", { method: "POST" });
    if (!r.ok) {
      // Static host fallback message
      toast("Live session needs the ATI desk server");
      setLog(await r.text());
      return;
    }
    const data = await r.json();
    if (data.snapshot) render(data.snapshot);
    setLog(data.session || data);
    toast("Session complete");
  } catch (err) {
    toast("Session API unavailable on static host");
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

$("btnSession").addEventListener("click", runSession);
$("btnRefresh").addEventListener("click", refreshLive);
load();
setInterval(load, 120000);
