#!/usr/bin/env python
"""Generate the self-contained results dashboard (dashboard.html).

Reads the scored CSVs (batch1_100.csv + balanced_150.csv), computes the
metrics/story numbers, and emits a single offline HTML file with all data
embedded. Charts are hand-rolled SVG/Canvas via vanilla JS (no external deps).
"""
import json
import os

import pandas as pd

from . import config

CLASS_ORDER_3 = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
EMOTIONS_8 = ["anger", "anticipation", "disgust", "fear", "joy", "sadness",
              "surprise", "trust"]

# --------------------------------------------------------------------------
# Data prep
# --------------------------------------------------------------------------

def _strip_nan(v):
    if v is None:
        return None
    if isinstance(v, float) and v != v:  # NaN
        return None
    if isinstance(v, pd.Timestamp):
        return str(v.date())
    return v


def load_batch1():
    p = os.path.join(config.OUT_DIR, "batch1_100.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "rating": int(r["rating"]),
            "year": int(r["year"]),
            "verified": bool(r["verified_purchase"]),
            "helpful": int(r["helpful_vote"]),
            "gt": str(r["ground_truth"]),
            "pred": str(r["model_sentiment"]),
            "title": _strip_nan(r["title"]),
            "text": _strip_nan(r["text"]),
        })
    return rows


def _emo(v):
    """Normalize an emotion cell (empty/NaN -> empty string)."""
    if v is None:
        return ""
    s = str(v)
    return "" if s.strip() == "" or s.lower() == "nan" else s


def load_balanced():
    p = os.path.join(config.OUT_DIR, "balanced_150.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "rating": int(r["rating"]),
            "year": int(r["year"]),
            "verified": bool(r["verified_purchase"]),
            "helpful": int(r["helpful_vote"]),
            "gt": str(r["ground_truth"]),
            "pred": str(r["model_sentiment"]),
            "llm_emo": _emo(r["llm_emotion"]),
            "nrc_emo": _emo(r["nrc_emotion"]),
            "title": _strip_nan(r["title"]),
            "text": _strip_nan(r["text"]),
        })
    return rows


def stats(rows, classes):
    n = len(rows)
    correct = sum(r["gt"] == r["pred"] for r in rows)
    acc = correct / n if n else 0
    per_class = {}
    cm = {g: {p: 0 for p in classes} for g in classes}
    pred_counts = {c: 0 for c in classes}
    for r in rows:
        cm[r["gt"]][r["pred"]] += 1
        pred_counts[r["pred"]] = pred_counts.get(r["pred"], 0) + 1
    for c in classes:
        sub = [r for r in rows if r["gt"] == c]
        per_class[c] = {
            "n": len(sub),
            "correct": sum(r["gt"] == r["pred"] for r in sub),
            "acc": (sum(r["gt"] == r["pred"] for r in sub) / len(sub)) if sub else 0,
        }
    return {"n": n, "correct": correct, "acc": acc,
            "per_class": per_class, "confusion": cm, "pred_counts": pred_counts}


def star_dist(rows):
    d = {i: 0 for i in range(1, 6)}
    for r in rows:
        d[r["rating"]] += 1
    return d


def emotion_stats(rows):
    rows = [r for r in rows if r.get("llm_emo")]
    n = len(rows)
    agree = sum(str(r.get("llm_emo")) == str(r.get("nrc_emo")) for r in rows)
    nrc_unknown = sum(not str(r.get("nrc_emo")) for r in rows)
    # per-emotion: llm counts vs nrc counts
    llm_counts = {e: 0 for e in EMOTIONS_8}
    nrc_counts = {e: 0 for e in EMOTIONS_8}
    for r in rows:
        le = r.get("llm_emo")
        ne = r.get("nrc_emo")
        if le in llm_counts:
            llm_counts[le] += 1
        if ne in nrc_counts:
            nrc_counts[ne] += 1
    return {"n": n, "agree": agree,
            "agree_rate": (agree / n if n else 0),
            "nrc_unknown": nrc_unknown,
            "llm_counts": llm_counts, "nrc_counts": nrc_counts}


# --------------------------------------------------------------------------
# HTML template
# --------------------------------------------------------------------------

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gift-Card Review Sentiment — MBAX 6418</title>
<style>
:root{
  /* ---- theme tokens (recolor the whole page here) ---- */
  --bg:#f6f7f9; --surface:#ffffff; --surface-2:#eef1f6; --surface-3:#e4e8f0;
  --border:#e3e7ee; --text:#161d2b; --muted:#6b7487;
  --accent:#4f46e5; --accent-soft:rgba(79,70,229,.09); --accent-ink:#fff;
  --good:#0ea371; --good-soft:rgba(14,163,113,.12);
  --bad:#d63b3b;  --bad-soft:rgba(214,59,59,.11);
  --nei:#3b82f6;  --nei-soft:rgba(59,130,246,.12);
  --amb:#d99a06;  --amb-soft:rgba(217,154,6,.14);
  --neutral:#8a94a6; --shadow:0 1px 2px rgba(22,29,43,.05),0 8px 24px rgba(22,29,43,.06);
  --radius:14px;
  font-size:16px;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.5;-webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px;margin:0 auto;padding:40px 24px 80px}
/* header */
.masthead{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:16px;margin-bottom:8px}
h1{font-size:26px;margin:0;letter-spacing:-.02em;font-weight:750;display:flex;align-items:center;gap:10px}
h1 .dot{width:12px;height:12px;border-radius:50%;background:var(--accent);display:inline-block}
.sub{color:var(--muted);font-size:14px;margin:6px 0 0}
.pill{display:inline-flex;align-items:center;gap:8px;background:var(--surface);border:1px solid var(--border);
  padding:8px 14px;border-radius:999px;font-size:13px;font-weight:600;color:var(--muted);box-shadow:var(--shadow)}
.pill b{color:var(--text)}
/* dataset switcher */
.switch{display:flex;background:var(--surface-2);border:1px solid var(--border);border-radius:12px;padding:4px;gap:4px;margin:24px 0}
.switch button{border:0;background:transparent;padding:9px 18px;border-radius:9px;font-size:14px;font-weight:600;
  color:var(--muted);cursor:pointer;transition:.15s}
.switch button.active{background:var(--surface);color:var(--accent);box-shadow:var(--shadow)}
.sect{font-size:12px;text-transform:uppercase;letter-spacing:.09em;font-weight:700;color:var(--muted);
  margin:0 0 4px}
h2{font-size:20px;margin:0 0 16px;letter-spacing:-.01em}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px 22px;box-shadow:var(--shadow)}
.grid{display:grid;gap:16px}
.kpis{grid-template-columns:repeat(4,1fr);margin-bottom:16px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px 18px;box-shadow:var(--shadow)}
.kpi .lab{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;font-weight:600}
.kpi .val{font-size:28px;font-weight:780;letter-spacing:-.02em;margin-top:6px}
.kpi .val small{font-size:15px;font-weight:600;color:var(--muted)}
.kpi .sub2{font-size:12px;color:var(--muted);margin-top:4px}
.big{grid-template-columns:1.2fr .8fr}
.two{grid-template-columns:1fr 1fr}
/* bars */
.bar-row{display:grid;grid-template-columns:110px 1fr 54px;align-items:center;gap:10px;margin:9px 0;font-size:13px}
.bar-row .lab{color:var(--muted);font-weight:600;text-align:right}
.bar-track{background:var(--surface-2);border-radius:6px;height:20px;position:relative;overflow:hidden}
.bar-fill{height:100%;border-radius:6px;min-width:2px;transition:width .5s ease}
.bar-val{font-weight:700;text-align:right}
/* confusion matrix */
.cmtbl{border-collapse:collapse;width:100%;font-size:13px}
.cmtbl th,.cmtbl td{padding:8px 10px;text-align:center}
.cmtbl thead th{color:var(--muted);font-weight:600}
.cmtbl .rh{color:var(--muted);font-weight:600;text-align:right}
.cell{border-radius:8px;font-weight:700}
.cap{font-size:12px;color:var(--muted);text-align:center;margin-top:8px}
/* stacked legend */
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--muted);margin:10px 0 4px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{width:10px;height:10px;border-radius:3px;display:inline-block}
/* emotion */
.emo-grid{grid-template-columns:repeat(4,1fr)}
.emo-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px}
.emo-card .top{display:flex;justify-content:space-between;align-items:baseline;font-size:13px}
.emo-card .e{font-weight:700}
.emo-card .n{color:var(--muted)}
.emo-bar{height:6px;border-radius:4px;background:var(--surface-2);margin-top:10px;overflow:hidden}
.emo-fill{height:100%;border-radius:4px}
/* tables/table card */
.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:14px}
.filters{display:flex;flex-wrap:wrap;gap:8px}
.chip{border:1px solid var(--border);background:var(--surface);border-radius:999px;padding:7px 13px;
  font-size:13px;font-weight:600;color:var(--muted);cursor:pointer;transition:.12s}
.chip:hover{border-color:var(--accent);color:var(--accent)}
.chip.active{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.search{flex:1;min-width:220px;border:1px solid var(--border);background:var(--surface);
  border-radius:10px;padding:8px 12px;font-size:14px;color:var(--text)}
.search:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}
.count{font-size:13px;color:var(--muted);font-weight:600}
.count b{color:var(--text)}
.table-box{overflow:auto;max-height:520px;border:1px solid var(--border);border-radius:12px}
table.rev{border-collapse:collapse;width:100%;font-size:13px;min-width:860px}
table.rev thead{position:sticky;top:0;background:var(--surface);z-index:2}
table.rev th{text-align:left;padding:10px 12px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);border-bottom:1px solid var(--border);background:var(--surface)}
table.rev td{padding:10px 12px;border-bottom:1px solid var(--surface-2);vertical-align:top}
table.rev tr:hover td{background:var(--surface-2)}
.tag{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:700}
.tag.auto{line-height:1.7}
.t-pos{background:var(--good-soft);color:var(--good)}
.t-neu{background:var(--amb-soft);color:#9a6b00}
.t-neg{background:var(--bad-soft);color:var(--bad)}
.t-plain{background:var(--surface-3);color:var(--muted)}
.stars{color:var(--amb);letter-spacing:1px;white-space:nowrap}
.rev-title{font-weight:650}
.rev-text{color:var(--muted);max-width:340px}
.ok{color:var(--good);font-weight:700}
.no{color:var(--bad);font-weight:700}
.tiny{font-size:11px;color:var(--muted)}
.emo-pair{font-size:12px;white-space:nowrap}
.emo-pair .llm{color:var(--accent);font-weight:600}
.emo-pair .sep{color:var(--muted);margin:0 3px}
.emo-pair .nrc{color:#7c3aed;font-weight:600}
.foot{margin-top:48px;color:var(--muted);font-size:12px;text-align:center}
.foot a{color:var(--accent)}
@media(max-width:900px){.kpis{grid-template-columns:repeat(2,1fr)}.big,.two,.emo-grid{grid-template-columns:1fr}}
.note{font-size:12.5px;color:var(--muted);background:var(--surface-2);border:1px dashed var(--border);
  border-radius:10px;padding:8px 12px;margin-top:12px}
</style>
</head>
<body>
<div class="wrap">
  <header class="masthead">
    <div>
      <h1><span class="dot"></span>Gift-Card Review Sentiment</h1>
      <div class="sub">Amazon Reviews '23 · Gift Cards category &nbsp;·&nbsp; LLM: Qwen3.6-35B (OpenAI-compatible endpoint) &nbsp;·&nbsp; seed 6418</div>
    </div>
    <div class="pill">Reviews scored <b id="pillN">–</b></div>
  </header>

  <div class="switch" id="modelSwitch">
    <button data-m="balanced" class="active">Balanced · three-class (main)</button>
    <button data-m="batch1">First 100 · binary (imbalance demo)</button>
  </div>

  <div id="kpis" class="grid kpis"></div>

  <div class="grid big" style="margin-bottom:16px">
    <div class="card">
      <p class="sect">Star ratings in scored set</p>
      <div id="starChart"></div>
    </div>
    <div class="card">
      <p class="sect">Correct answer vs. model · confusion</p>
      <div id="confusion"></div>
    </div>
  </div>

  <div class="grid two" style="margin-bottom:16px">
    <div class="card">
      <p class="sect">Accuracy by class</p>
      <div id="classAcc"></div>
    </div>
    <div class="card">
      <p class="sect">Where the model goes wrong</p>
      <div id="mistakes"></div>
    </div>
  </div>

  <div class="card" id="emoCard" style="margin-bottom:16px">
    <p class="sect">Primary emotion — LLM vs. NRC word list</p>
    <h2 style="margin-bottom:6px">Two independent takes, compared</h2>
    <div id="emoSummary"></div>
    <div id="emoLegend" class="legend" style="margin-bottom:6px"></div>
    <div id="emoChart" class="grid emo-grid"></div>
  </div>

  <div class="card">
    <p class="sect">Explore the reviews</p>
    <h2 style="margin-bottom:6px">Per-review detail</h2>
    <div class="toolbar">
      <div class="filters" id="filters"></div>
      <input class="search" id="search" type="search" placeholder="Filter by title / review text…">
    </div>
    <div class="count">Showing <b id="showN">0</b> of <span id="totN">0</span> reviews</div>
    <div class="table-box"><table class="rev" id="revTbl">
      <thead><tr>
        <th>Rating</th><th>Ground truth</th><th>Model</th><th>Match</th>
        <th>Emotion (LLM → NRC)</th><th>Details</th>
      </tr></thead>
      <tbody></tbody>
    </table></div>
  </div>

  <div class="foot">Built with the <b>Amazon Reviews '23</b> dataset (McAuley Lab, UC San Diego) ·
  <a href="https://amazon-reviews-2023.github.io" target="_blank">amazon-reviews-2023.github.io</a></div>
</div>

<script>
const DATA = __DATA__;
let STATE = { model: "balanced", gt: null, match: null, q: "" };

function batches(){ return STATE.model==="balanced" ? DATA.balanced : DATA.batch1; }
function rows(){ return batches().rows; }
function cls(){ return batches().classes; }

function fmtPct(x){ return (100*x).toFixed(1)+"%"; }
function starsStr(n){ return "★".repeat(n) + "☆".repeat(5-n); }
function esc(s){ return String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function clsClass(c){return {POSITIVE:"t-pos",NEUTRAL:"t-neu",NEGATIVE:"t-neg"}[c]||"t-plain";}

function render(){
  const b = batches(), s = b.stats, d = s;
  document.getElementById("pillN").textContent = d.n;
  renderKpis(); renderStar(b); renderConfusion(b); renderClassAcc(b);
  renderMistakes(b); renderEmotions(b); renderFilters(b); renderTable(b);
}
/* ---------- KPIs ---------- */
function renderKpis(){
  const b = batches(), d = b.stats; let h = "";
  const mk=(lab,val,sub,c)=>`<div class="kpi"><div class="lab">${lab}</div><div class="val" style="color:${c}">${val}</div><div class="sub2">${sub}</div></div>`;
  h += mk("Reviews scored", d.n, "seed 6418 · repeatable sample", "var(--text)");
  h += mk("Overall accuracy", fmtPct(d.acc), d.correct+" of "+d.n+" match the rating", "var(--text)");
  const cols={POSITIVE:"var(--good)",NEUTRAL:"var(--amb)",NEGATIVE:"var(--bad)"};
  b.classes.forEach(c=>{
    const p=d.per_class[c];
    h += mk(c.charAt(0)+c.slice(1).toLowerCase()+" recall",
      fmtPct(p.acc), p.correct+" / "+p.n, cols[c]);
  });
  document.getElementById("kpis").innerHTML = h;
}
/* ---------- Star distribution ---------- */
function renderStar(b){
  const max = Math.max(...Object.values(b.star));
  let h="";
  for(let i=5;i>=1;i--){
    const n=b.star[i]; const w=(n/max)*100;
    h+=`<div class="bar-row"><div class="lab">${i} ★</div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(w,2)}%;background:${clsColor(i)}"></div></div>
      <div class="bar-val">${n}</div></div>`;
  }
  document.getElementById("starChart").innerHTML=h;
}
function clsColor(r){
  if(r>=4)return"var(--good)"; if(r===3)return"var(--amb)"; return"var(--bad)";
}
/* ---------- Confusion ---------- */
function renderConfusion(b){
  const s=b.stats, classes=b.classes;
  let h=`<table class="cmtbl"><thead><tr><th></th>`+classes.map(c=>`<th>Pred: ${c}</th>`).join("")+`<th>Row</th></tr></thead><tbody>`;
  classes.forEach(gt=>{
    let rowTot=0; classes.forEach(c=>rowTot+=s.confusion[gt][c]);
    h+=`<tr><td class="rh">${gt}</td>`;
    classes.forEach(pr=>{
      const v=s.confusion[gt][pr]; const cellOk=(gt===pr);
      const strength=rowTot? (v/rowTot):0;
      const cc=cellOk? "14,163,113":"214,59,59";
      const alpha=0.10+0.55*strength;
      h+=`<td><div class="cell" style="background:rgba(${cc},${alpha})">${v}</div></td>`;
    });
    h+=`<td class="rh">${rowTot}</td></tr>`;
  });
  h+=`</tbody></table><div class="cap">Rows = ground truth (from rating) · columns = model prediction · diagonal = correct</div>`;
  document.getElementById("confusion").innerHTML=h;
}
/* ---------- Accuracy by class ---------- */
function renderClassAcc(b){
  const s=b.stats; let h="";
  b.classes.forEach(c=>{
    const p=s.per_class[c];
    const w=p.acc*100;
    const col=c==="POSITIVE"?"var(--good)":c==="NEUTRAL"?"var(--amb)":"var(--bad)";
    h+=`<div class="bar-row"><div class="lab">${c}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(w,2)}%;background:${col}"></div></div>
      <div class="bar-val">${fmtPct(p.acc)}</div></div>`;
  });
  document.getElementById("classAcc").innerHTML=h;
}
/* ---------- Mistakes ---------- */
function renderMistakes(b){
  const s=b.stats; let parts=[];
  b.classes.forEach(gt=>{
    b.classes.forEach(pr=>{
      if(gt!==pr && s.confusion[gt][pr]>0){
        parts.push({gt,pr,v:s.confusion[gt][pr]});
      }
    });
  });
  parts.sort((a,c)=>c.v-a.v);
  const max=parts.length?parts[0].v:1;
  let h = parts.length? "":"<div class='note'>No mismatches in this set.</div>";
  parts.forEach(p=>{
    const w=(p.v/max)*100;
    h+=`<div class="bar-row"><div class="lab">${p.gt} → ${p.pr}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(w,4)}%;background:${p.pr==="POSITIVE"?"var(--good)":p.pr==="NEUTRAL"?"var(--amb)":"var(--bad)"}"></div></div>
      <div class="bar-val">${p.v}</div></div>`;
  });
  document.getElementById("mistakes").innerHTML=h;
}
/* ---------- Emotions ---------- */
function renderEmotions(b){
  const card=document.getElementById("emoCard");
  const emo=b.emotions;
  if(!emo || !emo.n){ card.style.display="none"; return; }
  card.style.display="block";
  const rate=fmtPct(emo.agree_rate);
  document.getElementById("emoSummary").innerHTML =
    `<div class="kpis" style="grid-template-columns:repeat(3,1fr);margin-bottom:14px">
      <div class="kpi"><div class="lab">Reviews w/ emotion</div><div class="val"><small>${emo.n}</small></div></div>
      <div class="kpi"><div class="lab">Agreement</div><div class="val" style="color:var(--accent)">${rate}</div><div class="sub2">${emo.agree} reviews agree</div></div>
      <div class="kpi"><div class="lab">NRC unmatched</div><div class="val"><small>${emo.nrc_unknown}</small></div><div class="sub2">no NRC emotion word found</div></div></div>`;
  document.getElementById("emoLegend").innerHTML =
    `<span><i style="background:var(--accent)"></i>LLM predicts</span><span><i style="background:#7c3aed"></i>NRC word-list derives</span>`;
  const order=["joy","trust","anticipation","surprise","sadness","anger","disgust","fear"];
  const colors={joy:"#eab308",trust:"#2563eb",anticipation:"#7c3aed",surprise:"#0891b2",
    sadness:"#475569",anger:"#dc2626",disgust:"#84cc16",fear:"#111827"};
  let h="";
  order.forEach(e=>{
    const llm=emo.llm_counts[e]||0, nrc=emo.nrc_counts[e]||0;
    const maxv=Math.max(llm,nrc,1);
    h+=`<div class="emo-card"><div class="top"><span class="e" style="color:${colors[e]}">${e}</span><span class="n">LLM ${llm} · NRC ${nrc}</span></div>
      <div class="emo-bar"><div class="emo-fill" style="width:${(llm/maxv)*100}%;background:var(--accent)"></div></div>
      <div class="emo-bar" style="margin-top:4px"><div class="emo-fill" style="width:${(nrc/maxv)*100}%;background:#7c3aed"></div></div>
      </div>`;
  });
  document.getElementById("emoChart").innerHTML=h;
}
/* ---------- Filters ---------- */
function renderFilters(b){
  const cont=document.getElementById("filters");
  const classes=b.classes;
  let h=`<button class="chip" data-f="gt" data-v="any">All ground truth</button>`+
    classes.map(c=>`<button class="chip" data-f="gt" data-v="${c}">Truth: ${c}</button>`).join("")+
    `<span style="align-self:center;width:1px;height:22px;background:var(--border)"></span>`+
    `<button class="chip" data-f="m" data-v="ok">✓ Correct</button>`+
    `<button class="chip" data-f="m" data-v="no">✗ Mismatch</button>`;
  cont.innerHTML=h;
  cont.querySelectorAll(".chip").forEach(bn=>{
    bn.addEventListener("click",()=>{
      const f=bn.dataset.f, v=bn.dataset.v;
      // group-wise toggle: clicking the active one turns the group off
      if(STATE[f]===v){ STATE[f]=null; }
      else { STATE[f]=v; }
      cont.querySelectorAll(`.chip[data-f="${f}"]`).forEach(x=>x.classList.toggle("active",x.dataset.v===STATE[f]));
      renderTable(b);
    });
  });
}
/* ---------- Table ---------- */
function renderTable(b){
  const q=STATE.q.trim().toLowerCase();
  const out=b.rows.filter(r=>{
    if(STATE.gt && r.gt!==STATE.gt) return false;
    if(STATE.m && ((STATE.m==="ok")!==(r.gt===r.pred))) return false;
    if(q && !((r.title||"").toLowerCase().includes(q)||(r.text||"").toLowerCase().includes(q))) return false;
    return true;
  });
  document.getElementById("totN").textContent=b.rows.length;
  document.getElementById("showN").textContent=out.length;
  document.getElementById("showN").style.color = out.length===b.rows.length?"var(--text)":"var(--accent)";
  let h="";
  out.forEach(r=>{
    const ok = r.gt===r.pred;
    const hasEmo = !!r.llm_emo;
    let emoCell = hasEmo
      ? `<span class="emo-pair"><span class="llm">${esc(r.llm_emo)}</span><span class="sep">→</span><span class="nrc">${esc(r.nrc_emo||"—")}</span></span>
         <div class="tiny">${r.llm_emo===r.nrc_emo?"agree":"disagree"}${(r.nrc_emo===""?" (no NRC word)":"")}</div>`
      : `<span class="tiny">—</span>`;
    h+=`<tr>
      <td><span class="stars">${starsStr(r.rating)}</span> <span class="tiny">${r.year}</span></td>
      <td><span class="tag ${clsClass(r.gt)}">${r.gt}</span></td>
      <td><span class="tag ${clsClass(r.pred)}">${r.pred}</span></td>
      <td>${ok?'<span class="ok">✓</span>':'<span class="no">✗</span>'}</td>
      <td>${emoCell}</td>
      <td><div class="rev-title">${esc(r.title)}</div><div class="rev-text">${esc(r.text)}</div></td>
    </tr>`;
  });
  document.getElementById("revTbl").querySelector("tbody").innerHTML=h || `<tr><td colspan="6" style="text-align:center;color:var(--muted);padding:24px">No reviews match the current filters.</td></tr>`;
}

/* ---------- switch ---------- */
document.getElementById("modelSwitch").querySelectorAll("button").forEach(bn=>{
  bn.addEventListener("click",()=>{
    document.getElementById("modelSwitch").querySelectorAll("button").forEach(x=>x.classList.remove("active"));
    bn.classList.add("active");
    STATE.model=bn.dataset.m;
    STATE.gt=null; STATE.m=null; STATE.q="";
    document.getElementById("search").value="";
    document.getElementById("filters").innerHTML="";
    render();
  });
});
document.getElementById("search").addEventListener("input",e=>{ STATE.q=e.target.value; renderTable(batches()); });

render();
</script>
</body>
</html>
"""


def build():
    b1 = load_batch1()
    bal = load_balanced()

    payload = {
        "balanced": None,
        "batch1": None,
    }
    if b1 is not None:
        payload["batch1"] = {
            "rows": b1,
            "classes": ["POSITIVE", "NEGATIVE"],
            "stats": stats(b1, ["POSITIVE", "NEGATIVE"]),
            "star": star_dist(b1),
            "emotions": None,
        }
    if bal is not None:
        payload["balanced"] = {
            "rows": bal,
            "classes": CLASS_ORDER_3,
            "stats": stats(bal, CLASS_ORDER_3),
            "star": star_dist(bal),
            "emotions": emotion_stats(bal),
        }

    blob = json.dumps(payload, separators=(",", ":"))
    html = TEMPLATE.replace("__DATA__", blob)
    out = os.path.join(config.ROOT, "dashboard.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"Wrote {out} ({len(html):,} bytes)")
    return out


if __name__ == "__main__":
    build()
