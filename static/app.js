/* 科研检索查询扩展系统 — 原生 JS 前端（无框架、无构建步骤） */
"use strict";

const state = {
  sessionId: null,
  revision: 0,
  suggestionId: null,
  suggestionRev: null,
  locked: [],
  excluded: [],
  lastResult: null,
  candidateSet: new Set(),
};

const $ = (id) => document.getElementById(id);

function setStatus(el, text, kind) {
  el.textContent = text;
  el.className = "status " + (kind || "");
}

function showError(el, text) {
  el.textContent = text;
  el.classList.remove("hidden");
}
function hideError(el) { el.classList.add("hidden"); }

function fmtPct(x) { return (x * 100).toFixed(1) + "%"; }

async function api(path, options = {}) {
  const resp = await fetch(path, {
    headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...options,
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const err = new Error(data.detail || data.error || `HTTP ${resp.status}`);
    err.status = resp.status;
    err.payload = data;
    throw err;
  }
  return data;
}

/* ------------------------------------------------------------------ */
/* ① 语料上传                                                          */
/* ------------------------------------------------------------------ */
$("upload-btn").addEventListener("click", async () => {
  const file = $("file-input").files[0];
  const statusEl = $("upload-status");
  if (!file) { setStatus(statusEl, "请先选择文件。", "err"); return; }
  const fd = new FormData();
  fd.append("file", file);
  setStatus(statusEl, "上传并构建 TF-IDF 索引中…", "");
  try {
    const data = await api("/api/sessions", { method: "POST", body: fd });
    bootstrapSession(data);
    setStatus(
      statusEl,
      `上传成功：${data.corpus_size} 篇（正例 ${data.n_relevant} / 负例 ${data.n_nonrelevant}），词表 ${data.vocab_size} 项。`,
      "ok"
    );
  } catch (err) {
    setStatus(statusEl, "上传失败：" + err.message, "err");
  }
});

$("demo-btn").addEventListener("click", async () => {
  const statusEl = $("upload-status");
  setStatus(statusEl, "加载内置演示语料…", "");
  try {
    const fd = new FormData();
    const text = await fetch("/api/demo-corpus").then((r) => {
      if (!r.ok) throw new Error("演示语料不可用");
      return r.text();
    });
    fd.append("file", new File([text], "demo_corpus.csv", { type: "text/csv" }));
    const data = await api("/api/sessions", { method: "POST", body: fd });
    bootstrapSession(data);
    setStatus(
      statusEl,
      `演示语料已加载：${data.corpus_size} 篇（正例 ${data.n_relevant} / 负例 ${data.n_nonrelevant}），词表 ${data.vocab_size} 项。`,
      "ok"
    );
  } catch (err) {
    setStatus(statusEl, "演示语料加载失败：" + err.message, "err");
  }
});

function bootstrapSession(data) {
  state.sessionId = data.session_id;
  state.revision = data.revision;
  state.suggestionId = null;
  state.locked = [];
  state.excluded = [];
  state.lastResult = null;
  $("query-card").classList.remove("hidden");
  $("terms-card").classList.add("hidden");
  $("result-card").classList.add("hidden");
  resetEvalState();
  updateRevisionBadge();
}

/* ------------------------------------------------------------------ */
/* ② 分析                                                              */
/* ------------------------------------------------------------------ */
$("analyze-btn").addEventListener("click", runAnalysis);

async function runAnalysis() {
  hideError($("query-error"));
  if (!state.sessionId) return;
  const maxFpRaw = $("max-fp").value.trim();
  const body = {
    session_id: state.sessionId,
    query: $("query-input").value,
    min_recall: parseFloat($("min-recall").value),
    max_fdr: parseFloat($("max-fdr").value),
    max_false_positives: maxFpRaw === "" ? null : parseInt(maxFpRaw, 10),
    beam_width: parseInt($("beam-width").value, 10),
    max_terms: parseInt($("max-terms").value, 10),
    top_k: parseInt($("top-k").value, 10),
    suggestion_id: state.suggestionId,
  };
  try {
    const data = await api("/api/analyze", {
      method: "POST",
      body: JSON.stringify(body),
    });
    state.suggestionId = data.suggestion_id;
    state.suggestionRev = data.revision;
    state.revision = data.revision;
    state.locked = data.current_locked_terms;
    state.excluded = data.current_excluded_terms;
    state.lastResult = data;
    state.candidateSet = new Set(data.candidate_pool.map((c) => c.term));
    renderTerms(data);
    renderResult(data);
    updateRevisionBadge();
  } catch (err) {
    if (err.status === 409) {
      // 旧建议已失效：清除本地建议 ID，提示用户重新分析
      state.suggestionId = null;
      state.suggestionRev = null;
      withdrawEvalConclusions();
      showError(
        $("query-error"),
        "⚠ 旧建议已失效（修订后未重新分析）。\n" + err.message +
        "\n请再次点击「运行扩展分析」以获取新建议。"
      );
    } else {
      showError($("query-error"), err.message);
    }
  }
}

/* ------------------------------------------------------------------ */
/* ③ 锁定 / 排除修订                                                    */
/* ------------------------------------------------------------------ */
function renderTerms(data) {
  $("terms-card").classList.remove("hidden");
  renderChips();
  const tbody = $("candidate-table").querySelector("tbody");
  tbody.innerHTML = "";
  for (const c of data.candidate_pool) {
    const tr = document.createElement("tr");
    const isLocked = c.locked || state.locked.includes(c.term);
    const isExcluded = c.excluded || state.excluded.includes(c.term);
    tr.innerHTML = `
      <td><strong>${escapeHtml(c.term)}</strong></td>
      <td>${c.n_relevant_hits}</td>
      <td>${c.n_nonrelevant_hits}</td>
      <td>${c.score.toFixed(4)}</td>
      <td class="hint">${escapeHtml(c.reason)}</td>
      <td class="row" style="margin:0;gap:6px;"></td>`;
    const actionTd = tr.lastElementChild;
    if (isLocked) {
      const btn = document.createElement("button");
      btn.className = "small undo";
      btn.textContent = "解除锁定";
      btn.onclick = () => revise({ unlock: [c.term] });
      actionTd.appendChild(btn);
      tr.style.background = "var(--green-bg)";
    } else if (isExcluded) {
      const btn = document.createElement("button");
      btn.className = "small undo";
      btn.textContent = "解除排除";
      btn.onclick = () => revise({ unexclude: [c.term] });
      actionTd.appendChild(btn);
      tr.style.background = "var(--red-bg)";
    } else {
      const lockBtn = document.createElement("button");
      lockBtn.className = "small lock";
      lockBtn.textContent = "锁定";
      lockBtn.onclick = () => revise({ lock: [c.term] });
      const exBtn = document.createElement("button");
      exBtn.className = "small exclude";
      exBtn.textContent = "排除";
      exBtn.onclick = () => revise({ exclude: [c.term] });
      actionTd.append(lockBtn, exBtn);
    }
    tbody.appendChild(tr);
  }
}

function renderChips() {
  const lockedBox = $("locked-chips");
  const excludedBox = $("excluded-chips");
  lockedBox.innerHTML = '<span class="hint">已锁定：</span>';
  excludedBox.innerHTML = '<span class="hint">已排除：</span>';
  if (state.locked.length === 0) lockedBox.innerHTML += '<span class="hint">（无）</span>';
  if (state.excluded.length === 0) excludedBox.innerHTML += '<span class="hint">（无）</span>';
  for (const term of state.locked) {
    lockedBox.appendChild(makeChip(term, "locked", () => revise({ unlock: [term] })));
  }
  for (const term of state.excluded) {
    excludedBox.appendChild(makeChip(term, "excluded", () => revise({ unexclude: [term] })));
  }
}

function makeChip(term, kind, onRemove) {
  const chip = document.createElement("span");
  chip.className = "chip " + kind;
  chip.textContent = term;
  const x = document.createElement("button");
  x.textContent = "×";
  x.title = "撤销";
  x.onclick = onRemove;
  chip.appendChild(x);
  return chip;
}

async function revise(patch) {
  hideError($("query-error"));
  try {
    const data = await api("/api/revise", {
      method: "POST",
      body: JSON.stringify({
        session_id: state.sessionId,
        stale_suggestion_id: state.suggestionId,
        ...patch,
      }),
    });
    state.revision = data.revision;
    state.locked = data.locked_terms;
    state.excluded = data.excluded_terms;
    // 修订即旧建议作废
    state.suggestionId = null;
    state.suggestionRev = null;
    // 建议失效：立即撤下评测结论（本批标注保留，新建议出来后自动重算）
    withdrawEvalConclusions();
    updateRevisionBadge();
    renderChips();
    if (state.lastResult) renderTerms(state.lastResult);
    // 自动重算
    await runAnalysis();
  } catch (err) {
    if (err.status === 409) {
      state.suggestionId = null;
      updateRevisionBadge();
      showError($("query-error"), "⚠ " + err.message);
    } else {
      showError($("query-error"), err.message);
    }
  }
}

function updateRevisionBadge() {
  const badge = $("revision-badge");
  badge.classList.remove("hidden");
  const stale = state.lastResult && state.suggestionRev !== state.revision;
  if (state.suggestionId && !stale) {
    badge.textContent = `修订号 rev=${state.revision} · 当前建议有效`;
    badge.className = "badge fresh";
  } else if (state.lastResult || state.locked.length || state.excluded.length) {
    badge.textContent = `修订号 rev=${state.revision} · 旧建议已失效，请重新分析`;
    badge.className = "badge stale";
  } else {
    badge.textContent = `修订号 rev=${state.revision}`;
    badge.className = "badge";
  }
}

/* ------------------------------------------------------------------ */
/* ④ 结果渲染                                                          */
/* ------------------------------------------------------------------ */
function renderResult(data) {
  $("result-card").classList.remove("hidden");
  const summary = $("result-summary");
  const base = data.baseline_metrics;
  const fin = data.metrics;
  const satisfied = data.status === "satisfied";
  summary.innerHTML = `
    <div class="${satisfied ? "satisfied-banner" : "failed-banner"}">
      ${satisfied
        ? `✅ 找到满足阈值的扩展方案：扩展词 ${data.expanded_terms.length} 个 → ${data.expanded_terms.map(escapeHtml).join(", ") || "（无需扩展）"}`
        : "❌ 找不到同时满足召回下限与误检上限的扩展方案，原因见下方诊断。"}
    </div>
    <div class="metric-grid">
      ${metric("基线召回", fmtPct(base.recall), base.recall >= data.thresholds.min_recall ? "good" : "bad")}
      ${metric("扩展后召回", fmtPct(fin.recall), fin.recall >= data.thresholds.min_recall ? "good" : "bad")}
      ${metric("召回下限", fmtPct(data.thresholds.min_recall))}
      ${metric("基线误检率", fmtPct(base.fdr), base.fdr <= data.thresholds.max_fdr ? "good" : "bad")}
      ${metric("扩展后误检率", fmtPct(fin.fdr), fin.fdr <= data.thresholds.max_fdr ? "good" : "bad")}
      ${metric("误检率上限", fmtPct(data.thresholds.max_fdr))}
      ${metric("命中文档", `${fin.n_retrieved} / ${data.corpus_size}`)}
      ${metric("正例命中 (TP)", `${fin.true_positives} / ${data.n_relevant}`)}
      ${metric("误检 (FP)", `${fin.false_positives}`)}
    </div>
    <p class="hint">
      基线查询词：${data.baseline_terms.map(escapeHtml).join(", ") || "（词表中无匹配）"}
      ${data.unknown_query_tokens.length
        ? "｜未登录词（已忽略）：" + data.unknown_query_tokens.map(escapeHtml).join(", ")
        : ""}
    </p>`;

  // trace table
  const traceBody = $("trace-table").querySelector("tbody");
  traceBody.innerHTML = "";
  for (const row of data.trace) {
    const tr = document.createElement("tr");
    if (row.is_baseline) tr.style.background = "#f1f5f9";
    tr.innerHTML = `
      <td><strong>${escapeHtml(row.term)}</strong></td>
      <td>${row.added_relevant}</td>
      <td>${row.added_nonrelevant}</td>
      <td>${row.cumulative_retrieved}</td>
      <td>${fmtPct(row.cumulative_recall)}</td>
      <td>${fmtPct(row.cumulative_fdr)}</td>
      <td>${row.cumulative_false_positives}</td>
      <td class="hint">${escapeHtml(row.note)}</td>`;
    traceBody.appendChild(tr);
  }

  // failure diagnosis
  const box = $("failure-box");
  if (data.failure) {
    box.classList.remove("hidden");
    const f = data.failure;
    box.innerHTML = `
      <h4>失败诊断</h4>
      <ul>${f.reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>
      <p class="hint" style="margin-top:8px;">
        无约束最高召回：<strong>${fmtPct(f.unconstrained_max_recall)}</strong> ｜
        误检约束内最高召回：<strong>${fmtPct(f.best_feasible_recall)}</strong>
        ${f.blocker_term
          ? `｜首个越界词：<code>${escapeHtml(f.blocker_term.term)}</code>`
          : ""}
      </p>
      ${f.missing_relevant_docs.length
        ? `<p class="hint">无法覆盖的正例：${f.missing_relevant_docs.map((d) => escapeHtml(d.id)).join(", ")}</p>`
        : ""}`;
  } else {
    box.classList.add("hidden");
    box.innerHTML = "";
  }

  // ranking
  $("ranking-meta").textContent =
    `（共 ${fin.n_retrieved} 篇命中，展示前 ${data.ranking.length} 篇；绿色=正例，红色=负例）`;
  const rankBody = $("ranking-table").querySelector("tbody");
  rankBody.innerHTML = "";
  const expansionSet = new Set(data.expanded_terms);
  for (const doc of data.ranking) {
    const tr = document.createElement("tr");
    const terms = doc.matched_terms
      .map((t) => `<span class="matched-term ${expansionSet.has(t) ? "expansion" : ""}">${escapeHtml(t)}</span>`)
      .join("");
    tr.innerHTML = `
      <td>${doc.rank}</td>
      <td>${escapeHtml(doc.id)}</td>
      <td><strong>${escapeHtml(doc.title)}</strong><br/><span class="hint">${escapeHtml(truncate(doc.abstract, 180))}</span></td>
      <td><span class="tag ${doc.relevant ? "rel" : "nonrel"}">${doc.relevant ? "正例" : "负例"}</span></td>
      <td>${terms}</td>
      <td>${doc.score.toFixed(4)}</td>`;
    rankBody.appendChild(tr);
  }

  // 评测入口随分析结果开放；若已载入评测集，用当前建议自动重算差值界
  $("eval-card").classList.remove("hidden");
  if (evalState.evalId) computeEval();
}

function metric(k, v, cls) {
  return `<div class="metric"><div class="k">${k}</div><div class="v ${cls || ""}">${v}</div></div>`;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function truncate(s, n) { return s.length > n ? s.slice(0, n) + "…" : s; }

/* ------------------------------------------------------------------ */
/* ⑤ 新批次评测：NDCG@K 差值上下界                                      */
/* ------------------------------------------------------------------ */
const evalState = {
  evalId: null,
  docs: [],            // [{id, title, abstract}]
  annotations: {},     // doc_id -> true | false（两榜共用同一份标注）
  result: null,        // 最近一次 eval-compute 响应
  epoch: 0,            // 单调递增：迟到响应若 epoch 过期直接丢弃
  withdrawn: false,    // 建议失效后结论已撤下（标注保留）
};

function resetEvalState() {
  evalState.evalId = null;
  evalState.docs = [];
  evalState.annotations = {};
  evalState.result = null;
  evalState.withdrawn = false;
  evalState.epoch++;                 // 使全部在途评测响应失效
  $("eval-card").classList.add("hidden");
  $("eval-body").classList.add("hidden");
  setStatus($("eval-status"), "", "");
  hideError($("eval-error"));
}

function fmtBound(x) {
  const v = Math.abs(x) < 5e-5 ? 0 : x;   // 避免显示 “-0.0000”
  return (v > 0 ? "+" : "") + v.toFixed(4);
}

/* ---- 上传评测集 ---- */
$("eval-upload-btn").addEventListener("click", () => {
  const file = $("eval-file-input").files[0];
  if (!file) { setStatus($("eval-status"), "请先选择评测集 CSV 文件。", "err"); return; }
  uploadEvalSet(file);
});

$("eval-demo-btn").addEventListener("click", async () => {
  setStatus($("eval-status"), "加载演示评测集…", "");
  hideError($("eval-error"));
  try {
    const text = await fetch("/api/demo-eval").then((r) => {
      if (!r.ok) throw new Error("演示评测集不可用");
      return r.text();
    });
    await uploadEvalSet(new File([text], "demo_eval.csv", { type: "text/csv" }));
  } catch (err) {
    setStatus($("eval-status"), "演示评测集加载失败：" + err.message, "err");
  }
});

async function uploadEvalSet(file) {
  if (!state.sessionId) return;
  const epoch = ++evalState.epoch;   // 新批次：旧批次在途结果一律作废
  const fd = new FormData();
  fd.append("file", file);
  setStatus($("eval-status"), "上传评测集…", "");
  hideError($("eval-error"));
  try {
    const data = await api(`/api/sessions/${state.sessionId}/eval-set`, {
      method: "POST", body: fd,
    });
    if (epoch !== evalState.epoch) return;          // 迟到结果丢弃
    evalState.evalId = data.eval_id;
    evalState.docs = data.docs;
    evalState.annotations = {};                     // 换评测集 → 重新标注
    evalState.result = null;
    evalState.withdrawn = false;
    $("eval-body").classList.remove("hidden");
    renderEvalDocs();
    clearEvalConclusions("新评测集已载入，请重新标注；正在计算差值界…");
    setStatus($("eval-status"), data.message, "ok");
    await computeEval(epoch);
  } catch (err) {
    if (epoch !== evalState.epoch) return;
    setStatus($("eval-status"), "", "");
    showError($("eval-error"), "评测集上传失败：" + err.message);
  }
}

/* ---- 标注（可更正；两榜共用） ---- */
async function judgeEvalDoc(docId, relevant) {   // relevant: true | false | null
  if (!evalState.evalId) return;
  const epoch = ++evalState.epoch;
  hideError($("eval-error"));
  try {
    const data = await api(`/api/sessions/${state.sessionId}/eval-judge`, {
      method: "POST",
      body: JSON.stringify({ eval_id: evalState.evalId, doc_id: docId, relevant }),
    });
    if (epoch !== evalState.epoch) return;          // 迟到结果丢弃
    evalState.annotations = data.annotations;
    renderEvalDocs();
    await computeEval(epoch);                       // 改标后重算
  } catch (err) {
    if (epoch !== evalState.epoch) return;
    showError($("eval-error"), "⚠ " + err.message);
  }
}

/* ---- 计算差值界 ---- */
$("eval-compute-btn").addEventListener("click", () => computeEval());
$("eval-k").addEventListener("change", () => computeEval());

async function computeEval(epoch) {
  if (!evalState.evalId || !state.suggestionId) return;
  if (epoch === undefined) epoch = ++evalState.epoch;
  const k = parseInt($("eval-k").value, 10);
  if (!Number.isInteger(k) || k < 1 || k > 20) {
    showError($("eval-error"), "K 必须是 1～20 的整数。");
    return;
  }
  hideError($("eval-error"));
  try {
    const data = await api(`/api/sessions/${state.sessionId}/eval-compute`, {
      method: "POST",
      body: JSON.stringify({
        eval_id: evalState.evalId,
        suggestion_id: state.suggestionId,
        k,
      }),
    });
    if (epoch !== evalState.epoch) return;          // 迟到结果不得恢复旧结论
    evalState.result = data;
    evalState.withdrawn = false;
    renderEvalResult(data);
  } catch (err) {
    if (epoch !== evalState.epoch) return;
    if (err.status === 409) {
      // 建议失效 / 批次已换：撤下结论，本批标注保留
      evalState.result = null;
      evalState.withdrawn = true;
      clearEvalConclusions("⚠ " + err.message + "（本批标注已保留）");
    } else {
      showError($("eval-error"), err.message);
    }
  }
}

/* 建议失效时：立即撤下结论（标注保留，等新建议重算） */
function withdrawEvalConclusions(note) {
  if (!evalState.evalId) return;
  evalState.epoch++;                 // 在途评测响应全部作废
  evalState.result = null;
  evalState.withdrawn = true;
  clearEvalConclusions(note || "建议已失效，评测结论已撤下；本批标注已保留，等待新建议重算。");
}

function clearEvalConclusions(note) {
  const v = $("eval-verdict");
  v.className = "verdict withdrawn";
  v.textContent = note;
  $("eval-metrics").innerHTML = "";
  $("eval-witness").innerHTML = "";
  for (const id of ["eval-list-a", "eval-list-b"]) {
    $(id).querySelector("tbody").innerHTML = "";
  }
  $("eval-lists-meta").textContent = "（暂无有效结论）";
}

/* ---- 渲染 ---- */
function judgmentTag(j) {
  if (j === true) return '<span class="tag rel">相关</span>';
  if (j === false) return '<span class="tag nonrel">不相关</span>';
  return '<span class="tag unjudged">未判</span>';
}

function judgeButtons(docId) {
  const wrap = document.createElement("span");
  wrap.className = "judge-btns";
  const cur = evalState.annotations[docId];
  const mk = (label, val, onCls) => {
    const b = document.createElement("button");
    b.textContent = label;
    if (cur === val) b.classList.add(onCls);
    b.onclick = () => judgeEvalDoc(docId, val);
    return b;
  };
  const clr = document.createElement("button");
  clr.textContent = "撤销";
  clr.title = "撤销判定，回到未审";
  clr.onclick = () => judgeEvalDoc(docId, null);
  wrap.append(mk("相关", true, "on-rel"), mk("不相关", false, "on-nonrel"), clr);
  return wrap;
}

function renderEvalDocs() {
  const tbody = $("eval-docs-table").querySelector("tbody");
  tbody.innerHTML = "";
  for (const doc of evalState.docs) {
    const tr = document.createElement("tr");
    const j = evalState.annotations[doc.id];
    tr.innerHTML = `
      <td>${escapeHtml(doc.id)}</td>
      <td><strong>${escapeHtml(doc.title)}</strong><br/>
          <span class="hint">${escapeHtml(truncate(doc.abstract, 160))}</span></td>
      <td class="jcell"></td>
      <td class="bcell"></td>`;
    tr.querySelector(".jcell").innerHTML = judgmentTag(j === undefined ? null : j);
    tr.querySelector(".bcell").appendChild(judgeButtons(doc.id));
    tbody.appendChild(tr);
  }
  const nJudged = Object.keys(evalState.annotations).length;
  const nRel = Object.values(evalState.annotations).filter(Boolean).length;
  $("eval-judge-meta").textContent =
    `（共 ${evalState.docs.length} 篇，已判 ${nJudged}：相关 ${nRel} / 不相关 ${nJudged - nRel}；` +
    "判定可更正，两榜共用同一份标注）";
}

function renderEvalList(tableId, rows) {
  const tbody = $(tableId).querySelector("tbody");
  tbody.innerHTML = "";
  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" class="hint">（该查询在评测集上无命中）</td></tr>';
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.rank}</td>
      <td>${escapeHtml(row.id)}</td>
      <td><strong>${escapeHtml(row.title)}</strong><br/>
          <span class="hint">相似度 ${row.score.toFixed(4)}｜命中词：${row.matched_terms.map(escapeHtml).join(", ")}</span></td>
      <td class="jcell"></td>`;
    tr.querySelector(".jcell").innerHTML = judgmentTag(row.judgment === undefined ? null : row.judgment);
    tbody.appendChild(tr);
  }
}

function witnessHtml(list) {
  if (list.length === 0) return "（无 —— 未判论文全部判不相关即达到该端点）";
  const shown = list.slice(0, 30)
    .map((x) => `<code>${escapeHtml(x.id)}</code>`).join(" ");
  return shown + (list.length > 30 ? ` …等 ${list.length} 篇` : "");
}

function renderEvalResult(data) {
  $("eval-body").classList.remove("hidden");
  const b = data.bounds;
  const t = data.tentative;
  const v = $("eval-verdict");
  if (b.certain_improvement) {
    v.className = "verdict improve";
    v.textContent = `✅ 确定改善：无论剩余 ${data.n_unjudged} 篇未判论文如何判定，` +
      `扩展榜 NDCG@${data.k} 都严格高于原榜（差值下界 ${fmtBound(b.lower)} > 0）。`;
  } else if (b.certain_regression) {
    v.className = "verdict regress";
    v.textContent = `❌ 确定变差：无论剩余 ${data.n_unjudged} 篇未判论文如何判定，` +
      `扩展榜都不优于原榜（差值上界 ${fmtBound(b.upper)} < 0）。`;
  } else if (Math.abs(b.upper - b.lower) < 1e-9) {
    v.className = "verdict neutral";
    v.textContent = `ℹ️ 差值已收敛为一点：Δ = ${fmtBound(b.lower)}；` +
      "下界未大于 0，不能判定确定改善。";
  } else {
    v.className = "verdict uncertain";
    v.textContent = `⏳ 尚不确定：差值 ∈ [${fmtBound(b.lower)}, ${fmtBound(b.upper)}]；` +
      `继续标注剩余 ${data.n_unjudged} 篇未判论文可收窄区间，下界 > 0 才能判定确定改善。`;
  }
  $("eval-metrics").innerHTML = [
    metric("差值下界", fmtBound(b.lower), b.lower > 0 ? "good" : ""),
    metric("差值上界", fmtBound(b.upper), b.upper < 0 ? "bad" : ""),
    metric("暂定差值（未判按不相关）", fmtBound(t.delta)),
    metric("暂定 NDCG · 原榜", t.ndcg_original.toFixed(4)),
    metric("暂定 NDCG · 扩展榜", t.ndcg_expanded.toFixed(4)),
    metric("已判 / 未判", `${data.n_judged} / ${data.n_unjudged}`),
  ].join("");
  $("eval-witness").innerHTML = `
    <div class="witness-box">
      <h4>达到上界 ${fmtBound(b.upper)} 时：应判相关的未判论文</h4>
      <p class="hint">${witnessHtml(b.upper_witness)}（其余未判论文判不相关）</p>
      <h4>达到下界 ${fmtBound(b.lower)} 时：应判相关的未判论文</h4>
      <p class="hint">${witnessHtml(b.lower_witness)}（其余未判论文判不相关）</p>
    </div>`;
  $("eval-list-a-title").textContent =
    `原查询 top-${data.k}（${data.query_terms_original.join(", ") || "无有效词"}）`;
  $("eval-list-b-title").textContent =
    `扩展查询 top-${data.k}（${data.query_terms_expanded.join(", ") || "无有效词"}）`;
  renderEvalList("eval-list-a", data.lists.original);
  renderEvalList("eval-list-b", data.lists.expanded);
  $("eval-lists-meta").textContent =
    `（基于 rev=${data.revision} 的当前建议；评测集 ${data.n_docs} 篇，理想榜按全集计算）`;
  // 标注表同步（compute 响应携带最新标注）
  evalState.annotations = data.annotations;
  renderEvalDocs();
}
