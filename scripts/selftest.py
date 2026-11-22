"""Delivery self-test: threshold-conflict and revision scenarios.

Runs the full HTTP stack in-process (FastAPI TestClient) against the demo
corpus and writes a Markdown report with the *measured* numbers to
reports/selftest_report.md.  Exits non-zero if any expectation breaks.

Scenarios covered
-----------------
S1  Baseline analysis: satisfied expansion with per-term coverage trace.
S2  Threshold conflict: contradictory recall/fdr limits -> failure +
    structured diagnosis (ceiling recall, blocker term, reasons).
S3  FDR relaxation resolves the same conflict (proves diagnosis is accurate).
S4  Revision — exclude terms: solution is recomputed without them and the
    old suggestion id is rejected with HTTP 409.
S5  Revision — lock a term: it is forced into every new solution.
S6  Lock/exclude contradiction is rejected; unlock restores feasibility.
S7  Eval set upload + NDCG@K difference bounds on the demo eval set.
S8  Bounds exactness: API bounds vs brute-force enumeration on a tiny set.
S9  Judging narrows the interval; fully judged set collapses to one value.
S10 Revision invalidates eval conclusions (409) but keeps the annotations.
S11 Replacing the eval set clears annotations; late calls with the old
    eval_id are rejected (HTTP 409).
"""

from __future__ import annotations

import itertools
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app.main import app

CORPUS_PATH = ROOT / "data" / "demo_corpus.csv"
EVAL_PATH = ROOT / "data" / "demo_eval.csv"
REPORT_DIR = ROOT / "reports"

client = TestClient(app)
results: list[dict] = []


def record(scenario: str, ok: bool, detail: str, payload=None) -> None:
    results.append({"scenario": scenario, "ok": ok, "detail": detail,
                    "payload": payload or {}})
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {scenario}: {detail}")


def upload_demo() -> str:
    raw = CORPUS_PATH.read_bytes()
    resp = client.post(
        "/api/sessions",
        files={"file": ("demo_corpus.csv", raw, "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


def analyze(sid: str, **overrides) -> dict:
    body = {
        "session_id": sid,
        "query": "quantum error correction",
        # 15/17 relevant docs are reachable (0.882); use 0.85 so the
        # satisfied path is exercised, and 0.95+ for ceiling failures.
        "min_recall": 0.85,
        "max_fdr": 0.2,
    }
    body.update(overrides)
    resp = client.post("/api/analyze", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


# --------------------------------------------------------------------------- #
# eval helpers
# --------------------------------------------------------------------------- #
def upload_eval(sid: str, raw: bytes | None = None,
                name: str = "demo_eval.csv") -> dict:
    data = raw if raw is not None else EVAL_PATH.read_bytes()
    resp = client.post(f"/api/sessions/{sid}/eval-set",
                       files={"file": (name, data, "text/csv")})
    assert resp.status_code == 200, resp.text
    return resp.json()


def eval_compute(sid: str, eval_id: str, suggestion_id: str, k: int = 10):
    return client.post(f"/api/sessions/{sid}/eval-compute", json={
        "eval_id": eval_id, "suggestion_id": suggestion_id, "k": k,
    })


def eval_judge(sid: str, eval_id: str, doc_id: str, relevant):
    return client.post(f"/api/sessions/{sid}/eval-judge", json={
        "eval_id": eval_id, "doc_id": doc_id, "relevant": relevant,
    })


def _w(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def brute_force_delta_bounds(doc_ids, judgments, list_a, list_b, k):
    """Enumerate all 2^U labellings (only feasible for tiny eval sets)."""
    unjudged = [d for d in doc_ids if d not in judgments]
    lo, hi = math.inf, -math.inf
    for bits in range(2 ** len(unjudged)):
        labels = dict(judgments)
        for i, d in enumerate(unjudged):
            labels[d] = bool(bits >> i & 1)
        r = sum(1 for v in labels.values() if v)
        if r == 0:
            delta = 0.0
        else:
            idcg = sum(_w(i) for i in range(1, min(r, k) + 1))
            dcg_a = sum(_w(i + 1) for i, d in enumerate(list_a) if labels[d])
            dcg_b = sum(_w(i + 1) for i, d in enumerate(list_b) if labels[d])
            delta = dcg_b / idcg - dcg_a / idcg
        lo, hi = min(lo, delta), max(hi, delta)
    return lo, hi


def main() -> int:
    failures = 0

    # --------------------------------------------------------------- S1
    sid = upload_demo()
    s1 = analyze(sid)
    ok = (
        s1["status"] == "satisfied"
        and s1["metrics"]["recall"] >= 0.85 - 1e-9
        and s1["metrics"]["fdr"] <= 0.2 + 1e-9
        and len(s1["trace"]) == 1 + len(s1["expanded_terms"])
    )
    failures += 0 if ok else 1
    record(
        "S1 基线扩展分析（满足阈值）", ok,
        f"扩展词={s1['expanded_terms']}；召回 "
        f"{s1['baseline_metrics']['recall']:.3f}→{s1['metrics']['recall']:.3f}；"
        f"fdr {s1['baseline_metrics']['fdr']:.3f}→{s1['metrics']['fdr']:.3f}；"
        f"trace 行数={len(s1['trace'])}",
        payload=s1,
    )

    # --------------------------------------------------------------- S2
    s2 = analyze(sid, min_recall=1.0, max_fdr=0.05)
    f = s2.get("failure") or {}
    # In this corpus the recall ceiling is itself below 100% (two relevant
    # abstracts use document-unique words), so S2 demonstrates the
    # vocabulary-coverage failure mode under strict thresholds.
    ok = (
        s2["status"] == "failed"
        and bool(f.get("reasons"))
        and "unconstrained_max_recall" in f
        and f["unconstrained_max_recall"] < 1.0
    )
    failures += 0 if ok else 1
    blocker = f.get("blocker_term") or {}
    record(
        "S2 阈值冲突（召回 100% & fdr≤5%，词表上限不足）", ok,
        f"状态={s2['status']}；无约束最高召回={f.get('unconstrained_max_recall')}；"
        f"约束内最高召回={f.get('best_feasible_recall')}；"
        f"无法覆盖正例={[d['id'] for d in f.get('missing_relevant_docs', [])]}；"
        f"诊断条数={len(f.get('reasons', []))}",
        payload=s2,
    )

    # --------------------------------------------------------------- S2b
    # A conflict where the recall target is reachable but the fdr budget is
    # not: clean terms reach 0.882 recall, and the remaining polluted terms
    # needed beyond it push fdr over a 1% budget.
    s2b = analyze(sid, min_recall=0.85, max_fdr=0.01)
    fb = s2b.get("failure") or {}
    ok = (
        s2b["status"] == "failed"
        and fb.get("unconstrained_max_recall", 0) >= 0.85 - 1e-12
        and fb.get("best_feasible_recall", 1) < 0.85
        and any("误检" in r for r in fb.get("reasons", []))
    )
    failures += 0 if ok else 1
    bb = fb.get("blocker_term") or {}
    record(
        "S2b 阈值冲突（召回可达但误检预算不足）", ok,
        f"状态={s2b['status']}；无约束最高召回={fb.get('unconstrained_max_recall')}；"
        f"约束内最高召回={fb.get('best_feasible_recall')}；"
        f"首个越界词={bb.get('term')}（新增正例 {bb.get('would_add_relevant')} / "
        f"负例 {bb.get('would_add_nonrelevant')}，fdr→{bb.get('resulting_fdr')}）",
        payload=s2b,
    )

    # --------------------------------------------------------------- S3
    # Pair with S2b: relaxing the fdr budget to 40% removes the conflict.
    s3 = analyze(sid, min_recall=0.85, max_fdr=0.4)
    ok = s3["status"] == "satisfied" and s3["metrics"]["recall"] >= 0.85 - 1e-12
    failures += 0 if ok else 1
    record(
        "S3 放宽误检上限至 40% 后 S2b 冲突消解", ok,
        f"状态={s3['status']}；召回={s3['metrics']['recall']:.3f}；"
        f"fdr={s3['metrics']['fdr']:.3f}；FP={s3['metrics']['false_positives']}",
        payload=s3,
    )

    # --------------------------------------------------------------- S4
    old_suggestion = s1["suggestion_id"]
    excluded_now = []
    rev_info = None
    for term in s1["expanded_terms"]:
        resp = client.post("/api/revise", json={
            "session_id": sid, "exclude": [term],
        })
        assert resp.status_code == 200, resp.text
        excluded_now.append(term)
        rev_info = resp.json()
    assert rev_info is not None, "S1 produced no expansion terms to exclude"
    stale = client.post("/api/analyze", json={
        "session_id": sid, "query": "quantum error correction",
        "min_recall": 0.85, "max_fdr": 0.2,
        "suggestion_id": old_suggestion,
    })
    s4 = analyze(sid)  # fresh call without stale id
    # With every effective term excluded, the solver must never reuse them;
    # dropping back to baseline recall (and possibly failing the threshold)
    # is the correct, expected consequence.
    ok = (
        stale.status_code == 409
        and rev_info["revision"] >= 1
        and not (set(s4["expanded_terms"]) & set(excluded_now))
        and s4["metrics"]["recall"] <= s1["metrics"]["recall"] + 1e-9
    )
    failures += 0 if ok else 1
    record(
        "S4 修订-排除词后重算 & 旧建议 409 失效", ok,
        f"排除={excluded_now}；旧建议 HTTP={stale.status_code}；"
        f"新修订号={rev_info['revision']}；新扩展词={s4['expanded_terms']}；"
        f"状态={s4['status']}；召回={s4['metrics']['recall']:.3f}"
        f"（排除全部有效词后回落到基线，符合预期）",
        payload={"stale": stale.json(), "fresh": s4},
    )

    # --------------------------------------------------------------- S5
    # Use a fresh session: in S4 we excluded decoherence, so locking it
    # there would (correctly) raise a lock/exclude contradiction.
    sid2 = upload_demo()
    lock_term = "decoherence"
    resp = client.post("/api/revise", json={
        "session_id": sid2, "lock": [lock_term],
    })
    assert resp.status_code == 200, resp.text
    s5 = analyze(sid2, min_recall=0.5, max_fdr=0.5)
    ok = lock_term in s5["selected_terms"]
    failures += 0 if ok else 1
    record(
        "S5 修订-锁定词强制入选（独立会话）", ok,
        f"锁定={lock_term}；selected_terms={s5['selected_terms']}；"
        f"召回={s5['metrics']['recall']:.3f}；fdr={s5['metrics']['fdr']:.3f}",
        payload=s5,
    )

    # --------------------------------------------------------------- S6
    conflict = client.post("/api/revise", json={
        "session_id": sid2, "exclude": [lock_term],
    })
    unlock = client.post("/api/revise", json={
        "session_id": sid2, "unlock": [lock_term],
    })
    s6 = analyze(sid2, min_recall=0.5, max_fdr=0.5)
    # The real assertion: contradiction rejected; after unlock term no longer
    # forced (it may still be picked naturally, so check locked list instead).
    ok = conflict.status_code == 400 and unlock.json()["locked_terms"] == []
    failures += 0 if ok else 1
    record(
        "S6 锁定/排除矛盾拒绝 & 解除锁定", ok,
        f"矛盾请求 HTTP={conflict.status_code}（{conflict.json().get('detail')}）；"
        f"解锁后 locked_terms={unlock.json()['locked_terms']}；"
        f"当前修订号={unlock.json()['revision']}",
        payload={"conflict": conflict.json(), "unlock": unlock.json()},
    )

    # --------------------------------------------------------------- S7
    # 评测：演示评测集上传 + NDCG@K 差值界计算
    sid3 = upload_demo()
    a3 = analyze(sid3)
    ev = upload_eval(sid3)
    r7 = eval_compute(sid3, ev["eval_id"], a3["suggestion_id"], k=10)
    assert r7.status_code == 200, r7.text
    e7 = r7.json()
    b7 = e7["bounds"]
    ok = (
        e7["n_docs"] == 40
        and len(e7["lists"]["original"]) <= 10
        and len(e7["lists"]["expanded"]) <= 10
        and len(e7["lists"]["expanded"]) >= len(e7["lists"]["original"])
        and b7["lower"] <= e7["tentative"]["delta"] <= b7["upper"]
        and b7["lower"] <= 0.0 <= b7["upper"]  # 全未判时全不相关补全给出 0
    )
    failures += 0 if ok else 1
    record(
        "S7 评测集上传与差值界计算（K=10）", ok,
        f"评测集 {e7['n_docs']} 篇；原榜 {len(e7['lists']['original'])} 篇 / "
        f"扩展榜 {len(e7['lists']['expanded'])} 篇；"
        f"差值界=[{b7['lower']:.4f}, {b7['upper']:.4f}]；"
        f"暂定差值={e7['tentative']['delta']:.4f}；"
        f"确定改善={b7['certain_improvement']}",
        payload=e7,
    )

    # --------------------------------------------------------------- S8
    # 精确性：小评测集上 API 界与 2^U 穷举完全一致
    tiny_csv = (
        "id,title,abstract\n"
        "t1,x,quantum error correction gates\n"
        "t2,x,surface code logical qubit\n"
        "t3,x,syndrome decoherence mitigation\n"
        "t4,x,quantum error correction circuits\n"
        "t5,x,neural network layers\n"
        "t6,x,crispr gene editing\n"
    ).encode()
    ev8 = upload_eval(sid3, raw=tiny_csv, name="tiny.csv")
    eval_judge(sid3, ev8["eval_id"], "t2", True)
    eval_judge(sid3, ev8["eval_id"], "t5", False)
    r8 = eval_compute(sid3, ev8["eval_id"], a3["suggestion_id"], k=3)
    assert r8.status_code == 200, r8.text
    e8 = r8.json()
    lo_bf, hi_bf = brute_force_delta_bounds(
        ["t1", "t2", "t3", "t4", "t5", "t6"],
        {"t2": True, "t5": False},
        [r["id"] for r in e8["lists"]["original"]],
        [r["id"] for r in e8["lists"]["expanded"]],
        3,
    )
    ok = (
        abs(e8["bounds"]["lower"] - lo_bf) < 1e-6
        and abs(e8["bounds"]["upper"] - hi_bf) < 1e-6
    )
    failures += 0 if ok else 1
    record(
        "S8 差值界与 2^U 穷举完全一致（6 篇小集，K=3）", ok,
        f"API 界=[{e8['bounds']['lower']:.6f}, {e8['bounds']['upper']:.6f}]；"
        f"穷举界=[{lo_bf:.6f}, {hi_bf:.6f}]",
        payload=e8,
    )

    # --------------------------------------------------------------- S9
    # 标注收窄区间；全部判定后上下界收敛为同一点
    ev9 = upload_eval(sid3)  # 40 篇演示评测集
    sug9 = a3["suggestion_id"]
    for i, doc in enumerate(ev9["docs"]):
        # 前 12 篇先判：标题含 (QEC) 判相关，否则不相关
        if i < 12:
            eval_judge(sid3, ev9["eval_id"], doc["id"],
                       "(QEC)" in doc["title"])
    mid = eval_compute(sid3, ev9["eval_id"], sug9, k=10).json()
    span_before = b7["upper"] - b7["lower"]
    span_mid = mid["bounds"]["upper"] - mid["bounds"]["lower"]
    for doc in ev9["docs"]:
        if eval_judge(sid3, ev9["eval_id"], doc["id"],
                      "(QEC)" in doc["title"]).status_code != 200:
            raise AssertionError("judge failed")
    full = eval_compute(sid3, ev9["eval_id"], sug9, k=10).json()
    ok = (
        span_mid <= span_before + 1e-9
        and full["n_unjudged"] == 0
        and abs(full["bounds"]["lower"] - full["bounds"]["upper"]) < 1e-9
        and abs(full["bounds"]["lower"] - full["tentative"]["delta"]) < 1e-9
    )
    failures += 0 if ok else 1
    record(
        "S9 标注收窄区间 & 全判定后界收敛", ok,
        f"区间宽 {span_before:.4f} →（判 12 篇）{span_mid:.4f} →"
        f"（判全部 40 篇）{full['bounds']['upper'] - full['bounds']['lower']:.6f}；"
        f"全判定后 Δ={full['bounds']['lower']:.4f}；"
        f"确定改善={full['bounds']['certain_improvement']}",
        payload=full,
    )

    # --------------------------------------------------------------- S10
    # 修订 → 评测结论撤下（旧建议 409），本批标注保留供新建议使用
    eval_judge(sid3, ev9["eval_id"], ev9["docs"][0]["id"], True)
    client.post("/api/revise", json={
        "session_id": sid3, "exclude": [a3["expanded_terms"][0]],
    })
    stale_eval = eval_compute(sid3, ev9["eval_id"], sug9, k=10)
    a3b = analyze(sid3)  # 新建议
    kept = eval_compute(sid3, ev9["eval_id"], a3b["suggestion_id"], k=10)
    ok = (
        stale_eval.status_code == 409
        and kept.status_code == 200
        and kept.json()["n_judged"] == 40  # 标注全部保留
    )
    failures += 0 if ok else 1
    record(
        "S10 修订使评测结论失效（409）但标注保留", ok,
        f"旧建议 compute HTTP={stale_eval.status_code}；"
        f"新建议 rev={a3b['revision']} 可计算；"
        f"保留标注数={kept.json()['n_judged'] if kept.status_code == 200 else 'N/A'}",
        payload={"stale": stale_eval.json(), "fresh_rev": a3b["revision"]},
    )

    # --------------------------------------------------------------- S11
    # 更换评测集 → 重新标注；迟到的旧 eval_id 请求不得恢复旧结论
    ev11 = upload_eval(sid3)
    old_eval_id = ev9["eval_id"]
    late_compute = eval_compute(sid3, old_eval_id, a3b["suggestion_id"], k=10)
    late_judge = eval_judge(sid3, old_eval_id, "E001", True)
    fresh = eval_compute(sid3, ev11["eval_id"], a3b["suggestion_id"], k=10)
    ok = (
        ev11["annotations"] == {}
        and late_compute.status_code == 409
        and late_judge.status_code == 409
        and fresh.status_code == 200
        and fresh.json()["n_unjudged"] == 40
    )
    failures += 0 if ok else 1
    record(
        "S11 更换评测集后重新标注 & 迟到请求被拒", ok,
        f"新批次标注数={len(ev11['annotations'])}；"
        f"旧 eval_id compute HTTP={late_compute.status_code}、"
        f"judge HTTP={late_judge.status_code}；"
        f"新批次 compute HTTP={fresh.status_code}",
        payload={"late_compute": late_compute.json()},
    )

    write_report(s1, s2, s2b, s3, s4, s5, s6, stale, failures,
                 eval_info={"s7": e7, "s8": (e8, lo_bf, hi_bf), "s9": full})
    print(f"\n{len(results) - failures}/{len(results)} scenarios passed.")
    return 1 if failures else 0


def _metrics_row(title: str, data: dict) -> str:
    m = data["metrics"]
    b = data["baseline_metrics"]
    return (
        f"| {title} | {data['status']} | {b['recall']:.3f} | {m['recall']:.3f} "
        f"| {b['fdr']:.3f} | {m['fdr']:.3f} | {m['n_retrieved']} "
        f"| {m['true_positives']} | {m['false_positives']} |"
    )


def write_report(s1, s2, s2b, s3, s4, s5, s6, stale, failures,
                 eval_info=None) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / "selftest_report.md"
    lines: list[str] = []
    lines.append("# 交付前自测报告（阈值冲突、修订与评测场景）")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append("- 语料：`data/demo_corpus.csv`（45 篇，正例 17 / 负例 28；"
                 "其中 2 篇正例只含文档级独有词，用于触发词表覆盖上限）")
    lines.append("- 评测集：`data/demo_eval.csv`（40 篇未审摘要，无标签列）")
    lines.append("- 初始查询：`quantum error correction`")
    lines.append("- 方式：FastAPI TestClient 全栈在进程内调用，无外部网络")
    lines.append("")
    lines.append("## 场景结果汇总")
    lines.append("")
    lines.append("| 场景 | 结果 | 实测细节 |")
    lines.append("|---|---|---|")
    for r in results:
        lines.append(f"| {r['scenario']} | {'✅ PASS' if r['ok'] else '❌ FAIL'} | {r['detail']} |")
    lines.append("")

    lines.append("## 关键指标实测")
    lines.append("")
    lines.append("| 场景 | 状态 | 基线召回 | 最终召回 | 基线fdr | 最终fdr | 命中数 | TP | FP |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    lines.append(_metrics_row("S1 常规扩展 (R≥0.85,F≤0.2)", s1))
    lines.append(_metrics_row("S2 词表上限冲突 (R=1.0,F≤0.05)", s2))
    lines.append(_metrics_row("S2b 误检预算冲突 (R≥0.85,F≤0.01)", s2b))
    lines.append(_metrics_row("S3 放宽 fdr (R≥0.85,F≤0.4)", s3))
    lines.append(_metrics_row("S4 排除全部原扩展词后重算", s4))
    lines.append(_metrics_row("S5 锁定 decoherence", s5))
    lines.append("")

    lines.append("## S2 失败诊断原文（词表覆盖上限）")
    lines.append("")
    for reason in s2["failure"]["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("## S2b 失败诊断原文（误检预算不足）")
    lines.append("")
    for reason in s2b["failure"]["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("## S1 逐词覆盖变化（trace 实测）")
    lines.append("")
    lines.append("| 步骤 | 新增正例 | 新增负例 | 累计召回 | 累计fdr | 累计FP |")
    lines.append("|---|---|---|---|---|---|")
    for t in s1["trace"]:
        lines.append(
            f"| {t['term']} | {t['added_relevant']} | {t['added_nonrelevant']} "
            f"| {t['cumulative_recall']:.3f} | {t['cumulative_fdr']:.3f} "
            f"| {t['cumulative_false_positives']} |"
        )
    lines.append("")
    lines.append("## 旧建议失效验证（S4）")
    lines.append("")
    lines.append(f"- 排除修订后携带旧 `suggestion_id` 调用 `/api/analyze` → "
                 f"HTTP **{stale.status_code}**")
    lines.append(f"- 响应内容：`{stale.json().get('detail')}`")
    lines.append("")

    if eval_info:
        e7 = eval_info["s7"]
        e8, lo_bf, hi_bf = eval_info["s8"]
        full = eval_info["s9"]
        b7 = e7["bounds"]
        lines.append("## 评测实测（新批次 NDCG@K 差值界）")
        lines.append("")
        lines.append("### S7 演示评测集（40 篇全未判，K=10）")
        lines.append("")
        lines.append(f"- 原查询词：`{', '.join(e7['query_terms_original'])}`")
        lines.append(f"- 扩展查询词：`{', '.join(e7['query_terms_expanded'])}`")
        lines.append(f"- 原榜命中 top-{e7['k']}："
                     f"{[r['id'] for r in e7['lists']['original']]}")
        lines.append(f"- 扩展榜命中 top-{e7['k']}："
                     f"{[r['id'] for r in e7['lists']['expanded']]}")
        lines.append(f"- 差值界：**[{b7['lower']:.4f}, {b7['upper']:.4f}]**；"
                     f"暂定差值（未判按不相关）{e7['tentative']['delta']:.4f}")
        lines.append(f"- 达上界应判相关的未判论文："
                     f"{[w['id'] for w in b7['upper_witness']]}")
        lines.append(f"- 达下界应判相关的未判论文："
                     f"{[w['id'] for w in b7['lower_witness']]}")
        lines.append("")
        lines.append("### S8 精确性对照（6 篇小集，K=3，2 篇已判）")
        lines.append("")
        lines.append(f"- API 界 = [{e8['bounds']['lower']:.6f}, "
                     f"{e8['bounds']['upper']:.6f}]")
        lines.append(f"- 穷举界 = [{lo_bf:.6f}, {hi_bf:.6f}]（2^4=16 种补全逐一计算）")
        lines.append("")
        lines.append("### S9 全部 40 篇判定后（界收敛）")
        lines.append("")
        lines.append(f"- 判定规则：标题含 (QEC) 判相关（{full['n_relevant_judged']} 篇），"
                     f"其余不相关")
        lines.append(f"- 收敛后 Δ = **{full['bounds']['lower']:.4f}**"
                     f"（下界 == 上界 == 暂定值）；"
                     f"确定改善={full['bounds']['certain_improvement']}；"
                     f"暂定 NDCG：原榜 {full['tentative']['ndcg_original']:.4f} / "
                     f"扩展榜 {full['tentative']['ndcg_expanded']:.4f}")
        lines.append("")

    lines.append(f"## 结论：**{len(results) - failures}/{len(results)} 通过**")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport written -> {path}")


if __name__ == "__main__":
    sys.exit(main())
