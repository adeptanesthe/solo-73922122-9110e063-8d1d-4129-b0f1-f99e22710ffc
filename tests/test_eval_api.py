"""End-to-end API tests for the evaluation endpoints (FastAPI TestClient)."""

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

EVAL_ROWS = [
    ("e1", "quantum error correction gates"),
    ("e2", "quantum error correction circuits"),
    ("e3", "surface code logical qubit"),
    ("e4", "syndrome decoherence mitigation"),
    ("e5", "neural network layers"),
    ("e6", "crispr gene editing"),
]


def _eval_csv(rows=EVAL_ROWS):
    buf = io.StringIO()
    buf.write("id,title,abstract\n")
    for rid, abstract in rows:
        buf.write(f'{rid},"title-{rid}","{abstract}"\n')
    return {"file": ("eval.csv", buf.getvalue(), "text/csv")}


@pytest.fixture()
def ready_session(records):
    """Session with an analysis suggestion and an uploaded eval set."""
    buf = io.StringIO()
    buf.write("id,title,abstract,relevant\n")
    for r in records:
        buf.write(
            f'{r["id"]},{r["title"]},"{r["abstract"]}",'
            f'{"true" if r["relevant"] else "false"}\n'
        )
    sid = client.post(
        "/api/sessions", files={"file": ("c.csv", buf.getvalue(), "text/csv")}
    ).json()["session_id"]
    sug = client.post("/api/analyze", json={
        "session_id": sid, "query": "quantum error correction",
        "min_recall": 0.9, "max_fdr": 0.2,
    }).json()["suggestion_id"]
    ev = client.post(f"/api/sessions/{sid}/eval-set", files=_eval_csv())
    assert ev.status_code == 200, ev.text
    return {"sid": sid, "suggestion": sug, "eval_id": ev.json()["eval_id"]}


def _compute(sid, eval_id, suggestion, k=4):
    return client.post(f"/api/sessions/{sid}/eval-compute", json={
        "eval_id": eval_id, "suggestion_id": suggestion, "k": k,
    })


def _judge(sid, eval_id, doc_id, relevant):
    return client.post(f"/api/sessions/{sid}/eval-judge", json={
        "eval_id": eval_id, "doc_id": doc_id, "relevant": relevant,
    })


# --------------------------------------------------------------------------- #
# upload validation
# --------------------------------------------------------------------------- #
def test_eval_upload_ok(ready_session):
    assert ready_session["eval_id"]


def _corpus_upload(records):
    buf = io.StringIO()
    buf.write("id,title,abstract,relevant\n")
    for r in records:
        buf.write(
            f'{r["id"]},{r["title"]},"{r["abstract"]}",'
            f'{"true" if r["relevant"] else "false"}\n'
        )
    return {"file": ("c.csv", buf.getvalue(), "text/csv")}


def test_eval_upload_rejects_too_many_and_duplicates(records):
    sid = client.post("/api/sessions", files=_corpus_upload(records)
                      ).json()["session_id"]
    many = [("e%d" % i, "quantum error correction") for i in range(201)]
    resp = client.post(f"/api/sessions/{sid}/eval-set", files=_eval_csv(many))
    assert resp.status_code == 400 and "最多" in resp.json()["detail"]
    dup = [("e1", "quantum error correction"), ("e1", "surface code")]
    resp = client.post(f"/api/sessions/{sid}/eval-set", files=_eval_csv(dup))
    assert resp.status_code == 400 and "唯一" in resp.json()["detail"]


def test_eval_upload_unknown_session_404():
    resp = client.post("/api/sessions/nope/eval-set", files=_eval_csv())
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# compute: lists, bounds, K validation
# --------------------------------------------------------------------------- #
def test_compute_returns_lists_and_bounds(ready_session):
    resp = _compute(ready_session["sid"], ready_session["eval_id"],
                    ready_session["suggestion"], k=4)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["k"] == 4 and data["n_docs"] == len(EVAL_ROWS)
    assert len(data["lists"]["original"]) <= 4
    assert len(data["lists"]["expanded"]) <= 4
    # 扩展查询命中应不少于原查询（扩展词只增命中）
    assert len(data["lists"]["expanded"]) >= len(data["lists"]["original"])
    b = data["bounds"]
    assert b["lower"] <= data["tentative"]["delta"] <= b["upper"]
    assert b["lower"] <= b["upper"]
    assert data["n_unjudged"] == len(EVAL_ROWS)
    # 全部未判且无已判相关时，全不相关补全给出 Δ=0 → 界跨过 0
    assert b["lower"] <= 0.0 <= b["upper"]


def test_compute_k_validation(ready_session):
    for bad_k in (0, 21, -1):
        resp = _compute(ready_session["sid"], ready_session["eval_id"],
                        ready_session["suggestion"], k=bad_k)
        assert resp.status_code == 422


def test_compute_requires_fresh_suggestion(ready_session):
    resp = _compute(ready_session["sid"], ready_session["eval_id"],
                    "not-a-real-suggestion")
    assert resp.status_code == 409


# --------------------------------------------------------------------------- #
# judging: shared annotations, correction, recomputation
# --------------------------------------------------------------------------- #
def test_judge_updates_annotations_and_bounds(ready_session):
    sid, ev, sug = ready_session["sid"], ready_session["eval_id"], \
        ready_session["suggestion"]
    before = _compute(sid, ev, sug).json()["bounds"]
    assert _judge(sid, ev, "e1", True).status_code == 200
    assert _judge(sid, ev, "e3", True).status_code == 200
    mid = _compute(sid, ev, sug).json()
    assert mid["n_judged"] == 2 and mid["n_relevant_judged"] == 2
    # 标注后区间不得变宽
    assert mid["bounds"]["lower"] >= before["lower"] - 1e-9
    assert mid["bounds"]["upper"] <= before["upper"] + 1e-9
    # 更正：e1 改判不相关
    assert _judge(sid, ev, "e1", False).status_code == 200
    # 撤销：e3 回到未审
    resp = _judge(sid, ev, "e3", None)
    assert resp.status_code == 200
    assert resp.json()["n_judged"] == 1
    assert "e3" not in resp.json()["annotations"]


def test_judge_unknown_doc_and_wrong_eval_id(ready_session):
    sid, ev = ready_session["sid"], ready_session["eval_id"]
    assert _judge(sid, ev, "nope", True).status_code == 400
    assert _judge(sid, "stale-eval-id", "e1", True).status_code == 409


def test_all_judged_collapses_bounds(ready_session):
    sid, ev, sug = ready_session["sid"], ready_session["eval_id"], \
        ready_session["suggestion"]
    for rid, _ in EVAL_ROWS:
        assert _judge(sid, ev, rid, rid in {"e1", "e3"}).status_code == 200
    data = _compute(sid, ev, sug).json()
    assert data["n_unjudged"] == 0
    assert data["bounds"]["lower"] == pytest.approx(data["bounds"]["upper"])
    assert data["bounds"]["lower"] == pytest.approx(
        data["tentative"]["delta"]
    )


# --------------------------------------------------------------------------- #
# suggestion lifecycle: 失效撤下结论、标注保留、换集重标、迟到拒绝
# --------------------------------------------------------------------------- #
def test_revision_withdraws_conclusions_but_keeps_annotations(ready_session):
    sid, ev, sug = ready_session["sid"], ready_session["eval_id"], \
        ready_session["suggestion"]
    assert _judge(sid, ev, "e1", True).status_code == 200
    # 修订 → 旧建议失效
    rev = client.post("/api/revise", json={
        "session_id": sid, "exclude": ["surface"],
    })
    assert rev.status_code == 200
    # 旧建议的评测结论必须撤下（409），但本批标注保留
    stale = _compute(sid, ev, sug)
    assert stale.status_code == 409
    # 重新分析 → 新建议可继续用同一份标注计算
    fresh = client.post("/api/analyze", json={
        "session_id": sid, "query": "quantum error correction",
        "min_recall": 0.9, "max_fdr": 0.2,
    }).json()
    data = _compute(sid, ev, fresh["suggestion_id"]).json()
    assert data["n_judged"] == 1
    assert data["annotations"] == {"e1": True}


def test_new_eval_set_requires_reannotation_and_rejects_late_calls(
        ready_session):
    sid, ev, sug = ready_session["sid"], ready_session["eval_id"], \
        ready_session["suggestion"]
    assert _judge(sid, ev, "e1", True).status_code == 200
    # 更换评测集 → 新批次，标注清零
    rows2 = [("n1", "quantum error correction"), ("n2", "surface code qubit")]
    ev2 = client.post(f"/api/sessions/{sid}/eval-set",
                      files=_eval_csv(rows2)).json()
    assert ev2["annotations"] == {} and ev2["eval_id"] != ev
    # 迟到请求（旧 eval_id）不得恢复旧结论：compute 与 judge 都 409
    assert _compute(sid, ev, sug).status_code == 409
    assert _judge(sid, ev, "e2", True).status_code == 409
    # 新批次可正常计算
    ok = _compute(sid, ev2["eval_id"], sug)
    assert ok.status_code == 200
    assert ok.json()["n_docs"] == 2


def test_compute_before_any_analysis(records):
    sid = client.post("/api/sessions", files=_corpus_upload(records)
                      ).json()["session_id"]
    ev = client.post(f"/api/sessions/{sid}/eval-set", files=_eval_csv()).json()
    resp = _compute(sid, ev["eval_id"], "whatever")
    assert resp.status_code == 409   # 无有效建议 → 拒绝
