"""Evaluation-layer tests.

The centrepiece is a brute-force cross-validation of the polynomial
NDCG@K-difference bounds against full enumeration of every labelling of
the unjudged papers (only feasible for tiny sets — exactly why the
production path must not enumerate).
"""

import itertools
import math
import random

import pytest

from app.engine import Corpus
from app.evaluate import (
    EvalBatch,
    EvalError,
    ndcg_delta_bounds,
    parse_eval_file,
    topk_list,
)


# --------------------------------------------------------------------------- #
# brute-force reference
# --------------------------------------------------------------------------- #
def _w(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def _delta_for(labels: dict, list_a: list, list_b: list, k: int) -> float:
    r = sum(1 for v in labels.values() if v)
    if r == 0:                       # 全集不相关：两榜记 0
        return 0.0
    idcg = sum(_w(i) for i in range(1, min(r, k) + 1))
    dcg_a = sum(_w(i + 1) for i, d in enumerate(list_a) if labels[d])
    dcg_b = sum(_w(i + 1) for i, d in enumerate(list_b) if labels[d])
    return dcg_b / idcg - dcg_a / idcg


def brute_force_bounds(doc_ids, judgments, list_a, list_b, k):
    unjudged = [d for d in doc_ids if d not in judgments]
    lo, hi = math.inf, -math.inf
    for bits in range(2 ** len(unjudged)):
        labels = dict(judgments)
        for i, d in enumerate(unjudged):
            labels[d] = bool(bits >> i & 1)
        delta = _delta_for(labels, list_a, list_b, k)
        lo, hi = min(lo, delta), max(hi, delta)
    return lo, hi


def _random_case(rng: random.Random, n_docs: int = 8):
    doc_ids = [f"d{i}" for i in range(n_docs)]
    k = rng.randint(1, 4)
    shuffled = doc_ids[:]
    rng.shuffle(shuffled)
    list_a = shuffled[: rng.randint(0, k)]
    rng.shuffle(shuffled)
    list_b = shuffled[: rng.randint(0, k)]
    judgments = {}
    for d in doc_ids:
        if rng.random() < 0.4:
            judgments[d] = rng.random() < 0.5
    return doc_ids, judgments, list_a, list_b, k


# --------------------------------------------------------------------------- #
# exactness vs enumeration
# --------------------------------------------------------------------------- #
def test_bounds_match_brute_force_on_random_cases():
    rng = random.Random(20261122)
    for case in range(400):
        doc_ids, judgments, list_a, list_b, k = _random_case(rng)
        res = ndcg_delta_bounds(doc_ids, judgments, list_a, list_b, k)
        lo, hi = brute_force_bounds(doc_ids, judgments, list_a, list_b, k)
        assert res["lower"] == pytest.approx(lo, abs=1e-9), f"case {case}"
        assert res["upper"] == pytest.approx(hi, abs=1e-9), f"case {case}"
        # 暂定值（未判全部按不相关）本身也是一种补全，必在界内
        assert lo - 1e-9 <= res["tentative"]["delta"] <= hi + 1e-9


def test_witnesses_attain_the_bounds():
    rng = random.Random(7)
    for _ in range(200):
        doc_ids, judgments, list_a, list_b, k = _random_case(rng)
        res = ndcg_delta_bounds(doc_ids, judgments, list_a, list_b, k)
        unjudged = [d for d in doc_ids if d not in judgments]
        for key, bound in (("lower_witness_ids", res["lower"]),
                           ("upper_witness_ids", res["upper"])):
            witness = set(res[key])
            assert witness <= set(unjudged)
            labels = dict(judgments)
            for d in unjudged:
                labels[d] = d in witness
            assert _delta_for(labels, list_a, list_b, k) == pytest.approx(
                bound, abs=1e-9
            )


# --------------------------------------------------------------------------- #
# targeted properties
# --------------------------------------------------------------------------- #
def test_identical_lists_have_zero_bounds():
    doc_ids = ["a", "b", "c", "d"]
    res = ndcg_delta_bounds(doc_ids, {}, ["a", "b"], ["a", "b"], 2)
    assert res["lower"] == pytest.approx(0.0)
    assert res["upper"] == pytest.approx(0.0)
    assert not res["certain_improvement"]


def test_all_judged_collapses_to_single_value():
    doc_ids = ["a", "b", "c"]
    judgments = {"a": True, "b": False, "c": True}
    res = ndcg_delta_bounds(doc_ids, judgments, ["a", "b"], ["b", "c"], 2)
    expect = _delta_for(judgments, ["a", "b"], ["b", "c"], 2)
    assert res["lower"] == pytest.approx(expect)
    assert res["upper"] == pytest.approx(expect)
    assert res["lower_witness_ids"] == [] and res["upper_witness_ids"] == []


def test_all_irrelevant_scores_zero():
    doc_ids = ["a", "b", "c"]
    judgments = {"a": False, "b": False, "c": False}
    res = ndcg_delta_bounds(doc_ids, judgments, ["a"], ["b"], 1)
    assert res["lower"] == 0.0 and res["upper"] == 0.0
    assert res["tentative"]["ndcg_original"] == 0.0
    assert res["tentative"]["ndcg_expanded"] == 0.0


def test_zero_known_relevant_keeps_zero_inside_bounds():
    # 无已判相关 + 存在未判：全不相关补全给出 Δ=0，界必跨过 0
    doc_ids = ["a", "b", "c", "d"]
    res = ndcg_delta_bounds(doc_ids, {"a": False}, ["b"], ["c"], 2)
    assert res["lower"] <= 0.0 <= res["upper"]


def test_expanded_superset_list_cannot_lose():
    # 扩展榜在同一相对顺序下是原榜超集 → δ_d ≥ 0，下界 ≥ 0
    doc_ids = ["a", "b", "c", "d", "e"]
    res = ndcg_delta_bounds(
        doc_ids, {"e": False}, ["a", "c"], ["a", "c", "b", "d"], 4
    )
    assert res["lower"] >= -1e-12


def test_certain_improvement_flag():
    # 扩展榜独占全部已判相关，原榜为空 → 任何补全下 Δ > 0
    doc_ids = ["a", "b"]
    res = ndcg_delta_bounds(doc_ids, {"a": True, "b": False}, [], ["a"], 1)
    assert res["lower"] > 0 and res["certain_improvement"]
    assert res["tentative"]["ndcg_original"] == 0.0
    assert res["tentative"]["ndcg_expanded"] == pytest.approx(1.0)


def test_witness_prefers_list_positions_over_idcg_only_docs():
    # 上界证人应优先取扩展榜高位次的未判论文，而非榜外论文；
    # 同一上界在多个 S 处达到时取所需标注最少的一个
    doc_ids = ["x", "y", "z"]
    res = ndcg_delta_bounds(doc_ids, {}, [], ["x", "y"], 2)
    assert res["upper"] == pytest.approx(1.0)
    assert res["upper_witness_ids"] == ["x"]   # S=1 即达上界，z 永不在证人中
    assert "z" not in res["upper_witness_ids"]


# --------------------------------------------------------------------------- #
# eval-set CSV parsing
# --------------------------------------------------------------------------- #
def _csv(rows, header="id,title,abstract"):
    lines = [header] + [",".join(r) for r in rows]
    return "\n".join(lines).encode()


def test_parse_eval_csv_ok_and_ignores_label_column():
    raw = _csv(
        [("e1", "t1", "quantum error correction", "true"),
         ("e2", "t2", "neural network", "false")],
        header="id,title,abstract,relevant",
    )
    recs = parse_eval_file(raw, "eval.csv")
    assert [r["id"] for r in recs] == ["e1", "e2"]
    assert all("relevant" not in r for r in recs)


def test_parse_eval_rejects_over_200():
    raw = _csv([(f"e{i}", "t", "abstract text") for i in range(201)])
    with pytest.raises(EvalError, match="最多"):
        parse_eval_file(raw, "eval.csv")
    raw200 = _csv([(f"e{i}", "t", "abstract text") for i in range(200)])
    assert len(parse_eval_file(raw200, "eval.csv")) == 200


def test_parse_eval_rejects_duplicate_and_empty_ids():
    with pytest.raises(EvalError, match="唯一"):
        parse_eval_file(_csv([("e1", "t", "a b"), ("e1", "t", "c d")]), "e.csv")
    with pytest.raises(EvalError, match="id 为空"):
        parse_eval_file(_csv([("", "t", "a b")]), "e.csv")


def test_parse_eval_rejects_missing_abstract_and_bad_type():
    with pytest.raises(EvalError, match="abstract"):
        parse_eval_file(b"id,title\ne1,t\n", "e.csv")
    with pytest.raises(EvalError, match="缺少 abstract"):
        parse_eval_file(b"id,title,abstract\ne1,t,\n", "e.csv")
    with pytest.raises(EvalError, match="csv"):
        parse_eval_file(b"[{}]", "e.json")
    with pytest.raises(EvalError, match="为空"):
        parse_eval_file(b"id,title,abstract\n", "e.csv")


# --------------------------------------------------------------------------- #
# ranking rules on the eval set reuse the corpus TF-IDF model
# --------------------------------------------------------------------------- #
def _eval_batch(corpus, rows):
    recs = [{"id": r[0], "title": r[0], "abstract": r[1]} for r in rows]
    return EvalBatch.from_records(recs, corpus)


def test_topk_uses_boolean_or_hits_and_cosine_order(corpus):
    batch = _eval_batch(corpus, [
        ("e1", "quantum error correction thresholds"),
        ("e2", "surface code logical qubit"),
        ("e3", "neural network graph"),
        ("e4", "quantum quantum quantum"),
    ])
    rows = topk_list(corpus, batch, ["quantum"], k=10)
    ids = [r["id"] for r in rows]
    assert set(ids) == {"e1", "e4"}          # 布尔 OR 命中
    assert ids == sorted(ids, key=lambda i: -next(
        r["score"] for r in rows if r["id"] == i))  # 余弦降序
    # 扩展词把 surface/code 文档纳入命中
    rows2 = topk_list(corpus, batch, ["quantum", "surface", "code"], k=10)
    assert {r["id"] for r in rows2} == {"e1", "e2", "e4"}
    # top-k 截断
    assert len(topk_list(corpus, batch, ["quantum"], k=1)) == 1
    # 词表外词被丢弃；全词表外 → 无命中
    assert topk_list(corpus, batch, ["zzz-not-in-vocab"], k=5) == []


def test_topk_tie_break_is_deterministic_by_index(corpus):
    batch = _eval_batch(corpus, [
        ("e1", "quantum error correction"),
        ("e2", "quantum error correction"),
    ])
    rows = topk_list(corpus, batch, ["quantum"], k=2)
    assert [r["id"] for r in rows] == ["e1", "e2"]   # 同分按评测集顺序
