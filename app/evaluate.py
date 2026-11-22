"""Evaluate a query-expansion suggestion on a new batch of unjudged abstracts.

The researcher uploads up to 200 unjudged abstracts (CSV, unique ids), picks
K (1..20), and the current suggestion's original / expanded queries are run
on this evaluation set, reusing the *original* corpus TF-IDF model and the
original hit & ranking rules (boolean-OR hits over shared vocabulary,
TF-IDF cosine ranking, document-index tie-break — same as
``engine._rank_documents``).

Each paper can be judged relevant / not relevant at any time; both lists
share one annotation set and judgments can be corrected.  Unjudged papers
are treated as possibly relevant or irrelevant, and this module computes
*exact* upper/lower bounds of

    Δ = NDCG@K(expanded list) − NDCG@K(original list)

over all 2^U labellings of the U unjudged papers **without enumerating
them**, and reports which unjudged papers must be judged relevant to
attain each endpoint.  The ideal list is computed over the whole
evaluation set; when no paper is relevant both lists score 0 (hence
Δ = 0 for that completion).

Method
------
For a completion x ∈ {0,1}^U (x_d = 1 ⟺ unjudged paper d is relevant):

    Δ(x) = ( c_B − c_A + Σ_d δ_d · x_d ) / f(R_known + Σ_d x_d)

where c_A / c_B are DCG contributions of already-judged-relevant papers,
δ_d = w_B(d) − w_A(d) with w = 1/log2(rank+1) (0 when absent from a
list), R_known is the number of judged-relevant papers, and
f(r) = Σ_{i=1..min(r,K)} 1/log2(i+1) is the ideal DCG@K.

The denominator depends on x only through the count S = Σ x_d, so for a
fixed S the numerator is maximised by judging relevant the S papers with
the largest δ_d (prefix sums of δ sorted descending) and minimised by
the S smallest.  Scanning S = 0..U is therefore exact and costs
O(U log U) — no enumeration of label combinations.
"""

from __future__ import annotations

import csv
import io
import math
import uuid
from dataclasses import dataclass, field
from typing import Optional

from sklearn.metrics.pairwise import cosine_similarity

from .engine import Corpus, EngineError

MAX_EVAL_DOCS = 200
MIN_K, MAX_K = 1, 20
# 判定“确定改善/确定变差”时对浮点零的容差
_EPS = 1e-9


class EvalError(EngineError):
    """User-facing evaluation errors (bad CSV, unknown ids, ...)."""


# --------------------------------------------------------------------------- #
# Eval-set parsing
# --------------------------------------------------------------------------- #
def parse_eval_file(raw: bytes, filename: str,
                    max_docs: int = MAX_EVAL_DOCS) -> list[dict]:
    """Parse an unjudged-abstract CSV/TSV upload.

    Required columns: ``id``, ``abstract`` (``title`` optional).  A
    ``relevant`` column, if present, is ignored — every paper starts
    unjudged by definition.
    """
    name = (filename or "").lower()
    if not name.endswith((".csv", ".tsv", ".txt")):
        raise EvalError("评测集只支持 .csv / .tsv 文件（表头含 id,title,abstract）。")
    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise EvalError(f"评测集文件不是有效的 UTF-8 文本：{exc}") from exc
    delim = "\t" if name.endswith(".tsv") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    if reader.fieldnames is None:
        raise EvalError("评测集文件缺少表头。")
    field_map = {h.strip().lower(): h for h in reader.fieldnames if h}
    if "id" not in field_map or "abstract" not in field_map:
        raise EvalError("评测集表头必须包含 id 与 abstract 两列（title 可选）。")

    def cell(row: dict, name: str) -> str:
        if name not in field_map:
            return ""
        value = row.get(field_map[name])
        return value.strip() if isinstance(value, str) else ""

    records: list[dict] = []
    for row in reader:
        if row is None:
            continue
        rec = {
            "id": cell(row, "id"),
            "title": cell(row, "title"),
            "abstract": cell(row, "abstract"),
        }
        if not rec["id"] and not rec["abstract"]:
            continue  # tolerate blank lines
        records.append(rec)

    if not records:
        raise EvalError("评测集为空：至少需要 1 篇未审摘要。")
    if len(records) > max_docs:
        raise EvalError(
            f"评测集最多 {max_docs} 篇，本次上传了 {len(records)} 篇，请删减后重试。"
        )

    seen: set[str] = set()
    duplicates: list[str] = []
    for i, rec in enumerate(records):
        if not rec["id"]:
            raise EvalError(f"第 {i + 1} 行的 id 为空；每篇论文编号必须非空且唯一。")
        if rec["id"] in seen:
            duplicates.append(rec["id"])
        seen.add(rec["id"])
        if not rec["abstract"]:
            raise EvalError(f"第 {i + 1} 行（id={rec['id']}）缺少 abstract 摘要内容。")
    if duplicates:
        raise EvalError(
            "评测集编号必须唯一，以下 id 重复：" + ", ".join(sorted(set(duplicates)))
        )
    for i, rec in enumerate(records):
        if not rec["title"]:
            rec["title"] = f"paper-{rec['id']}"
    return records


# --------------------------------------------------------------------------- #
# Eval batch (uploaded unjudged set + shared annotations)
# --------------------------------------------------------------------------- #
@dataclass
class EvalBatch:
    """One uploaded evaluation set with its own annotation batch.

    Annotations live on the batch (not on the suggestion), so a revised
    suggestion reuses them, while uploading a new batch starts a fresh
    empty annotation set.
    """

    eval_id: str
    doc_ids: list[str]
    titles: list[str]
    abstracts: list[str]
    tfidf: object                       # transformed with the *corpus* vectorizer
    bool_mat: object                    # boolean copy of tfidf (shared vocab)
    annotations: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_records(cls, records: list[dict], corpus: Corpus) -> "EvalBatch":
        abstracts = [r["abstract"] for r in records]
        # 沿用原语料拟合的 TF-IDF：只 transform，不重新 fit
        tfidf = corpus.vectorizer.transform(abstracts)
        bool_mat = (tfidf > 0).astype(int)
        return cls(
            eval_id=uuid.uuid4().hex,
            doc_ids=[r["id"] for r in records],
            titles=[r["title"] for r in records],
            abstracts=abstracts,
            tfidf=tfidf,
            bool_mat=bool_mat,
        )

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    def docs_for_term(self, corpus: Corpus, term: str) -> set[int]:
        j = corpus.term_index(term)
        if j is None:
            return set()
        return set(self.bool_mat.tocsr().getcol(j).nonzero()[0].tolist())

    def judge(self, doc_id: str, relevant: Optional[bool]) -> None:
        """Set / correct / clear one judgment (shared by both lists)."""
        if doc_id not in set(self.doc_ids):
            raise EvalError(f"评测集中不存在编号为 {doc_id!r} 的论文。")
        if relevant is None:
            self.annotations.pop(doc_id, None)
        else:
            self.annotations[doc_id] = bool(relevant)


# --------------------------------------------------------------------------- #
# Ranking on the eval set — mirrors engine._rank_documents exactly
# --------------------------------------------------------------------------- #
def topk_list(corpus: Corpus, batch: EvalBatch,
              query_terms: list[str], k: int) -> list[dict]:
    """Boolean-OR hits + TF-IDF cosine ranking, top-k (1-based ranks)."""
    terms = [t for t in dict.fromkeys(query_terms) if corpus.term_index(t) is not None]
    if not terms:
        return []
    qvec = corpus.vectorizer.transform([" ".join(terms)])
    sims = cosine_similarity(qvec, batch.tfidf).ravel()
    hits: set[int] = set()
    for t in terms:
        hits |= batch.docs_for_term(corpus, t)
    ranked = sorted(hits, key=lambda d: (-sims[d], d))  # 与原排序规则一致
    rows = []
    for rank, d in enumerate(ranked[:k], start=1):
        rows.append({
            "rank": rank,
            "id": batch.doc_ids[d],
            "title": batch.titles[d],
            "score": round(float(sims[d]), 6),
            "matched_terms": [
                t for t in terms if d in batch.docs_for_term(corpus, t)
            ],
        })
    return rows


# --------------------------------------------------------------------------- #
# NDCG@K difference bounds (exact, polynomial — see module docstring)
# --------------------------------------------------------------------------- #
def _dcg_weight(rank: int) -> float:
    """Discount for a 1-based rank."""
    return 1.0 / math.log2(rank + 1)


def _ideal_dcg(n_relevant: int, k: int) -> float:
    """IDCG@K of the ideal list over the whole evaluation set."""
    return sum(_dcg_weight(i) for i in range(1, min(n_relevant, k) + 1))


def ndcg_delta_bounds(
    doc_ids: list[str],
    judgments: dict[str, bool],
    list_a: list[str],
    list_b: list[str],
    k: int,
) -> dict:
    """Exact [lower, upper] bounds of NDCG@K(B) − NDCG@K(A).

    ``judgments`` maps judged doc ids to bool; every id in ``doc_ids``
    missing from it is unjudged and free to be relevant or irrelevant.
    Returns the bounds, the unjudged papers to judge relevant at each
    endpoint (all other unjudged papers are judged irrelevant there), and
    the tentative NDCG values under the current partial judgments
    (unjudged counted as irrelevant).
    """
    pos_a = {d: i + 1 for i, d in enumerate(list_a)}
    pos_b = {d: i + 1 for i, d in enumerate(list_b)}
    w_a = {d: _dcg_weight(p) for d, p in pos_a.items()}
    w_b = {d: _dcg_weight(p) for d, p in pos_b.items()}

    judged_rel = [d for d, v in judgments.items() if v]
    r_known = len(judged_rel)
    c_a = sum(w_a.get(d, 0.0) for d in judged_rel)
    c_b = sum(w_b.get(d, 0.0) for d in judged_rel)

    unjudged = [d for d in doc_ids if d not in judgments]
    delta = {d: w_b.get(d, 0.0) - w_a.get(d, 0.0) for d in unjudged}

    # 固定相关总数 S 时，分子最大 = 取 δ 降序前 S 项；最小 = 取升序前 S 项
    desc = sorted(unjudged, key=lambda d: (-delta[d], d))
    asc = sorted(unjudged, key=lambda d: (delta[d], d))
    pref_desc = [0.0]
    for d in desc:
        pref_desc.append(pref_desc[-1] + delta[d])
    pref_asc = [0.0]
    for d in asc:
        pref_asc.append(pref_asc[-1] + delta[d])

    base = c_b - c_a
    upper, upper_s = -math.inf, 0
    lower, lower_s = math.inf, 0
    for s in range(len(unjudged) + 1):
        r = r_known + s
        denom = _ideal_dcg(r, k)
        # r == 0：全集无相关论文，两榜 NDCG 均记 0 → Δ = 0
        v_up = 0.0 if r == 0 else (base + pref_desc[s]) / denom
        v_lo = 0.0 if r == 0 else (base + pref_asc[s]) / denom
        if v_up > upper:
            upper, upper_s = v_up, s
        if v_lo < lower:
            lower, lower_s = v_lo, s

    idcg_now = _ideal_dcg(r_known, k)
    ndcg_a_now = 0.0 if r_known == 0 else c_a / idcg_now
    ndcg_b_now = 0.0 if r_known == 0 else c_b / idcg_now

    return {
        "lower": lower,
        "upper": upper,
        "certain_improvement": lower > _EPS,
        "certain_regression": upper < -_EPS,
        "lower_witness_ids": asc[:lower_s],
        "upper_witness_ids": desc[:upper_s],
        "tentative": {
            "ndcg_original": ndcg_a_now,
            "ndcg_expanded": ndcg_b_now,
            "delta": ndcg_b_now - ndcg_a_now,
        },
    }


# --------------------------------------------------------------------------- #
# Full evaluation payload
# --------------------------------------------------------------------------- #
def evaluate_batch(
    corpus: Corpus,
    batch: EvalBatch,
    baseline_terms: list[str],
    selected_terms: list[str],
    k: int,
) -> dict:
    """Run both queries on the eval set and compute the Δ bounds."""
    if not MIN_K <= k <= MAX_K:
        raise EvalError(f"K 必须在 {MIN_K}～{MAX_K} 之间。")
    original_terms = list(baseline_terms)
    expanded_terms = list(baseline_terms) + [
        t for t in selected_terms if t not in set(baseline_terms)
    ]
    list_a = topk_list(corpus, batch, original_terms, k)
    list_b = topk_list(corpus, batch, expanded_terms, k)
    bounds = ndcg_delta_bounds(
        batch.doc_ids, batch.annotations,
        [r["id"] for r in list_a], [r["id"] for r in list_b], k,
    )

    title_by_id = dict(zip(batch.doc_ids, batch.titles))
    in_a = {r["id"] for r in list_a}
    in_b = {r["id"] for r in list_b}

    def witness(ids: list[str]) -> list[dict]:
        return [
            {
                "id": d,
                "title": title_by_id[d],
                "in_original": d in in_a,
                "in_expanded": d in in_b,
            }
            for d in ids
        ]

    def annotate(rows: list[dict]) -> list[dict]:
        for row in rows:
            row["judgment"] = batch.annotations.get(row["id"])  # True/False/None
        return rows

    n_judged = len(batch.annotations)
    return {
        "k": k,
        "n_docs": batch.n_docs,
        "query_terms_original": original_terms,
        "query_terms_expanded": expanded_terms,
        "lists": {"original": annotate(list_a), "expanded": annotate(list_b)},
        "annotations": dict(batch.annotations),
        "n_judged": n_judged,
        "n_unjudged": batch.n_docs - n_judged,
        "n_relevant_judged": sum(1 for v in batch.annotations.values() if v),
        "tentative": {key: round(val, 6) for key, val in bounds["tentative"].items()},
        "bounds": {
            "lower": round(bounds["lower"], 6),
            "upper": round(bounds["upper"], 6),
            "certain_improvement": bounds["certain_improvement"],
            "certain_regression": bounds["certain_regression"],
            "lower_witness": witness(bounds["lower_witness_ids"]),
            "upper_witness": witness(bounds["upper_witness_ids"]),
        },
    }
