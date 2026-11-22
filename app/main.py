"""FastAPI application: corpus upload, query expansion, term revisions."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import (
    AnalysisRequest,
    Corpus,
    CorpusError,
    EngineError,
    RevisionError,
    ThresholdError,
    analyze,
)
from .evaluate import (
    MAX_EVAL_DOCS,
    EvalBatch,
    EvalError,
    evaluate_batch,
    parse_eval_file,
)
from .session import SessionStore, store

app = FastAPI(
    title="科研检索查询扩展系统",
    description=(
        "上传带相关性标签的论文摘要，选择正例候选词扩展查询，"
        "在召回下限与误检上限约束下重排结果并解释每个扩展词的覆盖变化。"
    ),
    version="1.0.0",
)
session_store: SessionStore = store

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class AnalyzeBody(BaseModel):
    session_id: str
    query: str = Field(..., min_length=1)
    min_recall: float = Field(..., ge=0.0, le=1.0)
    max_fdr: float = Field(..., ge=0.0, le=1.0)
    max_false_positives: int | None = Field(default=None, ge=0)
    beam_width: int = Field(default=8, ge=1, le=64)
    max_terms: int = Field(default=12, ge=1, le=30)
    top_k: int = Field(default=50, ge=1, le=500)


class ReviseBody(BaseModel):
    session_id: str
    lock: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    unlock: list[str] = Field(default_factory=list)
    unexclude: list[str] = Field(default_factory=list)
    # 若携带旧建议 ID，服务端校验其是否已被本次修订作废
    stale_suggestion_id: str | None = None


class AnalyzeWithSuggestionBody(AnalyzeBody):
    suggestion_id: str | None = None


class EvalJudgeBody(BaseModel):
    eval_id: str
    doc_id: str
    # true=相关，false=不相关，null=撤销判定（回到未审）
    relevant: bool | None = None


class EvalComputeBody(BaseModel):
    eval_id: str
    suggestion_id: str
    k: int = Field(..., ge=1, le=20)


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
@app.exception_handler(EngineError)
async def engine_error_handler(request, exc: EngineError):
    status = 409 if isinstance(exc, RevisionError) else 400
    return _json_error(status, exc)


@app.exception_handler(CorpusError)
async def corpus_error_handler(request, exc: CorpusError):
    return _json_error(400, exc)


def _json_error(status: int, exc: Exception):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/demo-corpus")
def demo_corpus():
    """Serve the built-in demo corpus so the browser demo needs no upload."""
    path = Path(__file__).resolve().parent.parent / "data" / "demo_corpus.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="演示语料尚未生成。")
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        path.read_text(encoding="utf-8"),
        headers={"Content-Disposition": 'attachment; filename="demo_corpus.csv"'},
    )


@app.post("/api/sessions")
async def create_session(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    try:
        corpus = Corpus.from_file(raw, file.filename or "upload.csv")
    except CorpusError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # malformed csv/json etc.
        raise HTTPException(
            status_code=400, detail=f"文件解析失败：{exc}"
        ) from exc
    session = session_store.create(corpus)
    return {
        "session_id": session.session_id,
        "revision": session.revision,
        "corpus_size": len(corpus.labels),
        "n_relevant": corpus.n_relevant,
        "n_nonrelevant": corpus.n_nonrelevant,
        "vocab_size": len(corpus.vocab),
    }


@app.post("/api/analyze")
def analyze_endpoint(body: AnalyzeWithSuggestionBody):
    try:
        session = session_store.get(body.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="会话不存在或已过期，请重新上传语料。")

    # 旧建议校验：如果前端带着一个 suggestion_id，先确认它没有被修订作废
    if body.suggestion_id:
        try:
            session_store.ensure_fresh(session, body.suggestion_id)
        except RevisionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    req = AnalysisRequest(
        query=body.query,
        min_recall=body.min_recall,
        max_fdr=body.max_fdr,
        max_false_positives=body.max_false_positives,
        locked_terms=session.locked_terms,
        excluded_terms=session.excluded_terms,
        beam_width=body.beam_width,
        max_terms=body.max_terms,
        top_k=body.top_k,
    )
    try:
        result = analyze(session.corpus, req)
    except (ThresholdError, CorpusError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    suggestion_id, revision = session_store.bind_suggestion(
        session,
        analysis_terms={
            "query": result["query"],
            "baseline_terms": result["baseline_terms"],
            "selected_terms": result["selected_terms"],
        },
    )
    result["suggestion_id"] = suggestion_id
    result["revision"] = revision
    result["current_locked_terms"] = list(session.locked_terms)
    result["current_excluded_terms"] = list(session.excluded_terms)
    return result


@app.post("/api/revise")
def revise_endpoint(body: ReviseBody):
    try:
        session = session_store.get(body.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="会话不存在或已过期，请重新上传语料。")

    # 先用旧 ID 演示“旧建议失效”的显式报错路径
    if body.stale_suggestion_id:
        try:
            session_store.ensure_fresh(session, body.stale_suggestion_id)
        except RevisionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        session = session_store.revise(
            body.session_id,
            lock=body.lock,
            exclude=body.exclude,
            unlock=body.unlock,
            unexclude=body.unexclude,
        )
    except ThresholdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "session_id": session.session_id,
        "revision": session.revision,
        "locked_terms": list(session.locked_terms),
        "excluded_terms": list(session.excluded_terms),
        "previous_suggestion_invalidated": True,
        "message": (
            "锁定/排除词已更新，修订号 +1；此前生成的建议已失效，"
            "请重新发起分析。"
        ),
    }


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    try:
        session = session_store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="会话不存在或已过期。")
    return {
        "session_id": session.session_id,
        "revision": session.revision,
        "locked_terms": list(session.locked_terms),
        "excluded_terms": list(session.excluded_terms),
        "last_suggestion_id": session.last_suggestion_id,
        "last_suggestion_rev": session.last_suggestion_rev,
    }


# --------------------------------------------------------------------------- #
# 评测：新批次未审摘要上的 NDCG@K 差值界
# --------------------------------------------------------------------------- #
@app.get("/api/demo-eval")
def demo_eval():
    """Serve the built-in demo evaluation set (unjudged abstracts)."""
    path = Path(__file__).resolve().parent.parent / "data" / "demo_eval.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="演示评测集尚未生成。")
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        path.read_text(encoding="utf-8"),
        headers={"Content-Disposition": 'attachment; filename="demo_eval.csv"'},
    )


def _get_session_or_404(session_id: str):
    try:
        return session_store.get(session_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail="会话不存在或已过期，请重新上传语料。"
        )


@app.post("/api/sessions/{session_id}/eval-set")
async def upload_eval_set(session_id: str, file: UploadFile = File(...)):
    """Upload a new unjudged eval set (<=200 unique-id abstracts).

    A new upload always starts a fresh batch: previous annotations are
    discarded and must be redone on the new set.
    """
    session = _get_session_or_404(session_id)
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    try:
        records = parse_eval_file(raw, file.filename or "eval.csv")
    except EvalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"文件解析失败：{exc}") from exc
    batch = EvalBatch.from_records(records, session.corpus)
    session_store.set_eval_batch(session_id, batch)
    return {
        "eval_id": batch.eval_id,
        "n_docs": batch.n_docs,
        "max_docs": MAX_EVAL_DOCS,
        "docs": [
            {"id": d, "title": t, "abstract": a}
            for d, t, a in zip(batch.doc_ids, batch.titles, batch.abstracts)
        ],
        "annotations": {},
        "message": (
            f"评测集已载入（{batch.n_docs} 篇，全部未审）；"
            "既往批次的标注已清空，请重新标注。"
        ),
    }


@app.post("/api/sessions/{session_id}/eval-judge")
def judge_eval_doc(session_id: str, body: EvalJudgeBody):
    """Judge / correct / clear one paper; both lists share the annotation."""
    session = _get_session_or_404(session_id)
    try:
        batch = session_store.judge_eval(
            session, body.eval_id, body.doc_id, body.relevant
        )
    except RevisionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EvalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "eval_id": batch.eval_id,
        "annotations": dict(batch.annotations),
        "n_judged": len(batch.annotations),
        "n_unjudged": batch.n_docs - len(batch.annotations),
        "n_relevant_judged": sum(1 for v in batch.annotations.values() if v),
    }


@app.post("/api/sessions/{session_id}/eval-compute")
def compute_eval(session_id: str, body: EvalComputeBody):
    """Top-K lists + exact NDCG@K difference bounds for the fresh suggestion."""
    session = _get_session_or_404(session_id)
    # 校验与计算在同一临界区：迟到/越批请求不得恢复旧结论，
    # 建议失效就撤下结论（本批标注保留待新建议使用）
    with session.lock:
        try:
            batch = session_store.current_eval_batch(session, body.eval_id)
        except RevisionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except EvalError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            session_store.ensure_fresh(session, body.suggestion_id)
        except RevisionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        terms = session.last_analysis_terms
        if not terms:
            raise HTTPException(
                status_code=400, detail="尚未生成建议，请先运行扩展分析再评测。"
            )
        result = evaluate_batch(
            session.corpus, batch,
            terms["baseline_terms"], terms["selected_terms"], body.k,
        )
        revision = session.revision
    result.update({
        "eval_id": batch.eval_id,
        "suggestion_id": body.suggestion_id,
        "revision": revision,
        "query": terms["query"],
    })
    return result


# --------------------------------------------------------------------------- #
# Static frontend
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
