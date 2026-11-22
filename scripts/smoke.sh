#!/usr/bin/env bash
# End-to-end HTTP smoke test against a running server.
# Usage: scripts/smoke.sh [base_url]   (default http://127.0.0.1:8000)
set -euo pipefail

BASE="${1:-http://127.0.0.1:8000}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== health =="
curl -fsS "$BASE/api/health"; echo

echo "== demo corpus endpoint =="
curl -fsS "$BASE/api/demo-corpus" -o "$TMP/demo.csv"
wc -l "$TMP/demo.csv"

echo "== create session =="
SID=$(curl -fsS -F "file=@$TMP/demo.csv;type=text/csv" "$BASE/api/sessions" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
echo "session=$SID"

echo "== analyze (satisfied expected) =="
curl -fsS -X POST "$BASE/api/analyze" -H "Content-Type: application/json" -d "{
  \"session_id\": \"$SID\", \"query\": \"quantum error correction\",
  \"min_recall\": 0.85, \"max_fdr\": 0.2
}" -o "$TMP/a1.json"
python3 - "$TMP/a1.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["status"] == "satisfied", d["status"]
print("status:", d["status"], "| expanded:", d["expanded_terms"],
      "| recall:", round(d["metrics"]["recall"], 3),
      "| fdr:", round(d["metrics"]["fdr"], 3))
print("suggestion_id:", d["suggestion_id"], "| rev:", d["revision"])
PY
SUG=$(python3 -c 'import json; print(json.load(open("'$TMP'/a1.json"))["suggestion_id"])')
TERM=$(python3 -c 'import json; print(json.load(open("'$TMP'/a1.json"))["expanded_terms"][0])')

echo "== revise: exclude $TERM (revision bumps) =="
curl -fsS -X POST "$BASE/api/revise" -H "Content-Type: application/json" -d "{
  \"session_id\": \"$SID\", \"exclude\": [\"$TERM\"]
}" -o "$TMP/rev.json"
python3 - "$TMP/rev.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["revision"] == 1 and d["previous_suggestion_invalidated"]
print("rev:", d["revision"], "| excluded:", d["excluded_terms"])
PY

echo "== stale suggestion must return 409 =="
CODE=$(curl -s -o "$TMP/stale.json" -w "%{http_code}" -X POST "$BASE/api/analyze" \
  -H "Content-Type: application/json" -d "{
  \"session_id\": \"$SID\", \"query\": \"quantum error correction\",
  \"min_recall\": 0.85, \"max_fdr\": 0.2, \"suggestion_id\": \"$SUG\"}")
echo "HTTP $CODE"; test "$CODE" = "409"
python3 -c 'import json; print("detail:", json.load(open("'$TMP'/stale.json"))["detail"])'

echo "== fresh analyze after revision =="
curl -fsS -X POST "$BASE/api/analyze" -H "Content-Type: application/json" -d "{
  \"session_id\": \"$SID\", \"query\": \"quantum error correction\",
  \"min_recall\": 0.85, \"max_fdr\": 0.2
}" -o "$TMP/a2.json"
python3 - "$TMP/a2.json" "$TERM" <<'PY'
import json, sys
d, term = json.load(open(sys.argv[1])), sys.argv[2]
assert term not in d["expanded_terms"], d["expanded_terms"]
print("recomputed expanded:", d["expanded_terms"],
      "| new suggestion:", d["suggestion_id"][:8], "| rev:", d["revision"])
PY

echo "== conflict threshold (failed + diagnosis) =="
curl -fsS -X POST "$BASE/api/analyze" -H "Content-Type: application/json" -d "{
  \"session_id\": \"$SID\", \"query\": \"quantum error correction\",
  \"min_recall\": 1.0, \"max_fdr\": 0.05
}" -o "$TMP/a3.json"
python3 - "$TMP/a3.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["status"] == "failed" and d["failure"]["reasons"]
print("failed as expected; ceiling recall:",
      d["failure"]["unconstrained_max_recall"],
      "| reasons:", len(d["failure"]["reasons"]))
PY

echo "== evaluation: upload eval set =="
curl -fsS "$BASE/api/demo-eval" -o "$TMP/eval.csv"
wc -l "$TMP/eval.csv"
curl -fsS -F "file=@$TMP/eval.csv;type=text/csv" "$BASE/api/sessions/$SID/eval-set" \
  -o "$TMP/ev.json"
python3 - "$TMP/ev.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["n_docs"] == 40 and d["annotations"] == {}
print("eval_id:", d["eval_id"][:8], "| docs:", d["n_docs"])
PY
EV=$(python3 -c 'import json; print(json.load(open("'$TMP'/ev.json"))["eval_id"])')

echo "== evaluation: fresh suggestion for compute =="
curl -fsS -X POST "$BASE/api/analyze" -H "Content-Type: application/json" -d "{
  \"session_id\": \"$SID\", \"query\": \"quantum error correction\",
  \"min_recall\": 0.85, \"max_fdr\": 0.2
}" -o "$TMP/a4.json"
SUG2=$(python3 -c 'import json; print(json.load(open("'$TMP'/a4.json"))["suggestion_id"])')

echo "== evaluation: compute NDCG@10 difference bounds =="
curl -fsS -X POST "$BASE/api/sessions/$SID/eval-compute" \
  -H "Content-Type: application/json" -d "{
  \"eval_id\": \"$EV\", \"suggestion_id\": \"$SUG2\", \"k\": 10
}" -o "$TMP/ec.json"
python3 - "$TMP/ec.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
b = d["bounds"]
assert d["n_unjudged"] == 40
assert b["lower"] <= d["tentative"]["delta"] <= b["upper"]
assert b["lower"] <= 0.0 <= b["upper"]   # 全未判时全不相关补全给出 0
print("bounds: [%.4f, %.4f]" % (b["lower"], b["upper"]),
      "| tentative:", d["tentative"]["delta"],
      "| lists:", len(d["lists"]["original"]), "/", len(d["lists"]["expanded"]))
PY

echo "== evaluation: judge all docs (QEC titles relevant) then bounds collapse =="
python3 - "$BASE" "$SID" "$EV" "$TMP/ev.json" <<'PY'
import json, sys, urllib.request
base, sid, ev = sys.argv[1], sys.argv[2], sys.argv[3]
docs = json.load(open(sys.argv[4]))["docs"]
for doc in docs:
    body = json.dumps({
        "eval_id": ev, "doc_id": doc["id"],
        "relevant": "(QEC)" in doc["title"],
    }).encode()
    req = urllib.request.Request(
        f"{base}/api/sessions/{sid}/eval-judge", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
print("judged", len(docs), "docs")
PY
curl -fsS -X POST "$BASE/api/sessions/$SID/eval-compute" \
  -H "Content-Type: application/json" -d "{
  \"eval_id\": \"$EV\", \"suggestion_id\": \"$SUG2\", \"k\": 10
}" -o "$TMP/ec2.json"
python3 - "$TMP/ec2.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
b = d["bounds"]
assert d["n_unjudged"] == 0
assert abs(b["lower"] - b["upper"]) < 1e-9
assert b["certain_improvement"] and b["lower"] > 0
print("collapsed delta: %.4f" % b["lower"], "| certain improvement:",
      b["certain_improvement"])
PY

echo "== evaluation: stale suggestion compute -> 409 =="
curl -fsS -X POST "$BASE/api/revise" -H "Content-Type: application/json" -d "{
  \"session_id\": \"$SID\", \"exclude\": [\"syndrome\"]
}" -o /dev/null
CODE=$(curl -s -o "$TMP/ec3.json" -w "%{http_code}" -X POST \
  "$BASE/api/sessions/$SID/eval-compute" -H "Content-Type: application/json" -d "{
  \"eval_id\": \"$EV\", \"suggestion_id\": \"$SUG2\", \"k\": 10
}")
echo "HTTP $CODE"; test "$CODE" = "409"
python3 -c 'import json; print("detail:", json.load(open("'$TMP'/ec3.json"))["detail"])'

echo "== evaluation: annotations kept for the new suggestion =="
curl -fsS -X POST "$BASE/api/analyze" -H "Content-Type: application/json" -d "{
  \"session_id\": \"$SID\", \"query\": \"quantum error correction\",
  \"min_recall\": 0.85, \"max_fdr\": 0.2
}" -o "$TMP/a5.json"
SUG3=$(python3 -c 'import json; print(json.load(open("'$TMP'/a5.json"))["suggestion_id"])')
curl -fsS -X POST "$BASE/api/sessions/$SID/eval-compute" \
  -H "Content-Type: application/json" -d "{
  \"eval_id\": \"$EV\", \"suggestion_id\": \"$SUG3\", \"k\": 10
}" -o "$TMP/ec4.json"
python3 - "$TMP/ec4.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["n_judged"] == 40   # 标注在修订后保留
print("annotations kept:", d["n_judged"], "| new rev:", d["revision"])
PY

echo "== evaluation: replaced eval set rejects late old-batch calls =="
printf 'id,title,abstract\nx1,t,quantum error correction\nx2,t,surface code\n' \
  > "$TMP/eval2.csv"
curl -fsS -F "file=@$TMP/eval2.csv;type=text/csv" "$BASE/api/sessions/$SID/eval-set" \
  -o "$TMP/ev2.json"
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  "$BASE/api/sessions/$SID/eval-compute" -H "Content-Type: application/json" -d "{
  \"eval_id\": \"$EV\", \"suggestion_id\": \"$SUG3\", \"k\": 10
}")
echo "late compute with old eval_id -> HTTP $CODE"; test "$CODE" = "409"

echo "ALL SMOKE CHECKS PASSED"
