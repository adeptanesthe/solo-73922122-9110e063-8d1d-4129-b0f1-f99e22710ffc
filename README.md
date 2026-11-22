# 科研检索查询扩展 Web 应用

上传带相关性标签的论文摘要集合，输入初始查询、**召回下限**与**误检率上限**；
系统仅从**正例文档**中提取候选扩展词，在约束下用束搜索（beam search）选择扩展项，
对检索结果做 TF-IDF 余弦重排，并逐词解释每个扩展词带来的覆盖变化。
找不到满足阈值的扩展时返回结构化失败原因；用户可**锁定 / 排除**任意词，
系统重算结果并通过修订号使旧建议立即失效。

扩展结果页内置**新批次评测**入口：上传至多 200 篇未审摘要（CSV、编号唯一），
选择 K（1～20），系统沿用原语料拟合的 TF-IDF 与原有命中/排序规则，
用当前建议的原查询与扩展后查询在评测集上各取前 K 篇；研究员逐篇判「相关 / 不相关」
（可更正，两榜共用同一份标注），系统把未判论文视为可能相关或不相关，
**精确**计算「扩展榜 − 原榜」NDCG@K 差值的上下界（不枚举标签组合），
并列出达到各端点时应判相关的未判论文；下界 > 0 才判定「确定改善」。

技术栈：FastAPI + scikit-learn（TF-IDF / 稀疏矩阵 / 余弦相似度）+ 原生 HTML/CSS/JS，
无前端构建步骤。

## 目录结构

```
app/
  engine.py     纯算法层：语料、候选词、束搜索、逐词 trace、失败诊断
  evaluate.py   评测算法层：评测集解析、top-K 复算、NDCG@K 差值精确界
  session.py    内存会话与修订号（revision）/ 旧建议失效 / 评测批次
  main.py       FastAPI 路由与静态资源
static/         原生前端（index.html / styles.css / app.js）
tests/          pytest 单元与 API 测试（60 项）
scripts/
  generate_demo_data.py  生成内置演示语料与演示评测集
  selftest.py            交付前自测（12 个阈值冲突/修订/评测场景，产出报告）
  smoke.sh               运行中服务的 HTTP 端到端冒烟脚本
data/demo_corpus.csv     45 篇演示摘要（17 正例 / 28 负例）
data/demo_eval.csv       40 篇未审演示评测集（无标签列）
reports/                 自测输出
```

## 快速开始（Docker Compose）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

页面点击「使用内置演示语料」即可一键载入，再点「运行扩展分析」。

## 本地运行（无 Docker）

```bash
python3 -m pip install -r requirements.txt
python3 scripts/generate_demo_data.py          # 生成演示语料
python3 -m uvicorn app.main:app --port 8000
```

## 输入格式

CSV / TSV（表头 `id,title,abstract,relevant`）或 JSON / JSONL 对象数组。
`relevant` 支持 `true/false`、`1/0`、`yes/no`、`relevant/irrelevant`。

## 检索与约束模型

- 文档命中：查询词与扩展词的布尔 **OR** 并集；TF-IDF 与布尔矩阵共用同一词表。
- 召回 recall = 命中正例 / 全部正例；误检率 fdr = 命中负例 / 全部命中文档；
  可选 `max_false_positives` 限制误检绝对数量。
- 候选词：仅取正例中出现、词频 `min_df=2`、非基线词、未被排除的词，
  按「正例覆盖率 × 纯度 × IDF」打分。
- 束搜索逐层加入候选词，剪去突破 fdr / fp 上限的分支，按召回→fdr→词数排序保留 beam。
- 锁定词强制前置纳入；若锁定词本身越界，直接进入失败诊断。

## 失败诊断（找不到满足阈值的扩展时）

返回：无约束词表覆盖上限、误检约束内可达最高召回、无法覆盖的正例文档、
首个越界词及其新增正/负例数与导致的 fdr、人类可读原因列表；
同时仍返回尽力而为（best-effort）的重排结果。

## 修订与旧建议失效

- 每次有效锁定/排除/解锁/取消排除都会使会话 `revision +1`。
- 分析响应绑定 `suggestion_id` 与当时的修订号；修订后再携带旧 ID 调用
  `/api/analyze` 或 `/api/revise` 会得到 **HTTP 409**，提示重新分析。
- 同一词同时锁定与排除返回 **HTTP 400**。

## 新批次评测（NDCG@K 差值精确界）

扩展结果页（卡片⑤）提供评测入口：

1. **上传评测集**：CSV / TSV（表头 `id,title,abstract`），至多 200 篇、
   编号唯一、摘要非空；若含 `relevant` 列会被忽略（全部按未审开始）。
   每次上传开启新批次：既往标注清空，需重新标注。
2. **选择 K（1～20）**：沿用原语料拟合的 TF-IDF（只 transform 不重新 fit）
   与原有命中/排序规则（布尔 OR 命中、TF-IDF 余弦降序、文档序号决胜），
   用当前建议的原查询词与扩展后查询词在评测集上各取前 K 篇。
3. **逐篇判定**：每篇可判相关 / 不相关 / 撤销，随时更正；两榜共用同一份标注。
4. **差值上下界**：未判论文视为可能相关或不相关。对任一标签补全 x，

   `Δ(x) = (c_B − c_A + Σ δ_d·x_d) / f(R_known + Σx_d)`

   其中 `c_A/c_B` 为已判相关论文的 DCG 贡献，`δ_d = w_B(d) − w_A(d)`
   （`w = 1/log2(rank+1)`，未入榜为 0），`f(r)` 为按**评测全集**计算的
   IDCG@K；全集不相关时两榜 NDCG 均记 0（该补全 Δ=0）。
   分母只依赖相关总数 S，故固定 S 时分子由 δ 排序后的前缀和取得最值，
   扫描 S = 0..U 即得**精确**上下界，复杂度 O(U log U)，**不枚举** 2^U 种组合。
5. **端点证人**：页面列出达到上界 / 下界时应判为相关的未审论文
   （其余未判论文在该端点判不相关；同一端点在多个 S 处达到时取所需标注最少者）。
6. **结论判定**：仅当**下界 > 0** 时显示「确定改善」（上界 < 0 时显示「确定变差」）；
   每次改标后自动重算。
7. **失效与迟到保护**：锁定/排除词修订使建议失效 → 评测结论立即撤下，
   但本批标注保留，新建议生成后自动用同一标注重算；携带旧建议或旧批次
   `eval_id` 的请求返回 **HTTP 409**，前端另以单调递增的 epoch 丢弃迟到响应，
   旧结论不会被恢复。

## API 摘要

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/sessions` | 上传语料，创建会话（multipart file） |
| GET  | `/api/demo-corpus` | 下载内置演示语料 |
| POST | `/api/analyze` | 运行扩展分析（可带 suggestion_id 校验时效） |
| POST | `/api/revise` | 锁定/排除/解锁/取消排除，修订号 +1 |
| GET  | `/api/sessions/{id}` | 查看会话修订状态 |
| GET  | `/api/demo-eval` | 下载内置演示评测集（40 篇未审摘要） |
| POST | `/api/sessions/{id}/eval-set` | 上传评测集（≤200 篇、编号唯一；开启新批次并清空标注） |
| POST | `/api/sessions/{id}/eval-judge` | 判定/更正/撤销单篇（relevant: true/false/null） |
| POST | `/api/sessions/{id}/eval-compute` | 计算两榜 top-K 与 NDCG@K 差值精确界（k: 1～20） |
| GET  | `/api/health` | 健康检查 |

## 测试与自测

```bash
python3 -m pytest tests/ -q          # 60 项单元 + API 测试
python3 scripts/selftest.py          # 12 个场景并写 reports/selftest_report.md
# 服务启动后：
bash scripts/smoke.sh http://127.0.0.1:8000
```

`selftest.py` 覆盖：常规满足、词表覆盖上限冲突、误检预算冲突、放宽上限消解、
排除词重算 + 409、锁定词强制入选、锁定/排除矛盾与解锁、评测集上传与差值界计算、
差值界与 2^U 穷举对照、标注收窄与全判定收敛、修订使评测结论失效但标注保留、
更换评测集后迟到请求被拒。
