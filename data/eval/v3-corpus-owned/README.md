# v3 评测集（扩库后：本人库存 + 好友补档）

对应知识库：`data/processed/` 当前约 **760** 款（本人 + 好友缺档补抓完成）。

相对冻结的 `v2-corpus77/`（@74 时代）：

- 语料扩大（本人库存补档 + 好友库存缺档补抓）
- 题集 = v2 主集 40 题 + library 3 题
- library 题依赖 `data/library/me_owned.json`

## 内容

| 文件 | 说明 |
|------|------|
| `cases.jsonl` | 全量 43 题 |
| `cases_library.jsonl` | 仅 library 3 题 |
| `history/runs.jsonl` | 本目录时代的跑分（初始为空，评测追加到全局 `data/eval/history/` 亦可） |
| `last_eval_summary.json` | 首轮全量基线（跑完后写入） |

## 首轮基线（@760）

| 指标 | 结果 |
|------|------|
| run_id | `20260820-123646` |
| 路由准确率 | 41/43 = **95%** |
| 检索命中率（Hit@Top3） | 17/38 = **45%** |
| 实时拒答 | 2/2 = 100% |
| 幻觉率 | 2/43 = 5% |

对照：v2 冻结基线（@74）Hit **23/35≈66%**。语料从 ~74 扩到 760 后稀释明显，**不可与 v2 直接比绝对分**。

## 跑评测

```bash
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases.jsonl --label v3-corpus-owned-baseline --rebuild-index
```

## 已抓取列表

跳过逻辑以 `data/processed/*.md` 为准；旁路清单：`data/library/fetched_appids.json`（当前 count=760）。
