# UI 调试前正式冻结（Pre-UI Baseline）

**时间**：2026-08-20  
**标签**：`v3-tag-overlap-baseline`  
**用途**：Streamlit UI（`app_ui.py`）上线/调试前的检索+路由成绩快照。后续改 UI 或改检索时，用本包对照，避免「页面改了却说不清分数变没变」。

## 分数一览

| 题集 | 文件 | Hit@Top3 | 路由 | 幻觉 | run_id |
|------|------|----------|------|------|--------|
| 主集 20 | `cases.jsonl` | **12/20 = 60%** | 95% | 0% | `20260820-164406` |
| 回归 43 | `cases_regression.jsonl` | **24/39 = 62%** | 95% | 0% | `20260820-164714` |
| 好友 8 | `cases_friend_rec.jsonl` | **6/8 = 75%** | 100% | 0% | `20260820-164811` |

## 当时配置（不可混用别的开关对照）

- 语料：~760 款，`user_tags` 已进索引
- `use_query_rewrite = false`
- `use_rewrite_hard_aliases = false`
- `use_user_tag_overlap_boost = true`（字面重叠 RRF 加分）

## 本目录文件

| 文件 | 说明 |
|------|------|
| `summary.json` | 机器可读汇总 |
| `details_rec20.jsonl` | 主集逐题 |
| `details_regression43.jsonl` | 回归逐题 |
| `details_friend_rec8.jsonl` | 好友逐题 |
| `README.md` | 本说明 |

同源明细也在：`data/eval/history/details/20260820-164*.jsonl`。

## 和 v1 / v2 冻结的关系

| 目录 | 时代 | 作用 |
|------|------|------|
| `data/eval/v1-seed13/` | seed13 | 小语料回归 |
| `data/eval/v2-corpus77/` | @74 | 扩库前冻结 |
| **本包** | @760 + tag overlap | **UI 前 v3 正式基线** |

## 复现（不重建索引）

```bash
cd D:\rag\steam-game-advisor
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases.jsonl --label v3-rec20-tag-overlap
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases_regression.jsonl --label v3-regression-tag-overlap
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases_friend_rec.jsonl --label v3-friend-rec-tag-overlap
```

UI 调试后若改了检索逻辑，请**新 label** 再跑，不要覆盖本目录文件。
