# jieba + 标签字面召回冻结（当前正式基线）

**时间**：2026-08-21  
**标签**：`v3-jieba-tagliteral-baseline`  
**用途**：BM25 改为 jieba 分词，并增加 `user_tags` 字面第三路后的成绩快照。后续改检索时对照本包；**勿覆盖**，新实验用新目录。

相对 Pre-UI（`baseline_tag_overlap_20260820/`：60% / 62% / 75%）全面上升。

## 分数一览

| 题集 | 文件 | Hit@Top3 | 路由 | 幻觉 | run_id |
|------|------|----------|------|------|--------|
| 主集 20 | `cases.jsonl` | **13/20 = 65%** | 95% | 0% | `20260821-100453` |
| 回归 43 | `cases_regression.jsonl` | **30/39 = 77%** | 95% | 0% | `20260821-100752` |
| 好友 8 | `cases_friend_rec.jsonl` | **7/8 = 88%** | 100% | 0% | `20260821-100846` |

## 当时配置

- 语料：~760 款，`user_tags` 已进索引
- `use_query_rewrite = false`
- `use_rewrite_hard_aliases = false`
- `use_user_tag_overlap_boost = true`
- **BM25**：jieba + 场景停用词 + 分数 ≤ 0 丢弃
- **`use_tag_literal_recall = true`**（问句命中标签的游戏单独进 RRF）

## 本目录文件

| 文件 | 说明 |
|------|------|
| `summary.json` | 机器可读汇总 |
| `details_rec20.jsonl` | 主集逐题 |
| `details_regression43.jsonl` | 回归逐题 |
| `details_friend_rec8.jsonl` | 好友逐题 |
| `README.md` | 本说明 |

同源明细：`data/eval/history/details/20260821-100*.jsonl`。

## 复现（不重建索引）

```bash
cd D:\rag\steam-game-advisor
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases.jsonl --label v3-jieba-tagliteral-rec20
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases_regression.jsonl --label v3-jieba-tagliteral-regression
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases_friend_rec.jsonl --label v3-jieba-tagliteral-friend
```
