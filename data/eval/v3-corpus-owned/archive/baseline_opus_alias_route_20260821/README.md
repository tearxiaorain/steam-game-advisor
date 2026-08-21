# Opus 本地化 + 别名/幸存者面/路由冻结（当前正式基线）

**时间**：2026-08-21  
**标签**：`v3-opus-alias-route-baseline`  
**用途**：英文档 Opus 译入 processed、重建索引后，再加以撒/幸存者like 检索补强、幸存者意图面压 FPS、「想玩+描述」路由修正后的成绩快照。后续改检索/路由时对照本包；**勿覆盖**，新实验用新目录。

相对上一正式基线 `baseline_jieba_tagliteral_20260821/`（65% / 77% / 88%）主集大幅上升。

## 分数一览

| 题集 | 文件 | Hit@Top5 | 路由 | 幻觉 | run_id |
|------|------|----------|------|------|--------|
| 主集 20 | `cases.jsonl` | **20/20 = 100%** | 100% | 0% | `20260821-140500` |
| 回归 43 | `cases_regression.jsonl` | **32/39 = 82%** | 95% | 0% | `20260821-140717` |
| 好友 8 | `cases_friend_rec.jsonl` | **7/8 = 88%** | 100% | 0% | `20260821-140801` |

## 当时配置

- 语料：~760 款；121 款英文档经 `local_opus` 写入 processed 后重建索引
- `top_k = 5`
- `use_query_rewrite = false`
- `use_rewrite_hard_aliases = false`
- `use_tag_literal_recall = true`（jieba BM25 + 标签字面第三路）
- 多意图覆盖 RRF / 标签字面加权：开
- `GAME_QUERY_HINTS`：以撒、幸存者like（及小丑牌）
- 幸存者意图面：有幸存者线索时去掉泛「射击」面
- 路由 prompt：描述「想玩哪款」走 recommend

## 本目录文件

| 文件 | 说明 |
|------|------|
| `summary.json` | 机器可读汇总 |
| `details_rec20.jsonl` | 主集逐题 |
| `details_regression43.jsonl` | 回归逐题 |
| `details_friend_rec8.jsonl` | 好友逐题 |
| `README.md` | 本说明 |

同源明细：`data/eval/history/details/20260821-140*.jsonl`。

## 复现（不重建索引；需已含 Opus processed）

```bash
cd D:\rag\steam-game-advisor
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases.jsonl --label v3-opus-alias-route-rec20
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases_regression.jsonl --label v3-opus-alias-route-regression
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases_friend_rec.jsonl --label v3-opus-alias-route-friend
```
