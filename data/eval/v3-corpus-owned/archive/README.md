# v3 基线归档（@760 首轮，改主集前冻结）

冻结时刻：扩库后、新推荐主集落地前。

| 文件 | 说明 |
|------|------|
| `cases_baseline43.jsonl` | 旧全量 43 题原文（v2 主集 40 + library 3），一字未改 |
| `last_eval_summary_20260820-123646.json` | 首轮基线汇总：Hit 17/38=45%，路由 41/43=95% |
| `details_20260820-123646.jsonl` | 该轮逐题明细 |

再跑回归请用上级目录的 `cases_regression.jsonl`（相对归档原文，仅修正「库外题已入库」边界，避免假失败）。

**较新快照**：

- **当前正式**：`baseline_jieba_tagliteral_20260821/` — jieba BM25 + 标签字面（主集 65% / 回归 77% / 好友 88%），见 `CURRENT_BASELINE.md`
- **Pre-UI 对照**：`baseline_tag_overlap_20260820/` — tag overlap only（主集 60% / 回归 62% / 好友 75%）
