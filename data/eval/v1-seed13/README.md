# v1 评测集（seed13 语料）

对应知识库：`data/processed/` 中 **13 款** seed 游戏（与 `src/ingest/fetch_steam_games.py` 的 `EVAL_SEED_APPIDS` 一致）。

## 内容

| 文件 | 说明 |
|------|------|
| `cases.jsonl` | 24 题固定评测集（2026-08-19 归档） |
| `history/runs.jsonl` | 截至归档时的全部评测汇总 |
| `history/details/*.jsonl` | 每轮明细存档 |
| `last_eval_summary.json` | 归档前最后一轮（rewrite-guard-v4） |

## 跑 v1 回归

```bash
cd D:\rag\steam-game-advisor
python src/eval_run.py --cases data/eval/v1-seed13/cases.jsonl --label seed13-regression
```

**前提**：`data/processed/` 仍包含 cases 里期望的 13 款，且 `vector_index/` 与当前 processed 一致（见项目根 README / 下文「扩库后重建索引」）。

## 基准成绩（rewrite-guard-v4 @ seed13）

- 检索命中：16/20 = 80%
- 胡编率：1/24 = 4%
- 路由：23/24 = 96%

扩库后若仍用本集，命中率可能因干扰项增多而下降，属正常现象。
