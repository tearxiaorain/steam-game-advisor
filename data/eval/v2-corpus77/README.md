# v2 评测集冻结（corpus77 / @74 时代）

对应知识库：扩库存补档**之前**的约 **74–77** 款游戏语料（过滤非游戏后常标 `@74`）。

> 2026-08-20 已用本人库存补档，当前 `data/processed/` 约 **226** 款。  
> **本目录是冻结快照**，分数与题集对齐「扩库前」；新基线请另开目录（如 v3）或新跑全量。

## 内容

| 文件 | 说明 |
|------|------|
| `cases.jsonl` | 冻结主集（推荐/详情/边界，**不含** library 题） |
| `cases_library.jsonl` | 库存策略题（依赖 `me_owned.json` + 扩库后语料，单独跑） |
| `history/runs.jsonl` | 本时代全部全量 v2 评测汇总 |
| `history/details/*.jsonl` | 每轮明细存档 |
| `last_eval_summary.json` | 冻结基准轮（taxonomy-scrub-v1@74） |

## 跑 v2 回归（冻结题集）

```bash
cd D:\rag\steam-game-advisor
python src/eval_run.py --cases data/eval/v2-corpus77/cases.jsonl --label v2-corpus77-regression
```

库存题（可选）：

```bash
python src/eval_run.py --cases data/eval/v2-corpus77/cases_library.jsonl --label v2-library-regression
```

## 基准成绩（taxonomy-scrub-v1@74）

- 检索命中：23/35 = 65.7%
- 胡编率：1/40 = 2.5%
- 路由：38/40 = 95.0%

对照：扩库前实用基线 `genre-filter+name-boost-v1@74` 为 22/35≈63%；taxonomy 为当时最好全量分。

## 与 v1-seed13

| | v1-seed13 | v2-corpus77（本冻结） |
|--|-----------|----------------------|
| 语料规模 | 13 | ~74–77 |
| 题量 | 24 | 40（主集） |
| Hit 分母 | 20 | 35 |
| 冻结基准 | rewrite-guard-v4 80% | taxonomy-scrub 66% |

归档时复制 runs：**11** 条；明细文件：**11** 个。
