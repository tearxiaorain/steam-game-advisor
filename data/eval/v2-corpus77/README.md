# v2 评测集（corpus77）

对应知识库：`data/processed/` 约 **77 款**（2026-08-19 扩库）。

## 与 v1-seed13 的关系

| 文件 | 说明 |
|------|------|
| `cases.jsonl` | **完整 v2**（seed 修正版 24 题 + 新增 14 题 = 38 题） |
| `../v1-seed13/cases.jsonl` | 冻结基线，不改期望，用于「种子在大库里是否仍召回」 |

### 相对 seed13 的期望修正（仅 v2）

- **rec-c01**：`730` → `730,440`（CS2 或 TF2 均可）
- **trd-04**：库内已有 MHW(582010)；`forbidden_names` 仅禁「荒野」，可推荐世界

## 跑评测

```bash
python src/eval_run.py --cases data/eval/v2-corpus77/cases.jsonl --label v2-corpus77
```

## 题目标签

- `seed13`：沿用 v1 题面（期望已按上表微调）
- `v2-new`：针对 77 款库新增
- `boundary` / `multi-answer`：见各题 `tags`

新增题的 `expect_app_ids` 经 manifest + 档案人工核对；多解题以 **集合交集命中** 为准。
