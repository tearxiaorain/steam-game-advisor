# 当前正式基线指针

Opus 本地化 + 以撒/幸存者别名 + 幸存者意图面 + 战神路由（2026-08-21）：

→ [`baseline_opus_alias_route_20260821/`](./baseline_opus_alias_route_20260821/)

主集 **100%** / 回归 **82%** / 好友 **88%**（Hit@Top5）。说明见该目录 `README.md` 与 `summary.json`。

## 历史对照

| 包 | 主集 / 回归 / 好友 |
|----|-------------------|
| [`baseline_opus_alias_route_20260821/`](./baseline_opus_alias_route_20260821/)（**当前**） | **100% / 82% / 88%** |
| [`baseline_jieba_tagliteral_20260821/`](./baseline_jieba_tagliteral_20260821/) | 65% / 77% / 88% |
| [`PRE_UI_BASELINE.md`](./PRE_UI_BASELINE.md) → `baseline_tag_overlap_20260820/` | 60% / 62% / 75% |

中间实验笔记（已并入当前基线栈，仅作过程记录）：

- [`notes_opus_localize_20260821.md`](./notes_opus_localize_20260821.md)
- [`notes_alias_survivor_route_20260821.md`](./notes_alias_survivor_route_20260821.md)

冻结后补丁（未另开基线包）：帕鲁 `GAME_QUERY_HINTS` + 压射击面 → 好友 Hit **8/8 = 100%**（`20260821-142413` / `v3-palworld-hint-friend`）。
