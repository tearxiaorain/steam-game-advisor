# v3 基线快照：tag overlap（2026-08-20）

**冻结** = 把某一时刻的**题集 + 配置 + 分数 + 逐题明细**固定下来，以后改代码再跑，可以和这个数字对照，看是变好了还是变差了。不是再抓一遍语料，也不改游戏库。

## 本快照对应配置

- 语料：@760，`user_tags` 已入库
- `use_query_rewrite=false`，`use_rewrite_hard_aliases=false`
- `use_user_tag_overlap_boost=true`（问句与社区标签字面重叠 → RRF 加分）

## 分数（冻结）

| 题集 | 文件 | Hit | 路由 | 幻觉 |
|------|------|-----|------|------|
| 主集 20 | `cases.jsonl` | **12/20=60%** | 95% | 0% |
| 回归 43 | `cases_regression.jsonl` | **24/39=62%** | 95% | 0% |
| 好友 8 | `cases_friend_rec.jsonl` | **6/8=75%** | 100% | 0% |

## 目录内文件

| 文件 | 说明 |
|------|------|
| `summary.json` | 三套汇总 + 配置说明 |
| `details_rec20.jsonl` | 主集逐题（run `20260820-164406`） |
| `details_regression43.jsonl` | 回归逐题（run `20260820-164714`） |
| `details_friend_rec8.jsonl` | 好友逐题（run `20260820-164811`） |

复现命令见上级 `../README.md` 的「冻结基线」一节。
