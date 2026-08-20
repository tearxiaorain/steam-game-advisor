# v3 评测集（@760：本人 + 好友补档语料）

对应知识库：`data/processed/` ≈ **760** 款。

## 题集怎么拆

| 文件 | 题数 | 用途 |
|------|------|------|
| `cases.jsonl`（= `cases_rec_v3.jsonl`） | **20** | **v3 主集**：适配扩库后的新推荐题，迭代盯这个 |
| `cases_regression.jsonl` | 43 | 回归：旧详情/搜索/拒答/旧推荐/library；仅修了「原库外现已入库」2 题 |
| `cases_library.jsonl` | 3 | library 子集（也含在 regression 里） |
| `archive/` | — | **冻结原文**：旧 43 题一字未改 + 首轮基线 `20260820-123646`（Hit 45%） |

详情 / 搜索 / 拒答 / library **都保留在 regression**，没有删；主集只加新推荐题。

## 跑评测

```bash
# v3 主集（默认）
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases.jsonl --label v3-rec20-baseline

# 回归（旧能力）
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases_regression.jsonl --label v3-regression43
```

## 已抓取列表

跳过逻辑以 `data/processed/*.md` 为准；旁路清单：`data/library/fetched_appids.json`（本地，gitignore）。

个人/好友库存快照不入库。

## 好友向偏置（关键词闸门）

主路由仍是 **LLM** 四分类；`library` 子策略与「是否好友向」是 **关键词**。

拥有度偏置仅当：`recommend` + 问句含好友/朋友/开黑/一起玩等。  
好友向小题：`cases_friend_rec.jsonl`（8 题）。

硬映射（`REWRITE_HARD_ALIAS_RULES`）在 LLM 改写后并入强检索词。

```bash
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases.jsonl --label v3-rec20-hard-alias
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases_friend_rec.jsonl --label v3-friend-rec-hard-alias
```

近期：`hard-alias-v1` 主集 **17/20=85%**，好友 **6/8=75%**。


| 指标 | 结果 |
|------|------|
| 路由 | 19/20 = 95% |
| Hit@Top3 | **8/20 = 40%** |
| 幻觉 | 0/20 |

说明：新主集仍难，后续优先做「本人/好友偏置」再盯这 20 题。
