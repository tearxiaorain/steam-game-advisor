# v3 评测集（@760：本人 + 好友补档语料）

对应知识库：`data/processed/` ≈ **760** 款。

> **阅读入口（题目清单）**：[测试集一览.md](./测试集一览.md) — 三套题的完整表格与当前基线分数。

## 题集怎么拆

| 文件 | 题数 | 用途 |
|------|------|------|
| `cases.jsonl`（= `cases_rec_v3.jsonl`） | **20** | **v3 主集**：适配扩库后的新推荐题，迭代盯这个 |
| `cases_regression.jsonl` | 43 | 回归：旧详情/搜索/拒答/旧推荐/library；仅修了「原库外现已入库」2 题 |
| `cases_library.jsonl` | 3 | library 子集（也含在 regression 里） |
| `archive/` | — | **冻结快照**；**当前正式基线** → [`archive/CURRENT_BASELINE.md`](archive/CURRENT_BASELINE.md)；Pre-UI 对照 → [`archive/PRE_UI_BASELINE.md`](archive/PRE_UI_BASELINE.md) |

详情 / 搜索 / 拒答 / library **都保留在 regression**，没有删；主集只加新推荐题。

## 冻结基线（当前：`baseline_jieba_tagliteral_20260821`）

**冻结** = 固定「题集 + 当时配置 + 分数 + 明细」，方便以后对照，**不是**再建索引或改预期答案。  
相对 Pre-UI（`baseline_tag_overlap_20260820`：60% / 62% / 75%）本包为 **65% / 77% / 88%**。

| 题集 | Hit | 路由 | 幻觉 |
|------|-----|------|------|
| 主集 20 | **65%** | 95% | 0% |
| 回归 43 | **77%** | 95% | 0% |
| 好友 8 | **88%** | 100% | 0% |

配置：`user_tags` 入库 + tag overlap 加分 + **jieba BM25** + **标签字面第三路**；LLM 改写 / 硬映射 **关**。

快照目录：`archive/baseline_jieba_tagliteral_20260821/`（含 `summary.json` 与三套 `details_*.jsonl`）。

```bash
# 复现当前冻结基线（不重建索引）
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases.jsonl --label v3-jieba-tagliteral-rec20
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases_regression.jsonl --label v3-jieba-tagliteral-regression
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases_friend_rec.jsonl --label v3-jieba-tagliteral-friend
```

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

硬映射（`REWRITE_HARD_ALIAS_RULES`）与 LLM 改写均可通过 `config.py` 开关对比；**当前默认均关**。

```bash
# 当前基线（user_tags，无改写）
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases.jsonl --label v3-user-tags-no-rewrite --rebuild-index

# 对照：硬映射（易过拟合主集）
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases.jsonl --label v3-rec20-hard-alias
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases_friend_rec.jsonl --label v3-friend-rec-hard-alias
```

## 当前基线（已冻结 → `archive/baseline_tag_overlap_20260820/`）

见上文「冻结基线」表。旧首轮 43 题基线（45%）仍在 `archive/cases_baseline43.jsonl`。
