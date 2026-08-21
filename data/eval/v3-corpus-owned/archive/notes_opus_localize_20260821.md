# Opus 本地化后评测（2026-08-21）

## 做了什么

1. 对 **121** 款 `description_lang=en` 的 raw 跑 `prepare_processed --only-en --backend local_opus`
2. 强制重建 FAISS / BM25 索引
3. 三套题评测（金标已含此前扩期望：DS1/2/3、免费竞技射击多解；彩六题已改为「干员」）

## 分数对照

| 题集 | 冻结基线 (jieba+tagliteral) | 本轮 top_k=5+多意图（译前） | **本轮 Opus 本地化后** |
|------|------------------------------|----------------------------|------------------------|
| 主集 Hit | 65% | 80% | **90%** (18/20) |
| 回归 Hit | 77% | 79% | **82%** (32/39) |
| 好友 Hit | 88% | 88% | **88%** (7/8) |
| 路由 | 95/95/100 | 95/95/100 | **95/95/100** |
| 胡编 | 0 | 0 | **0** |

label：`v3-opus-localize-rec20` / `v3-opus-localize-regression` / `v3-opus-localize-friend`  
run_id：`20260821-130445` / `20260821-130707` / `20260821-130755`

## 主集仍 miss

| id | 说明 |
|----|------|
| `v3-rec-09` 以撒 | 译后仍未进 Top5（Tiny Rogues / Monster Train / Noita 等） |
| `v3-rec-18` 幸存者 like | 仍被 Crosshair / FPS 噪声挤掉 |
| `v3-rec-15` 战神 | **检索命中**，路由仍判 detail（非检索问题） |

扩期望后可过：黑暗魂系列、（回归）免费竞技射击。

## 说明

- 未改正式冻结指针 `CURRENT_BASELINE.md`（仍为 65/77/88）；本文件为实验记录。
- Opus 机翻质量一般，但对部分中文问句有帮助（主集 +10pp vs 译前）。
- 译文缓存目录：`data/cache/translations/`（gitignore）。
