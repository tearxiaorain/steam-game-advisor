"""将 v2-corpus77 评测时代（~74–77 款语料）按 v1-seed13 方式冻结归档。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
SRC_RUNS = EVAL / "history" / "runs.jsonl"
SRC_DETAILS = EVAL / "history" / "details"
DST = EVAL / "v2-corpus77"
DST_HIST = DST / "history"
DST_DETAILS = DST_HIST / "details"


def main() -> None:
    DST_DETAILS.mkdir(parents=True, exist_ok=True)

    runs = []
    for line in SRC_RUNS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases = str(row.get("cases") or "").replace("\\", "/")
        label = str(row.get("label") or "")
        # 全量 v2 题集上的跑分；排除仅 3 题的 library smoke
        if "v2-corpus77" not in cases:
            continue
        if "lib_smoke" in cases or "_lib_smoke" in cases:
            continue
        if label.startswith("library-after-owned-fill") or label.startswith("library-tonight"):
            continue
        runs.append(row)

    runs_path = DST_HIST / "runs.jsonl"
    runs_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in runs) + ("\n" if runs else ""),
        encoding="utf-8",
    )

    copied = 0
    for row in runs:
        archive = row.get("detail_archive") or ""
        if not archive:
            continue
        src = Path(archive)
        if not src.is_file():
            # 兼容相对路径
            alt = SRC_DETAILS / src.name
            src = alt if alt.is_file() else src
        if src.is_file():
            shutil.copy2(src, DST_DETAILS / src.name)
            copied += 1

    # 基准：taxonomy-scrub-v1@74（扩库前全量 v2 最好一轮）
    baseline = next(
        (r for r in reversed(runs) if r.get("label") == "taxonomy-scrub-v1@74"),
        runs[-1] if runs else None,
    )
    if baseline is None:
        raise SystemExit("未找到可冻结的 v2 runs")

    summary = {
        "run_id": baseline["run_id"],
        "ts": baseline["ts"],
        "label": baseline["label"],
        "cases": "data/eval/v2-corpus77/cases.jsonl",
        "n": baseline["n"],
        "route_accuracy": baseline["route_accuracy"],
        "retrieval_hit_rate": baseline["retrieval_hit_rate"],
        "live_refuse_rate": baseline["live_refuse_rate"],
        "invent_rate": baseline["invent_rate"],
        "detail_latest": None,
        "detail_archive": f"data/eval/v2-corpus77/history/details/{baseline['run_id']}.jsonl",
        "notes": (
            "冻结基准：taxonomy-scrub-v1@74（约 74 款有效游戏语料）。"
            "此后库存补档扩至 ~226，新实验请另开评测目录或新 label。"
        ),
    }
    (DST / "last_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # cases：拆出 library 题，主集保持「扩库前 v2」形态
    cases_path = DST / "cases.jsonl"
    all_cases = [
        json.loads(l)
        for l in cases_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    main_cases = [c for c in all_cases if not str(c.get("id", "")).startswith("lib-")]
    lib_cases = [c for c in all_cases if str(c.get("id", "")).startswith("lib-")]
    cases_path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in main_cases) + "\n",
        encoding="utf-8",
    )
    if lib_cases:
        (DST / "cases_library.jsonl").write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in lib_cases) + "\n",
            encoding="utf-8",
        )

    hit = summary["retrieval_hit_rate"]
    invent = summary["invent_rate"]
    route = summary["route_accuracy"]
    readme = f"""# v2 评测集冻结（corpus77 / @74 时代）

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
cd D:\\rag\\steam-game-advisor
python src/eval_run.py --cases data/eval/v2-corpus77/cases.jsonl --label v2-corpus77-regression
```

库存题（可选）：

```bash
python src/eval_run.py --cases data/eval/v2-corpus77/cases_library.jsonl --label v2-library-regression
```

## 基准成绩（taxonomy-scrub-v1@74）

- 检索命中：{hit['ok']}/{hit['total']} = {hit['ratio']:.1%}
- 胡编率：{invent['ok']}/{invent['total']} = {invent['ratio']:.1%}
- 路由：{route['ok']}/{route['total']} = {route['ratio']:.1%}

对照：扩库前实用基线 `genre-filter+name-boost-v1@74` 为 22/35≈63%；taxonomy 为当时最好全量分。

## 与 v1-seed13

| | v1-seed13 | v2-corpus77（本冻结） |
|--|-----------|----------------------|
| 语料规模 | 13 | ~74–77 |
| 题量 | 24 | {len(main_cases)}（主集） |
| Hit 分母 | 20 | 35 |
| 冻结基准 | rewrite-guard-v4 80% | taxonomy-scrub 66% |

归档时复制 runs：**{len(runs)}** 条；明细文件：**{copied}** 个。
"""
    (DST / "README.md").write_text(readme, encoding="utf-8")
    print(f"frozen runs={len(runs)} details={copied} main_cases={len(main_cases)} lib_cases={len(lib_cases)}")
    print(f"baseline={baseline['label']} hit={hit['ok']}/{hit['total']}")


if __name__ == "__main__":
    main()
