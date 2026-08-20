"""创建 v3 评测目录：当前扩库语料（库存+好友补档后）基线。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
V2 = EVAL / "v2-corpus77"
V3 = EVAL / "v3-corpus-owned"


def main() -> None:
    V3.mkdir(parents=True, exist_ok=True)
    (V3 / "history" / "details").mkdir(parents=True, exist_ok=True)

    processed_n = len(list((ROOT / "data" / "processed").glob("*.md")))
    main_cases = [
        json.loads(l)
        for l in (V2 / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    lib_path = V2 / "cases_library.jsonl"
    lib_cases = []
    if lib_path.is_file():
        lib_cases = [
            json.loads(l)
            for l in lib_path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]

    # v3 全量题 = v2 主集 + library
    all_cases = main_cases + lib_cases
    (V3 / "cases.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in all_cases) + "\n",
        encoding="utf-8",
    )
    if lib_cases:
        shutil.copy2(lib_path, V3 / "cases_library.jsonl")

    (V3 / "history" / "runs.jsonl").write_text("", encoding="utf-8")

    readme = f"""# v3 评测集（扩库后：本人库存 + 好友补档）

对应知识库：`data/processed/` 当前约 **{processed_n}** 款（随补抓增长）。

相对冻结的 `v2-corpus77/`（@74 时代）：

- 语料扩大（本人库存补档 + 好友库存缺档补抓）
- 题集 = v2 主集 40 题 + library 3 题
- library 题依赖 `data/library/me_owned.json`

## 内容

| 文件 | 说明 |
|------|------|
| `cases.jsonl` | 全量 43 题 |
| `cases_library.jsonl` | 仅 library 3 题 |
| `history/runs.jsonl` | 本目录时代的跑分（初始为空，评测追加到全局 `data/eval/history/` 亦可） |
| `last_eval_summary.json` | 首轮全量基线（跑完后写入） |

## 跑评测

```bash
python src/eval_run.py --cases data/eval/v3-corpus-owned/cases.jsonl --label v3-corpus-owned-baseline --rebuild-index
```

## 已抓取列表

跳过逻辑以 `data/processed/*.md` 为准；旁路清单：`data/library/fetched_appids.json`。
"""
    (V3 / "README.md").write_text(readme, encoding="utf-8")
    print(f"v3 ready cases={len(all_cases)} processed≈{processed_n}")


if __name__ == "__main__":
    main()
