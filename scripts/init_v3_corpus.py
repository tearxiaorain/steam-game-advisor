"""维护 v3 评测目录结构（归档 / 回归 / 新主集）。

不会覆盖 archive/ 冻结文件；不会重写 cases_rec_v3.jsonl。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
V3 = EVAL / "v3-corpus-owned"
ARCHIVE = V3 / "archive"


def main() -> None:
    V3.mkdir(parents=True, exist_ok=True)
    (V3 / "history" / "details").mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    processed_n = len(list((ROOT / "data" / "processed").glob("*.md")))
    rec_path = V3 / "cases_rec_v3.jsonl"
    reg_path = V3 / "cases_regression.jsonl"
    if not rec_path.is_file():
        raise SystemExit(f"missing {rec_path}")
    if not reg_path.is_file():
        raise SystemExit(f"missing {reg_path}")

    # 默认主集 = 新推荐题
    shutil.copy2(rec_path, V3 / "cases.jsonl")
    rec_n = sum(1 for l in rec_path.read_text(encoding="utf-8").splitlines() if l.strip())
    reg_n = sum(1 for l in reg_path.read_text(encoding="utf-8").splitlines() if l.strip())

    print(
        json.dumps(
            {
                "processed": processed_n,
                "main_rec": rec_n,
                "regression": reg_n,
                "archive_ok": (ARCHIVE / "cases_baseline43.jsonl").is_file(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
