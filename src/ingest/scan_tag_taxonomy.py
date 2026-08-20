"""扫描语料中未登记的 genres/categories，扩库后跑一次便于补词表。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG, PROJECT_ROOT
from rag_modules.data_preparation import DataPreparationModule
from rag_modules.tag_taxonomy import get_taxonomy


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描未登记 Steam 标签")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=PROJECT_ROOT / "data" / "library" / "tag_taxonomy.json",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(DEFAULT_CONFIG.data_path),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "library" / "tag_unknowns.json",
        help="未登记标签输出路径",
    )
    args = parser.parse_args()

    tax = get_taxonomy(args.taxonomy, reload=True)
    data = DataPreparationModule(str(args.data))
    data.load_documents()
    report = tax.scan_documents(data.documents)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"词表: {args.taxonomy}")
    print(f"文档: {len(data.documents)}")
    print(f"词表规模: {report['counts']}")
    print(f"unknown genres ({len(report['unknown_genres'])}): {report['unknown_genres']}")
    print(
        f"unknown categories ({len(report['unknown_categories'])}): "
        f"{report['unknown_categories']}"
    )
    print(f"已写入: {args.out}")


if __name__ == "__main__":
    main()
