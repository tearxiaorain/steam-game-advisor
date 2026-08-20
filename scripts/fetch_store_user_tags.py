"""从 Steam 商店页抓取「热门用户自定义标签」。

例:
  python scripts/fetch_store_user_tags.py 413150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest.fetch_steam_games import fetch_store_user_tags  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("appid", type=int, nargs="?", default=413150)
    parser.add_argument("--lang", default="schinese")
    args = parser.parse_args()
    names = fetch_store_user_tags(args.appid, args.lang)
    print(
        json.dumps(
            {"appid": args.appid, "lang": args.lang, "count": len(names), "names": names},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
