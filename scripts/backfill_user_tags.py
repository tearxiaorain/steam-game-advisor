"""给已有 processed/*.md 补抓 Steam 热门用户标签（不重拉详情）。"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest.fetch_steam_games import (  # noqa: E402
    PROCESSED_DIR,
    fetch_store_user_tags,
    yaml_list,
)


def upsert_user_tags_in_md(text: str, user_tags: list[str]) -> str:
    tags_yaml = yaml_list(user_tags)
    tags_line = "用户标签: " + (", ".join(user_tags) if user_tags else "（无）")

    if re.search(r"^user_tags:\s*", text, flags=re.M):
        text = re.sub(
            r"^user_tags:\s*.*$",
            f"user_tags: {tags_yaml}",
            text,
            count=1,
            flags=re.M,
        )
    else:
        # 插在 tags: 或 genres: 后
        m = re.search(r"^(tags:\s*.*)$", text, flags=re.M)
        if not m:
            m = re.search(r"^(genres:\s*.*)$", text, flags=re.M)
        if m:
            insert_at = m.end()
            text = text[:insert_at] + f"\nuser_tags: {tags_yaml}" + text[insert_at:]

    if re.search(r"^用户标签:\s*", text, flags=re.M):
        text = re.sub(r"^用户标签:\s*.*$", tags_line, text, count=1, flags=re.M)
    else:
        m2 = re.search(r"^(分类:\s*.*)$", text, flags=re.M)
        if m2:
            insert_at = m2.end()
            text = text[:insert_at] + "\n" + tags_line + text[insert_at:]
        else:
            m3 = re.search(r"^(## 类型与标签\s*)$", text, flags=re.M)
            if m3:
                insert_at = m3.end()
                text = text[:insert_at] + "\n" + tags_line + text[insert_at:]
    return text


def has_user_tags(text: str) -> bool:
    m = re.search(r"^user_tags:\s*(\[.*\])\s*$", text, flags=re.M)
    if not m:
        return False
    raw = m.group(1).strip()
    return raw not in {"[]", "[ ]"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=float, default=1.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--refresh", action="store_true", help="已有 user_tags 也重抓")
    parser.add_argument("--appid", type=int, default=0, help="只处理单个 appid")
    args = parser.parse_args()

    files = sorted(
        PROCESSED_DIR.glob("*.md"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )
    if args.appid:
        files = [PROCESSED_DIR / f"{args.appid}.md"]
        files = [p for p in files if p.exists()]

    ok = skip = fail = 0
    for i, path in enumerate(files, 1):
        if args.limit and ok >= args.limit:
            break
        if not path.stem.isdigit():
            continue
        text = path.read_text(encoding="utf-8")
        if has_user_tags(text) and not args.refresh:
            skip += 1
            continue
        appid = int(path.stem)
        print(f"[{i}/{len(files)}] user_tags {appid} ...")
        try:
            tags = fetch_store_user_tags(appid)
        except Exception as exc:
            print(f"  fail: {exc}")
            fail += 1
            time.sleep(args.sleep)
            continue
        new_text = upsert_user_tags_in_md(text, tags)
        path.write_text(new_text, encoding="utf-8")
        # raw sidecar optional
        raw_path = ROOT / "data" / "raw" / f"{appid}.json"
        if raw_path.exists():
            try:
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                raw["user_tags"] = tags
                raw_path.write_text(
                    json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception:
                pass
        print(f"  ok {len(tags)} tags: {', '.join(tags[:8])}{'...' if len(tags)>8 else ''}")
        ok += 1
        time.sleep(args.sleep)

    print(json.dumps({"updated": ok, "skipped": skip, "failed": fail}, ensure_ascii=False))


if __name__ == "__main__":
    main()
