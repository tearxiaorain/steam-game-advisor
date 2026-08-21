"""从 data/raw 生成 data/processed（进索引语料）。

默认翻译后端为 passthrough（不调用云 API）。
英文档会原样写入 processed，并在 frontmatter 标记 localization_source。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import PROJECT_ROOT
from ingest.fetch_steam_games import (
    PROCESSED_DIR,
    RAW_DIR,
    collect_manifest_from_processed,
    to_markdown,
    write_fetched_appids,
    write_games_manifest,
)
from ingest.translate import Translator, get_translator

TEXT_FIELDS = ("short_description", "detailed_description")


def needs_translation(record: Dict[str, Any]) -> bool:
    short_lang = str(record.get("description_lang_short") or "").lower()
    detail_lang = str(record.get("description_lang_detail") or "").lower()
    return short_lang == "en" or detail_lang == "en"


def localize_record(record: Dict[str, Any], translator: Translator) -> Dict[str, Any]:
    out = dict(record)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not needs_translation(record) or translator.name == "passthrough":
        # 中文透传，或本轮未真正翻译
        if needs_translation(record) and translator.name == "passthrough":
            out["localization_source"] = "pending"
        else:
            out["localization_source"] = "passthrough"
        out["localized_at"] = now
        return out

    for field in TEXT_FIELDS:
        text = out.get(field) or ""
        lang_key = (
            "description_lang_short"
            if field == "short_description"
            else "description_lang_detail"
        )
        if str(out.get(lang_key) or "").lower() != "en":
            continue
        if not str(text).strip():
            continue
        out[field] = translator.translate(str(text), source_lang="en", target_lang="zh")
    name_cn = (out.get("name_cn") or "").strip()
    if not name_cn and (out.get("name") or "").strip():
        # 英文常用名：尝试译一版中文名（质量一般，可手改）
        out["name_cn"] = translator.translate(
            str(out["name"]), source_lang="en", target_lang="zh"
        )
    out["localization_source"] = translator.name
    out["localized_at"] = now
    return out


def list_raw_app_ids(*, only_en: bool = False) -> List[str]:
    ids: List[str] = []
    for path in sorted(
        RAW_DIR.glob("*.json"),
        key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem),
    ):
        if not path.stem.isdigit():
            continue
        if only_en:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not needs_translation(record):
                continue
        ids.append(path.stem)
    return ids


def prepare_one(app_id: str, translator: Translator) -> Path:
    raw_path = RAW_DIR / f"{app_id}.json"
    if not raw_path.is_file():
        raise FileNotFoundError(f"缺少 raw: {raw_path}")
    record = json.loads(raw_path.read_text(encoding="utf-8"))
    localized = localize_record(record, translator)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    md_path = PROCESSED_DIR / f"{app_id}.md"
    md_path.write_text(to_markdown(localized), encoding="utf-8")
    return md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="raw → processed（进索引准备）")
    parser.add_argument("--app-id", type=str, default="", help="只处理指定 app_id")
    parser.add_argument("--only-en", action="store_true", help="只处理 description_lang 为 en 的 raw")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少款（0=不限制）")
    parser.add_argument("--dry-run", action="store_true", help="只列出将处理的 app_id")
    parser.add_argument(
        "--backend",
        type=str,
        default="passthrough",
        help="翻译后端：passthrough（默认）| local_opus",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=r"D:\model\Helsinki-NLP_opus-mt-en-zh",
        help="local_opus 模型目录",
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="结束后根据 processed 刷新 games_manifest / fetched_appids",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    translator = get_translator(args.backend, model_dir=args.model_dir)

    if args.app_id:
        app_ids = [str(args.app_id).strip()]
    else:
        app_ids = list_raw_app_ids(only_en=args.only_en)

    if args.limit and args.limit > 0:
        app_ids = app_ids[: args.limit]

    print(f"翻译后端: {translator.name}")
    print(f"待处理: {len(app_ids)} 款" + ("（仅 en）" if args.only_en and not args.app_id else ""))
    if args.dry_run:
        for aid in app_ids[:50]:
            print(f"  {aid}")
        if len(app_ids) > 50:
            print(f"  ... 另有 {len(app_ids) - 50} 款")
        return

    ok = 0
    for i, aid in enumerate(app_ids, 1):
        try:
            path = prepare_one(aid, translator)
            ok += 1
            if i <= 5 or i == len(app_ids) or i % 50 == 0:
                print(f"[{i}/{len(app_ids)}] {path.name}")
        except NotImplementedError as exc:
            raise SystemExit(str(exc)) from exc
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            print(f"[{i}/{len(app_ids)}] {aid} 失败: {exc}")

    print(f"完成：成功 {ok}/{len(app_ids)}")
    if args.refresh_manifest:
        rows = collect_manifest_from_processed()
        write_games_manifest(
            rows,
            {
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": "prepare_processed",
            },
        )
        fetched = write_fetched_appids()
        print(f"已刷新清单 {len(rows)} 款；{fetched}")


if __name__ == "__main__":
    main()
