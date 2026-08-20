"""采集 Steam 游戏档案：热门候选 + 可选库存，再拉商店详情与评测摘要。"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import PROJECT_ROOT

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LIBRARY_DIR = PROJECT_ROOT / "data" / "library"
LIBRARY_PATH = LIBRARY_DIR / "owned_appids.json"
MANIFEST_JSON = LIBRARY_DIR / "games_manifest.json"
MANIFEST_MD = LIBRARY_DIR / "games_manifest.md"

STEAM_API = "https://api.steampowered.com"
STORE_API = "https://store.steampowered.com"

# 评测集里会用到的常见游戏，避免热门榜/库存都没覆盖到。
EVAL_SEED_APPIDS = [
    730,  # Counter-Strike 2
    105600,  # Terraria
    413150,  # Stardew Valley
    1091500,  # Cyberpunk 2077
    1245620,  # Elden Ring
    367520,  # Hollow Knight
    1086940,  # Baldur's Gate 3
    632470,  # Disco Elysium
    837470,  # Untitled Goose Game
    1623730,  # Palworld
    1203620,  # Enshrouded
    1203220,  # NARAKA: BLADEPOINT
    632360,  # Risk of Rain 2
]

# 简介中 CJK 占比低于此阈值时，改用英文商店页正文
MIN_CJK_RATIO = 0.12


def http_get_json(url: str, timeout: int = 30) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "steam-game-advisor/0.1 (personal research)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p>", "\n\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def yaml_list(values: Iterable[Any]) -> str:
    items = [str(item).replace('"', "") for item in values if str(item).strip()]
    inner = ", ".join(f'"{item}"' for item in items)
    return f"[{inner}]"


def fetch_most_played(api_key: Optional[str], limit: int) -> List[int]:
    params = {}
    if api_key:
        params["key"] = api_key
    url = f"{STEAM_API}/ISteamChartsService/GetMostPlayedGames/v1/"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    payload = http_get_json(url)
    ranks = (
        payload.get("response", {}).get("ranks")
        or payload.get("response", {}).get("rollup_list")
        or []
    )
    appids = []
    for row in ranks:
        appid = row.get("appid") or row.get("appid_owner") or (row.get("item") or {}).get("appid")
        if appid:
            appids.append(int(appid))
        if len(appids) >= limit:
            break
    return appids


def fetch_owned_games(api_key: str, steam_id: str, limit: int) -> List[int]:
    query = urllib.parse.urlencode(
        {
            "key": api_key,
            "steamid": steam_id,
            "include_appinfo": 1,
            "include_played_free_games": 1,
            "format": "json",
        }
    )
    url = f"{STEAM_API}/IPlayerService/GetOwnedGames/v1/?{query}"
    payload = http_get_json(url)
    games = payload.get("response", {}).get("games") or []
    games.sort(key=lambda item: int(item.get("playtime_forever") or 0), reverse=True)
    return [int(item["appid"]) for item in games[:limit] if item.get("appid")]


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk / max(len(text), 1)


def fetch_appdetails(appid: int, lang: str = "schinese") -> Optional[Dict[str, Any]]:
    query = urllib.parse.urlencode({"appids": appid, "l": lang, "cc": "cn"})
    payload = http_get_json(f"{STORE_API}/api/appdetails?{query}")
    node = payload.get(str(appid)) or {}
    if not node.get("success"):
        return None
    data = node.get("data") or {}
    if data.get("type") != "game":
        return None
    return data


def fetch_game_details(appid: int) -> Optional[Dict[str, Any]]:
    """拉简体与英文两版详情，合并名称并择优选简介语言。"""
    zh = fetch_appdetails(appid, "schinese")
    if not zh:
        return None
    en: Optional[Dict[str, Any]] = None
    try:
        en = fetch_appdetails(appid, "english")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        en = None

    name_en = ((en or {}).get("name") or zh.get("name") or str(appid)).strip()
    name_zh = (zh.get("name") or "").strip()
    if cjk_ratio(name_zh) >= MIN_CJK_RATIO and name_zh and name_zh != name_en:
        name_cn = name_zh
    elif cjk_ratio(name_en) >= MIN_CJK_RATIO:
        name_cn = name_en
        if not name_zh:
            name_zh = name_en
    else:
        name_cn = name_zh if name_zh and name_zh != name_en else ""

    zh_short = strip_html(zh.get("short_description") or "")
    en_short = strip_html((en or {}).get("short_description") or "")
    short_description, short_lang = pick_localized_text(zh_short, en_short)

    zh_detail = strip_html(zh.get("detailed_description") or zh.get("about_the_game") or "")
    en_detail = strip_html(
        (en or {}).get("detailed_description") or (en or {}).get("about_the_game") or ""
    )
    detailed_description, detail_lang = pick_localized_text(zh_detail, en_detail)
    # 长简介仍偏英文、但短简介已是中文时：游玩方式用英文，简介保留中文一句
    if detail_lang == "en" and short_lang == "zh" and en_detail:
        detailed_description = en_detail

    merged = dict(zh)
    merged["name"] = name_en
    merged["name_cn"] = name_cn
    merged["name_zh_store"] = name_zh
    merged["short_description"] = short_description
    merged["detailed_description"] = detailed_description
    merged["description_lang_short"] = short_lang
    merged["description_lang_detail"] = detail_lang
    return merged


def pick_localized_text(zh_text: str, en_text: str) -> tuple[str, str]:
    zh_text = (zh_text or "").strip()
    en_text = (en_text or "").strip()
    if zh_text and cjk_ratio(zh_text) >= MIN_CJK_RATIO:
        return zh_text, "zh"
    if en_text:
        return en_text, "en"
    return zh_text or en_text, "mixed"


def fetch_review_summary(appid: int) -> Dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "json": 1,
            "language": "all",
            "purchase_type": "all",
            "num_per_page": 0,
            "filter": "summary",
        }
    )
    payload = http_get_json(f"{STORE_API}/appreviews/{appid}?{query}")
    summary = payload.get("query_summary") or {}
    total = int(summary.get("total_reviews") or 0)
    positive = int(summary.get("total_positive") or 0)
    percent = round(100.0 * positive / total, 1) if total else None
    return {
        "review_desc": summary.get("review_score_desc"),
        "review_count": total or None,
        "review_percentage": percent,
        "review_score": summary.get("review_score"),
    }


def parse_languages(raw: str) -> List[str]:
    text = strip_html(raw or "")
    if not text:
        return []
    return [part.strip() for part in re.split(r"[,/、]", text) if part.strip()]


def parse_price_cny(data: Dict[str, Any]) -> Optional[float]:
    if data.get("is_free"):
        return 0.0
    overview = data.get("price_overview") or {}
    if overview.get("currency") == "CNY" and overview.get("final") is not None:
        return round(int(overview["final"]) / 100.0, 2)
    if overview.get("final") is not None:
        return round(int(overview["final"]) / 100.0, 2)
    return None


def to_record(appid: int, data: Dict[str, Any], reviews: Dict[str, Any]) -> Dict[str, Any]:
    platforms = data.get("platforms") or {}
    platform_names = [name.capitalize() for name, enabled in platforms.items() if enabled]
    categories = [item.get("description") for item in data.get("categories") or [] if item.get("description")]
    genres = [item.get("description") for item in data.get("genres") or [] if item.get("description")]
    pc_min = data.get("pc_requirements") or {}
    if isinstance(pc_min, dict):
        pc_min_text = strip_html(pc_min.get("minimum") or "")
        pc_rec_text = strip_html(pc_min.get("recommended") or "")
    else:
        pc_min_text = ""
        pc_rec_text = ""
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "app_id": str(appid),
        "name": data.get("name") or str(appid),
        "name_cn": (data.get("name_cn") or "").strip(),
        "short_description": strip_html(data.get("short_description") or ""),
        "detailed_description": strip_html(
            data.get("detailed_description") or data.get("about_the_game") or ""
        ),
        "description_lang_short": data.get("description_lang_short") or "mixed",
        "description_lang_detail": data.get("description_lang_detail") or "mixed",
        "genres": genres,
        "tags": genres,
        "categories": categories,
        "developers": data.get("developers") or [],
        "publishers": data.get("publishers") or [],
        "release_date": (data.get("release_date") or {}).get("date"),
        "platforms": platform_names,
        "supported_languages": parse_languages(data.get("supported_languages") or ""),
        "is_free": bool(data.get("is_free")),
        "price_cny": parse_price_cny(data),
        "pc_requirements_min": pc_min_text,
        "pc_requirements_rec": pc_rec_text,
        "fetched_at": fetched_at,
        **reviews,
    }


def to_markdown(record: Dict[str, Any]) -> str:
    front = [
        "---",
        f'app_id: "{record["app_id"]}"',
        f'name: "{record["name"].replace(chr(34), "")}"',
    ]
    if record.get("name_cn"):
        front.append(f'name_cn: "{str(record["name_cn"]).replace(chr(34), "")}"')
    front.extend([
        f"genres: {yaml_list(record.get('genres') or [])}",
        f"tags: {yaml_list(record.get('tags') or [])}",
        f"categories: {yaml_list(record.get('categories') or [])}",
        f"developers: {yaml_list(record.get('developers') or [])}",
        f"publishers: {yaml_list(record.get('publishers') or [])}",
        f"platforms: {yaml_list(record.get('platforms') or [])}",
        f"supported_languages: {yaml_list(record.get('supported_languages') or [])}",
        f"is_free: {str(bool(record.get('is_free'))).lower()}",
    ])
    if record.get("price_cny") is not None:
        front.append(f"price_cny: {record['price_cny']}")
    if record.get("review_percentage") is not None:
        front.append(f"review_percentage: {record['review_percentage']}")
    if record.get("review_count") is not None:
        front.append(f"review_count: {record['review_count']}")
    if record.get("review_desc"):
        front.append(f'review_desc: "{record["review_desc"]}"')
    if record.get("release_date"):
        front.append(f'release_date: "{record["release_date"]}"')
    front.append(f'description_lang_short: "{record.get("description_lang_short", "mixed")}"')
    front.append(f'description_lang_detail: "{record.get("description_lang_detail", "mixed")}"')
    front.append(f'fetched_at: "{record["fetched_at"]}"')
    front.append("---")

    title = record["name"]
    if record.get("name_cn") and record["name_cn"] != record["name"]:
        title = f"{record['name_cn']} / {record['name']}"

    body = [
        f"# {title}（app_id={record['app_id']}）",
        "",
        "## 简介",
        record.get("short_description") or "（无简介）",
        "",
        "## 类型与标签",
        "类型: " + ", ".join(record.get("genres") or []) or "（无）",
        "分类: " + ", ".join(record.get("categories") or []) or "（无）",
        "",
        "## 游玩方式",
        record.get("detailed_description") or "（无详细介绍）",
        "",
        "## 配置与平台",
        "平台: " + ", ".join(record.get("platforms") or []) or "（无）",
        "",
        record.get("pc_requirements_min") or "（无最低配置信息）",
        "",
        "## 语言",
        ", ".join(record.get("supported_languages") or []) or "（无）",
        "",
        "## 评价摘要",
        f"档位: {record.get('review_desc') or '未知'}",
        f"好评率: {record.get('review_percentage')}",
        f"评测数: {record.get('review_count')}",
        f"价格(CNY): {record.get('price_cny')}",
        f"抓取时间: {record.get('fetched_at')}",
        "",
    ]
    return "\n".join(front + [""] + body)


def unique(appids: Iterable[int]) -> List[int]:
    seen = set()
    result = []
    for appid in appids:
        if appid in seen:
            continue
        seen.add(appid)
        result.append(appid)
    return result


def safe_print(message: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def write_library(appids: List[int]) -> None:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    LIBRARY_PATH.write_text(json.dumps(appids, ensure_ascii=False, indent=2), encoding="utf-8")


def write_fetched_appids() -> Path:
    """维护已抓取游戏列表：与 data/processed/*.md 对齐。"""
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    app_ids = sorted(
        (p.stem for p in PROCESSED_DIR.glob("*.md") if p.stem.isdigit()),
        key=lambda x: int(x),
    )
    path = LIBRARY_DIR / "fetched_appids.json"
    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(app_ids),
        "source": "data/processed/*.md",
        "app_ids": [int(x) for x in app_ids],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_fetched_appid_set() -> set[str]:
    """已抓取集合：processed 为准，fetched_appids.json 作旁路清单。"""
    existing = {p.stem for p in PROCESSED_DIR.glob("*.md")}
    fetched_path = LIBRARY_DIR / "fetched_appids.json"
    if fetched_path.is_file():
        try:
            raw = json.loads(fetched_path.read_text(encoding="utf-8"))
            ids = raw.get("app_ids") if isinstance(raw, dict) else raw
            existing.update(str(x) for x in (ids or []))
        except (json.JSONDecodeError, OSError):
            pass
    return existing


def read_md_manifest_entry(md_path: Path) -> Optional[Dict[str, Any]]:
    text = md_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end < 0:
        return None
    front = text[3:end].strip()
    meta: Dict[str, Any] = {"app_id": md_path.stem}
    for line in front.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"')
        if key in {"app_id", "name", "name_cn", "description_lang_short", "description_lang_detail"}:
            meta[key] = value
    if not meta.get("name"):
        return None
    meta.setdefault("name_cn", "")
    return meta


def write_games_manifest(entries: List[Dict[str, Any]], run_meta: Dict[str, Any]) -> None:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    entries = sorted(entries, key=lambda item: int(item["app_id"]))
    payload = {
        "generated_at": run_meta.get("generated_at"),
        "source": run_meta.get("source"),
        "count": len(entries),
        "description_policy": (
            "商店页先拉 schinese，再拉 english；简介/长文 CJK 占比不足时回退英文。"
            f"阈值={MIN_CJK_RATIO}"
        ),
        "games": entries,
    }
    MANIFEST_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Steam Game Advisor 游戏清单",
        "",
        f"- 生成时间: {payload['generated_at']}",
        f"- 来源: {payload['source']}",
        f"- 游戏数: {payload['count']}",
        f"- 简介策略: {payload['description_policy']}",
        "",
        "| App ID | 英文名 | 中文名 | 简介语言 | 长文语言 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in entries:
        lines.append(
            "| {app_id} | {name} | {name_cn} | {short} | {detail} |".format(
                app_id=item["app_id"],
                name=(item.get("name") or "").replace("|", "\\|"),
                name_cn=(item.get("name_cn") or "—").replace("|", "\\|"),
                short=item.get("description_lang_short") or "—",
                detail=item.get("description_lang_detail") or "—",
            )
        )
    lines.append("")
    MANIFEST_MD.write_text("\n".join(lines), encoding="utf-8")


def collect_manifest_from_processed() -> List[Dict[str, Any]]:
    entries = []
    for md_path in sorted(PROCESSED_DIR.glob("*.md"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        item = read_md_manifest_entry(md_path)
        if item:
            entries.append(item)
    return entries


def manifest_entry_from_record(record: Dict[str, Any], status: str = "fetched") -> Dict[str, Any]:
    return {
        "app_id": record["app_id"],
        "name": record.get("name") or "",
        "name_cn": record.get("name_cn") or "",
        "description_lang_short": record.get("description_lang_short") or "mixed",
        "description_lang_detail": record.get("description_lang_detail") or "mixed",
        "status": status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集 Steam 游戏档案到 data/processed")
    parser.add_argument(
        "--source",
        choices=["seed", "charts", "library", "both", "none"],
        default="seed",
        help="候选来源：种子/热门/库存/合并；none=仅用 --owned-missing 或 --candidates-file",
    )
    parser.add_argument("--limit", type=int, default=80, help="最多写入多少款游戏")
    parser.add_argument("--candidate-limit", type=int, default=120, help="每路来源最多取多少 appid")
    parser.add_argument("--min-reviews", type=int, default=500, help="评测数下限，便于先审阅高关注游戏")
    parser.add_argument("--min-positive", type=float, default=70.0, help="好评率下限（百分比）")
    parser.add_argument("--sleep", type=float, default=1.5, help="商店请求间隔秒数")
    parser.add_argument("--include-eval-seed", action="store_true", help="并入评测集常用 appid")
    parser.add_argument("--refresh", action="store_true", help="已存在的 processed 也重新拉取（更新简介/中文名）")
    parser.add_argument("--candidates-only", action="store_true", help="只写出候选 appid，不拉详情")
    parser.add_argument(
        "--candidates-file",
        type=str,
        default="",
        help="从 JSON 文件读候选 appid 列表（数组，或 {\"app_ids\":[...]}）",
    )
    parser.add_argument(
        "--owned-missing",
        action="store_true",
        help="从 data/library/me_owned.json 取尚未入库的 appid（按时长降序）",
    )
    parser.add_argument(
        "--friends-missing",
        action="store_true",
        help="从 data/library/friends/ 汇总好友库存中尚未入库的 appid（按拥有好友数降序）",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    api_key = os.getenv("STEAM_API_KEY")
    steam_id = os.getenv("STEAM_ID")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    candidates: List[int] = []
    if args.source == "seed" or args.include_eval_seed:
        candidates.extend(EVAL_SEED_APPIDS)

    if args.owned_missing:
        me_path = LIBRARY_DIR / "me_owned.json"
        if not me_path.exists():
            raise SystemExit(f"未找到 {me_path}，请先跑 fetch_steam_libraries.py")
        me = json.loads(me_path.read_text(encoding="utf-8"))
        existing = load_fetched_appid_set()
        games = list(me.get("games") or [])
        games.sort(key=lambda g: int(g.get("playtime_forever") or 0), reverse=True)
        owned_missing = [
            int(g["app_id"])
            for g in games
            if g.get("app_id") is not None and str(g["app_id"]) not in existing
        ]
        print(f"库存缺档 {len(owned_missing)} 款（按时长降序；已抓取跳过）")
        candidates.extend(owned_missing)

    if args.friends_missing:
        friends_dir = LIBRARY_DIR / "friends" / "by_steamid"
        if not friends_dir.is_dir():
            raise SystemExit(f"未找到 {friends_dir}，请先跑 fetch_steam_libraries.py")
        existing = load_fetched_appid_set()
        owner_count: Dict[int, int] = {}
        for path in friends_dir.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            for app_id in data.get("app_ids") or []:
                aid = int(app_id)
                if str(aid) in existing:
                    continue
                owner_count[aid] = owner_count.get(aid, 0) + 1
        friends_missing = sorted(
            owner_count.keys(),
            key=lambda aid: (owner_count[aid], aid),
            reverse=True,
        )
        print(
            f"好友库存缺档 {len(friends_missing)} 款"
            f"（按拥有好友数降序；已抓取 {len(existing)} 款跳过）"
        )
        candidates.extend(friends_missing)

    if args.candidates_file:
        cpath = Path(args.candidates_file)
        if not cpath.is_file():
            raise SystemExit(f"候选文件不存在: {cpath}")
        payload = json.loads(cpath.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("app_ids") or payload.get("candidates") or []
        candidates.extend(int(x) for x in payload)

    if args.source in {"charts", "both"}:
        print("正在拉取 Steam 热门榜（GetMostPlayedGames）...")
        try:
            charts = fetch_most_played(api_key, args.candidate_limit)
            print(f"热门榜得到 {len(charts)} 个 appid")
            candidates.extend(charts)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"热门榜失败: {exc}")
            if args.source == "charts" and not args.include_eval_seed:
                raise

    if args.source in {"library", "both"}:
        if not api_key or not steam_id:
            raise SystemExit("使用库存来源时请在 .env 中设置 STEAM_API_KEY 和 STEAM_ID")
        print("正在拉取库存（GetOwnedGames）...")
        owned = fetch_owned_games(api_key, steam_id, args.candidate_limit)
        print(f"库存按时长取前 {len(owned)} 个 appid")
        write_library(owned)
        candidates.extend(owned)

    candidates = unique(candidates)
    (RAW_DIR / "candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"去重后候选 {len(candidates)} 款，已写入 data/raw/candidates.json")
    if args.candidates_only:
        return

    kept = []
    manifest_rows: List[Dict[str, Any]] = []
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for index, appid in enumerate(candidates, 1):
        if len(kept) >= args.limit and not (args.refresh and (PROCESSED_DIR / f"{appid}.md").exists()):
            break
        print(f"[{index}/{len(candidates)}] 拉取 {appid} ...")
        md_path = PROCESSED_DIR / f"{appid}.md"
        if md_path.exists() and not args.refresh:
            kept.append(str(appid))
            safe_print(f"  已存在，跳过 {md_path.name}")
            continue
        if md_path.exists() and args.refresh:
            safe_print(f"  --refresh：覆盖 {md_path.name}")
        try:
            details = fetch_game_details(appid)
            time.sleep(args.sleep)
            if not details:
                print("  跳过：不是游戏或详情失败")
                continue
            reviews = fetch_review_summary(appid)
            time.sleep(args.sleep)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"  失败: {exc}")
            time.sleep(args.sleep)
            continue

        review_count = reviews.get("review_count") or 0
        review_percentage = reviews.get("review_percentage")
        if review_count < args.min_reviews:
            print(f"  跳过：评测数 {review_count} < {args.min_reviews}")
            continue
        if review_percentage is None or review_percentage < args.min_positive:
            print(f"  跳过：好评率 {review_percentage} < {args.min_positive}")
            continue

        record = to_record(appid, details, reviews)
        raw_path = RAW_DIR / f"{appid}.json"
        raw_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(to_markdown(record), encoding="utf-8")
        if record["app_id"] not in kept:
            kept.append(record["app_id"])
        manifest_rows.append(manifest_entry_from_record(record, status="fetched"))
        lang_note = f"{record.get('description_lang_short')}/{record.get('description_lang_detail')}"
        safe_print(
            f"  已写入 {md_path.name}  {record.get('name_cn') or record['name']}  "
            f"({lang_note})  {reviews.get('review_desc')}"
        )

    summary_path = RAW_DIR / "fetch_summary.json"
    summary_path.write_text(
        json.dumps({"kept": kept, "count": len(kept), "fetched_at": fetched_at}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 清单覆盖库内全部 processed（含跳过/历史已有）
    all_manifest = collect_manifest_from_processed()
    write_games_manifest(
        all_manifest,
        {
            "generated_at": fetched_at,
            "source": args.source + ("+eval-seed" if args.include_eval_seed else ""),
        },
    )
    fetched_path = write_fetched_appids()
    print(f"完成：库内共 {len(all_manifest)} 款（本轮新写入/刷新 {len(manifest_rows)} 款）")
    print(f"  摘要: {summary_path}")
    print(f"  已抓取列表: {fetched_path}")
    print(f"  清单 JSON: {MANIFEST_JSON}")
    print(f"  清单 Markdown: {MANIFEST_MD}")


if __name__ == "__main__":
    main()
