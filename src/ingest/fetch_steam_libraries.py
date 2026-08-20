"""拉取本人与好友 Steam 库存，写入 data/library 备用。

输出结构（便于按好友名 / steamid 查找）：
  data/library/me_owned.json              # 本人完整库存
  data/library/owned_appids.json          # 给 library 路由用的 app_id 列表
  data/library/friends/index.json         # 好友索引（by_steamid / by_persona / by_vanity）
  data/library/friends/by_steamid/<id>.json
  data/library/friends/summary.md         # 可读清单

依赖 .env：STEAM_API_KEY、STEAM_ID（或 STEAM_VANITY / --steamid / --vanity）
"""

from __future__ import annotations

import argparse
import http.client
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
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import PROJECT_ROOT

LIBRARY_DIR = PROJECT_ROOT / "data" / "library"
ME_PATH = LIBRARY_DIR / "me_owned.json"
OWNED_APPIDS_PATH = LIBRARY_DIR / "owned_appids.json"
FRIENDS_DIR = LIBRARY_DIR / "friends"
FRIENDS_BY_ID_DIR = FRIENDS_DIR / "by_steamid"
FRIENDS_INDEX_PATH = FRIENDS_DIR / "index.json"
FRIENDS_SUMMARY_PATH = FRIENDS_DIR / "summary.md"

STEAM_API = "https://api.steampowered.com"


def http_get_json(url: str, timeout: int = 60, retries: int = 3) -> Any:
    last_err: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "steam-game-advisor/0.1 (personal research)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            http.client.HTTPException,
            OSError,
        ) as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
                continue
            raise
    raise RuntimeError(str(last_err))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def normalize_name(name: str) -> str:
    text = (name or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def vanity_from_profile_url(profile_url: str) -> str:
    if not profile_url:
        return ""
    m = re.search(r"/id/([^/]+)/?", profile_url)
    return m.group(1) if m else ""


def resolve_vanity(api_key: str, vanity: str) -> str:
    query = urllib.parse.urlencode({"key": api_key, "vanityurl": vanity})
    url = f"{STEAM_API}/ISteamUser/ResolveVanityURL/v1/?{query}"
    payload = http_get_json(url)
    resp = payload.get("response") or {}
    if int(resp.get("success") or 0) != 1 or not resp.get("steamid"):
        raise SystemExit(f"无法解析自定义 URL: {vanity!r} -> {resp}")
    return str(resp["steamid"])


def fetch_player_summaries(api_key: str, steamids: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(steamids), 100):
        batch = steamids[i : i + 100]
        query = urllib.parse.urlencode({"key": api_key, "steamids": ",".join(batch)})
        url = f"{STEAM_API}/ISteamUser/GetPlayerSummaries/v2/?{query}"
        payload = http_get_json(url)
        for player in (payload.get("response") or {}).get("players") or []:
            sid = str(player.get("steamid") or "")
            if sid:
                out[sid] = player
    return out


def fetch_friend_list(api_key: str, steam_id: str) -> List[Dict[str, Any]]:
    query = urllib.parse.urlencode(
        {"key": api_key, "steamid": steam_id, "relationship": "friend"}
    )
    url = f"{STEAM_API}/ISteamUser/GetFriendList/v1/?{query}"
    try:
        payload = http_get_json(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise SystemExit(
                "好友列表不可用（401）。请在 Steam 隐私设置里把「好友列表」设为公开，"
                "或确认 STEAM_ID 是这把 API Key 对应账户。"
            ) from exc
        raise
    return list((payload.get("friendslist") or {}).get("friends") or [])


def fetch_owned_games_full(
    api_key: str, steam_id: str
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """返回 (games_rows, error)。隐私不可见时 games 为空列表且 error 说明原因。"""
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
    try:
        payload = http_get_json(url)
    except urllib.error.HTTPError as exc:
        return None, f"http_{exc.code}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, str(exc)

    resp = payload.get("response") or {}
    games = resp.get("games")
    if games is None:
        # 常见：库存设为私密时返回空 response / 无 games 字段
        return [], "private_or_empty"
    rows: List[Dict[str, Any]] = []
    for item in games:
        appid = item.get("appid")
        if not appid:
            continue
        rows.append(
            {
                "app_id": int(appid),
                "name": item.get("name") or "",
                "playtime_forever": int(item.get("playtime_forever") or 0),
                "playtime_2weeks": int(item.get("playtime_2weeks") or 0),
            }
        )
    rows.sort(key=lambda g: g["playtime_forever"], reverse=True)
    return rows, None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_owner_record(
    steam_id: str,
    summary: Dict[str, Any],
    games: List[Dict[str, Any]],
    *,
    role: str,
    friend_since: Optional[int] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    persona = summary.get("personaname") or ""
    profile_url = summary.get("profileurl") or ""
    vanity = vanity_from_profile_url(profile_url)
    return {
        "role": role,
        "steamid": steam_id,
        "persona_name": persona,
        "persona_name_normalized": normalize_name(persona),
        "vanity": vanity,
        "profile_url": profile_url,
        "avatar": summary.get("avatarfull") or summary.get("avatarmedium") or "",
        "community_visibility_state": summary.get("communityvisibilitystate"),
        "friend_since": friend_since,
        "game_count": len(games),
        "games_visible": error is None,
        "error": error,
        "app_ids": [g["app_id"] for g in games],
        "games": games,
        "fetched_at": now_iso(),
    }


def write_friends_summary(friends: List[Dict[str, Any]]) -> None:
    lines = [
        "# 好友库存快照",
        "",
        f"- 生成时间: {now_iso()}",
        f"- 好友数: {len(friends)}",
        f"- 可见库存: {sum(1 for f in friends if f.get('games_visible'))}",
        f"- 不可见/失败: {sum(1 for f in friends if not f.get('games_visible'))}",
        "",
        "| # | 昵称 | SteamID64 | vanity | 游戏数 | 状态 | 档案 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, f in enumerate(friends, 1):
        status = "ok" if f.get("games_visible") else (f.get("error") or "hidden")
        lines.append(
            "| {i} | {name} | `{sid}` | {vanity} | {n} | {status} | {url} |".format(
                i=i,
                name=(f.get("persona_name") or "—").replace("|", "\\|"),
                sid=f.get("steamid") or "",
                vanity=(f.get("vanity") or "—").replace("|", "\\|"),
                n=f.get("game_count") or 0,
                status=status,
                url=f.get("profile_url") or "",
            )
        )
    lines.append("")
    lines.append("按昵称查找：打开 `index.json` 的 `by_persona_name`。")
    lines.append("按 SteamID 打开：`by_steamid/<steamid>.json`。")
    lines.append("")
    FRIENDS_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拉取本人与好友 Steam 库存备用")
    parser.add_argument("--steamid", type=str, default="", help="本人 SteamID64，覆盖 .env")
    parser.add_argument("--vanity", type=str, default="", help="本人自定义 URL，覆盖 .env STEAM_VANITY")
    parser.add_argument(
        "--friend-limit",
        type=int,
        default=20,
        help="最多拉取多少位好友的库存（0=全部）",
    )
    parser.add_argument("--sleep", type=float, default=0.35, help="请求间隔秒数")
    parser.add_argument("--skip-friends", action="store_true", help="只拉本人库存")
    parser.add_argument("--skip-me", action="store_true", help="只拉好友（需已有 STEAM_ID）")
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()

    api_key = os.getenv("STEAM_API_KEY") or ""
    if not api_key:
        raise SystemExit("请在 .env 设置 STEAM_API_KEY")

    steam_id = (args.steamid or os.getenv("STEAM_ID") or "").strip()
    vanity = (args.vanity or os.getenv("STEAM_VANITY") or "").strip()
    if not steam_id and vanity:
        print(f"正在解析 vanity={vanity!r} ...")
        steam_id = resolve_vanity(api_key, vanity)
        print(f"得到 STEAM_ID={steam_id}")
    if not steam_id:
        raise SystemExit(
            "请设置 STEAM_ID（64 位数字）或 STEAM_VANITY（个人资料自定义 URL），"
            "也可传 --steamid / --vanity。"
            "可在资料页 URL 或 https://steamid.io 查看。"
        )

    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    FRIENDS_BY_ID_DIR.mkdir(parents=True, exist_ok=True)

    me_summary = fetch_player_summaries(api_key, [steam_id]).get(steam_id) or {}
    if not args.skip_me:
        print("正在拉取本人库存（GetOwnedGames）...")
        games, err = fetch_owned_games_full(api_key, steam_id)
        if games is None:
            raise SystemExit(f"本人库存拉取失败: {err}")
        me = build_owner_record(steam_id, me_summary, games, role="me", error=err)
        write_json(ME_PATH, me)
        write_json(
            OWNED_APPIDS_PATH,
            {
                "steamid": steam_id,
                "persona_name": me.get("persona_name") or "",
                "updated_at": me["fetched_at"],
                "count": len(me["app_ids"]),
                "app_ids": me["app_ids"],
            },
        )
        print(f"本人库存 {me['game_count']} 款 -> {ME_PATH.name} / {OWNED_APPIDS_PATH.name}")
        if err:
            print(f"警告: {err}")

    if args.skip_friends:
        print("已跳过好友。")
        return

    print("正在拉取好友列表（GetFriendList）...")
    friends_raw = fetch_friend_list(api_key, steam_id)
    friends_raw.sort(key=lambda f: int(f.get("friend_since") or 0))
    if args.friend_limit and args.friend_limit > 0:
        friends_raw = friends_raw[: args.friend_limit]
    print(f"将处理 {len(friends_raw)} 位好友")

    friend_ids = [str(f["steamid"]) for f in friends_raw if f.get("steamid")]
    summaries = fetch_player_summaries(api_key, friend_ids)

    friend_records: List[Dict[str, Any]] = []
    by_steamid: Dict[str, str] = {}
    by_persona: Dict[str, List[str]] = {}
    by_vanity: Dict[str, str] = {}

    for i, fr in enumerate(friends_raw, 1):
        sid = str(fr.get("steamid") or "")
        if not sid:
            continue
        summary = summaries.get(sid) or {}
        persona = summary.get("personaname") or sid
        print(f"[{i}/{len(friends_raw)}] {persona} ({sid}) ...", end=" ", flush=True)
        games, err = fetch_owned_games_full(api_key, sid)
        if games is None:
            games = []
            err = err or "fetch_failed"
        record = build_owner_record(
            sid,
            summary,
            games,
            role="friend",
            friend_since=int(fr.get("friend_since") or 0) or None,
            error=err,
        )
        rel = FRIENDS_BY_ID_DIR / f"{sid}.json"
        write_json(rel, record)
        friend_records.append(record)

        by_steamid[sid] = f"by_steamid/{sid}.json"
        key = record["persona_name_normalized"] or sid
        by_persona.setdefault(key, []).append(sid)
        if record.get("vanity"):
            by_vanity[str(record["vanity"]).lower()] = sid

        status = f"{record['game_count']} games" if record["games_visible"] else f"skip ({err})"
        print(status)
        time.sleep(max(args.sleep, 0.0))

    index = {
        "owner_steamid": steam_id,
        "owner_persona_name": me_summary.get("personaname") or "",
        "fetched_at": now_iso(),
        "friend_count": len(friend_records),
        "visible_count": sum(1 for f in friend_records if f.get("games_visible")),
        "lookup_fields": [
            "steamid",
            "persona_name",
            "persona_name_normalized",
            "vanity",
            "profile_url",
            "friend_since",
            "game_count",
            "app_ids",
        ],
        "by_steamid": by_steamid,
        "by_persona_name": by_persona,
        "by_vanity": by_vanity,
        "friends": [
            {
                "steamid": f["steamid"],
                "persona_name": f["persona_name"],
                "persona_name_normalized": f["persona_name_normalized"],
                "vanity": f.get("vanity") or "",
                "profile_url": f.get("profile_url") or "",
                "friend_since": f.get("friend_since"),
                "game_count": f.get("game_count") or 0,
                "games_visible": bool(f.get("games_visible")),
                "error": f.get("error"),
                "path": by_steamid.get(f["steamid"], ""),
            }
            for f in friend_records
        ],
    }
    write_json(FRIENDS_INDEX_PATH, index)
    write_friends_summary(friend_records)
    print(
        f"好友索引已写 {FRIENDS_INDEX_PATH.relative_to(PROJECT_ROOT)} "
        f"（可见 {index['visible_count']}/{index['friend_count']}）"
    )


if __name__ == "__main__":
    main()
