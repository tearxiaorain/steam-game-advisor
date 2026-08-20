"""本人库存画像：时长加载 + library 子策略识别与候选排序。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


# tonight/recent：近两周有玩优先；否则按时长总榜
# backlog：库内 playtime_forever==0
# owned：检索后限制在库存内
# unowned：检索后排除库存（还没买）
LibraryMode = str


@dataclass
class OwnedGame:
    app_id: str
    name: str = ""
    playtime_forever: int = 0
    playtime_2weeks: int = 0


@dataclass
class OwnedLibrary:
    steamid: str = ""
    persona_name: str = ""
    games: Dict[str, OwnedGame] = field(default_factory=dict)

    @property
    def app_ids(self) -> List[str]:
        return list(self.games.keys())

    def get(self, app_id: str) -> Optional[OwnedGame]:
        return self.games.get(str(app_id))


def load_owned_library(me_owned_path: str, fallback_appids_path: str = "") -> OwnedLibrary:
    """优先读 me_owned.json（含时长）；否则退回 owned_appids.json。"""
    me_path = Path(me_owned_path) if me_owned_path else None
    if me_path and me_path.is_file():
        raw = json.loads(me_path.read_text(encoding="utf-8"))
        games: Dict[str, OwnedGame] = {}
        for item in raw.get("games") or []:
            app_id = str(item.get("app_id") or item.get("appid") or "")
            if not app_id:
                continue
            games[app_id] = OwnedGame(
                app_id=app_id,
                name=str(item.get("name") or ""),
                playtime_forever=int(item.get("playtime_forever") or 0),
                playtime_2weeks=int(item.get("playtime_2weeks") or 0),
            )
        if not games:
            for app_id in raw.get("app_ids") or []:
                games[str(app_id)] = OwnedGame(app_id=str(app_id))
        return OwnedLibrary(
            steamid=str(raw.get("steamid") or ""),
            persona_name=str(raw.get("persona_name") or ""),
            games=games,
        )

    fb = Path(fallback_appids_path) if fallback_appids_path else None
    if fb and fb.is_file():
        raw = json.loads(fb.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            ids = raw.get("app_ids") or raw.get("owned") or []
        else:
            ids = raw
        games = {str(i): OwnedGame(app_id=str(i)) for i in ids if i is not None}
        return OwnedLibrary(games=games)

    return OwnedLibrary()


def detect_library_mode(question: str) -> LibraryMode:
    """从问法识别库存子策略。"""
    q = question or ""

    # 还没买 / 库外推荐（不要用单独的「没玩过」，易与 backlog 冲突）
    if any(k in q for k in ("还没买", "没买过", "未购买", "想买一款", "推荐一款没买")):
        return "unowned"

    if any(
        k in q
        for k in (
            "库里没玩",
            "库存里没玩",
            "买了没玩",
            "买了从来没",
            "尘封",
            "backlog",
            "积压",
            "从没打开",
            "从来没玩",
            "库里从来没",
            "库存里从来没",
            "没碰过的库",
        )
    ):
        return "backlog"

    if any(k in q for k in ("最近玩", "这两周", "近两周", "最近在玩", "接着玩", "继续玩")):
        return "recent"

    if any(k in q for k in ("今晚", "现在玩哪个", "玩啥好", "库里玩哪个", "库存玩哪个", "今晚玩")):
        return "tonight"

    return "owned"


def _rank_recent(games: Iterable[OwnedGame]) -> List[OwnedGame]:
    recent = [g for g in games if g.playtime_2weeks > 0]
    recent.sort(key=lambda g: (g.playtime_2weeks, g.playtime_forever), reverse=True)
    return recent


def _rank_forever(games: Iterable[OwnedGame]) -> List[OwnedGame]:
    rows = [g for g in games if g.playtime_forever > 0]
    rows.sort(key=lambda g: g.playtime_forever, reverse=True)
    return rows


def _rank_backlog(games: Iterable[OwnedGame]) -> List[OwnedGame]:
    rows = [g for g in games if g.playtime_forever <= 0]
    rows.sort(key=lambda g: (g.name or g.app_id))
    return rows


def select_owned_candidates(
    library: OwnedLibrary,
    mode: LibraryMode,
    *,
    available_app_ids: Optional[Sequence[str]] = None,
    limit: int = 3,
) -> List[OwnedGame]:
    """按子策略选出库存候选（可与知识库 app_id 求交）。"""
    if not library.games:
        return []

    pool = list(library.games.values())
    if available_app_ids is not None:
        allow = {str(a) for a in available_app_ids}
        pool = [g for g in pool if g.app_id in allow]

    if mode == "backlog":
        ranked = _rank_backlog(pool)
    elif mode == "recent":
        ranked = _rank_recent(pool) or _rank_forever(pool)
    elif mode == "tonight":
        ranked = _rank_recent(pool) or _rank_forever(pool)
    else:
        # owned / unowned：默认按时长给一点稳定顺序，检索路径仍会再过滤
        ranked = _rank_forever(pool) or pool

    return ranked[: max(1, limit)]


def attach_playtime_metadata(doc: Any, game: Optional[OwnedGame]) -> Any:
    """把时长写进 Document.metadata，供 allowlist / prompt 展示。"""
    if game is None:
        return doc
    meta = dict(doc.metadata or {})
    meta["playtime_forever"] = game.playtime_forever
    meta["playtime_2weeks"] = game.playtime_2weeks
    meta["owned"] = True
    doc.metadata = meta
    return doc


LIBRARY_MODE_HINTS = {
    "tonight": (
        "策略=今晚从库存挑一款。优先近两周有时长的；若没有则按时长总榜。"
        "给一个主推荐，并点明依据是近两周时长或历史总时长。"
    ),
    "recent": (
        "策略=最近在玩。优先 playtime_2weeks>0 的库存游戏；"
        "说明近两周游玩情况。"
    ),
    "backlog": (
        "策略=库内从没玩过（playtime_forever=0）。"
        "只谈允许列表里的 backlog；没有就说明库存与知识库交集里没有未玩条目。"
    ),
    "owned": (
        "策略=在已有库存中按玩家约束筛选。"
        "只讨论允许列表（均为库内游戏）。"
    ),
    "unowned": (
        "策略=推荐还没买进库的游戏。"
        "允许列表应已排除库存；不要推荐玩家已拥有的。"
    ),
}
