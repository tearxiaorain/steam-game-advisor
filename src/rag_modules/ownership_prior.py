"""库存/好友拥有度先验：推荐检索时抬本人与多人重叠，压长尾。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# 好友向推荐：关键词闸门（主路由仍是 LLM；此处只决定要不要开拥有度偏置）
FRIEND_RECOMMEND_KEYWORDS: tuple[str, ...] = (
    "好友",
    "朋友",
    "室友",
    "开黑",
    "一起玩",
    "联机玩",
    "和同学",
    "和队友",
    "好友库",
    "朋友都有",
    "好友都有",
    "好友在玩",
    "朋友在玩",
    "合玩",
    "双人成行",  # 常见合玩表述，可与游戏名撞车，下方会再看语境
)


def detect_friend_recommend_intent(question: str) -> bool:
    """问法是否偏向「和好友/朋友一起」的推荐（关键词，非 LLM）。"""
    q = (question or "").strip()
    if not q:
        return False
    # 「双人成行」单独出现多为点名游戏，不算好友意图
    if "双人成行" in q and not any(
        k in q for k in ("好友", "朋友", "开黑", "一起", "联机", "合玩")
    ):
        return False
    return any(k in q for k in FRIEND_RECOMMEND_KEYWORDS)


@dataclass
class OwnershipPrior:
    """app_id -> 拥有情况。"""

    me: set[str] = field(default_factory=set)
    friend_owners: Dict[str, int] = field(default_factory=dict)

    def friend_count(self, app_id: str) -> int:
        return int(self.friend_owners.get(str(app_id), 0))

    def factor(
        self,
        app_id: str,
        *,
        me_factor: float = 1.25,
        multi_friend_factor: float = 1.18,
        duo_friend_factor: float = 1.08,
        longtail_factor: float = 0.72,
    ) -> float:
        aid = str(app_id)
        if aid in self.me:
            return float(me_factor)
        n = self.friend_count(aid)
        if n >= 3:
            return float(multi_friend_factor)
        if n == 2:
            return float(duo_friend_factor)
        if n == 1:
            return float(longtail_factor)
        return 1.0


def load_ownership_prior(
    me_owned_path: str,
    friends_dir: str,
    fallback_appids_path: str = "",
) -> OwnershipPrior:
    prior = OwnershipPrior()
    me_path = Path(me_owned_path) if me_owned_path else None
    if me_path and me_path.is_file():
        try:
            raw = json.loads(me_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                ids = raw.get("app_ids") or []
                if not ids and isinstance(raw.get("games"), list):
                    ids = [
                        g.get("app_id") or g.get("appid")
                        for g in raw["games"]
                        if isinstance(g, dict)
                    ]
            elif isinstance(raw, list):
                ids = raw
            else:
                ids = []
            prior.me = {str(i) for i in ids if i is not None}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("读取本人库存失败: %s", exc)

    if not prior.me and fallback_appids_path:
        fb = Path(fallback_appids_path)
        if fb.is_file():
            try:
                raw = json.loads(fb.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    ids = raw.get("app_ids") or raw.get("owned") or []
                else:
                    ids = raw if isinstance(raw, list) else []
                prior.me = {str(i) for i in ids if i is not None}
            except (json.JSONDecodeError, OSError):
                pass

    fdir = Path(friends_dir) if friends_dir else None
    counts: Dict[str, int] = {}
    if fdir and fdir.is_dir():
        for path in fdir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            ids = data.get("app_ids") or []
            if not ids and isinstance(data.get("games"), list):
                ids = [
                    g.get("app_id") or g.get("appid")
                    for g in data["games"]
                    if isinstance(g, dict)
                ]
            for aid in {str(i) for i in ids if i is not None}:
                counts[aid] = counts.get(aid, 0) + 1
    prior.friend_owners = counts

    logger.info(
        "拥有度先验: me=%s friend_apps=%s",
        len(prior.me),
        len(prior.friend_owners),
    )
    return prior


def filter_longtail_docs(
    docs: Sequence[Document],
    prior: OwnershipPrior,
    *,
    min_keep: int = 8,
) -> List[Document]:
    """推荐候选：去掉仅 1 个好友拥有的长尾；过少则回退原列表。"""
    if not docs:
        return []
    kept: List[Document] = []
    dropped = 0
    for doc in docs:
        aid = str(doc.metadata.get("app_id") or "")
        n = prior.friend_count(aid)
        # 本人有 / 多人有 / 不在好友图里（种子等）→ 保留；仅 1 好友 → 丢
        if aid in prior.me or n != 1:
            kept.append(doc)
        else:
            dropped += 1
    if len(kept) < max(1, int(min_keep)):
        logger.info(
            "长尾过滤后仅 %s 条（丢 %s），回退原候选",
            len(kept),
            dropped,
        )
        return list(docs)
    logger.info("长尾过滤: 保留 %s / 丢弃 %s", len(kept), dropped)
    return kept


def apply_ownership_bias(
    docs: Sequence[Document],
    prior: OwnershipPrior,
    *,
    me_factor: float = 1.25,
    multi_friend_factor: float = 1.18,
    duo_friend_factor: float = 1.08,
    longtail_factor: float = 0.72,
) -> List[Document]:
    """按拥有度乘到 rrf_score（无则按名次给伪分），再降序。"""
    if not docs:
        return []
    n = len(docs)
    out: List[Document] = []
    for i, doc in enumerate(docs):
        aid = str(doc.metadata.get("app_id") or "")
        factor = prior.factor(
            aid,
            me_factor=me_factor,
            multi_friend_factor=multi_friend_factor,
            duo_friend_factor=duo_friend_factor,
            longtail_factor=longtail_factor,
        )
        base = doc.metadata.get("rrf_score")
        if base is None:
            base = float(n - i)
        else:
            base = float(base)
        meta = dict(doc.metadata)
        meta["ownership_factor"] = factor
        meta["rrf_score"] = base * factor
        out.append(Document(page_content=doc.page_content, metadata=meta))
    out.sort(key=lambda d: float(d.metadata.get("rrf_score", 0.0)), reverse=True)
    return out
