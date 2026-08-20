"""标签受控词表：加载、分类、索引清洗、未登记扫描。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

logger = logging.getLogger(__name__)

DEFAULT_TAXONOMY_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "library"
    / "tag_taxonomy.json"
)


@dataclass
class TagTaxonomy:
    play_genres: Set[str] = field(default_factory=set)
    non_game_genres: Set[str] = field(default_factory=set)
    play_categories: Set[str] = field(default_factory=set)
    platform_features: Set[str] = field(default_factory=set)
    broad_terms: Set[str] = field(default_factory=set)
    category_aliases: Dict[str, str] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path | str | None = None) -> "TagTaxonomy":
        tax_path = Path(path) if path else DEFAULT_TAXONOMY_PATH
        if not tax_path.exists():
            logger.warning("标签词表不存在: %s，使用空词表", tax_path)
            return cls(path=tax_path)
        raw = json.loads(tax_path.read_text(encoding="utf-8"))
        return cls(
            play_genres={str(x) for x in raw.get("play_genres") or []},
            non_game_genres={str(x) for x in raw.get("non_game_genres") or []},
            play_categories={str(x) for x in raw.get("play_categories") or []},
            platform_features={str(x) for x in raw.get("platform_features") or []},
            broad_terms={str(x) for x in raw.get("broad_terms") or []},
            category_aliases={
                str(k): str(v) for k, v in (raw.get("category_aliases") or {}).items()
            },
            path=tax_path,
        )

    def normalize_category(self, name: str) -> str:
        name = str(name or "").strip()
        return self.category_aliases.get(name, name)

    def classify_genre(self, name: str) -> str:
        name = str(name or "").strip()
        if not name:
            return "empty"
        if name in self.non_game_genres:
            return "non_game"
        if name in self.play_genres:
            return "play"
        return "unknown"

    def classify_category(self, name: str) -> str:
        name = self.normalize_category(name)
        if not name:
            return "empty"
        if name in self.play_categories:
            return "play"
        if name in self.platform_features:
            return "platform"
        return "unknown"

    def play_categories_only(self, categories: Sequence[Any]) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for item in categories or []:
            name = self.normalize_category(str(item))
            if self.classify_category(name) != "play":
                continue
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def play_genres_only(self, genres: Sequence[Any]) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for item in genres or []:
            name = str(item).strip()
            if self.classify_genre(name) != "play":
                continue
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def scrub_genre_section_text(self, text: str, metadata: Dict[str, Any] | None = None) -> str:
        """索引用：类型与标签块只保留玩法 genres + 玩法 categories。"""
        meta = metadata or {}
        genres = self.play_genres_only(meta.get("genres") or [])
        if not genres:
            # 正文里的「类型:」行作兜底
            m = re.search(r"类型:\s*(.+)", text)
            if m:
                raw = [x.strip() for x in m.group(1).split(",") if x.strip()]
                genres = self.play_genres_only(raw) or raw
        cats = self.play_categories_only(meta.get("categories") or [])

        lines = [line for line in text.splitlines() if line.startswith("#")]
        lines.append("类型: " + (", ".join(genres) if genres else "（无）"))
        if cats:
            lines.append("分类: " + ", ".join(cats))
        return "\n".join(lines).strip()

    def scan_documents(self, documents: Iterable[Any]) -> Dict[str, Any]:
        unknown_genres: Dict[str, int] = {}
        unknown_categories: Dict[str, int] = {}
        for doc in documents:
            meta = getattr(doc, "metadata", None) or {}
            for g in meta.get("genres") or []:
                name = str(g).strip()
                if self.classify_genre(name) == "unknown":
                    unknown_genres[name] = unknown_genres.get(name, 0) + 1
            for c in meta.get("categories") or []:
                name = self.normalize_category(str(c))
                if self.classify_category(name) == "unknown":
                    unknown_categories[name] = unknown_categories.get(name, 0) + 1
        return {
            "unknown_genres": dict(sorted(unknown_genres.items(), key=lambda x: (-x[1], x[0]))),
            "unknown_categories": dict(
                sorted(unknown_categories.items(), key=lambda x: (-x[1], x[0]))
            ),
            "taxonomy_path": str(self.path) if self.path else None,
            "counts": {
                "play_genres": len(self.play_genres),
                "non_game_genres": len(self.non_game_genres),
                "play_categories": len(self.play_categories),
                "platform_features": len(self.platform_features),
            },
        }


_TAXONOMY: TagTaxonomy | None = None


def get_taxonomy(path: Path | str | None = None, reload: bool = False) -> TagTaxonomy:
    global _TAXONOMY
    if _TAXONOMY is None or reload or path is not None:
        _TAXONOMY = TagTaxonomy.load(path)
    return _TAXONOMY
