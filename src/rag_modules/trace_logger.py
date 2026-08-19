"""问答轨迹：路由、改写、命中与最终回答，按行追加 JSONL。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class TraceLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        question: str,
        route: str,
        rewritten_query: str,
        filters: Optional[Dict[str, Any]],
        hits: List[Dict[str, Any]],
        answer: str,
        stream: bool = False,
        notes: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        record = {
            "ts": timestamp
            or datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "question": question,
            "route": route,
            "rewritten_query": rewritten_query,
            "filters": filters or {},
            "hits": hits,
            "answer": answer,
            "stream": stream,
        }
        if notes:
            record["notes"] = notes
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
