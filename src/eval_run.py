"""按 data/eval/cases.jsonl 跑评测。

输出：
- data/eval/last_eval.jsonl —— 本轮明细（覆盖写）
- data/eval/last_eval_summary.json —— 本轮汇总（覆盖写）
- data/eval/history/runs.jsonl —— 每次汇总追加一行，用来看趋势
- data/eval/history/details/<run_id>.jsonl —— 该轮完整明细存档
- data/eval/traces/traces.jsonl —— 问答流水账（追加）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

from config import PROJECT_ROOT
from main import SteamGameAdvisor

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

APP_ID_LABEL_RE = re.compile(
    r"(?:App\s*ID|app[_ ]?id)\s*[:：]?\s*(\d{3,7})",
    re.IGNORECASE,
)
REFUSE_LIVE_HINTS = ("没有", "无法", "未接入", "实时", "暂不", "不能提供", "知识库没有", "做不到")
HISTORY_DIR = PROJECT_ROOT / "data" / "eval" / "history"
HISTORY_RUNS = HISTORY_DIR / "runs.jsonl"
HISTORY_DETAILS = HISTORY_DIR / "details"


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


def hit_ok(expect_ids: list[str], hit_ids: list[str]) -> bool | None:
    if not expect_ids:
        return None
    expect = set(expect_ids)
    return bool(expect & set(hit_ids))


def route_ok(expect: str, actual: str) -> bool:
    return expect == actual


def refuse_live_ok(answer: str) -> bool:
    return any(h in answer for h in REFUSE_LIVE_HINTS)


def extract_labeled_app_ids(answer: str) -> set[str]:
    return set(APP_ID_LABEL_RE.findall(answer))


def invent_fail(case: dict, answer: str, corpus_ids: set[str], hit_ids: list[str]) -> bool:
    """True = 判定为胡编。"""
    if case.get("must_not_invent"):
        for name in case.get("forbidden_names") or []:
            if name and name in answer:
                neg = any(
                    phrase in answer
                    for phrase in (
                        f"没有{name}",
                        f"没有《{name}",
                        f"不包含{name}",
                        f"库里没有",
                        f"知识库没有",
                        f"没有收录",
                        f"找不到{name}",
                        f"没有这款",
                        "并未收录",
                    )
                )
                if not neg and ("推荐" in answer or "App ID" in answer or "app_id" in answer.lower()):
                    return True
                if not neg and re.search(rf"{re.escape(name)}.{{0,20}}(可以|适合|推荐)", answer):
                    return True

    mentioned = extract_labeled_app_ids(answer)
    foreign = {i for i in mentioned if i not in corpus_ids}
    if foreign:
        return True

    if case.get("expect_route") in {"recommend", "library"} and hit_ids:
        in_corpus_mentioned = {i for i in mentioned if i in corpus_ids}
        if in_corpus_mentioned - set(hit_ids):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Steam Game Advisor 评测")
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "cases.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "last_eval.jsonl",
    )
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题，0 表示全部")
    parser.add_argument("--label", type=str, default="", help="本轮备注，如 grounding-v1")
    parser.add_argument(
        "--multi-query",
        action="store_true",
        help="启用多重查询（expand_queries + 多路 RRF），覆盖 config 默认",
    )
    parser.add_argument(
        "--mmr",
        action="store_true",
        help="启用 MMR 游戏级重排（RRF 候选池内多样化），覆盖 config 默认",
    )
    parser.add_argument(
        "--detail-name-boost",
        action="store_true",
        help="详情题启用游戏名/别名精确匹配加分",
    )
    parser.add_argument(
        "--no-genre-filter",
        action="store_true",
        help="关闭非游戏 genre 过滤（默认开启）",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="强制重建向量索引",
    )
    args = parser.parse_args()

    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit("请先在 .env 设置 DEEPSEEK_API_KEY")

    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]

    advisor = SteamGameAdvisor()
    if args.multi_query:
        advisor.config.use_multi_query = True
    if args.mmr:
        advisor.config.use_mmr = True
    if args.detail_name_boost:
        advisor.config.detail_name_boost = True
    if args.no_genre_filter:
        advisor.config.exclude_non_game_genres = False
    advisor.initialize_system()
    advisor.build_knowledge_base(force_rebuild=args.rebuild_index)

    corpus_ids = {
        str(doc.metadata.get("app_id"))
        for doc in advisor.data_module.documents
        if doc.metadata.get("app_id")
    }

    rows = []
    counters = Counter()
    scored = Counter()

    print(f"评测集: {args.cases} 共 {len(cases)} 题\n")

    for case in cases:
        cid = case["id"]
        question = case["question"]
        print(f"[{cid}] {question}")

        route = advisor.generation_module.query_router(question)
        rewritten = question
        query_variants = [question]
        if route in {"recommend", "detail"}:
            if advisor.config.use_multi_query:
                query_variants = advisor.generation_module.expand_queries(
                    question, n=advisor.config.multi_query_count
                )
                rewritten = " | ".join(query_variants)
            else:
                rewritten = advisor.generation_module.query_rewrite(question)
                query_variants = [rewritten]

        filters = advisor._extract_filters_from_query(question)
        if route == "trending":
            answer = advisor.generation_module.trending_unavailable_answer()
            hits = []
            hit_ids = []
        else:
            if route == "library":
                from rag_modules.library_profile import detect_library_mode

                lib_mode = detect_library_mode(question)
                if lib_mode in {"tonight", "recent", "backlog"}:
                    docs = advisor._select_library_docs(lib_mode)
                    if not docs:
                        chunks = advisor._retrieve_chunks(
                            route, question, rewritten, query_variants, filters
                        )
                        docs = advisor._apply_library_constraint(
                            question, chunks, mode="owned"
                        )
                else:
                    chunks = advisor._retrieve_chunks(
                        route, question, rewritten, query_variants, filters
                    )
                    docs = advisor._apply_library_constraint(
                        question, chunks, mode=lib_mode
                    )
            else:
                chunks = advisor._retrieve_chunks(
                    route, question, rewritten, query_variants, filters
                )
                docs = advisor.data_module.get_parent_documents(chunks)
            hits = [
                {
                    "name": d.metadata.get("name"),
                    "name_cn": d.metadata.get("name_cn"),
                    "app_id": str(d.metadata.get("app_id", "")),
                }
                for d in docs
            ]
            hit_ids = [h["app_id"] for h in hits if h.get("app_id")]

            if not docs:
                answer = "没有找到相关游戏档案。可以换关键词，或检查 data/processed 是否已放入语料。"
            elif route == "library":
                from rag_modules.library_profile import detect_library_mode

                answer = advisor.generation_module.generate_library_answer(
                    question, docs, library_mode=detect_library_mode(question)
                )
            elif route == "detail":
                answer = advisor.generation_module.generate_detail_answer(question, docs)
            else:
                answer = advisor.generation_module.generate_recommend_answer(question, docs)

        advisor.trace_logger.append(
            question=question,
            route=route,
            rewritten_query=rewritten,
            filters=filters,
            hits=hits,
            answer=answer,
            stream=False,
            notes=f"eval:{cid}",
        )

        r_ok = route_ok(case["expect_route"], route)
        h_ok = hit_ok(case.get("expect_app_ids") or [], hit_ids)
        refuse_ok = None
        if case.get("expect_refuse_live"):
            refuse_ok = refuse_live_ok(answer)
        invented = invent_fail(case, answer, corpus_ids, hit_ids)

        scored["route"] += 1
        counters["route_ok"] += int(r_ok)
        if h_ok is not None:
            scored["hit"] += 1
            counters["hit_ok"] += int(h_ok)
        if refuse_ok is not None:
            scored["refuse"] += 1
            counters["refuse_ok"] += int(refuse_ok)
        scored["invent"] += 1
        counters["invent_fail"] += int(invented)

        row = {
            "id": cid,
            "question": question,
            "expect_route": case["expect_route"],
            "actual_route": route,
            "route_ok": r_ok,
            "expect_app_ids": case.get("expect_app_ids") or [],
            "hit_ids": hit_ids,
            "hit_ok": h_ok,
            "refuse_live_ok": refuse_ok,
            "invent_fail": invented,
            "rewritten_query": rewritten,
            "answer": answer,
        }
        rows.append(row)
        flags = []
        flags.append("路由OK" if r_ok else "路由FAIL")
        if h_ok is not None:
            flags.append("命中OK" if h_ok else "命中FAIL")
        if refuse_ok is not None:
            flags.append("拒答OK" if refuse_ok else "拒答FAIL")
        flags.append("胡编FAIL" if invented else "胡编OK")
        print("  " + " ".join(flags) + f"  hits={hit_ids}\n")

    now = datetime.now(timezone.utc).astimezone()
    run_id = now.strftime("%Y%m%d-%H%M%S")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    HISTORY_DETAILS.mkdir(parents=True, exist_ok=True)
    detail_archive = HISTORY_DETAILS / f"{run_id}.jsonl"
    detail_archive.write_text(args.out.read_text(encoding="utf-8"), encoding="utf-8")

    def rate(num: int, den: int) -> dict:
        return {
            "ok": num,
            "total": den,
            "ratio": (round(num / den, 4) if den else None),
        }

    summary = {
        "run_id": run_id,
        "ts": now.isoformat(timespec="seconds"),
        "label": args.label or None,
        "cases": str(args.cases),
        "n": len(rows),
        "route_accuracy": rate(counters["route_ok"], scored["route"]),
        "retrieval_hit_rate": rate(counters["hit_ok"], scored["hit"]),
        "live_refuse_rate": rate(counters["refuse_ok"], scored["refuse"]),
        "invent_rate": rate(counters["invent_fail"], scored["invent"]),
        "detail_latest": str(args.out),
        "detail_archive": str(detail_archive),
    }
    summary_path = args.out.with_name("last_eval_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_RUNS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    def pct(num: int, den: int) -> str:
        if den == 0:
            return "n/a"
        return f"{num}/{den} = {num / den:.0%}"

    print("=" * 50)
    print(f"汇总 run_id={run_id}" + (f" label={args.label}" if args.label else ""))
    print(f"  路由准确率: {pct(counters['route_ok'], scored['route'])}")
    print(f"  检索命中率: {pct(counters['hit_ok'], scored['hit'])}")
    print(f"  实时拒答率: {pct(counters['refuse_ok'], scored['refuse'])}")
    print(f"  胡编率:     {pct(counters['invent_fail'], scored['invent'])}（越低越好）")
    print(f"本轮明细: {args.out}")
    print(f"本轮汇总: {summary_path}")
    print(f"历史趋势: {HISTORY_RUNS}（追加）")
    print(f"本轮存档: {detail_archive}")


if __name__ == "__main__":
    main()
