"""混合检索：向量 + BM25，RRF 融合，元数据过滤；支持多重查询与 MMR 重排。"""

import hashlib
import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import jieba
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# 口语 → 意图面（同面合并，如「构筑」「卡牌」算同一约束）
INTENT_FACET_OF: Dict[str, str] = {
    "肉鸽": "肉鸽",
    "roguelike": "肉鸽",
    "rogue": "肉鸽",
    "卡牌": "卡牌",
    "构筑": "卡牌",
    "吃鸡": "大逃杀",
    "大逃杀": "大逃杀",
    "视觉小说": "视觉小说",
    "种田": "种田",
    "联机": "联机",
    "合作": "联机",
    "开黑": "联机",
    "恐怖": "恐怖",
    "射击": "射击",
    "开放世界": "开放世界",
}

# 意图面 → Steam user_tags 匹配片段（小写）；用于召回与多约束覆盖计分
INTENT_TAG_PATTERNS: Dict[str, List[str]] = {
    "肉鸽": ["类 rogue", "轻度 rogue", "动作类 rogue", "牌组构建式类 rogue", "rogue"],
    "卡牌": ["卡牌游戏", "卡牌战斗", "牌组构建", "牌组构建式类 rogue", "卡牌"],
    "大逃杀": ["大逃杀"],
    "视觉小说": ["视觉小说"],
    "种田": ["农场模拟", "生活模拟", "种植", "农场"],
    "联机": ["在线合作", "合作", "多人", "同屏"],
    "恐怖": ["恐怖", "生存恐怖", "心理恐怖"],
    "射击": ["射击", "第一人称射击", "第三人称射击"],
    "开放世界": ["开放世界"],
}

# 兼容旧名：口语 token → 标签同义词（由 INTENT_* 派生）
TAG_QUERY_SYNONYMS: Dict[str, List[str]] = {
    tok: INTENT_TAG_PATTERNS[facet]
    for tok, facet in INTENT_FACET_OF.items()
    if facet in INTENT_TAG_PATTERNS
}

# 口语游戏名 → 检索补强词（按需加，不写题材 if）
GAME_QUERY_HINTS: Dict[str, str] = {
    "小丑牌": "Balatro 小丑牌 卡牌 肉鸽 扑克",
    "balatro": "Balatro 小丑牌 卡牌 肉鸽",
}

# 多意图覆盖：标签字面分 / RRF 加分（命中面数 ≥2 才启用）
_MULTI_INTENT_LITERAL_PER = 12.0
_MULTI_INTENT_LITERAL_FULL = 16.0
_MULTI_INTENT_RRF_PER = 0.012
_MULTI_INTENT_RRF_FULL = 0.012

# 与菜谱 C9 同思路：场景向中文停用词（不引第三方包）
_CHINESE_STOPWORDS = set(
    """
的 了 和 是 在 我 有 就 不 也 都 还 这 那 一 个 与 及 等 上 下 中 为 以 于 从 把 被 让 使 又 而 但 或
什么 怎么 如何 哪些 哪个 哪里 谁 多少 几 你 他 她 它 我们 他们 她们 它们
请问 请 想 要 需要 能 可以 应该 会 啊 呢 吧 嘛 吗 哦 呀 哈
之 其 此 该 即 各 每 些 种 类 时 后 前 里 外 内 间 已经 正在 一些 一下
推荐 一下 有没有 求 找 玩玩 玩法 游戏
""".split()
)

# 玩法/品类词：避免被 jieba 拆碎（如「大/逃/杀」）
_GAME_DICT_WORDS = (
    "大逃杀",
    "吃鸡",
    "肉鸽",
    "类魂",
    "魂类",
    "视觉小说",
    "开放世界",
    "卡牌构筑",
    "联机",
    "开黑",
    "银河战士恶魔城",
    "类银河战士恶魔城",
    "Roguelike",
    "roguelike",
)

_jieba_dict_loaded = False


def _ensure_jieba_dict() -> None:
    global _jieba_dict_loaded
    if _jieba_dict_loaded:
        return
    for word in _GAME_DICT_WORDS:
        jieba.add_word(word)
    _jieba_dict_loaded = True


def tokenize_chinese(text: str) -> List[str]:
    """jieba 精确分词 + 停用词 / 空白 / 单字符过滤（对齐菜谱 C9）。"""
    if not text:
        return []
    _ensure_jieba_dict()
    tokens = jieba.lcut(text)
    return [
        t
        for t in tokens
        if t.strip()
        and t not in _CHINESE_STOPWORDS
        and not t.isspace()
        and len(t.strip()) > 1
    ]


class RetrievalOptimizationModule:
    def __init__(
        self,
        vectorstore: FAISS,
        chunks: List[Document],
        *,
        use_mmr: bool = False,
        mmr_lambda: float = 0.7,
        mmr_pool_size: int = 24,
        use_section_weights: bool = True,
        use_tag_breadth_penalty: bool = True,
        tag_breadth_free: int = 2,
        tag_breadth_alpha: float = 0.1,
        use_user_tag_overlap_boost: bool = False,
        user_tag_overlap_bonus: float = 0.012,
        user_tag_overlap_max: int = 4,
        use_tag_literal_recall: bool = False,
        tag_literal_min_len: int = 2,
    ):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.mmr_pool_size = mmr_pool_size
        self.use_section_weights = use_section_weights
        self.use_tag_breadth_penalty = use_tag_breadth_penalty
        self.tag_breadth_free = tag_breadth_free
        self.tag_breadth_alpha = tag_breadth_alpha
        self.use_user_tag_overlap_boost = use_user_tag_overlap_boost
        self.user_tag_overlap_bonus = user_tag_overlap_bonus
        self.user_tag_overlap_max = max(0, int(user_tag_overlap_max))
        self.use_tag_literal_recall = use_tag_literal_recall
        self.tag_literal_min_len = max(1, int(tag_literal_min_len))
        self.setup_retrievers()

    def setup_retrievers(self):
        logger.info("正在设置检索器...")
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        # 自建 BM25Okapi + jieba（LangChain 默认按空格分词，中文几乎失效）
        self._bm25_docs = list(self.chunks)
        tokenized = [tokenize_chinese(d.page_content) for d in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized) if self._bm25_docs else None
        avg_tok = (
            sum(len(t) for t in tokenized) / max(1, len(tokenized)) if tokenized else 0.0
        )
        logger.info(
            "BM25(jieba+停用词) 就绪：文档 %s，平均 token %.1f",
            len(self._bm25_docs),
            avg_tok,
        )
        self._tag_postings = self._build_tag_postings()
        logger.info(
            "标签字面倒排：%s 个标签，覆盖 %s 款游戏（开关=%s）",
            len(self._tag_postings),
            len({aid for docs in self._tag_postings.values() for aid, _, _ in docs}),
            "开" if self.use_tag_literal_recall else "关",
        )
        logger.info("检索器设置完成")

    def _build_tag_postings(
        self,
    ) -> Dict[str, List[Tuple[str, Document, int]]]:
        """tag_lower -> [(app_id, doc, review_count), ...]；每游戏每标签一条。"""
        # app_id -> (preferred doc, review_count, tags)
        per_game: Dict[str, Tuple[Document, int, List[str]]] = {}
        for doc in self.chunks:
            meta = doc.metadata or {}
            aid = str(meta.get("app_id") or "").strip()
            if not aid:
                continue
            tags = meta.get("user_tags") or []
            if not isinstance(tags, list):
                tags = [tags]
            tags = [str(t).strip() for t in tags if str(t).strip()]
            if not tags:
                continue
            try:
                reviews = int(float(meta.get("review_count") or 0))
            except (TypeError, ValueError):
                reviews = 0
            section = str(meta.get("section") or "")
            prev = per_game.get(aid)
            # 优先保留「类型与标签」块
            if prev is None or (
                section == "类型与标签" and str(prev[0].metadata.get("section") or "") != "类型与标签"
            ):
                per_game[aid] = (doc, reviews, tags)

        postings: Dict[str, List[Tuple[str, Document, int]]] = defaultdict(list)
        min_len = self.tag_literal_min_len
        for aid, (doc, reviews, tags) in per_game.items():
            seen: set[str] = set()
            for tag in tags:
                key = tag.lower()
                if len(key) < min_len or key in seen:
                    continue
                seen.add(key)
                postings[key].append((aid, doc, reviews))
        return dict(postings)

    def _query_intent_facets(self, query: str) -> List[str]:
        """问句激活的意图面（去重，保序）。"""
        q = query or ""
        tokens = {t.lower() for t in tokenize_chinese(q)}
        facets: List[str] = []
        seen: set[str] = set()
        for tok in tokens:
            facet = INTENT_FACET_OF.get(tok)
            if facet and facet not in seen and facet in INTENT_TAG_PATTERNS:
                seen.add(facet)
                facets.append(facet)
        # 整句兜底：分词漏掉但词表键在原文里
        low = q.lower()
        for tok, facet in INTENT_FACET_OF.items():
            if facet in seen or facet not in INTENT_TAG_PATTERNS:
                continue
            if tok in low or tok in q:
                seen.add(facet)
                facets.append(facet)
        return facets

    @staticmethod
    def _doc_tag_blob(doc: Document) -> str:
        tags = doc.metadata.get("user_tags") or []
        return " ".join(str(t).lower() for t in tags)

    def _facet_matched_in_blob(self, facet: str, blob: str) -> bool:
        for pat in INTENT_TAG_PATTERNS.get(facet, []):
            if pat.lower() in blob:
                return True
        return False

    def _multi_intent_coverage(self, facets: Sequence[str], doc: Document) -> int:
        if not facets:
            return 0
        blob = self._doc_tag_blob(doc)
        return sum(1 for f in facets if self._facet_matched_in_blob(f, blob))

    def _query_tag_hits(self, query: str) -> List[str]:
        """问句命中的标签：整词/子串/意图面同义词。长标签优先。"""
        q = (query or "").strip().lower()
        if not q or not self._tag_postings:
            return []
        tokens = {t.lower() for t in tokenize_chinese(query)}
        hits: set[str] = set()

        for tag in self._tag_postings:
            if tag in tokens or tag in q:
                hits.add(tag)
                continue
            for tok in tokens:
                if len(tok) < 2:
                    continue
                if tok in tag or tag in tok:
                    hits.add(tag)
                    break

        for tok in tokens:
            for syn in TAG_QUERY_SYNONYMS.get(tok, []):
                key = syn.lower()
                if key in self._tag_postings:
                    hits.add(key)
                else:
                    for tag in self._tag_postings:
                        if key in tag or tag in key:
                            hits.add(tag)

        return sorted(hits, key=len, reverse=True)

    def _augment_query_for_search(self, query: str) -> str:
        """游戏名别名 + 各意图面的代表标签，扩 BM25/向量召回。"""
        q = query or ""
        low = q.lower()
        extras: List[str] = []
        for tip, hint in GAME_QUERY_HINTS.items():
            if tip in low or tip in q:
                extras.append(hint)
        for facet in self._query_intent_facets(q):
            # 每面取前两个模式当检索词，避免塞太长
            extras.extend(INTENT_TAG_PATTERNS.get(facet, [])[:2])
        if not extras:
            return q
        # 去重保序
        seen: set[str] = set()
        uniq_ex: List[str] = []
        for x in extras:
            if x not in seen:
                seen.add(x)
                uniq_ex.append(x)
        return f"{q} {' '.join(uniq_ex)}"

    def _tag_literal_search(self, query: str, k: int) -> List[Document]:
        """按 user_tags 字面命中召回；多意图面同时出现时按覆盖面数加权。"""
        if not self.use_tag_literal_recall or k <= 0:
            return []
        matched_tags = self._query_tag_hits(query)
        if not matched_tags:
            return []
        facets = self._query_intent_facets(query)

        best: Dict[str, Tuple[float, int, Document]] = {}
        for tag in matched_tags:
            for aid, doc, reviews in self._tag_postings.get(tag, []):
                score = float(len(tag))
                prev = best.get(aid)
                if prev is None:
                    best[aid] = (score, reviews, doc)
                else:
                    best[aid] = (prev[0] + score, max(prev[1], reviews), prev[2])

        if len(facets) >= 2:
            n = len(facets)
            for aid, (score, reviews, doc) in list(best.items()):
                cov = self._multi_intent_coverage(facets, doc)
                if cov <= 0:
                    continue
                score += cov * _MULTI_INTENT_LITERAL_PER
                if cov >= n:
                    score += _MULTI_INTENT_LITERAL_FULL
                best[aid] = (score, reviews, doc)

        ranked = sorted(
            best.items(),
            key=lambda x: (x[1][0], x[1][1]),
            reverse=True,
        )
        return [doc for _, (_, _, doc) in ranked[:k]]

    def _apply_multi_intent_tag_boost(
        self, query: str, docs: List[Document]
    ) -> List[Document]:
        """问句激活 ≥2 个意图面时，按游戏覆盖面数给 RRF 加分（题材无关）。"""
        if not docs:
            return docs
        for doc in docs:
            doc.metadata.pop("multi_intent_boost", None)
            doc.metadata.pop("multi_intent_coverage", None)
        facets = self._query_intent_facets(query)
        if len(facets) < 2:
            return docs
        n = len(facets)
        for doc in docs:
            cov = self._multi_intent_coverage(facets, doc)
            if cov <= 0:
                continue
            bonus = cov * _MULTI_INTENT_RRF_PER
            if cov >= n:
                bonus += _MULTI_INTENT_RRF_FULL
            base = float(doc.metadata.get("rrf_score", 0.0))
            doc.metadata["rrf_score"] = base + bonus
            doc.metadata["multi_intent_boost"] = bonus
            doc.metadata["multi_intent_coverage"] = cov
        return sorted(
            docs, key=lambda d: float(d.metadata.get("rrf_score", 0.0)), reverse=True
        )

    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """关键词检索；分数 ≤ 0 丢弃，避免无匹配时按语料顺序灌噪声。"""
        if self._bm25 is None or not self._bm25_docs:
            return []
        tokens = tokenize_chinese(query)
        if not tokens:
            logger.debug("BM25 问句分词为空，跳过: %s", query)
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: List[Document] = []
        for i in order:
            if float(scores[i]) <= 0:
                break
            out.append(self._bm25_docs[i])
            if len(out) >= k:
                break
        return out

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        return self.multi_query_search([query], top_k=top_k)

    def multi_query_search(self, queries: Sequence[str], top_k: int = 3) -> List[Document]:
        ranked_lists: List[List[Document]] = []
        uniq: List[str] = []
        for q in queries:
            q = (q or "").strip()
            if q and q not in uniq:
                uniq.append(q)
        if not uniq:
            return []

        # 过滤场景会传入较大 top_k；底层检索器默认 k=5，必须同步放大否则候选池打不满
        fetch_k = max(int(top_k), 8)
        if self.use_mmr:
            fetch_k = max(fetch_k, self.mmr_pool_size)
        for q in uniq:
            q_search = self._augment_query_for_search(q)
            ranked_lists.append(self.vectorstore.similarity_search(q_search, k=fetch_k))
            ranked_lists.append(self._bm25_search(q_search, fetch_k))
            if self.use_tag_literal_recall:
                ranked_lists.append(self._tag_literal_search(q, fetch_k))

        fused = self._rrf_fuse(ranked_lists)
        if self.use_section_weights:
            fused = self._apply_section_weights(fused)
        if self.use_tag_breadth_penalty:
            fused = self._apply_tag_breadth_penalty(fused)
        if self.use_user_tag_overlap_boost:
            fused = self._apply_user_tag_overlap_boost(uniq, fused)
        fused = self._apply_multi_intent_tag_boost(uniq[0], fused)
        if self.use_mmr:
            diversified = self._game_level_mmr(uniq[0], fused, top_k=top_k)
        else:
            diversified = self._diversify_by_game(fused, top_k=top_k)
        logger.info(
            "多重查询融合完成: %s 条查询, %s 路列表, 合并后 %s 条, %s后返回 %s",
            len(uniq),
            len(ranked_lists),
            len(fused),
            "MMR" if self.use_mmr else "去重游戏",
            len(diversified),
        )
        return diversified

    @staticmethod
    def _apply_section_weights(docs: List[Document]) -> List[Document]:
        """按切块 section 乘权重后重排（简介/标签抬高，游玩方式/配置压低）。"""
        for doc in docs:
            weight = float(doc.metadata.get("section_weight", 1.0))
            if "section_weight" not in doc.metadata:
                section = str(
                    doc.metadata.get("section")
                    or doc.metadata.get("二级标题")
                    or ""
                ).strip()
                # 延迟导入避免循环；缺省 1.0
                try:
                    from config import SECTION_WEIGHTS

                    weight = float(SECTION_WEIGHTS.get(section, 1.0))
                except Exception:
                    weight = 1.0
                doc.metadata["section_weight"] = weight
            base = float(doc.metadata.get("rrf_score", 0.0))
            doc.metadata["rrf_score"] = base * weight
        return sorted(
            docs, key=lambda d: float(d.metadata.get("rrf_score", 0.0)), reverse=True
        )

    @staticmethod
    def _query_terms_for_tag_overlap(queries: Sequence[str]) -> tuple[str, set[str]]:
        parts: set[str] = set()
        for q in queries:
            text = (q or "").strip().lower()
            if not text:
                continue
            parts.add(text)
            for piece in re.split(r"[，,、；;。！？!?/\s]+", text):
                piece = piece.strip()
                if len(piece) >= 2:
                    parts.add(piece)
        blob = " ".join(sorted(parts, key=len, reverse=True))
        return blob, parts

    @staticmethod
    def _count_user_tag_overlaps(
        queries: Sequence[str], tags: Sequence[Any]
    ) -> int:
        blob, parts = RetrievalOptimizationModule._query_terms_for_tag_overlap(
            queries
        )
        if not blob:
            return 0
        matched = 0
        seen: set[str] = set()
        for raw in tags:
            tag = str(raw).strip()
            if len(tag) < 2:
                continue
            tl = tag.lower()
            if tl in seen:
                continue
            hit = tl in blob or any(
                len(p) >= 2 and (p in tl or tl in p) for p in parts
            )
            if hit:
                seen.add(tl)
                matched += 1
        return matched

    def _apply_user_tag_overlap_boost(
        self, queries: Sequence[str], docs: List[Document]
    ) -> List[Document]:
        """问句与 metadata.user_tags 字面重叠时，对 RRF 分做小幅加分。"""
        bonus = max(0.0, float(self.user_tag_overlap_bonus))
        cap = self.user_tag_overlap_max
        if bonus <= 0 or not docs:
            return docs
        for doc in docs:
            tags = doc.metadata.get("user_tags") or []
            if not isinstance(tags, list):
                tags = [tags]
            overlap = self._count_user_tag_overlaps(queries, tags)
            if cap:
                overlap = min(overlap, cap)
            if overlap <= 0:
                continue
            base = float(doc.metadata.get("rrf_score", 0.0))
            doc.metadata["user_tag_overlap"] = overlap
            doc.metadata["rrf_score"] = base + bonus * overlap
        return sorted(
            docs, key=lambda d: float(d.metadata.get("rrf_score", 0.0)), reverse=True
        )

    def _apply_tag_breadth_penalty(self, docs: List[Document]) -> List[Document]:
        """genres 越多，类型匹配越「泛」，对 RRF 分乘 <1 的因子。

        只用 genres，不用 categories：Steam 分类含成就/创意工坊等，几乎人人一堆，
        会误伤正常游戏并相对抬高「genres 很少但分类很多」的噪声款（如 BTD6）。

        factor = 1 / (1 + alpha * max(0, n - free))
        """
        free = max(0, int(self.tag_breadth_free))
        alpha = max(0.0, float(self.tag_breadth_alpha))
        for doc in docs:
            genres = doc.metadata.get("genres") or []
            if not isinstance(genres, list):
                genres = [genres]
            n = len([g for g in genres if str(g).strip()])
            factor = 1.0 / (1.0 + alpha * max(0, n - free))
            base = float(doc.metadata.get("rrf_score", 0.0))
            doc.metadata["tag_breadth_n"] = n
            doc.metadata["tag_breadth_factor"] = factor
            doc.metadata["rrf_score"] = base * factor
        return sorted(
            docs, key=lambda d: float(d.metadata.get("rrf_score", 0.0)), reverse=True
        )

    @staticmethod
    def diversify_by_game(docs: List[Document], top_k: int) -> List[Document]:
        return RetrievalOptimizationModule._diversify_by_game(docs, top_k)

    @staticmethod
    def _diversify_by_game(docs: List[Document], top_k: int) -> List[Document]:
        """同一游戏多块时只取最高分的一块，避免 Top-K 被同一款占满。"""
        seen = set()
        out: List[Document] = []
        for doc in docs:
            key = str(doc.metadata.get("app_id") or doc.metadata.get("parent_id") or "")
            if not key:
                key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append(doc)
            if len(out) >= top_k:
                break
        return out

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _game_level_mmr(
        self, query: str, fused: List[Document], top_k: int
    ) -> List[Document]:
        """RRF 候选池内按游戏做 MMR，在相关性与多样性之间折中。"""
        best_by_game: Dict[str, Document] = {}
        game_order: List[str] = []
        for doc in fused:
            key = str(doc.metadata.get("app_id") or doc.metadata.get("parent_id") or "")
            if not key:
                key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            if key not in best_by_game:
                game_order.append(key)
                best_by_game[key] = doc
            elif doc.metadata.get("rrf_score", 0) > best_by_game[key].metadata.get(
                "rrf_score", 0
            ):
                best_by_game[key] = doc

        pool = min(self.mmr_pool_size, len(game_order))
        candidates = [best_by_game[k] for k in game_order[:pool]]
        if len(candidates) <= top_k:
            return candidates[:top_k]

        embeddings = self.vectorstore.embeddings
        q_vec = embeddings.embed_query(query)
        d_vecs = embeddings.embed_documents([d.page_content for d in candidates])

        rrf_scores = [float(d.metadata.get("rrf_score", 0.0)) for d in candidates]
        rmin, rmax = min(rrf_scores), max(rrf_scores)

        def relevance(i: int) -> float:
            rrf_norm = 1.0 if rmax == rmin else (rrf_scores[i] - rmin) / (rmax - rmin)
            return 0.5 * rrf_norm + 0.5 * self._cosine(q_vec, d_vecs[i])

        selected: List[Document] = []
        selected_idx: List[int] = []
        remaining = list(range(len(candidates)))
        lam = self.mmr_lambda

        while len(selected) < top_k and remaining:
            best_i = remaining[0]
            best_score = float("-inf")
            for i in remaining:
                rel = relevance(i)
                penalty = 0.0
                if selected_idx:
                    penalty = max(self._cosine(d_vecs[i], d_vecs[j]) for j in selected_idx)
                score = lam * rel - (1.0 - lam) * penalty
                if score > best_score:
                    best_score = score
                    best_i = i
            selected_idx.append(best_i)
            selected.append(candidates[best_i])
            remaining.remove(best_i)

        return selected

    def metadata_filtered_search(
        self, query: str, filters: Dict[str, Any], top_k: int = 5
    ) -> List[Document]:
        return self.metadata_filtered_multi_search([query], filters, top_k=top_k)

    def metadata_filtered_multi_search(
        self,
        queries: Sequence[str],
        filters: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Document]:
        # 过滤会丢掉候选，需略放大召回；但过大（接近全库）会引入弱匹配、打乱 RRF 序
        pool = max(top_k * 4, 12)
        if filters:
            pool = max(pool, 16)
        docs = self.multi_query_search(queries, top_k=pool)
        filtered_docs = []
        for doc in docs:
            if self._match_filters(doc.metadata, filters):
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break
        return filtered_docs

    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, expected in filters.items():
            if key == "price_max":
                price = metadata.get("price_cny")
                if price is None or float(price) > float(expected):
                    return False
                continue
            if key == "review_min":
                score = metadata.get("review_percentage")
                if score is None or float(score) < float(expected):
                    return False
                continue
            if key not in metadata:
                return False
            actual = metadata[key]
            if isinstance(actual, list):
                if isinstance(expected, list):
                    if not any(self._value_in_list(item, actual) for item in expected):
                        return False
                elif not self._value_in_list(expected, actual):
                    return False
            elif isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    @staticmethod
    def _value_in_list(value: Any, items: List[Any]) -> bool:
        text = str(value).lower()
        return any(text == str(item).lower() or text in str(item).lower() for item in items)

    def _rrf_rerank(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        k: int = 60,
    ) -> List[Document]:
        return self._rrf_fuse([vector_docs, bm25_docs], k=k)

    def _rrf_fuse(
        self, ranked_lists: Sequence[Sequence[Document]], k: int = 60
    ) -> List[Document]:
        doc_scores: Dict[str, float] = {}
        doc_objects: Dict[str, Document] = {}

        for docs in ranked_lists:
            for rank, doc in enumerate(docs):
                doc_id = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
                doc_objects[doc_id] = doc
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            doc = doc_objects[doc_id]
            doc.metadata["rrf_score"] = final_score
            reranked_docs.append(doc)
        return reranked_docs
