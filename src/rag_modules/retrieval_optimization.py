"""混合检索：向量 + BM25，RRF 融合，元数据过滤；支持多重查询与 MMR 重排。"""

import hashlib
import logging
import math
import re
from typing import Any, Dict, List, Sequence

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


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
        self.setup_retrievers()

    def setup_retrievers(self):
        logger.info("正在设置检索器...")
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks, k=5)
        logger.info("检索器设置完成")

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
        old_bm25_k = getattr(self.bm25_retriever, "k", 5)
        self.bm25_retriever.k = fetch_k
        try:
            for q in uniq:
                ranked_lists.append(self.vectorstore.similarity_search(q, k=fetch_k))
                ranked_lists.append(self.bm25_retriever.invoke(q))
        finally:
            self.bm25_retriever.k = old_bm25_k

        fused = self._rrf_fuse(ranked_lists)
        if self.use_section_weights:
            fused = self._apply_section_weights(fused)
        if self.use_tag_breadth_penalty:
            fused = self._apply_tag_breadth_penalty(fused)
        if self.use_user_tag_overlap_boost:
            fused = self._apply_user_tag_overlap_boost(uniq, fused)
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
