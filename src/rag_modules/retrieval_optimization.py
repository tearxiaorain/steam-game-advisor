"""混合检索：向量 + BM25，RRF 融合，元数据过滤。"""

import hashlib
import logging
from typing import Any, Dict, List

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks
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
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]

    def metadata_filtered_search(
        self, query: str, filters: Dict[str, Any], top_k: int = 5
    ) -> List[Document]:
        docs = self.hybrid_search(query, top_k * 3)
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
        doc_scores = {}
        doc_objects = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            doc_objects[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            doc_id = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            doc_objects[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            doc = doc_objects[doc_id]
            doc.metadata["rrf_score"] = final_score
            reranked_docs.append(doc)

        logger.info(
            "RRF 重排完成: 向量 %s 条, BM25 %s 条, 合并后 %s 条",
            len(vector_docs),
            len(bm25_docs),
            len(reranked_docs),
        )
        return reranked_docs
