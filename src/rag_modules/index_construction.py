"""索引构建：嵌入与 FAISS 持久化。"""

import logging
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class IndexConstructionModule:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        index_save_path: str = "./vector_index",
    ):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()

    def setup_embeddings(self):
        logger.info("正在初始化嵌入模型: %s", self.model_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("嵌入模型初始化完成")

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        logger.info("正在构建 FAISS 向量索引...")
        if not chunks:
            raise ValueError("文档块列表不能为空")
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )
        logger.info("向量索引构建完成，包含 %s 个向量", len(chunks))
        return self.vectorstore

    def add_documents(self, new_chunks: List[Document]):
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")
        logger.info("正在添加 %s 个新文档到索引...", len(new_chunks))
        self.vectorstore.add_documents(new_chunks)

    def save_index(self):
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(self.index_save_path)
        logger.info("向量索引已保存到: %s", self.index_save_path)

    def load_index(self):
        if not self.embeddings:
            self.setup_embeddings()
        if not Path(self.index_save_path).exists():
            logger.info("索引路径不存在: %s，将构建新索引", self.index_save_path)
            return None
        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("向量索引已从 %s 加载", self.index_save_path)
            return self.vectorstore
        except Exception as exc:
            logger.warning("加载向量索引失败: %s，将构建新索引", exc)
            return None

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")
        return self.vectorstore.similarity_search(query, k=k)
