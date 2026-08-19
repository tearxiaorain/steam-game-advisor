"""数据准备：加载游戏 Markdown，按标题切分子块。"""

import ast
import hashlib
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)

SKIP_FILENAMES = {"readme.md", ".gitkeep"}


class DataPreparationModule:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.parent_child_map: Dict[str, str] = {}

    def load_documents(self) -> List[Document]:
        logger.info("正在从 %s 加载文档...", self.data_path)
        documents = []
        data_root = Path(self.data_path)
        if not data_root.exists():
            raise FileNotFoundError(f"数据路径不存在: {self.data_path}")

        for md_file in sorted(data_root.rglob("*.md")):
            if md_file.name.lower() in SKIP_FILENAMES:
                continue
            try:
                content = md_file.read_text(encoding="utf-8")
                try:
                    relative_path = md_file.resolve().relative_to(data_root.resolve()).as_posix()
                except ValueError:
                    relative_path = md_file.as_posix()
                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
                body, extra_meta = self._split_front_matter(content)
                metadata = {
                    "source": str(md_file),
                    "parent_id": parent_id,
                    "doc_type": "parent",
                }
                metadata.update(extra_meta)
                self._fill_identity_metadata(metadata, md_file, body)
                documents.append(Document(page_content=body, metadata=metadata))
            except Exception as exc:
                logger.warning("读取文件 %s 失败: %s", md_file, exc)

        self.documents = documents
        logger.info("成功加载 %s 个文档", len(documents))
        return documents

    def _split_front_matter(self, content: str) -> tuple[str, Dict[str, Any]]:
        if not content.startswith("---"):
            return content, {}
        parts = content.split("---", 2)
        if len(parts) < 3:
            return content, {}
        meta: Dict[str, Any] = {}
        for raw_line in parts[1].splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = self._parse_scalar(value.strip())
        return parts[2].lstrip("\n"), meta

    @staticmethod
    def _parse_scalar(value: str) -> Any:
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except (SyntaxError, ValueError):
                return [item.strip() for item in value[1:-1].split(",") if item.strip()]
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value.strip('"').strip("'")

    def _fill_identity_metadata(self, metadata: Dict[str, Any], md_file: Path, body: str):
        heading = re.search(
            r"^#\s+(.+?)(?:（app_id=([^）]+)）|\(app_id=([^)]+)\))?\s*$",
            body,
            flags=re.MULTILINE,
        )
        if heading:
            metadata.setdefault("name", heading.group(1).strip())
            app_id = heading.group(2) or heading.group(3)
            if app_id:
                metadata.setdefault("app_id", str(app_id).strip())
        if "app_id" not in metadata and md_file.stem.isdigit():
            metadata["app_id"] = md_file.stem
        metadata.setdefault("name", md_file.stem)
        metadata.setdefault("app_id", md_file.stem)
        metadata["app_id"] = str(metadata["app_id"])
        for list_key in ("genres", "tags", "categories", "platforms", "supported_languages", "developers", "publishers"):
            value = metadata.get(list_key)
            if isinstance(value, str):
                metadata[list_key] = [item.strip() for item in value.split(",") if item.strip()]

    def chunk_documents(self) -> List[Document]:
        logger.info("正在按 Markdown 标题分块...")
        if not self.documents:
            raise ValueError("请先加载文档")
        chunks = self._markdown_header_split()
        for i, chunk in enumerate(chunks):
            chunk.metadata.setdefault("chunk_id", str(uuid.uuid4()))
            chunk.metadata["batch_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        self.chunks = chunks
        logger.info("分块完成，共 %s 个 chunk", len(chunks))
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "主标题"),
                ("##", "二级标题"),
                ("###", "三级标题"),
            ],
            strip_headers=False,
        )
        all_chunks = []
        for doc in self.documents:
            try:
                md_chunks = splitter.split_text(doc.page_content)
                if len(md_chunks) <= 1:
                    logger.warning(
                        "文档 %s 未能按标题分割",
                        doc.metadata.get("name", "未知"),
                    )
                parent_id = doc.metadata["parent_id"]
                for i, chunk in enumerate(md_chunks):
                    child_id = str(uuid.uuid4())
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update(
                        {
                            "chunk_id": child_id,
                            "parent_id": parent_id,
                            "doc_type": "child",
                            "chunk_index": i,
                        }
                    )
                    self.parent_child_map[child_id] = parent_id
                all_chunks.extend(md_chunks)
            except Exception as exc:
                logger.warning("文档 %s 分割失败: %s", doc.metadata.get("source"), exc)
                all_chunks.append(doc)
        return all_chunks

    def get_statistics(self) -> Dict[str, Any]:
        if not self.documents:
            return {}
        genres: Dict[str, int] = {}
        for doc in self.documents:
            for genre in doc.metadata.get("genres") or ["未标注"]:
                genres[str(genre)] = genres.get(str(genre), 0) + 1
        avg_chunk = 0
        if self.chunks:
            avg_chunk = sum(c.metadata.get("chunk_size", 0) for c in self.chunks) / len(self.chunks)
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "genres": genres,
            "avg_chunk_size": avg_chunk,
        }

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        parent_relevance: Dict[str, int] = {}
        parent_docs_map: Dict[str, Document] = {}
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if not parent_id:
                continue
            parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1
            if parent_id not in parent_docs_map:
                for doc in self.documents:
                    if doc.metadata.get("parent_id") == parent_id:
                        parent_docs_map[parent_id] = doc
                        break
        sorted_ids = sorted(parent_relevance, key=parent_relevance.get, reverse=True)
        parent_docs = [parent_docs_map[pid] for pid in sorted_ids if pid in parent_docs_map]
        names = [
            f"{doc.metadata.get('name', '未知')}({parent_relevance.get(doc.metadata.get('parent_id'), 0)}块)"
            for doc in parent_docs
        ]
        logger.info("从 %s 个子块得到 %s 个父文档: %s", len(child_chunks), len(parent_docs), ", ".join(names))
        return parent_docs

    @staticmethod
    def load_owned_app_ids(library_path: str) -> List[str]:
        path = Path(library_path)
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("app_ids") or raw.get("owned") or []
        return [str(item) for item in raw]
