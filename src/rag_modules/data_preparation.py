"""数据准备：加载游戏 Markdown，按标题切分子块。"""

import ast
import hashlib
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from config import (
    DETAIL_NAME_ALIASES,
    INDEX_EXCLUDE_SECTIONS,
    PLAYSTYLE_DENOISE_MAX_CHARS,
    PLAYSTYLE_DROP_LINE_PATTERNS,
    SECTION_WEIGHTS,
)
from .tag_taxonomy import get_taxonomy

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
                body = self._prepend_display_names(body, metadata)
                documents.append(Document(page_content=body, metadata=metadata))
            except Exception as exc:
                logger.warning("读取文件 %s 失败: %s", md_file, exc)

        self.documents = documents
        logger.info("成功加载 %s 个文档", len(documents))
        return documents

    @staticmethod
    def is_indexable_game(metadata: Dict[str, Any]) -> bool:
        tax = get_taxonomy()
        genres = {str(g) for g in (metadata.get("genres") or [])}
        if not genres:
            return True
        has_non_game = bool(genres & tax.non_game_genres)
        has_play = bool(genres & tax.play_genres)
        if has_non_game and not has_play:
            return False
        return True

    def apply_game_only_filter(self) -> List[Document]:
        kept, dropped = [], []
        for doc in self.documents:
            if self.is_indexable_game(doc.metadata):
                kept.append(doc)
            else:
                dropped.append(doc)
        if dropped:
            names = ", ".join(
                f"{d.metadata.get('name_cn') or d.metadata.get('name')}({d.metadata.get('app_id')})"
                for d in dropped
            )
            logger.info("非游戏 genre 过滤: 排除 %s 款 → %s", len(dropped), names)
        self.documents = kept
        return kept

    @staticmethod
    def _normalize_match_text(text: str) -> str:
        text = (text or "").lower().strip()
        text = re.sub(r"[《》「」\"'（）()\[\]·:：,，.!！?？\s]+", "", text)
        return text

    def match_documents_for_detail(self, *texts: str) -> List[Document]:
        """详情题：从 query/改写中匹配游戏名或别名，返回父文档。"""
        merged = " ".join(t for t in texts if t).lower()
        norm_query = self._normalize_match_text(merged)
        if not norm_query:
            return []

        matched_ids: List[str] = []
        seen_ids: set[str] = set()

        for alias, app_ids in DETAIL_NAME_ALIASES.items():
            if alias.lower() in merged or self._normalize_match_text(alias) in norm_query:
                for aid in app_ids:
                    if aid not in seen_ids:
                        seen_ids.add(aid)
                        matched_ids.append(aid)

        for doc in self.documents:
            app_id = str(doc.metadata.get("app_id") or "")
            keys: List[str] = []
            for field in ("name_cn", "name"):
                raw = str(doc.metadata.get(field) or "").strip()
                if not raw:
                    continue
                keys.append(raw)
                if "：" in raw:
                    keys.append(raw.split("：", 1)[0])
                if ":" in raw:
                    keys.append(raw.split(":", 1)[0])
                if " / " in raw:
                    keys.extend(part.strip() for part in raw.split(" / ") if part.strip())
            for key in keys:
                nk = self._normalize_match_text(key)
                if len(nk) >= 2 and (nk in norm_query or key.lower() in merged):
                    if app_id and app_id not in seen_ids:
                        seen_ids.add(app_id)
                        matched_ids.append(app_id)
                    break

        out: List[Document] = []
        id_to_doc = {str(d.metadata.get("app_id")): d for d in self.documents}
        for aid in matched_ids:
            doc = id_to_doc.get(aid)
            if doc:
                out.append(doc)
        if out:
            logger.info(
                "详情名匹配: %s",
                ", ".join(f"{d.metadata.get('name_cn') or d.metadata.get('name')}({d.metadata.get('app_id')})" for d in out),
            )
        return out

    def get_chunks_for_app_ids(self, app_ids: Sequence[str]) -> List[Document]:
        want = {str(a) for a in app_ids}
        out: List[Document] = []
        seen: set[str] = set()
        for chunk in self.chunks:
            aid = str(chunk.metadata.get("app_id") or "")
            if aid in want and aid not in seen:
                seen.add(aid)
                out.append(chunk)
        return out

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

    @staticmethod
    def _prepend_display_names(body: str, metadata: Dict[str, Any]) -> str:
        names = []
        for key in ("name_cn", "name"):
            value = str(metadata.get(key) or "").strip()
            if value and value not in names:
                names.append(value)
        if not names or body.startswith("游戏名:"):
            return body
        return "游戏名: " + " / ".join(names) + "\n\n" + body

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

    def filter_chunks_for_index(self, chunks: List[Document] | None = None) -> List[Document]:
        """丢掉 section_weight=0 的块（如配置与平台），不参与向量/BM25 索引。"""
        src = chunks if chunks is not None else self.chunks
        kept = [
            c
            for c in src
            if float(c.metadata.get("section_weight", 1.0)) > 0
        ]
        dropped = len(src) - len(kept)
        if dropped:
            logger.info(
                "切块索引过滤: 排除 weight=0 的 %s 块，保留 %s",
                dropped,
                len(kept),
            )
        return kept

    @classmethod
    def denoise_playstyle_text(
        cls, text: str, max_chars: int = PLAYSTYLE_DENOISE_MAX_CHARS
    ) -> str:
        """去掉营销/更新/法务模板句，只保留开头核心段落，供检索索引使用。"""
        # 法务分隔线之后整段丢掉
        cut = re.split(r"\n\*{3,}\n", text, maxsplit=1)
        text = cut[0]
        drop_re = [
            re.compile(p, re.IGNORECASE) for p in PLAYSTYLE_DROP_LINE_PATTERNS
        ]
        kept_lines: List[str] = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                if kept_lines and kept_lines[-1] != "":
                    kept_lines.append("")
                continue
            if any(r.search(s) for r in drop_re):
                continue
            kept_lines.append(line.rstrip())
        body = "\n".join(kept_lines).strip()
        body = re.sub(r"\n{3,}", "\n\n", body)

        # 拆段落：保留标题行 + 首个实质玩法段，丢掉后面功能清单
        lines = body.splitlines()
        header: List[str] = []
        rest: List[str] = []
        for line in lines:
            if not rest and (
                line.startswith("#") or line.startswith("游戏名:")
            ):
                header.append(line)
            else:
                rest.append(line)
        rest_text = "\n".join(rest).strip()
        paras = [p.strip() for p in re.split(r"\n\s*\n", rest_text) if p.strip()]
        core = paras[0] if paras else rest_text
        body = "\n".join(header + ([core] if core else [])).strip()

        if max_chars > 0 and len(body) > max_chars:
            trimmed = body[:max_chars]
            if "\n" in trimmed:
                trimmed = trimmed.rsplit("\n", 1)[0]
            body = trimmed.rstrip() + "…"
        return body

    def prepare_index_chunks(
        self,
        chunks: List[Document] | None = None,
        *,
        use_section_weights: bool = False,
        use_playstyle_denoise: bool = True,
        playstyle_max_chars: int = PLAYSTYLE_DENOISE_MAX_CHARS,
        use_taxonomy_scrub: bool = True,
    ) -> List[Document]:
        """生成仅用于检索的 chunk 副本：排除配置块，游玩方式/类型块清洗。

        不修改父文档；生成回答仍读 self.documents 全文。
        """
        src = chunks if chunks is not None else self.chunks
        exclude = {str(s) for s in INDEX_EXCLUDE_SECTIONS}
        tax = get_taxonomy() if use_taxonomy_scrub else None
        out: List[Document] = []
        denoised = 0
        scrubbed = 0
        for chunk in src:
            section = str(
                chunk.metadata.get("section") or chunk.metadata.get("二级标题") or ""
            ).strip()
            if section in exclude:
                continue
            if use_section_weights and float(chunk.metadata.get("section_weight", 1.0)) <= 0:
                continue
            meta = dict(chunk.metadata)
            content = chunk.page_content
            if use_playstyle_denoise and section == "游玩方式":
                cleaned = self.denoise_playstyle_text(content, max_chars=playstyle_max_chars)
                if cleaned != content:
                    meta["index_denoised"] = True
                    meta["index_orig_len"] = len(content)
                    meta["index_denoised_len"] = len(cleaned)
                    denoised += 1
                content = cleaned or content[:playstyle_max_chars]
            if tax is not None and section == "类型与标签":
                cleaned = tax.scrub_genre_section_text(content, meta)
                if cleaned != content:
                    meta["index_taxonomy_scrubbed"] = True
                    scrubbed += 1
                content = cleaned
            out.append(Document(page_content=content, metadata=meta))
        logger.info(
            "索引切块准备完成: 输入 %s → 保留 %s（游玩方式降噪 %s，类型块清洗 %s）",
            len(src),
            len(out),
            denoised,
            scrubbed,
        )
        return out

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
                    section = str(chunk.metadata.get("二级标题") or "").strip()
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update(
                        {
                            "chunk_id": child_id,
                            "parent_id": parent_id,
                            "doc_type": "child",
                            "chunk_index": i,
                            "section": section,
                            "section_weight": float(
                                SECTION_WEIGHTS.get(section, 1.0)
                            ),
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
