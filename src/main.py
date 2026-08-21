"""Steam 游戏顾问入口。"""

import logging
import os
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv

from config import DEFAULT_CONFIG, RAGConfig, SECTION_WEIGHTS, INDEX_EXCLUDE_SECTIONS
from rag_modules import (
    DataPreparationModule,
    GenerationIntegrationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    TraceLogger,
)
from rag_modules.tag_taxonomy import get_taxonomy
from rag_modules.library_profile import (
    OwnedLibrary,
    attach_playtime_metadata,
    detect_library_mode,
    load_owned_library,
    select_owned_candidates,
)
from rag_modules.ownership_prior import (
    OwnershipPrior,
    apply_ownership_bias,
    detect_friend_recommend_intent,
    filter_longtail_docs,
    load_ownership_prior,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SteamGameAdvisor:
    def __init__(self, config: RAGConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None
        self.owned_app_ids = []
        self.owned_library = OwnedLibrary()
        self.ownership_prior = OwnershipPrior()
        self.trace_logger = TraceLogger(self.config.trace_path)

        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

    def initialize_system(self):
        print("正在初始化系统...")
        self.data_module = DataPreparationModule(self.config.data_path)
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path,
        )
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        self.owned_library = load_owned_library(
            self.config.me_owned_path, self.config.library_path
        )
        self.owned_app_ids = self.owned_library.app_ids or DataPreparationModule.load_owned_app_ids(
            self.config.library_path
        )
        if self.owned_app_ids:
            print(f"已加载库存 {len(self.owned_app_ids)} 款（含时长画像）。")
        self.ownership_prior = load_ownership_prior(
            self.config.me_owned_path,
            self.config.friends_dir,
            fallback_appids_path=self.config.library_path,
        )
        if self.config.use_ownership_bias:
            print(
                "拥有度偏置: "
                f"me={len(self.ownership_prior.me)} "
                f"friend_apps={len(self.ownership_prior.friend_owners)} "
                f"pool={self.config.ownership_pool_size}"
            )
        print(
            "查询改写: "
            + ("开" if self.config.use_query_rewrite else "关（Prompt 见 rewrite_prompts.py）")
            + "；硬映射: "
            + ("开" if self.config.use_rewrite_hard_aliases else "关")
        )
        print("系统初始化完成。")

    def build_knowledge_base(self, force_rebuild: bool = False):
        print("\n正在构建知识库...")
        self.data_module.load_documents()
        if self.config.exclude_non_game_genres:
            self.data_module.apply_game_only_filter()
        all_chunks = self.data_module.chunk_documents()
        chunks = self.data_module.prepare_index_chunks(
            all_chunks,
            use_section_weights=self.config.use_section_weights,
            use_playstyle_denoise=self.config.use_playstyle_denoise,
            playstyle_max_chars=self.config.playstyle_denoise_max_chars,
            use_taxonomy_scrub=self.config.use_taxonomy_scrub,
        )
        self.data_module.chunks = chunks

        index_meta = {
            "chunk_count": len(chunks),
            "document_count": len(self.data_module.documents),
            "exclude_non_game_genres": self.config.exclude_non_game_genres,
            "use_section_weights": self.config.use_section_weights,
            "use_tag_breadth_penalty": self.config.use_tag_breadth_penalty,
            "use_playstyle_denoise": self.config.use_playstyle_denoise,
            "playstyle_denoise_max_chars": self.config.playstyle_denoise_max_chars,
            "use_taxonomy_scrub": self.config.use_taxonomy_scrub,
            "taxonomy_path": self.config.taxonomy_path,
            "index_exclude_sections": list(INDEX_EXCLUDE_SECTIONS),
            "section_weights": dict(SECTION_WEIGHTS)
            if self.config.use_section_weights
            else {},
        }
        saved_meta = None if force_rebuild else self.index_module.load_index_meta(
            self.config.index_save_path
        )
        vectorstore = None if force_rebuild else self.index_module.load_index()
        if vectorstore is not None and saved_meta != index_meta:
            print("索引元数据与当前语料不一致，将重建向量索引…")
            vectorstore = None

        if vectorstore is not None:
            print("已加载本地向量索引。")
        else:
            if not self.data_module.documents:
                raise ValueError(
                    f"{self.config.data_path} 中没有可索引的游戏 Markdown。"
                )
            print("未找到匹配索引，开始构建…")
            vectorstore = self.index_module.build_vector_index(chunks)
            self.index_module.save_index(index_meta)

        self.retrieval_module = RetrievalOptimizationModule(
            vectorstore,
            chunks,
            use_mmr=self.config.use_mmr,
            mmr_lambda=self.config.mmr_lambda,
            mmr_pool_size=self.config.mmr_pool_size,
            use_section_weights=self.config.use_section_weights,
            use_tag_breadth_penalty=self.config.use_tag_breadth_penalty,
            tag_breadth_free=self.config.tag_breadth_free,
            tag_breadth_alpha=self.config.tag_breadth_alpha,
            use_user_tag_overlap_boost=self.config.use_user_tag_overlap_boost,
            user_tag_overlap_bonus=self.config.user_tag_overlap_bonus,
            user_tag_overlap_max=self.config.user_tag_overlap_max,
            use_tag_literal_recall=self.config.use_tag_literal_recall,
            tag_literal_min_len=self.config.tag_literal_min_len,
        )
        stats = self.data_module.get_statistics()
        print("\n知识库统计:")
        print(f"  游戏数: {stats.get('total_documents', 0)}")
        print(f"  文本块: {stats.get('total_chunks', 0)}")
        print(f"  类型: {list((stats.get('genres') or {}).keys())}")
        tax = get_taxonomy(self.config.taxonomy_path, reload=True)
        report = tax.scan_documents(self.data_module.documents)
        unk_g = report.get("unknown_genres") or {}
        unk_c = report.get("unknown_categories") or {}
        if unk_g or unk_c:
            print(
                f"  未登记标签: genres={len(unk_g)} categories={len(unk_c)}"
                "（见 data/library/tag_taxonomy.json，可用 scan_tag_taxonomy 查看）"
            )
            if unk_g:
                print(f"    unknown genres: {list(unk_g.keys())[:8]}")
            if unk_c:
                print(f"    unknown categories: {list(unk_c.keys())[:8]}")
        print("知识库就绪。")

    def ask(self, question: str) -> dict:
        """结构化问答，供 UI / API 使用。返回 route、hits、answer 等。"""
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        print(f"\n问题: {question}")
        route_type = self.generation_module.query_router(question)
        print(f"路由: {route_type}")

        if route_type == "trending":
            answer = self.generation_module.trending_unavailable_answer()
            result = {
                "question": question,
                "route": route_type,
                "rewritten_query": question,
                "filters": {},
                "library_mode": None,
                "hits": [],
                "answer": answer,
            }
            self.trace_logger.append(
                question=question,
                route=route_type,
                rewritten_query=question,
                filters={},
                hits=[],
                answer=answer,
                stream=False,
            )
            return result

        rewritten_query = question
        query_variants = [question]
        if route_type in {"recommend", "detail"}:
            if self.config.use_multi_query:
                query_variants = self.generation_module.expand_queries(
                    question, n=self.config.multi_query_count
                )
                rewritten_query = " | ".join(query_variants)
                print(f"多重查询: {query_variants}")
            else:
                rewritten_query = self.generation_module.query_rewrite(question)
                query_variants = [rewritten_query]

        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"过滤条件: {filters}")

        library_mode = None
        if route_type == "library":
            library_mode = detect_library_mode(question)
            print(f"库存策略: {library_mode}")
            if library_mode in {"tonight", "recent", "backlog"}:
                relevant_docs = self._select_library_docs(library_mode)
                rewritten_query = f"[library:{library_mode}] {question}"
                if not relevant_docs:
                    relevant_chunks = self._retrieve_chunks(
                        route_type, question, question, [question], filters
                    )
                    relevant_docs = self._apply_library_constraint(
                        question, relevant_chunks, mode="owned"
                    )
            else:
                relevant_chunks = self._retrieve_chunks(
                    route_type, question, rewritten_query, query_variants, filters
                )
                relevant_docs = self._apply_library_constraint(
                    question, relevant_chunks, mode=library_mode
                )
        else:
            relevant_chunks = self._retrieve_chunks(
                route_type, question, rewritten_query, query_variants, filters
            )
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

        hits = [
            {
                "name": doc.metadata.get("name"),
                "name_cn": doc.metadata.get("name_cn"),
                "app_id": str(doc.metadata.get("app_id", "")),
                "user_tags": list(doc.metadata.get("user_tags") or [])[:8],
            }
            for doc in relevant_docs
        ]
        names = [h.get("name_cn") or h.get("name") or "未知" for h in hits]
        if names:
            print(f"命中游戏: {', '.join(names)}")

        if not relevant_docs:
            answer = "没有找到相关游戏档案。可以换关键词，或检查 data/processed 是否已放入语料。"
        elif route_type == "library":
            answer = self.generation_module.generate_library_answer(
                question, relevant_docs, library_mode=library_mode or "owned"
            )
        elif route_type == "detail":
            answer = self.generation_module.generate_detail_answer(question, relevant_docs)
        else:
            answer = self.generation_module.generate_recommend_answer(question, relevant_docs)

        result = {
            "question": question,
            "route": route_type,
            "rewritten_query": rewritten_query,
            "filters": filters,
            "library_mode": library_mode,
            "hits": hits,
            "answer": answer,
        }
        self.trace_logger.append(
            question=question,
            route=route_type,
            rewritten_query=rewritten_query,
            filters=filters,
            hits=[{k: v for k, v in h.items() if k != "user_tags"} for h in hits],
            answer=answer,
            stream=False,
        )
        return result

    def ask_question(self, question: str, stream: bool = False):
        if stream:
            return self._ask_question_stream(question)
        return self.ask(question)["answer"]

    def _ask_question_stream(self, question: str):
        """流式回答（CLI）；UI 请用 ask()。"""
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        print(f"\n问题: {question}")
        route_type = self.generation_module.query_router(question)
        print(f"路由: {route_type}")

        if route_type == "trending":
            answer = self.generation_module.trending_unavailable_answer()
            self.trace_logger.append(
                question=question,
                route=route_type,
                rewritten_query=question,
                filters={},
                hits=[],
                answer=answer,
                stream=True,
            )
            return answer

        rewritten_query = question
        query_variants = [question]
        if route_type in {"recommend", "detail"}:
            if self.config.use_multi_query:
                query_variants = self.generation_module.expand_queries(
                    question, n=self.config.multi_query_count
                )
                rewritten_query = " | ".join(query_variants)
            else:
                rewritten_query = self.generation_module.query_rewrite(question)
                query_variants = [rewritten_query]

        filters = self._extract_filters_from_query(question)
        library_mode = None
        if route_type == "library":
            library_mode = detect_library_mode(question)
            if library_mode in {"tonight", "recent", "backlog"}:
                relevant_docs = self._select_library_docs(library_mode)
                rewritten_query = f"[library:{library_mode}] {question}"
                if not relevant_docs:
                    relevant_chunks = self._retrieve_chunks(
                        route_type, question, question, [question], filters
                    )
                    relevant_docs = self._apply_library_constraint(
                        question, relevant_chunks, mode="owned"
                    )
            else:
                relevant_chunks = self._retrieve_chunks(
                    route_type, question, rewritten_query, query_variants, filters
                )
                relevant_docs = self._apply_library_constraint(
                    question, relevant_chunks, mode=library_mode
                )
        else:
            relevant_chunks = self._retrieve_chunks(
                route_type, question, rewritten_query, query_variants, filters
            )
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

        hits = [
            {
                "name": doc.metadata.get("name"),
                "name_cn": doc.metadata.get("name_cn"),
                "app_id": str(doc.metadata.get("app_id", "")),
            }
            for doc in relevant_docs
        ]

        if not relevant_docs:
            answer = "没有找到相关游戏档案。可以换关键词，或检查 data/processed 是否已放入语料。"
            self.trace_logger.append(
                question=question,
                route=route_type,
                rewritten_query=rewritten_query,
                filters=filters,
                hits=hits,
                answer=answer,
                stream=True,
            )
            return answer

        if route_type == "detail":
            chunks = self.generation_module.generate_detail_answer_stream(
                question, relevant_docs
            )
        elif route_type == "library":
            # library 暂无非流式
            answer = self.generation_module.generate_library_answer(
                question, relevant_docs, library_mode=library_mode or "owned"
            )
            self.trace_logger.append(
                question=question,
                route=route_type,
                rewritten_query=rewritten_query,
                filters=filters,
                hits=hits,
                answer=answer,
                stream=True,
            )
            return answer
        else:
            chunks = self.generation_module.generate_recommend_answer_stream(
                question, relevant_docs
            )
        return self._stream_and_trace(
            chunks,
            question=question,
            route=route_type,
            rewritten_query=rewritten_query,
            filters=filters,
            hits=hits,
        )

    def _stream_and_trace(self, chunks, *, question, route, rewritten_query, filters, hits):
        parts = []

        def _gen():
            for chunk in chunks:
                parts.append(chunk)
                yield chunk
            self.trace_logger.append(
                question=question,
                route=route,
                rewritten_query=rewritten_query,
                filters=filters,
                hits=hits,
                answer="".join(parts),
                stream=True,
            )

        return _gen()

    def _select_library_docs(self, mode: str):
        """tonight/recent/backlog：直接按库存时长从知识库父文档里取候选。"""
        id_to_doc = {
            str(d.metadata.get("app_id")): d for d in self.data_module.documents
        }
        kb_ids = list(id_to_doc.keys())
        pool_limit = 500 if mode == "backlog" else max(self.config.top_k, 3)
        candidates = select_owned_candidates(
            self.owned_library,
            mode,
            available_app_ids=kb_ids,
            limit=pool_limit,
        )
        if mode == "backlog":
            # 库内未玩：优先好评率高的档案，避免按名字字典序乱序
            def _review_score(game):
                doc = id_to_doc.get(game.app_id)
                if doc is None:
                    return -1.0
                try:
                    return float(doc.metadata.get("review_percentage") or -1)
                except (TypeError, ValueError):
                    return -1.0

            candidates = sorted(candidates, key=_review_score, reverse=True)
        candidates = candidates[: max(self.config.top_k, 3)]

        docs = []
        for game in candidates:
            doc = id_to_doc.get(game.app_id)
            if doc is None:
                continue
            from langchain_core.documents import Document

            clone = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            docs.append(attach_playtime_metadata(clone, game))
        if not docs:
            print(f"库存策略 {mode} 与知识库无交集。")
        return docs

    def _apply_library_constraint(self, question: str, chunks, mode: str = None):
        docs = self.data_module.get_parent_documents(chunks)
        owned = set(self.owned_app_ids)
        if not owned:
            print("未导入库存，library 路由将按普通检索结果作答。")
            return docs

        mode = mode or detect_library_mode(question)
        if mode == "unowned":
            filtered = [doc for doc in docs if str(doc.metadata.get("app_id")) not in owned]
            return filtered

        filtered = [doc for doc in docs if str(doc.metadata.get("app_id")) in owned]
        if not filtered:
            # 检索没命中库内档案时，回退到时长排序的库内知识库游戏
            return self._select_library_docs("tonight")

        enriched = []
        for doc in filtered:
            from langchain_core.documents import Document

            clone = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            game = self.owned_library.get(str(doc.metadata.get("app_id")))
            enriched.append(attach_playtime_metadata(clone, game))
        return enriched or docs

    def _retrieve_chunks(
        self,
        route_type: str,
        question: str,
        rewritten_query: str,
        query_variants: list,
        filters: dict,
    ):
        top_k = self.config.top_k
        friend_intent = detect_friend_recommend_intent(question)
        apply_bias = (
            route_type == "recommend"
            and self.config.use_ownership_bias
            and (
                not self.config.ownership_bias_friends_only or friend_intent
            )
        )
        if apply_bias:
            print(
                "拥有度偏置: 开"
                + ("（好友向）" if friend_intent else "")
            )
        fetch_k = (
            max(int(self.config.ownership_pool_size), top_k) if apply_bias else top_k
        )
        if filters:
            chunks = self.retrieval_module.metadata_filtered_multi_search(
                query_variants, filters, top_k=fetch_k
            )
        else:
            chunks = self.retrieval_module.multi_query_search(
                query_variants, top_k=fetch_k
            )
        if apply_bias and chunks:
            chunks = filter_longtail_docs(
                chunks,
                self.ownership_prior,
                min_keep=max(top_k * 2, 6),
            ) if self.config.ownership_filter_longtail else list(chunks)
            if self.config.ownership_use_score_boost:
                chunks = apply_ownership_bias(
                    chunks,
                    self.ownership_prior,
                    me_factor=self.config.ownership_me_factor,
                    multi_friend_factor=self.config.ownership_multi_friend_factor,
                    duo_friend_factor=self.config.ownership_duo_friend_factor,
                    longtail_factor=self.config.ownership_longtail_factor,
                )
            chunks = self.retrieval_module.diversify_by_game(chunks, top_k=top_k)
        if route_type == "detail" and self.config.detail_name_boost:
            chunks = self._apply_detail_name_boost(
                question, rewritten_query, query_variants, chunks
            )
        return chunks

    def _apply_detail_name_boost(
        self,
        question: str,
        rewritten_query: str,
        query_variants: list,
        chunks: list,
    ):
        texts = [question, rewritten_query, *query_variants]
        matched = self.data_module.match_documents_for_detail(*texts)
        if not matched:
            return chunks
        app_ids = [str(d.metadata.get("app_id")) for d in matched if d.metadata.get("app_id")]
        boost_chunks = self.data_module.get_chunks_for_app_ids(app_ids)
        merged = boost_chunks + [
            c for c in chunks if str(c.metadata.get("app_id")) not in set(app_ids)
        ]
        return self.retrieval_module.diversify_by_game(merged, top_k=self.config.top_k)

    def _extract_filters_from_query(self, query: str) -> dict:
        filters = {}
        if "免费" in query:
            filters["is_free"] = True
        if "中文" in query:
            filters["supported_languages"] = ["简体中文", "中文", "schinese", "chinese"]
        if "单人" in query:
            filters["categories"] = ["单人", "Single-player", "Singleplayer"]
        # 「合作/开黑」偏 Co-op；「联机/朋友/多人」覆盖竞技多人，避免误杀 CS2 这类无 Co-op 标签的游戏
        if "合作" in query or "开黑" in query:
            filters["categories"] = ["合作", "Co-op", "Online Co-Op", "多人", "Multi-player", "Multiplayer"]
        elif "联机" in query or "多人" in query or "朋友" in query:
            filters["categories"] = [
                "多人",
                "Multi-player",
                "Multiplayer",
                "线上玩家对战",
                "玩家对战",
                "跨平台多人",
                "合作",
                "Co-op",
                "Online Co-Op",
            ]
        if "Mac" in query or "macOS" in query or "苹果" in query:
            filters["platforms"] = ["Mac", "macOS", "Mac OS"]
        price_match = re.search(r"(?:不超过|别超过|低于|以内)\s*(\d+)", query)
        if not price_match:
            price_match = re.search(r"(\d+)\s*块", query)
        if price_match:
            filters["price_max"] = float(price_match.group(1))
        return filters

    def run_interactive(self):
        print("=" * 60)
        print("Steam Game Advisor")
        print("=" * 60)
        self.initialize_system()
        self.build_knowledge_base()
        print("\n输入问题开始对话，输入退出结束。")
        while True:
            try:
                user_input = input("\n问题: ").strip()
                if user_input.lower() in {"退出", "quit", "exit", ""}:
                    break
                stream_choice = input("流式输出? (y/n, 默认 y): ").strip().lower()
                use_stream = stream_choice != "n"
                print("\n回答:")
                if use_stream:
                    result = self.ask_question(user_input, stream=True)
                    if isinstance(result, str):
                        print(result)
                    else:
                        for chunk in result:
                            print(chunk, end="", flush=True)
                        print("\n")
                else:
                    print(self.ask_question(user_input, stream=False))
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"处理问题时出错: {exc}")
        print("\n已退出。")


def main():
    try:
        SteamGameAdvisor().run_interactive()
    except Exception as exc:
        logger.error("系统运行出错: %s", exc)
        print(f"系统错误: {exc}")


if __name__ == "__main__":
    main()
