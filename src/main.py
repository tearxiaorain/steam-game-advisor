"""Steam 游戏顾问入口。"""

import logging
import os
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv

from config import DEFAULT_CONFIG, RAGConfig, SECTION_WEIGHTS
from rag_modules import (
    DataPreparationModule,
    GenerationIntegrationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    TraceLogger,
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
        self.owned_app_ids = DataPreparationModule.load_owned_app_ids(self.config.library_path)
        print("系统初始化完成。")

    def build_knowledge_base(self, force_rebuild: bool = False):
        print("\n正在构建知识库...")
        self.data_module.load_documents()
        if self.config.exclude_non_game_genres:
            self.data_module.apply_game_only_filter()
        all_chunks = self.data_module.chunk_documents()
        if self.config.use_section_weights:
            chunks = self.data_module.filter_chunks_for_index(all_chunks)
            self.data_module.chunks = chunks
        else:
            chunks = all_chunks

        index_meta = {
            "chunk_count": len(chunks),
            "document_count": len(self.data_module.documents),
            "exclude_non_game_genres": self.config.exclude_non_game_genres,
            "use_section_weights": self.config.use_section_weights,
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
        )
        stats = self.data_module.get_statistics()
        print("\n知识库统计:")
        print(f"  游戏数: {stats.get('total_documents', 0)}")
        print(f"  文本块: {stats.get('total_chunks', 0)}")
        print(f"  类型: {list((stats.get('genres') or {}).keys())}")
        print("知识库就绪。")

    def ask_question(self, question: str, stream: bool = False):
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
                stream=stream,
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
                print(f"多重查询: {query_variants}")
            else:
                rewritten_query = self.generation_module.query_rewrite(question)
                query_variants = [rewritten_query]

        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"过滤条件: {filters}")
        relevant_chunks = self._retrieve_chunks(
            route_type, question, rewritten_query, query_variants, filters
        )

        if route_type == "library":
            relevant_docs = self._apply_library_constraint(question, relevant_chunks)
        else:
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

        hits = [
            {
                "name": doc.metadata.get("name"),
                "name_cn": doc.metadata.get("name_cn"),
                "app_id": str(doc.metadata.get("app_id", "")),
            }
            for doc in relevant_docs
        ]
        names = [h.get("name_cn") or h.get("name") or "未知" for h in hits]
        if names:
            print(f"命中游戏: {', '.join(names)}")

        if not relevant_docs:
            answer = "没有找到相关游戏档案。可以换关键词，或检查 data/processed 是否已放入语料。"
            self.trace_logger.append(
                question=question,
                route=route_type,
                rewritten_query=rewritten_query,
                filters=filters,
                hits=hits,
                answer=answer,
                stream=stream,
            )
            return answer

        if stream:
            if route_type == "detail":
                chunks = self.generation_module.generate_detail_answer_stream(
                    question, relevant_docs
                )
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

        if route_type == "library":
            answer = self.generation_module.generate_library_answer(question, relevant_docs)
        elif route_type == "detail":
            answer = self.generation_module.generate_detail_answer(question, relevant_docs)
        else:
            answer = self.generation_module.generate_recommend_answer(question, relevant_docs)

        self.trace_logger.append(
            question=question,
            route=route_type,
            rewritten_query=rewritten_query,
            filters=filters,
            hits=hits,
            answer=answer,
            stream=False,
        )
        return answer

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

    def _apply_library_constraint(self, question: str, chunks):
        docs = self.data_module.get_parent_documents(chunks)
        owned = set(self.owned_app_ids)
        if not owned:
            print("未导入库存，library 路由将按普通检索结果作答。")
            return docs
        exclude_owned = any(key in question for key in ("没玩过", "未玩", "还没买", "推荐一款没"))
        if exclude_owned:
            return [doc for doc in docs if str(doc.metadata.get("app_id")) not in owned]
        return [doc for doc in docs if str(doc.metadata.get("app_id")) in owned] or docs

    def _retrieve_chunks(
        self,
        route_type: str,
        question: str,
        rewritten_query: str,
        query_variants: list,
        filters: dict,
    ):
        if filters:
            chunks = self.retrieval_module.metadata_filtered_multi_search(
                query_variants, filters, top_k=self.config.top_k
            )
        else:
            chunks = self.retrieval_module.multi_query_search(
                query_variants, top_k=self.config.top_k
            )
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
