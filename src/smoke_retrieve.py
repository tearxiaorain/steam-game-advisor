"""不调用大模型，只检查语料加载、嵌入模型和检索是否能跑通。"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from config import DEFAULT_CONFIG
from rag_modules.data_preparation import DataPreparationModule
from rag_modules.index_construction import IndexConstructionModule
from rag_modules.retrieval_optimization import RetrievalOptimizationModule


def main() -> None:
    print("Python:", sys.version.split()[0])
    print("数据目录:", DEFAULT_CONFIG.data_path)

    data = DataPreparationModule(DEFAULT_CONFIG.data_path)
    docs = data.load_documents()
    print(f"加载文档: {len(docs)}")
    if not docs:
        raise SystemExit("没有语料，请先放入 data/processed/*.md")

    sample = docs[0].metadata
    print(f"示例元数据 name={sample.get('name')} name_cn={sample.get('name_cn')} app_id={sample.get('app_id')}")

    chunks = data.chunk_documents()
    print(f"分块: {len(chunks)}")

    print("开始加载嵌入模型并建索引（首次会下载模型，可能较久）...")
    index = IndexConstructionModule(
        model_name=DEFAULT_CONFIG.embedding_model,
        index_save_path=str(Path(DEFAULT_CONFIG.index_save_path) / "smoke"),
    )
    vectorstore = index.build_vector_index(chunks)
    retrieval = RetrievalOptimizationModule(vectorstore, chunks)

    queries = ["星露谷物语怎么玩", "像极乐迪斯科那种叙事游戏", "无标题大鹅"]
    for query in queries:
        hits = retrieval.hybrid_search(query, top_k=3)
        names = [
            f"{hit.metadata.get('name_cn') or hit.metadata.get('name')}({hit.metadata.get('app_id')})"
            for hit in hits
        ]
        print(f"查询「{query}」-> {names}")

    print("检索冒烟测试完成。完整问答还需要 .env 中的 DEEPSEEK_API_KEY。")


if __name__ == "__main__":
    main()
