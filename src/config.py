"""项目配置。"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_EMBEDDING_DIR = PROJECT_ROOT / "models" / "bge-small-zh-v1.5"

# 带这些 genre 且不含核心玩法 genre 的条目视为非游戏，不参与检索索引
NON_GAME_EXCLUDE_GENRES: List[str] = [
    "实用工具",
    "软件培训",
    "视频制作",
    "动画制作和建模",
    "设计和插画",
    "照片编辑",
    "音频制作",
]
CORE_GAME_GENRES: List[str] = [
    "动作",
    "冒险",
    "角色扮演",
    "模拟",
    "策略",
    "竞速",
    "体育",
    "大型多人在线",
    "免费开玩",
    "抢先体验",
]

# 详情题：query 子串 → app_id（弥补 name/name_cn 覆盖不足）
DETAIL_NAME_ALIASES: Dict[str, List[str]] = {
    "反恐精英2": ["730"],
    "反恐精英": ["730"],
    "cs2": ["730"],
    "counter-strike 2": ["730"],
    "无标题大鹅": ["837470"],
    "大鹅": ["837470"],
    "空洞骑士": ["367520"],
    "博德之门3": ["1086940"],
    "博德之门": ["1086940"],
    "永劫无间": ["1203220"],
    "黑神话悟空": ["2358720"],
    "黑神话": ["2358720"],
    "巫师3": ["292030"],
    "巫师 3": ["292030"],
}

# 切块 section 权重：简介/标签抬高，配置等降权；0=不进检索索引
SECTION_WEIGHTS: Dict[str, float] = {
    "简介": 1.5,
    "类型与标签": 1.4,
    "语言": 1.1,
    "评价摘要": 0.9,
    "游玩方式": 0.65,
    "配置与平台": 0.0,
}


def _embedding_model_path() -> str:
    if (LOCAL_EMBEDDING_DIR / "model.safetensors").exists():
        return str(LOCAL_EMBEDDING_DIR)
    return "BAAI/bge-small-zh-v1.5"


@dataclass
class RAGConfig:
    data_path: str = str(PROJECT_ROOT / "data" / "processed")
    library_path: str = str(PROJECT_ROOT / "data" / "library" / "owned_appids.json")
    index_save_path: str = str(PROJECT_ROOT / "vector_index")
    trace_path: str = str(PROJECT_ROOT / "data" / "eval" / "traces" / "traces.jsonl")

    embedding_model: str = _embedding_model_path()
    llm_model: str = "deepseek-chat"

    top_k: int = 3
    temperature: float = 0.1
    max_tokens: int = 2048

    # 多重查询：可提升部分题召回，但在当前 13 款语料上评测曾低于单次改写；默认关
    use_multi_query: bool = False
    multi_query_count: int = 2

    # MMR：RRF 候选池内按游戏多样化重排，缓解泛文档霸榜；默认关
    use_mmr: bool = False
    mmr_lambda: float = 0.7
    mmr_pool_size: int = 24

    # 语料卫生：剔除非游戏 genre；详情题对游戏名做精确匹配加分
    exclude_non_game_genres: bool = True
    detail_name_boost: bool = True

    # 切块类型加权：按二级标题抬/降 RRF；weight=0 不进索引。v2 评测曾 63%→57%，默认关
    use_section_weights: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = RAGConfig()
