"""项目配置。"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_EMBEDDING_DIR = PROJECT_ROOT / "models" / "bge-small-zh-v1.5"


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

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = RAGConfig()
