"""进索引前的文本本地化接口（本轮不接云端 API）。

后端：
- passthrough：原样返回（默认）
- local_opus：预留；模型下载到 D:\\model\\Helsinki-NLP_opus-mt-en-zh 后再实现
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class Translator(ABC):
    name: str = "base"

    @abstractmethod
    def translate(self, text: str, *, source_lang: str = "en", target_lang: str = "zh") -> str:
        raise NotImplementedError


class PassthroughTranslator(Translator):
    """不翻译，原样返回。用于管线打通与中文档透传。"""

    name = "passthrough"

    def translate(self, text: str, *, source_lang: str = "en", target_lang: str = "zh") -> str:
        return text or ""


class LocalOpusTranslator(Translator):
    """本地 Helsinki-NLP/opus-mt-en-zh。本轮仅占位，下载模型后再接线。"""

    name = "local_opus"

    def __init__(self, model_dir: Optional[str] = None) -> None:
        self.model_dir = model_dir or r"D:\model\Helsinki-NLP_opus-mt-en-zh"

    def translate(self, text: str, *, source_lang: str = "en", target_lang: str = "zh") -> str:
        raise NotImplementedError(
            "local_opus 尚未接线。"
            f"请先将模型放到 {self.model_dir}，再实现 MarianMT 调用。"
        )


def get_translator(backend: str = "passthrough", **kwargs) -> Translator:
    key = (backend or "passthrough").strip().lower()
    if key in {"", "none", "passthrough", "pass"}:
        return PassthroughTranslator()
    if key in {"local_opus", "opus", "opus-mt"}:
        return LocalOpusTranslator(model_dir=kwargs.get("model_dir"))
    if key in {"deepseek", "ds"}:
        raise ValueError("本轮未接入 DeepSeek 翻译后端；请用 passthrough 或后续 local_opus。")
    raise ValueError(f"未知翻译后端: {backend}")
