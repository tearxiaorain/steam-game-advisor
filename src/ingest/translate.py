"""进索引前的文本本地化接口。

后端：
- passthrough：原样返回（默认）
- local_opus：Helsinki-NLP/opus-mt-en-zh（本地 MarianMT + 磁盘缓存）
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_OPUS_DIR = r"D:\model\Helsinki-NLP_opus-mt-en-zh"
# Opus-MT 常见长度上限偏紧；按字符粗切，留余量
_OPUS_MAX_CHARS = 400


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


def _cache_key(backend: str, model: str, text: str, source_lang: str, target_lang: str) -> str:
    payload = f"{backend}|{model}|{source_lang}|{target_lang}|{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _split_chunks(text: str, max_chars: int = _OPUS_MAX_CHARS) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # 先按段落，再按句号类标点
    parts = re.split(r"(\n{2,}|(?<=[.!?。！？])\s+)", text)
    chunks: List[str] = []
    buf = ""
    for part in parts:
        if part is None or part == "":
            continue
        if len(buf) + len(part) <= max_chars:
            buf += part
            continue
        if buf.strip():
            chunks.append(buf.strip())
        if len(part) <= max_chars:
            buf = part
        else:
            # 超长硬切
            for i in range(0, len(part), max_chars):
                piece = part[i : i + max_chars].strip()
                if piece:
                    chunks.append(piece)
            buf = ""
    if buf.strip():
        chunks.append(buf.strip())
    return chunks or [text[:max_chars]]


class LocalOpusTranslator(Translator):
    """本地 Helsinki-NLP/opus-mt-en-zh（MarianMT，CPU）。"""

    name = "local_opus"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        *,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir or DEFAULT_OPUS_DIR)
        self.cache_dir = Path(
            cache_dir
            or (Path(__file__).resolve().parents[2] / "data" / "cache" / "translations")
        )
        self.use_cache = use_cache
        self._tokenizer = None
        self._model = None
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"找不到 Opus-MT 模型目录: {self.model_dir}")
        if not (self.model_dir / "config.json").is_file():
            raise FileNotFoundError(f"模型目录不完整（缺 config.json）: {self.model_dir}")

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        from transformers import MarianMTModel, MarianTokenizer

        logger.info("加载 Opus-MT: %s", self.model_dir)
        self._tokenizer = MarianTokenizer.from_pretrained(str(self.model_dir))
        self._model = MarianMTModel.from_pretrained(str(self.model_dir))
        self._model.eval()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[str]:
        if not self.use_cache:
            return None
        path = self._cache_path(key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            text = data.get("text")
            return text if isinstance(text, str) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, text: str, source: str) -> None:
        if not self.use_cache:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(key)
        path.write_text(
            json.dumps(
                {
                    "backend": self.name,
                    "model_dir": str(self.model_dir),
                    "source": source,
                    "text": text,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _translate_chunk(self, chunk: str) -> str:
        import torch

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        inputs = self._tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
            )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def translate(self, text: str, *, source_lang: str = "en", target_lang: str = "zh") -> str:
        text = text or ""
        if not text.strip():
            return ""
        if source_lang.lower().startswith("zh") or target_lang.lower() not in {
            "zh",
            "zh-cn",
            "zh_cn",
            "chinese",
        }:
            # 本模型只做 en→zh；其它方向原样返回
            if not source_lang.lower().startswith("en"):
                return text

        key = _cache_key(self.name, str(self.model_dir), text, source_lang, target_lang)
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        chunks = _split_chunks(text)
        translated_parts = [self._translate_chunk(c) for c in chunks]
        # 段落之间用空行拼回，句间用空格
        result = "\n\n".join(p for p in translated_parts if p)
        self._write_cache(key, result, text)
        return result


def get_translator(backend: str = "passthrough", **kwargs) -> Translator:
    key = (backend or "passthrough").strip().lower()
    if key in {"", "none", "passthrough", "pass"}:
        return PassthroughTranslator()
    if key in {"local_opus", "opus", "opus-mt"}:
        return LocalOpusTranslator(
            model_dir=kwargs.get("model_dir"),
            cache_dir=kwargs.get("cache_dir"),
            use_cache=kwargs.get("use_cache", True),
        )
    if key in {"deepseek", "ds"}:
        raise ValueError("本轮未接入 DeepSeek 翻译后端；请用 passthrough 或 local_opus。")
    raise ValueError(f"未知翻译后端: {backend}")
