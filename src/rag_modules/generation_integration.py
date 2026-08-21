"""生成集成：路由、重写、基于检索上下文作答。"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Set

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

VALID_ROUTES = {"recommend", "detail", "library", "trending"}

APP_ID_LABEL_RE = re.compile(
    r"(?:App\s*ID|app[_ ]?id)\s*[:：]?\s*(\d{3,7})",
    re.IGNORECASE,
)

GROUNDING_RULES = """
硬性约束（违反即错误）：
- 只能谈论下方「候选游戏」以及档案正文里写到的信息；禁止写出候选以外的游戏名或 App ID。
- 禁止用训练记忆补充档案里没有的游戏或事实。
- 价格、语言、平台、好评率、App ID 必须能在档案原文中找到；找不到就说档案没写，不要猜。
- 若用户点名的游戏不在候选里，说「知识库没有这款」，不要用无关游戏冒充。
- 若没有真正符合需求的游戏，直接说没有合适推荐，不要硬凑。
- 推荐时说明理由，并点明依据来自简介、标签或评价摘要中的哪一类。

口吻（面向玩家）：
- 不要写「允许列表」「您提供的档案」「补充说明里逐条否定」等内部/调试说法。
- 只写推荐的游戏及简短理由；不合适的游戏直接跳过，不要逐个解释为何不推荐。
- 候选里已给出档案正文的游戏，禁止说「档案中未出现 / 未提供 / 无法确认」。
"""


class GenerationIntegrationModule:
    def __init__(
        self,
        model_name: str = "deepseek-chat",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        logger.info("正在初始化 LLM: %s", self.model_name)
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        logger.info("LLM 初始化完成")

    def query_router(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            """将玩家问题分成下面四类之一，只返回类别单词：

recommend - 找游戏、按条件筛选、要相似款、比较哪款更适合、游戏荒了玩什么
detail - 问某一两款游戏是什么、配置、语言、好评、App ID、叫什么名字
library - 基于「我的库存 / 库里有的 / 今晚玩哪个 / 库里买了没玩 / 最近在玩」
trending - 本周榜、现在最热、同时在线、刚打折的实时热度

用户问题: {query}

分类:"""
        )
        chain = {"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
        result = chain.invoke(query).strip().lower()
        for route in VALID_ROUTES:
            if route in result:
                return route
        return "recommend"

    def query_rewrite(self, query: str) -> str:
        """单次检索改写：扩通用玩法词，并做硬约束清洗。

        默认可由 config.use_query_rewrite 关闭；Prompt 见 rewrite_prompts.py。
        """
        from config import DEFAULT_CONFIG

        if not DEFAULT_CONFIG.use_query_rewrite:
            text = query
            if DEFAULT_CONFIG.use_rewrite_hard_aliases:
                text = self.apply_rewrite_hard_aliases(query, query)
            return text or query

        from .rewrite_prompts import QUERY_REWRITE_PROMPT

        prompt = PromptTemplate(
            template=QUERY_REWRITE_PROMPT,
            input_variables=["query"],
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query}).strip()
        response = response.strip("\"'`").splitlines()[0].strip()
        response = self._sanitize_rewrite(query, response)
        response = self.apply_rewrite_hard_aliases(query, response)
        if response != query:
            logger.info("查询已重写: '%s' → '%s'", query, response)
        return response or query

    def expand_queries(self, query: str, n: int = 2) -> List[str]:
        """一次 LLM 调用生成至多 n 条检索变体；返回 [原句, ...变体]，去重。"""
        from config import DEFAULT_CONFIG

        if not DEFAULT_CONFIG.use_query_rewrite:
            return [query]

        from .rewrite_prompts import MULTI_QUERY_PROMPT

        n = max(1, min(int(n), 3))
        prompt = PromptTemplate(
            template=MULTI_QUERY_PROMPT,
            input_variables=["query", "n"],
        )
        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke({"query": query, "n": n}).strip()
        variants: List[str] = []
        for line in raw.splitlines():
            line = line.strip().lstrip("0123456789.-、)） ").strip("\"'`")
            line = self._sanitize_rewrite(query, line)
            if DEFAULT_CONFIG.use_rewrite_hard_aliases:
                line = self.apply_rewrite_hard_aliases(query, line)
            if not line or line == query:
                continue
            if line not in variants:
                variants.append(line)
            if len(variants) >= n:
                break

        result = [query]
        for item in variants:
            if item not in result:
                result.append(item)
        logger.info("多重查询变体: %s", result)
        return result

    @staticmethod
    def _sanitize_rewrite(original: str, rewritten: str) -> str:
        """去掉原句未出现的约束词，避免改写模型擅自加过滤条件。"""
        if not rewritten:
            return original
        text = rewritten
        guarded = [
            ("免费", ("免费开玩", "免费")),
            ("中文", ("简体中文", "繁体中文", "简体", "繁体", "中文")),
            ("合作", ("在线合作", "同屏合作", "合作")),
            ("开黑", ("开黑",)),
            ("Mac", ("macOS", "Mac OS", "Mac")),
            ("苹果", ("苹果",)),
        ]
        for trigger, tokens in guarded:
            if trigger in original:
                continue
            for token in sorted(tokens, key=len, reverse=True):
                text = text.replace(token, " ")

        if "单人" not in original:
            text = text.replace("单人", " ")
        if "单机" not in original and "单人" not in original:
            text = text.replace("单机", " ")

        text = re.sub(r"[，,]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" ，,")

        # 原句有的硬约束若被模型删掉，补回（过滤依赖原句，但 BM25/向量仍需要这些词）
        keep: List[str] = []
        if "免费" in original and "免费" not in text:
            keep.append("免费")
        if ("中文" in original or "简体" in original) and (
            "中文" not in text and "简体" not in text
        ):
            keep.append("中文")
        if ("联机" in original or "多人" in original) and (
            "联机" not in text and "多人" not in text
        ):
            keep.append("联机")
        if "合作" in original and "合作" not in text:
            keep.append("合作")
        if keep:
            text = f"{text} {' '.join(keep)}".strip()

        return text or original

    @staticmethod
    def apply_rewrite_hard_aliases(original: str, rewritten: str) -> str:
        """高置信问法 → 强制并入档案里更稳的检索词（不替代 LLM 改写）。"""
        try:
            from config import DEFAULT_CONFIG, REWRITE_HARD_ALIAS_RULES

            if not DEFAULT_CONFIG.use_rewrite_hard_aliases:
                return rewritten or original
        except Exception:
            return rewritten or original

        q = original or ""
        text = rewritten or original or ""
        added: List[str] = []
        for rule in REWRITE_HARD_ALIAS_RULES:
            exclude = rule.get("exclude_any") or []
            if any(tok in q for tok in exclude):
                continue
            groups = rule.get("need_any") or []
            if not groups:
                continue
            if not all(any(tok in q for tok in group) for group in groups):
                continue
            boost = str(rule.get("boost") or "").strip()
            if not boost:
                continue
            for tok in boost.split():
                if tok and tok not in text and tok not in added:
                    added.append(tok)
            logger.info("改写硬映射命中: %s", rule.get("id"))
        if not added:
            return text
        merged = f"{text} {' '.join(added)}".strip()
        return re.sub(r"\s+", " ", merged)

    def generate_recommend_answer(self, query: str, context_docs: List[Document]) -> str:
        if not context_docs:
            return "没有检索到符合条件的游戏档案。请换一种问法，或放宽价格、语言、平台等条件。"
        return self._generate_grounded(
            query,
            context_docs,
            kind="recommend",
        )

    def generate_detail_answer(self, query: str, context_docs: List[Document]) -> str:
        if not context_docs:
            return "没有检索到相关游戏档案，无法回答详情问题。"
        return self._generate_grounded(query, context_docs, kind="detail")

    def generate_library_answer(
        self,
        query: str,
        context_docs: List[Document],
        library_mode: str = "owned",
    ) -> str:
        if not context_docs:
            return "没有检索到库存相关的游戏档案。"
        return self._generate_grounded(
            query, context_docs, kind="library", library_mode=library_mode
        )

    def trending_unavailable_answer(self) -> str:
        return (
            "当前没有接入 Steam 实时榜单或在线人数接口，"
            "无法回答本周销量第一、此刻最热这类时效问题。"
            "你可以改问某类游戏推荐，或某一款游戏的档案信息。"
        )

    def generate_recommend_answer_stream(self, query: str, context_docs: List[Document]):
        # 先完整生成并做 grounding 校验，再一次性输出，避免流式半截无法校验
        yield self.generate_recommend_answer(query, context_docs)

    def generate_detail_answer_stream(self, query: str, context_docs: List[Document]):
        yield self.generate_detail_answer(query, context_docs)

    def _generate_grounded(
        self,
        query: str,
        context_docs: List[Document],
        kind: str,
        library_mode: str = "owned",
    ) -> str:
        from .library_profile import LIBRARY_MODE_HINTS

        context = self._build_context(context_docs)
        allow_block = self._format_allowlist(context_docs)
        task = {
            "recommend": (
                "根据档案为玩家推荐至多 3 款真正符合需求的游戏。"
                "每款写：游戏名、App ID、为什么符合需求（对应档案原文），并点明依据类型。"
                "只写推荐；不匹配的跳过，不要写筛选过程或否定清单。"
                "没有合适的就直接说没有合适推荐。"
            ),
            "detail": (
                "根据档案回答关于具体游戏的问题。"
                "比较题需要两边档案都提到；缺一边就说明缺哪边。"
                "若用户问的游戏不在候选里，直接说知识库没有。"
            ),
            "library": (
                "玩家问的是自己库存相关的问题。"
                + LIBRARY_MODE_HINTS.get(library_mode, LIBRARY_MODE_HINTS["owned"])
                + "候选若带有「近两周时长/总时长」，回答里要引用这些数字。"
                "档案没有单局时长时，可谨慎推断并标明是推断。"
            ),
        }[kind]

        prompt = ChatPromptTemplate.from_template(
            """你是 Steam 游戏顾问，用简体中文、简洁顾问口吻回答。
{task}
{rules}

候选游戏（只能谈这些）:
{allowlist}

用户问题: {question}

游戏档案:
{context}

回答:"""
        )
        chain = (
            {
                "question": RunnablePassthrough(),
                "context": lambda _: context,
                "rules": lambda _: GROUNDING_RULES,
                "allowlist": lambda _: allow_block,
                "task": lambda _: task,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        answer = chain.invoke(query)
        answer = self._enforce_app_id_grounding(answer, context_docs, query=query, kind=kind)
        return answer

    def _enforce_app_id_grounding(
        self,
        answer: str,
        docs: List[Document],
        *,
        query: str,
        kind: str,
    ) -> str:
        allowed = self._allowed_app_ids(docs)
        bad = self._labeled_app_ids(answer) - allowed
        if not bad:
            return answer

        logger.warning("Grounding 违规 App ID=%s，允许=%s，将重试一次", sorted(bad), sorted(allowed))
        context = self._build_context(docs)
        allow_block = self._format_allowlist(docs)
        retry_prompt = ChatPromptTemplate.from_template(
            """你上次的回答引用了不允许的 App ID: {bad_ids}。
请重写。硬性要求：
- 只能使用候选游戏中的游戏名与 App ID
- 不要出现任何其它游戏名或 App ID
- 若都不合适，明确说没有合适推荐
- 面向玩家：不要提「允许列表」，不要逐条否定不推荐的游戏
{rules}

候选游戏:
{allowlist}

用户问题: {question}

游戏档案:
{context}

回答:"""
        )
        chain = (
            {
                "question": RunnablePassthrough(),
                "context": lambda _: context,
                "rules": lambda _: GROUNDING_RULES,
                "allowlist": lambda _: allow_block,
                "bad_ids": lambda _: ", ".join(sorted(bad)),
            }
            | retry_prompt
            | self.llm
            | StrOutputParser()
        )
        retry = chain.invoke(query)
        bad2 = self._labeled_app_ids(retry) - allowed
        if not bad2:
            return retry

        logger.warning("Grounding 重试仍违规 App ID=%s，回退为安全摘要", sorted(bad2))
        return self._safe_fallback_answer(query, docs, kind=kind)

    def _safe_fallback_answer(self, query: str, docs: List[Document], *, kind: str) -> str:
        lines = [
            "根据当前检索到的档案，我只能基于以下游戏作答：",
            "",
        ]
        for doc in docs:
            name = doc.metadata.get("name_cn") or doc.metadata.get("name") or "未知"
            app_id = doc.metadata.get("app_id", "")
            price = doc.metadata.get("price_cny")
            review = doc.metadata.get("review_desc")
            bits = [f"- {name}（App ID: {app_id}）"]
            if price is not None:
                bits.append(f"价格约 {price}")
            if review:
                bits.append(str(review))
            lines.append("；".join(bits))
        lines.append("")
        if kind == "detail":
            lines.append(f"关于「{query}」，请对照上方档案中的简介、标签、评价摘要查看；档案没有写到的内容我不能补充。")
        else:
            lines.append(
                f"关于「{query}」，若上列游戏不符合你的需求，说明当前知识库缺少更匹配的档案，"
                "而不是我可以凭记忆再推荐其它游戏。"
            )
        return "\n".join(lines)

    @staticmethod
    def _allowed_app_ids(docs: List[Document]) -> Set[str]:
        return {str(d.metadata.get("app_id")) for d in docs if d.metadata.get("app_id")}

    @staticmethod
    def _labeled_app_ids(text: str) -> Set[str]:
        return set(APP_ID_LABEL_RE.findall(text or ""))

    @staticmethod
    def _format_allowlist(docs: List[Document]) -> str:
        rows = []
        for doc in docs:
            name = doc.metadata.get("name_cn") or doc.metadata.get("name") or "未知"
            app_id = doc.metadata.get("app_id", "")
            en = doc.metadata.get("name")
            if en and en != name:
                line = f"- {name} / {en} | App ID={app_id}"
            else:
                line = f"- {name} | App ID={app_id}"
            forever = doc.metadata.get("playtime_forever")
            weeks = doc.metadata.get("playtime_2weeks")
            if forever is not None or weeks is not None:
                line += (
                    f" | 总时长={int(forever or 0)}分钟"
                    f" | 近两周={int(weeks or 0)}分钟"
                )
            rows.append(line)
        return "\n".join(rows) if rows else "- （空）"

    def _build_context(self, docs: List[Document], max_length: int = 9000) -> str:
        """为每款候选保留一段正文，避免「名单有、正文被截断」导致模型瞎说档案未出现。"""
        if not docs:
            return "暂无相关游戏档案。"
        per = max(900, max_length // max(len(docs), 1))
        parts = []
        current_length = 0
        for i, doc in enumerate(docs, 1):
            header = (
                f"【游戏 {i}】 "
                f"{doc.metadata.get('name_cn') or doc.metadata.get('name', '未知')}"
            )
            if doc.metadata.get("app_id"):
                header += f" | app_id={doc.metadata['app_id']}"
            if doc.metadata.get("price_cny") is not None:
                header += f" | 价格约 {doc.metadata['price_cny']}"
            if doc.metadata.get("review_desc"):
                header += f" | {doc.metadata['review_desc']}"
            body = (doc.page_content or "").strip()
            if len(body) > per:
                body = body[: per - 20].rstrip() + "\n…（档案节选）"
            block = f"{header}\n{body}\n"
            if current_length + len(block) > max_length and parts:
                short_per = max(400, (max_length - current_length) - 80)
                if short_per < 200:
                    break
                body = (doc.page_content or "").strip()[:short_per]
                block = f"{header}\n{body}\n"
            parts.append(block)
            current_length += len(block)
        divider = "\n" + "=" * 50 + "\n"
        return divider + divider.join(parts)
