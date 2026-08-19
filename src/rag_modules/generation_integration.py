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
- 只能使用下方「允许列表」里的游戏；禁止写出列表外的游戏名或 App ID。
- 禁止用训练记忆补充未出现在档案中的游戏（例如守墓人、波西亚时光、浮岛物语等）。
- 价格、语言、平台、好评率、App ID 必须能在档案原文中找到；找不到就说档案没有。
- 若用户点名的游戏不在允许列表，明确说「知识库没有这款」，不要用无关游戏冒充。
- 若允许列表里没有真正符合需求的游戏，直接说明没有合适推荐，不要硬凑或编造。
- 推荐时说明理由，并点出依据来自简介、标签或评价摘要的哪一类信息。
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
library - 基于「我的库存 / 库里有的 / 今晚玩哪个 / 没玩过的」
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
        prompt = PromptTemplate(
            template="""你是游戏检索查询改写器。把玩家原话改写成更利于「中文游戏档案检索」的查询。

目标：提高召回，而不是改写玩家人设。

规则：
1. 保留原意与约束（免费、中文、价格、联机、单机等一定留下）。
2. 把圈内黑话/简称扩成商店简介里更常见的**通用描述词**（类型、玩法、氛围、视角、难度），不要只重复黑话。
3. 禁止发明或点名具体游戏名、App ID（除非原句已经出现该名字，则保留）。
4. 已是明确点名某游戏的详情问法：可轻微补「评价/类型/价格/语言」等检索词，不要扩成推荐向长句。
5. 输出一行中文关键词/短句即可，空格或逗号分隔，不要解释、不要引号、不要编号。

扩写参考（有则用，无则跳过）：
- 魂系/类魂 → 高难度 动作角色扮演 Boss战 探索 惩罚死亡 硬核
- 种田/治愈/养老 → 农场 种植 经营 模拟 放松 慢节奏 生活模拟
- 虫子王国/类银河战士空洞感 → 2D 动作冒险 平台跳跃 探索 独立 地图互联
- 侦探/人格对话/技能检定感 → 角色扮演 剧情 选择 文字 叙事 侦探 对话
- Roguelike 射击 → Roguelike 射击 合作 随机 通关失败重来
- 赛博/义体 → 赛博朋克 开放世界 第一人称 角色扮演 改造
- 抓宠打工/帕鲁感 → 生存 建造 捕捉 宠物 联机 制作
- 大鹅捣乱 → 鹅 恶作剧 休闲 独立 喜剧
- CRPG/D&D → 回合制 角色扮演 龙与地下城 队友 剧情分支
- 共斗狩猎 → 合作 多人 动作 战斗 Boss 联机

原始查询: {query}

改写结果:""",
            input_variables=["query"],
        )
        chain = {"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
        response = chain.invoke(query).strip()
        response = response.strip("\"'`").splitlines()[0].strip()
        if response != query:
            logger.info("查询已重写: '%s' → '%s'", query, response)
        return response or query

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

    def generate_library_answer(self, query: str, context_docs: List[Document]) -> str:
        if not context_docs:
            return "没有检索到库存相关的游戏档案。"
        return self._generate_grounded(query, context_docs, kind="library")

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
    ) -> str:
        context = self._build_context(context_docs)
        allow_block = self._format_allowlist(context_docs)
        task = {
            "recommend": (
                "根据档案为玩家推荐 3 款以内游戏。"
                "每款必须写：游戏名、App ID、为什么符合需求（对应档案原文）。"
                "只能从允许列表中选；列表里都不合适就说明没有合适推荐。"
            ),
            "detail": (
                "根据档案回答关于具体游戏的问题。"
                "比较题需要两边档案都提到；缺一边就说明缺哪边。"
                "若用户问的游戏不在允许列表，直接说知识库没有。"
            ),
            "library": (
                "玩家问的是自己库存相关的问题。"
                "只讨论允许列表中的游戏；若问今晚玩哪个，给一个主推荐并说明原因。"
                "档案没有单局时长时，可谨慎推断并标明是推断。"
            ),
        }[kind]

        prompt = ChatPromptTemplate.from_template(
            """你是 Steam 游戏顾问。
{task}
{rules}

允许列表（只能谈这些游戏）:
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
请重写回答。硬性要求：
- 只能使用允许列表中的游戏与 App ID
- 不要出现任何其它游戏名或 App ID
- 若都不合适，明确说知识库没有合适推荐
{rules}

允许列表:
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
            "根据当前检索到的档案，我只能基于以下游戏作答（已拦截档案外的编造推荐）：",
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
                rows.append(f"- {name} / {en} | App ID={app_id}")
            else:
                rows.append(f"- {name} | App ID={app_id}")
        return "\n".join(rows) if rows else "- （空）"

    def _build_context(self, docs: List[Document], max_length: int = 4000) -> str:
        if not docs:
            return "暂无相关游戏档案。"
        parts = []
        current_length = 0
        for i, doc in enumerate(docs, 1):
            header = f"【游戏 {i}】 {doc.metadata.get('name', '未知')}"
            if doc.metadata.get("app_id"):
                header += f" | app_id={doc.metadata['app_id']}"
            if doc.metadata.get("price_cny") is not None:
                header += f" | 价格约 {doc.metadata['price_cny']}"
            if doc.metadata.get("review_desc"):
                header += f" | {doc.metadata['review_desc']}"
            block = f"{header}\n{doc.page_content}\n"
            if current_length + len(block) > max_length:
                break
            parts.append(block)
            current_length += len(block)
        divider = "\n" + "=" * 50 + "\n"
        return divider + divider.join(parts)
