"""生成集成：路由、重写、基于检索上下文作答。"""

import logging
import os
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

VALID_ROUTES = {"recommend", "detail", "library", "trending"}

GROUNDING_RULES = """
约束：
- 只依据给定游戏档案回答。价格、语言、平台、好评率、App ID 必须能在档案中找到。
- 档案没有的剧情考据、战斗公式、补丁说明、实时榜单，明确说知识库没有，不要编造。
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
            template="""把模糊的玩游戏需求改写成更利于检索的中文查询。
已经很具体的问题（含游戏名、明确标签、明确配置）保持原句。
可以补上类型、人数、时长、氛围等检索词，但不要改变原意，不要发明游戏名。

原始查询: {query}

只输出最终查询:""",
            input_variables=["query"],
        )
        chain = {"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
        response = chain.invoke(query).strip()
        if response != query:
            logger.info("查询已重写: '%s' → '%s'", query, response)
        return response

    def generate_recommend_answer(self, query: str, context_docs: List[Document]) -> str:
        if not context_docs:
            return "没有检索到符合条件的游戏档案。请换一种问法，或放宽价格、语言、平台等条件。"
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            """你是 Steam 游戏顾问。根据档案为玩家推荐 3 款以内游戏。
每款写：游戏名、App ID（若有）、为什么符合需求（必须能对应到档案原文）。
不要推荐档案里没有的游戏。
{rules}

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
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def generate_detail_answer(self, query: str, context_docs: List[Document]) -> str:
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            """你是 Steam 游戏顾问。根据档案回答关于具体游戏的问题。
比较题需要两边档案都提到；缺一边就说明缺哪边。
{rules}

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
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def generate_library_answer(self, query: str, context_docs: List[Document]) -> str:
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            """你是 Steam 游戏顾问。玩家问的是自己库存相关的问题。
只讨论下面这些已检索到的游戏档案，不要引入档案外的库内游戏。
若问题是「今晚玩哪个」，在这些候选里给一个主推荐并说明原因。
若档案没有单局时长，可以依据节奏做谨慎推断，并标明这是推断。
{rules}

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
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def trending_unavailable_answer(self) -> str:
        return (
            "当前没有接入 Steam 实时榜单或在线人数接口，"
            "无法回答本周销量第一、此刻最热这类时效问题。"
            "你可以改问某类游戏推荐，或某一款游戏的档案信息。"
        )

    def generate_recommend_answer_stream(self, query: str, context_docs: List[Document]):
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            """你是 Steam 游戏顾问。根据档案为玩家推荐 3 款以内游戏，并解释理由。
{rules}

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
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        yield from chain.stream(query)

    def generate_detail_answer_stream(self, query: str, context_docs: List[Document]):
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            """你是 Steam 游戏顾问。根据档案回答具体游戏问题。
{rules}

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
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        yield from chain.stream(query)

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
