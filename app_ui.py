"""本机 Streamlit UI：Steam 游戏顾问。

启动（在项目根目录）:
  pip install streamlit
  python -m streamlit run app_ui.py --server.fileWatcherType none
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

ROUTE_LABELS = {
    "recommend": "推荐",
    "detail": "详情",
    "library": "库存",
    "trending": "热门/实时（拒答）",
}

EXAMPLE_QUERIES = [
    "想玩大逃杀吃鸡，跳伞落地搜枪打架",
    "空洞骑士是什么类型的游戏",
    "想玩卡牌构筑 Roguelike，一层层爬塔选牌组羁绊",
    "这周 Steam 销量榜第一是哪个",
]


@st.cache_resource(show_spinner="正在加载知识库与嵌入模型…")
def load_advisor():
    # 延迟导入，避免 Streamlit 启动时扫 transformers 导致崩溃
    from main import SteamGameAdvisor

    advisor = SteamGameAdvisor()
    advisor.initialize_system()
    advisor.build_knowledge_base(force_rebuild=False)
    return advisor


def main() -> None:
    st.set_page_config(
        page_title="Steam 游戏顾问",
        page_icon="🎮",
        layout="centered",
    )
    st.title("Steam 游戏顾问")
    st.caption(
        "本地 RAG 演示 · 约 760 款档案 · 社区标签 overlap · "
        "无实时榜单 / 库存题依赖本机 me_owned"
    )

    try:
        advisor = load_advisor()
    except Exception as exc:
        st.error(f"初始化失败：{exc}")
        st.info("请确认已设置 `.env` 中的 `DEEPSEEK_API_KEY`，且 `data/processed` 与 `vector_index` 可用。")
        return

    with st.sidebar:
        st.subheader("示例问法")
        for q in EXAMPLE_QUERIES:
            if st.button(q, use_container_width=True, key=f"ex_{hash(q)}"):
                st.session_state["prefill"] = q
        st.divider()
        st.markdown(
            f"- 改写: `{'开' if advisor.config.use_query_rewrite else '关'}`\n"
            f"- 硬映射: `{'开' if advisor.config.use_rewrite_hard_aliases else '关'}`\n"
            f"- 标签重叠: `{'开' if advisor.config.use_user_tag_overlap_boost else '关'}`"
        )

    prefill = st.session_state.pop("prefill", None)
    question = st.text_area(
        "你的问题",
        value=prefill or "",
        height=100,
        placeholder="例如：想玩希腊神话 Roguelike，从冥界往上打…",
    )

    if st.button("提问", type="primary", use_container_width=True) and question.strip():
        with st.spinner("检索与生成中…"):
            try:
                result = advisor.ask(question.strip())
            except Exception as exc:
                st.exception(exc)
                return

        route = result.get("route") or ""
        st.markdown(f"**路由：** {ROUTE_LABELS.get(route, route)} (`{route}`)")
        if result.get("library_mode"):
            st.caption(f"库存策略：{result['library_mode']}")
        if result.get("filters"):
            st.caption(f"过滤：{result['filters']}")

        hits = result.get("hits") or []
        if hits:
            st.subheader("检索 Top 结果")
            for i, h in enumerate(hits, 1):
                title = h.get("name_cn") or h.get("name") or "未知"
                app_id = h.get("app_id") or ""
                tags = h.get("user_tags") or []
                line = f"{i}. **{title}** · App ID `{app_id}`"
                if tags:
                    line += f"  \n标签：{', '.join(tags)}"
                st.markdown(line)
        else:
            st.info("本轮无检索命中（拒答或空库）。")

        st.subheader("回答")
        st.markdown(result.get("answer") or "")


if __name__ == "__main__":
    main()
