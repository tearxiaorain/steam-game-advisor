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

# 推荐改写硬映射：原句同时命中 need_any 各组后，强制并入 boost 检索词（高置信少量）
# need_any: 每一组至少命中一个词；boost: 空格分词并入改写结果
REWRITE_HARD_ALIAS_RULES: List[Dict[str, Any]] = [
    {
        "id": "r6-siege",
        "need_any": [["破门"], ["运营商", "干员", "战术射击", "写实战术"]],
        "boost": "彩虹六号 围攻 战术射击 干员 人质 破门",
    },
    {
        "id": "lethal-company",
        "need_any": [["月亮", "月球"], ["公司", "捡垃圾", "配额", "废料"]],
        "boost": "致命公司 Lethal Company 月球 废料 配额 联机恐怖",
    },
    {
        "id": "pubg",
        "need_any": [["大逃杀", "吃鸡"], ["跳伞", "搜枪", "搜刮"]],
        "boost": "绝地求生 PUBG 大逃杀 跳伞 战术竞技",
    },
    {
        "id": "among-us",
        "need_any": [["内鬼"], ["任务", "社交推理", "投票"]],
        "boost": "Among Us 内鬼 太空 投票 社交推理 派对",
    },
    {
        "id": "moba-dota",
        "need_any": [["MOBA", "moba"], ["推塔", "打野", "团战"]],
        "boost": "Dota 刀塔 MOBA 推塔 打野 团战 英雄",
    },
    {
        "id": "vampire-survivors",
        "need_any": [["幸存者"], ["站桩", "清屏", "自动射击", "自动攻击"]],
        "boost": "吸血鬼幸存者 幸存者 站桩 自动攻击 清屏 割草",
    },
    {
        "id": "isaac",
        "need_any": [["以撒"]],
        "boost": "以撒的结合 弹幕 地牢 Roguelike 房间",
    },
    {
        "id": "hades",
        "need_any": [["希腊神话", "冥界"], ["Roguelike", "roguelike", "往上打", "出货"]],
        "boost": "哈迪斯 Hades 希腊神话 冥界 动作 Roguelike",
    },
    {
        "id": "hogwarts",
        "need_any": [["魔法学校"], ["咒语", "城堡", "上课"]],
        "boost": "霍格沃茨之遗 魔法学校 巫师 咒语 城堡",
    },
    {
        "id": "phasmophobia",
        "need_any": [["手电", "手电筒", "拿手电"], ["鬼屋", "灵异", "调查"]],
        "boost": "恐鬼症 手电筒 鬼屋 灵异 调查 证据",
    },
    {
        "id": "l4d",
        "need_any": [["僵尸"], ["过关", "四人", "四个人"], ["射击", "开黑", "合作"]],
        "boost": "求生之路 合作射击 僵尸 过关 四人",
    },
    {
        "id": "cs-free",
        "need_any": [["免费"], ["竞技", "枪战", "FPS", "射击"]],
        "boost": "反恐精英 Counter-Strike CS2 竞技 枪战",
        # 避免「免费 MOBA」误触：见 apply 里额外排除
        "exclude_any": ["MOBA", "moba", "推塔", "打野"],
    },
    {
        "id": "fallout",
        "need_any": [["废土"], ["定居点", "建定居"]],
        "boost": "辐射 废土 定居点 开放世界 RPG 第一人称",
    },
    {
        "id": "dark-souls",
        "need_any": [["黑暗魂", "黑暗之魂", "魂系列"], ["不是开放", "非开放", "关卡", "不骑马"]],
        "boost": "黑暗之魂 关卡制 线性 高难度 魂系",
    },
    {
        "id": "celeste",
        "need_any": [["精准", "山峰"], ["平台", "跳跃", "辅助模式"]],
        "boost": "蔚蓝 Celeste 精准平台 山峰 辅助模式",
    },
]

# 切块 section 权重（软版）：游玩方式不降；简介/标签微升；配置不进索引
SECTION_WEIGHTS: Dict[str, float] = {
    "简介": 1.2,
    "类型与标签": 1.15,
    "语言": 1.05,
    "评价摘要": 1.0,
    "游玩方式": 1.0,
    "配置与平台": 0.0,
}

# 游玩方式：索引侧规则降噪（父文档仍保留全文给生成）
PLAYSTYLE_DENOISE_MAX_CHARS: int = 420
PLAYSTYLE_DROP_LINE_PATTERNS: List[str] = [
    r"定期.*(更新|发布)",
    r"海量内容更新",
    r"每次更新都会",
    r"继续通过定期更新",
    r"远程畅玩",
    r"即使没有\s*WiFi",
    r"服务条款",
    r"隐私政策",
    r"内购",
    r"真实金钱",
    r"YouTube|Twitch",
    r"streamers@",
    r"请联系我们",
    r"Ninja Kiwi",
    r"奖杯商店",
    r"内容浏览器",
    r"可解锁的皮肤",
    r"我们十分尊重你投入的时间",
    r"如果你有任何意见和建议",
    r"^\*+$",
]
# 索引中直接排除的 section（不依赖 section_weight 开关）
INDEX_EXCLUDE_SECTIONS: List[str] = ["配置与平台"]


def _embedding_model_path() -> str:
    if (LOCAL_EMBEDDING_DIR / "model.safetensors").exists():
        return str(LOCAL_EMBEDDING_DIR)
    return "BAAI/bge-small-zh-v1.5"


@dataclass
class RAGConfig:
    data_path: str = str(PROJECT_ROOT / "data" / "processed")
    library_path: str = str(PROJECT_ROOT / "data" / "library" / "owned_appids.json")
    me_owned_path: str = str(PROJECT_ROOT / "data" / "library" / "me_owned.json")
    friends_dir: str = str(PROJECT_ROOT / "data" / "library" / "friends" / "by_steamid")
    index_save_path: str = str(PROJECT_ROOT / "vector_index")
    trace_path: str = str(PROJECT_ROOT / "data" / "eval" / "traces" / "traces.jsonl")

    # 拥有度偏置：默认仅「好友向推荐」关键词触发；全局 recommend 不开
    use_ownership_bias: bool = True
    ownership_bias_friends_only: bool = True
    ownership_pool_size: int = 24
    ownership_filter_longtail: bool = True
    ownership_use_score_boost: bool = True
    ownership_me_factor: float = 1.15  # 合玩时本人也有略加分
    ownership_multi_friend_factor: float = 1.25  # >=3 好友
    ownership_duo_friend_factor: float = 1.12  # 2 好友
    ownership_longtail_factor: float = 0.75  # 仅 1 好友

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

    # 切块类型加权（软版代码保留）；v2 评测 soft 仍 63%→57%，默认关
    use_section_weights: bool = False
    # 标签广度惩罚：仅按 genres 数量；误用 categories 曾掉到 37%，默认关
    use_tag_breadth_penalty: bool = False
    tag_breadth_free: int = 2  # genres 数 ≤ 此值不惩罚
    tag_breadth_alpha: float = 0.12  # 每多 1 个 genre 约降一成

    # 游玩方式索引降噪：规则去营销句 + 超长截断；生成仍用父文档全文
    use_playstyle_denoise: bool = True
    playstyle_denoise_max_chars: int = PLAYSTYLE_DENOISE_MAX_CHARS
    # 标签词表：类型块索引只保留玩法 genres/categories
    use_taxonomy_scrub: bool = True
    taxonomy_path: str = str(PROJECT_ROOT / "data" / "library" / "tag_taxonomy.json")
    # LLM 查询改写（Prompt 归档在 rewrite_prompts.py）；默认关
    use_query_rewrite: bool = False
    # 硬映射与 LLM 改写独立；默认关，便于对照「用户标签入库」效果
    use_rewrite_hard_aliases: bool = False

    # 问句与 user_tags 字面重叠时对 RRF 分加分（推荐/过滤检索均生效）
    use_user_tag_overlap_boost: bool = True
    user_tag_overlap_bonus: float = 0.012  # 每命中 1 个标签加至 rrf_score
    user_tag_overlap_max: int = 4  # 单块最多计分标签数

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = RAGConfig()
