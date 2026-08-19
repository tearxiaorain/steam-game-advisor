# Steam Game Advisor

面向 Steam 玩家的对话式游戏顾问：用自然语言描述想玩的类型、人数、时长或氛围，系统检索游戏档案并解释为什么推荐。

## 能做什么

- 按约束找游戏（价格、中文、合作、平台等）
- 找相似款，并引用简介或标签说明理由
- 基于本地导入的库存，回答「今晚玩哪个」
- 查询某款游戏的档案信息（简介、配置、好评统计）

实时销量榜、同时在线人数需要单独的时效接口，第一版未接入；问到会明确说明，不会用过期档案冒充榜单。

## 目录

```
steam-game-advisor/
├── docs/                 # 规格与评测
├── data/
│   ├── raw/              # 原始抓取/导出
│   ├── processed/        # 入库用 Markdown
│   ├── library/          # 本地库存 app_id 列表
│   └── eval/             # 评测集
├── src/                  # 检索与生成代码
└── README.md
```

## 运行

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

在项目根目录创建 `.env`：

```
DEEPSEEK_API_KEY=你的密钥
```

将游戏档案放入 `data/processed/`（Markdown，可带 YAML 头），然后：

```bash
python src/main.py
```

档案格式见 `docs/字段与问题分类参考.md`。

## 当前进度

已接入 DeepSeek、本地中文嵌入模型与 13 款种子游戏档案。运行 `python src/main.py` 可在终端问答；每次问答会追加到 `data/eval/traces/traces.jsonl`（路由、改写、命中、回答）。
