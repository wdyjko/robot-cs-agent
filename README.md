# 智能扫地机器人客服 Agent

基于 **FastAPI + LangGraph + LangChain + Streamlit** 的扫地 / 扫拖一体机器人智能客服。
向量库支持 **Chroma / 内置 NumPy 向量库**（自动探测降级），向量化支持 **云端 Embedding / 本地 bge-m3 / 内置哈希向量器**（逐级降级）。

支持故障排查、维护保养、使用技巧、选购建议，并结合**用户所在地、当前时间、天气、季节、湿度**给出针对性建议；
所有回答**先检索知识库、再总结、后生成**，不编造知识库中不存在的内容。

---

## 一、核心特性

| 能力 | 说明 |
| --- | --- |
| 知识库检索（RAG） | 6 份知识库文件按**条目/问答**切分（不是按字数），元数据含 `source_file` / `category` / `item_no` / `title` / `tags` |
| 混合检索 | 向量 Top10 + BM25 Top10 → RRF 融合去重 → Top5（重排接口已预留） |
| Agent 编排 | LangGraph 六节点：意图 → 工具 → 改写 → 检索 → 摘要 → 回答 |
| 实时工具 | 位置（手动 > 画像 > IP > 默认城市）、时间（日期/季节/星期）、天气（温度/湿度/降水/风力） |
| 环境适配 | 湿度 >80% 少湿拖、雨天/回南天除湿、>35℃ 避免直射充电、<5℃ 静置 30 分钟、沙尘天关窗 + HEPA、木地板低水量、宠物家庭防缠绕 |
| 安全兜底 | 电池鼓包 / 漏液 / 烧焦味 / 主板短路 → 强制提示**立即停用、断电、联系售后** |
| 记忆 | SQLite 保存多轮对话与用户画像（机型 / 地板 / 宠物 / 婴儿） |
| 向量化可选 | 云端 Embedding（OpenAI 兼容，低配机器推荐）→ 本地 `bge-m3` → 内置哈希向量器，逐级自动降级 |
| 降级可用 | 未配置 LLM Key 或天气 Key 时自动走纯检索 / mock 数据，项目始终可跑通 |

---

## 二、目录结构

```text
robot-cs-agent/
├─ backend/
│  ├─ app/
│  │  ├─ __init__.py
│  │  ├─ main.py                 # FastAPI 入口（CORS / 路由 / 启动建表 / /health）
│  │  ├─ api/
│  │  │  ├─ chat.py              # POST /api/chat、/api/chat/stream（SSE）、历史管理
│  │  │  ├─ kb.py                # POST /api/kb/rebuild、GET /api/kb/status
│  │  │  └─ tools.py             # /api/weather、/api/location、/api/datetime、/api/profile
│  │  ├─ core/
│  │  │  ├─ config.py            # pydantic-settings 读取 .env，单例 settings
│  │  │  ├─ prompts.py           # 4 个 Prompt 模板 + 环境映射规则 + 安全关键词
│  │  │  └─ logging.py           # 统一 logging
│  │  ├─ services/
│  │  │  ├─ rag_service.py       # 加载 / 清洗 / 按条目切分 / 建库 / 混合检索
│  │  │  ├─ agent_service.py     # LangGraph 六节点编排，run(req) -> ChatResponse
│  │  │  ├─ summary_service.py   # 检索片段 → 结构化 JSON（summary/steps/warnings/sources）
│  │  │  ├─ memory_service.py    # 对话历史 + 用户画像（SQLite）
│  │  │  ├─ embeddings.py        # Embedding 工厂（云端 → bge-m3 → 哈希，逐级降级）
│  │  │  ├─ api_embeddings.py    # 云端 Embedding（OpenAI 兼容 / DashScope 原生接口）
│  │  │  ├─ llm_service.py       # LLM 工厂（OpenAI 兼容，未配 Key 优雅降级）
│  │  │  ├─ vectorstore_service.py  # 向量库后端选择（子进程探测 Chroma / 自动切本地）
│  │  │  └─ local_vectorstore.py    # 内置 NumPy 向量库（Chroma 不可用时的兜底）
│  │  ├─ tools/
│  │  │  ├─ location.py          # get_user_location
│  │  │  ├─ weather.py           # get_weather
│  │  │  ├─ datetime_tool.py     # get_current_datetime
│  │  │  ├─ profile.py           # get_user_profile
│  │  │  └─ kb_tool.py           # search_knowledge_base
│  │  ├─ models/                 # 源码包：Pydantic 模型 + SQLAlchemy 表（**不是**模型权重目录）
│  │  │  ├─ schemas.py           # Pydantic 请求/响应模型
│  │  │  └─ db.py                # SQLAlchemy：chat_history、user_profile
│  │  └─ vectorstore/            # 占位目录（真实持久化目录见 CHROMA_PERSIST_DIR）
│  ├─ requirements.txt
│  └─ Dockerfile
├─ frontend/
│  ├─ streamlit_app.py
│  └─ Dockerfile
├─ data/
│  ├─ raw/                       # 6 份知识库文件
│  └─ processed/
├─ vectorstore/                  # 向量库持久化目录（根目录的 /vectorstore/ 已被 .gitignore 忽略）
├─ eval/                         # 检索效果评估：脚本 + 结果（唯一的效果证据来源）
│  ├─ README.md                  #   评测集构造方法、复现步骤、结论
│  ├─ eval_retrieval.py          #   逐题明细（可切换向量器 / 向量库）
│  ├─ eval_ablation.py           #   两种向量器 × 6 种融合策略的消融对照
│  └─ results/                   #   脚本原始输出（勿手改）
├─ tests/
│  └─ test_smoke.py              # 交付验证：6 项检查，不需要任何 API Key
├─ scripts/
│  └─ build_kb.py                # 独立建库脚本
├─ docker-compose.yml
├─ .env.example
├─ .gitignore
├─ setup.sh                      # macOS / Linux 一键安装
├─ setup.bat                     # Windows 一键安装
└─ README.md
```

> ⚠️ **`.gitignore` 里的目录规则必须带前导 `/`（根锚定）**，否则会误伤同名源码包：
>
> ```gitignore
> /vectorstore/     # ✅ 只忽略根目录下的向量库
> /vectorstore_*/   # ✅ 多套向量库目录（评测对照用）
> /models/          # ✅ 只忽略根目录下的模型权重缓存
>
> vectorstore/      # ❌ 会匹配任意层级，误伤 backend/app/vectorstore/
> models/           # ❌ 会匹配任意层级，误伤 backend/app/models/（源码包！）
> ```
>
> `models/` 这种**不带前导斜杠**的写法在 gitignore 语义里会匹配**任意深度**的同名目录。
> 本项目 `backend/app/models/` 是**应用自己的 Python 包**（`schemas.py` / `db.py`），
> 一旦被忽略，克隆仓库后后端会直接 `ModuleNotFoundError: No module named 'app.models'` 起不来。
> 同理 `backend/app/vectorstore/` 也是源码目录，不能写成裸 `vectorstore/`。
>
> 另外：**gitignore 不支持行尾注释**，注释必须单独成行（`/models/  # 注释` 会被当成路径的一部分）。

---

## 三、环境配置说明

### 3.1 前置要求

| 项目 | 要求 |
| --- | --- |
| Python | **3.10 ~ 3.13**（推荐 3.11，依赖预编译包最全） |
| 内存 | ≥ 8 GB（加载本地 bge-m3 时建议 16 GB） |
| 磁盘 | ≥ 6 GB（依赖约 3 GB + 本地 Embedding 模型约 2.3 GB） |
| 网络 | 首次安装依赖与下载本地 Embedding 模型需要联网（用云端 Embedding 时只需能访问接口） |

> **内存不足 8 GB 的机器也有可用路径**（例如 2 核 2G 的 ECS）：**默认配置会尝试加载本地
> bge-m3（2.3 GB 权重，进程内存超 3 GB）而被 OOM Killer 杀掉**，此时请改用下面两种方案之一：
>
> | 方案 | 配置 | 检索质量 | 说明 |
> | --- | --- | --- | --- |
> | **① 云端 Embedding（推荐）** | 填 `EMBEDDING_API_KEY` 等（见 3.4） | **优于**本地 bge-m3（可用生产级向量模型） | 不需要 torch，内存占用可忽略；建库与提问各需一次 HTTP 请求 |
> | ② 内置哈希向量器 | `EMBEDDING_ALLOW_DOWNLOAD=false` | 明显下降（**无语义能力**） | 零依赖、建库约 12 秒；关键词型提问尚可，语义型提问（如「这个月怎么保养」）会漏召回 |
>
> 两种方案的完整配置见 [3.5 关键环境变量一览](#35-关键环境变量一览) 与 [`.env.example`](.env.example) 末尾的「场景组合参考」。

> **本项目强制使用项目根目录下的 `.venv`，所有依赖只安装到 `.venv` 中，不做全局安装。**

### 3.2 一键安装（推荐）

**Windows（CMD / PowerShell）：**

```bat
cd robot-cs-agent
setup.bat
```

**macOS / Linux：**

```bash
cd robot-cs-agent
chmod +x setup.sh
bash setup.sh
```

脚本会自动完成：

1. 校验 Python 版本；
2. 在 **项目根目录** 创建 `.venv`；
3. 安装 **CPU 版 torch**（约 200 MB，比默认 CUDA 版小很多，本项目用 CPU 推理即可）；
4. 安装 `backend/requirements.txt` 全部依赖；
5. 若 `.env` 不存在，自动从 `.env.example` 复制一份。

可选参数：

```bat
setup.bat build-kb        REM 安装完成后顺便构建知识库
setup.bat --skip-torch    REM 已装过 torch，跳过以节省时间
```

```bash
bash setup.sh --build-kb
bash setup.sh --skip-torch
```

> 💡 **只用云端 Embedding（低配机器 / 不用本地模型）时可以加 `--skip-torch`**：
> 云端方案不加载 sentence-transformers，torch 装不装都不影响运行，能省下约 200 MB 与安装时间。
> 内置哈希向量器方案同理。

### 3.3 手动安装（等价步骤）

**Windows：**

```bat
cd robot-cs-agent
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
copy .env.example .env
```

**macOS / Linux：**

```bash
cd robot-cs-agent
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r backend/requirements.txt
cp .env.example .env
```

> 之后所有命令都用 `.venv` 里的解释器显式执行，**无需 activate**，也不会污染全局环境。
> 如需激活：Windows `.venv\Scripts\activate`，macOS/Linux `source .venv/bin/activate`。

### 3.4 配置 `.env`（需要你手动填写 API Key）

安装脚本只会生成 `.env` 文件，**Key 需要你自己去项目根目录下编辑 `.env` 手动填写**：

```env
# ---- LLM（OpenAI 兼容接口）----
OPENAI_API_KEY=sk-xxx                              # ← 在这里填你的 Key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# ---- Embedding ----
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_MODEL_PATH=                              # 本地已有模型时填目录，跳过联网下载
EMBEDDING_ALLOW_DOWNLOAD=true                      # 环境里 torch 不可用时改为 false（走内置向量器）

# ---- 向量库 / 数据 ----
CHROMA_PERSIST_DIR=./vectorstore
CHROMA_COLLECTION=robot_customer_service
VECTOR_BACKEND=auto                                # auto / chroma / local
DATA_RAW_DIR=./data/raw
SQLITE_URL=sqlite:///./robot_cs.db

# ---- 可选：天气 API ----
WEATHER_API_KEY=                                   # 不填则天气使用 mock 数据
DEFAULT_CITY=杭州
```

**常用模型服务商配置示例：**

| 服务商 | `OPENAI_BASE_URL` | `LLM_MODEL` 示例 |
| --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 阿里云百炼（Qwen） | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |

> **不配置 Key 也能跑**：LLM 相关能力自动降级为「知识库检索原文整理」，天气使用 mock 数据，
> 界面会明确提示当前处于降级模式，绝不编造内容。

#### 3.4.1 可选：改用云端 Embedding（低配机器的推荐方案）

本地 bge-m3 需要约 2.3 GB 显存/内存，**2 核 2G 的机器加载会被 OOM 杀掉**。填了
`EMBEDDING_API_KEY` 后，`get_embeddings()` 会**优先**走云端接口，不再加载本地模型；
云端不可用时自动回落到原有逻辑（本地模型 / 内置向量器），**不会让服务起不来**。

```env
# ---- 云端 Embedding（OpenAI 兼容接口，可选）----
EMBEDDING_API_KEY=sk-你的百炼Key
# 必须带 /compatible-mode/v1 后缀，否则请求路径不对
EMBEDDING_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_API_MODEL=text-embedding-v4
EMBEDDING_API_BATCH=10                             # 单次最多传几条文本（v3/v4 上限为 10）
EMBEDDING_API_DIM=                                 # 留空用模型默认维度
EMBEDDING_API_TIMEOUT=60
```

**可选模型**（阿里云百炼 OpenAI 兼容接口，见[官方文档](https://help.aliyun.com/zh/model-studio/embedding-interfaces-compatible-with-openai)）：

| 模型名 | 默认维度 | 可选维度 | 单次最大行数 |
| --- | --- | --- | --- |
| `qwen3.7-text-embedding` | 1024 | 2560 / 2048 / 1536 / 1024 / 768 / 512 / 256 | 20 |
| `qwen3.7-text-embedding-flash` | 1024 | 1024 / 768 / 512 / 256 | 20 |
| `text-embedding-v4` | 1024 | 2048 / 1536 / 1024 / 768 / 512 / 256 / 128 / 64 | 10 |
| `text-embedding-v3` | 1024 | 1024 / 768 / 512 / 256 / 128 / 64 | 10 |

也可以填**任意 OpenAI 兼容**的 Embedding 服务（如 OpenAI 的 `text-embedding-3-small`），
把 `EMBEDDING_API_BASE` 换成对应的 `.../v1` 即可。

> ⚠️ **两点注意**：
> 1. `EMBEDDING_API_BASE` 一定要带 `/compatible-mode/v1` 后缀。百炼**业务空间专属域名**
>    （`https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com`）也是 OpenAI 兼容入口，可以配合
>    该后缀使用；但**不要**直接把 `/embeddings` 拼在裸域名后面（会 404）。
> 2. **换了 Embedding 模型（或维度）必须重建知识库**，否则新旧向量维度不一致会检索失败。
>    `/api/kb/status` 的 `needs_rebuild` 会提示你。

### 3.5 关键环境变量一览

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 空 | 留空则禁用 LLM（有 mock 兜底） |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI 兼容地址 |
| `LLM_MODEL` | `gpt-4o-mini` | 模型名 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 本地向量模型；加载失败自动降级为内置哈希向量器 |
| `EMBEDDING_MODEL_PATH` | 空 | 指向本地模型目录，避免重复下载 |
| `EMBEDDING_ALLOW_DOWNLOAD` | `true` | 为 `false` 时直接用内置哈希向量器（不需要 torch） |
| `EMBEDDING_API_KEY` | 空 | **填了就走云端 Embedding**（优先于本地模型），低配机器推荐 |
| `EMBEDDING_API_BASE` | 百炼 compatible-mode | 云端 Embedding 的 OpenAI 兼容地址，须带 `/compatible-mode/v1` |
| `EMBEDDING_API_MODEL` | `text-embedding-v4` | 云端向量模型名（可选 `qwen3.7-text-embedding-flash` 等） |
| `EMBEDDING_API_BATCH` | `10` | 云端单次请求最多几条文本（v3/v4 上限为 10） |
| `EMBEDDING_API_DIM` | `0` | 云端向量维度；`0` 表示用模型默认维度 |
| `EMBEDDING_API_TIMEOUT` | `60` | 云端 Embedding 单次请求超时（秒） |
| `HF_HUB_OFFLINE` | `auto` | `auto` 时只要本地已有完整模型缓存就切离线模式，跳过逐文件联网检查（**能显著加快启动**）；也可强制 `true`/`false` |
| `HF_ENDPOINT` | 空 | HuggingFace 镜像地址，如 `https://hf-mirror.com`（国内网络建议配置） |
| `CHROMA_PERSIST_DIR` | `./vectorstore` | 向量库持久化目录 |
| `CHROMA_COLLECTION` | `robot_customer_service` | 集合名称 |
| `VECTOR_BACKEND` | `auto` | `auto` 自动探测 / `chroma` 强制 Chroma / `local` 内置 NumPy 向量库。**已确定不用 chromadb 时建议显式设 `local`**：`auto` 会在子进程里真跑一次 chromadb 写入+查询来探测可用性，虽稳健但属于多余的启动开销 |
| `DATA_RAW_DIR` | `./data/raw` | 知识库原始文件目录 |
| `SQLITE_URL` | `sqlite:///./robot_cs.db` | 记忆库（自动转成项目内绝对路径） |
| `WEATHER_API_KEY` | 空 | 留空则天气走 mock |
| `DEFAULT_CITY` | `杭州` | 无法定位时的默认城市 |
| `RERANK_ENABLED` | `false` | 是否启用 `bge-reranker-v2-m3` 重排（预留） |

### 3.6 运行环境自适应（重要）

这套代码刻意做到了「**依赖缺失也能跑通**」。正常环境（普通终端、Docker）下 `bge-m3` + Chroma 都能正常工作；
但在**受限运行环境**（受限沙箱、被禁止创建命名管道的进程、缺少 VC++ 运行库的系统）里，几个原生组件会整体失效：

| 组件 | 受限环境下的表现 | 本项目的处理方式 |
| --- | --- | --- |
| `torch` / `sentence-transformers` | `WinError 1114 动态链接库(DLL)初始化例程失败`（加载 `c10.dll` 失败） | Embedding 自动降级为内置 `HashingEmbeddings`（配置了云端 Embedding 时则直接走云端，根本不会碰 torch）；`langchain_text_splitters` / `langchain_community.document_loaders` 改为可选依赖，改用内置的递归切分器与 pypdf 加载器 |
| `chromadb` 原生引擎 | 写入时进程直接**访问违例崩溃**（`0xC0000005`），`try/except` 抓不住 | `VECTOR_BACKEND=auto`（默认）会在**子进程**里探测一次 chromadb，探测失败就自动切到内置 NumPy 向量库（`<collection>.vectors.npy` + `<collection>.records.json`） |
| `onnxruntime` | 导入即崩溃 | 代码从不使用 Chroma 的默认 ONNX Embedding，始终注入自己的 Embedding 函数 |
| LLM 未配置或调用失败 | 无法总结/生成自然回答 | 仍会检索知识库并把原文要点整理成结构化回答，并**区分提示**「未配置 Key」与「调用失败/超时」，绝不编造 |
| 未配置天气 Key | 无法获取真实天气 | 返回稳定可复现的 mock 天气 + 环境适配建议 |

> 实测结论：上述 `torch` / `chromadb` 报错在**解除沙箱限制后会自动消失**——换到普通 `cmd`/`PowerShell`、
> VSCode 终端或 Docker 里运行，`bge-m3`（约 2.3 GB，首次自动下载）与 Chroma 均可正常工作，无需额外安装 VC++ 运行库。

**降级后的检索质量**：双路检索是「向量 Top10 + BM25 Top10 → RRF 融合 → 词面覆盖度微调 → Top5」。
使用内置哈希向量器时语义能力弱，系统会自动给 BM25 更高权重；等环境可加载 `bge-m3` 后，
**记得重建一次知识库**，`/api/kb/status` 会返回 `needs_rebuild: true` 提醒你，前端侧边栏也会显示警告。

> ⚠️ 关于「词面覆盖度加权」这一步：**实测它在当前系数下对召回没有贡献，且在强向量器下有害**
> （1024 维 BGE-M3 下把 Recall@5 从 0.971 拉到 0.853）。详见[十一、检索效果评估](#十一检索效果评估实测数据)。
> 它唯一起作用的地方是 384 维降级模式下把 MRR 从 0.576 提到 0.627（只能微调排序，不能提升召回）。
> 下一步计划：关掉它，或改用代码中已预留的 `RERANK_ENABLED` + `bge-reranker-v2-m3`。

---

## 四、知识库文件

6 份文件已放在 `data/raw/`：

1. `故障排除.txt`
2. `扫地机器人100问.pdf`
3. `扫地机器人100问2.txt`
4. `扫拖一体机器人100问.txt`
5. `维护保养.txt`
6. `选购指南.txt`

---

## 五、构建知识库

```bat
REM Windows
.venv\Scripts\python.exe scripts\build_kb.py
```

```bash
# macOS / Linux
.venv/bin/python scripts/build_kb.py
```

首次运行会下载 `BAAI/bge-m3`（约 2.3 GB），请耐心等待；脚本会打印**每个文件的切分条数**与**检索验证结果**。

> 用云端 Embedding 时**不会**下载模型：910 条约 1~2 分钟（每批 10 条 ≈ 91 次请求）；
> 用内置哈希向量器约 12 秒。改过 Embedding 模型或维度后**必须重新执行本脚本**。

本项目的切分结果是（6 个文件共 **910 条**，每条都带 `source_file` / `category` / `item_no` / `title` / `tags`）：

| 文件 | 条目数 |
| --- | --- |
| `维护保养.txt` | 210 |
| `故障排除.txt` | 200 |
| `选购指南.txt` | 200 |
| `扫地机器人100问.pdf` | 100 |
| `扫地机器人100问2.txt` | 100 |
| `扫拖一体机器人100问.txt` | 100 |

常用参数：

```bat
.venv\Scripts\python.exe scripts\build_kb.py --query "电池鼓包怎么办"
.venv\Scripts\python.exe scripts\build_kb.py --no-verify
```

也可以在启动后通过前端侧边栏的「🏗️ 重建知识库」按钮，或 `POST /api/kb/rebuild` 触发重建。

> **想跑 `eval/` 里的对照实验**，需要两套向量库（正常路径 1024 维 + 降级路径 384 维）：
>
> ```bat
> REM 1) 正常路径：BGE-M3，1024 维
> set CHROMA_PERSIST_DIR=./vectorstore_bge1024
> set EMBEDDING_ALLOW_DOWNLOAD=true
> .venv\Scripts\python.exe scripts\build_kb.py
>
> REM 2) 降级路径：内置哈希向量器，384 维（无需 torch / 不下载模型）
> set CHROMA_PERSIST_DIR=./vectorstore
> set EMBEDDING_ALLOW_DOWNLOAD=false
> .venv\Scripts\python.exe scripts\build_kb.py
> ```
>
> 两个目录都已在 `.gitignore` 中排除。评测步骤见 [`eval/README.md`](eval/README.md)。

---

## 六、启动服务

### 6.1 启动后端

```bat
REM Windows（在项目根目录 robot-cs-agent 下执行）
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

```bash
# macOS / Linux
.venv/bin/python -m uvicorn backend.app.main:app --reload --port 8000
```

- API 文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/health>
- 知识库状态：<http://localhost:8000/api/kb/status>

### 6.2 启动前端（另开一个终端）

```bat
.venv\Scripts\python.exe -m streamlit run frontend\streamlit_app.py
```

```bash
.venv/bin/python -m streamlit run frontend/streamlit_app.py
```

浏览器访问 <http://localhost:8501>。

侧边栏可设置：用户 ID、城市、机型、地板类型、宠物 / 婴儿、保存画像、知识库状态、重建知识库、清空会话；
主聊天区显示回答，并提供「🛠️ 工具调用」和「📖 参考来源」折叠面板。

### 6.3 主要接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/chat` | 同步问答，返回 `answer` / `sources` / `tool_calls` |
| POST | `/api/chat/stream` | SSE 流式问答 |
| GET / DELETE | `/api/chat/history` | 查询 / 清空会话历史 |
| POST | `/api/kb/rebuild` | 重建向量知识库 |
| GET | `/api/kb/status` | 集合名、条目数、持久化目录 |
| GET | `/api/weather?city=杭州` | 天气 + 环境建议 |
| GET | `/api/location?user_id=u1` | 位置解析结果 |
| GET | `/api/datetime` | 当前日期 / 季节 / 星期 |
| GET / POST | `/api/profile` | 读取 / 保存用户画像 |
| GET | `/health` | 健康检查 |

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","user_id":"u1","message":"今天我这里湿度高，适合拖地吗？"}'
```

---

## 七、Docker 部署

> **必须在项目根目录执行 `docker compose`**：`docker-compose.yml` 里的
> `build: ./backend`、`env_file: .env`、`volumes: - ./data:/app/data` 都是**相对路径**，
> 换目录执行会解析到错误位置（典型报错 `env file ... not found` 或挂载出空目录）。
> 宝塔面板等「容器编排」功能会把 yaml 存到自己的目录再执行，因此**不要**直接用面板的
> 编排功能，请 SSH 到项目根目录用官方 CLI；或把相对路径改成绝对路径后再粘贴。

```bash
cd robot-cs-agent         # ← 必须在项目根目录
cp .env.example .env      # 填好 Key
docker compose up -d --build
```

- 前端：<http://localhost:8501>
- 后端：<http://localhost:8000/docs>

```bash
# 容器内构建知识库
docker compose exec backend python scripts/build_kb.py
docker compose logs -f backend
docker compose down
```

`docker-compose.yml` 会把 `./data` 和 `./vectorstore` 挂载进容器，重建容器不会丢失知识库与向量数据。

### 7.1 国内网络构建加速

两个 Dockerfile 默认已使用**阿里云公网镜像源**（pip），并且 torch 走
`mirrors.aliyun.com/pytorch-wheels/cpu`（失败自动回退官方源）。
阿里云 ECS 内网可换成更快的内网地址（**注意 `mirrors.cloud.aliyuncs.com` 只提供 HTTP，
写成 https 会因证书主机名不匹配而失败**）：

```bash
docker compose build \
  --build-arg PIP_INDEX_URL=http://mirrors.cloud.aliyuncs.com/pypi/simple/ \
  --build-arg PIP_TRUSTED_HOST=mirrors.cloud.aliyuncs.com
```

也可以把上面两个 build-arg 写进 `docker-compose.yml` 的 `build.args`，避免每次手输。
仓库里刻意**不硬编码内网地址**（非阿里云环境不可用）。

### 7.2 镜像体积与构建耗时

后端 Dockerfile 已做两项精简：

- **不再安装 `build-essential` / `gcc`**（省约 88 MB）。`chromadb` 的原生依赖
  `chroma-hnswlib` 以及 numpy / torch 在 amd64 与 arm64 上都有现成 manylinux wheel，
  pip 不会走源码编译。若你的平台确实缺 wheel（构建时报 `gcc: not found`），
  按 Dockerfile 里的注释把那两行放回来即可。
- **健康检查改用 Python 自带的 `urllib`**，不再为一次探活安装 `curl`。
- `backend/requirements.txt` **不再包含 `streamlit`**（后端代码从未 import 它，
  前端镜像单独安装），省掉 pyarrow / pandas / pillow 等重型依赖链。

> **想进一步瘦身**：若确定不使用本地 bge-m3 与 Chroma（即
> `EMBEDDING_ALLOW_DOWNLOAD=false` 或配置了云端 Embedding，且 `VECTOR_BACKEND=local`），
> 可从 `backend/requirements.txt` 移除 `torch`、`sentence-transformers`、`chromadb`、
> `langchain-huggingface`、`langchain-chroma`（这些包在代码里都是**函数内延迟导入**，
> 不用就完全不会触发）。代价是失去本地模型能力，请按需取舍。

### 7.3 改代码 vs 改配置：两种不同的生效方式

| 改动 | 生效方式 | 原因 |
| --- | --- | --- |
| 改了 `.env` | `docker compose up -d --force-recreate` | `.env` 是通过 compose 的 `env_file` 注入的**环境变量**。容器内 `config.py` 由 `__file__` 反推的项目根目录是 `/`，**应用不会去读挂载的 `.env` 文件**，所以单纯 `restart` 不生效 |
| 改了代码 | `docker compose up -d --build` | `COPY . /app` 在**构建期**执行，只重建容器不会带上新代码 |

> 新增配置项若只改 `.env` 而没在 `config.py` 里声明，pydantic-settings 会**静默忽略**
> （不报错但完全不生效）——排查时容易误判。

---

## 八、示例问题

- 机器人开机无反应怎么办？
- 今天我这里湿度高，适合拖地吗？
- 这个月怎么保养扫地机器人？
- 宠物家庭选什么扫地机器人？
- 滤网多久换一次？
- 机器人不回充是什么原因？
- 木地板拖地需要注意什么？
- 电池鼓包了还能用吗？（会触发安全兜底提示）

---

## 九、常见问题

**1) `pip install` 很慢或失败？**
使用国内镜像重试：

```bat
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**2) 提示 Embedding 模型加载失败 / 下载超时？**
程序会自动降级到内置哈希向量器（检索效果变弱但功能可用）。想要完整效果，请联网下载，或手动下载模型后在 `.env` 中设置：

```env
EMBEDDING_MODEL_PATH=D:/models/bge-m3
```

> 注意这里的 `D:/models/bge-m3` 只是**示例路径**：`models/` 在 `.gitignore` 里指的是
> **根目录下的模型权重缓存**（`/models/`，根锚定），与源码包 `backend/app/models/` 无关。
> 把权重放在仓库外的任意目录更稳妥。

**2.1) 内存只有 2G，加载 bge-m3 就被 OOM 杀掉怎么办？**
见 [3.1 前置要求](#31-前置要求) 的降级路径：**推荐配置云端 Embedding**（填 `EMBEDDING_API_KEY`，
内存占用可忽略且语义质量更好），完全离线时才用内置哈希向量器（关键词型提问尚可，
「这个月怎么保养」这类语义型提问会漏召回）。配置方法见 [3.4.1](#341-可选改用云端-embedding低配机器的推荐方案)。

**2.2) 配了 `EMBEDDING_API_KEY` 但看起来没生效？**
按顺序排查：

- 容器部署时 `.env` 是**环境变量注入**，改完要 `docker compose up -d --force-recreate`（`restart` 不生效）；
- 新增配置项必须在 `config.py` 里**声明过**，否则 pydantic-settings 会静默忽略；
- 启动日志会打印实际生效的后端：`使用云端 Embedding 接口: <模型> @ <地址>`；
  也可直接查 `GET /api/kb/status` 的 `embedding_backend`（云端方案形如 `OpenAICompatEmbeddings:模型名`）；
- 云端不可达（Key 错、模型名错、额度用尽、外网不通）时会**自动回落**到本地模型/哈希向量器并打 WARNING，
  所以"没报错"不等于"生效了"，请看日志确认。

**3) 知识库状态显示 0 条？**
先执行 `scripts/build_kb.py`；若 `data/raw` 文件缺失，请确认 6 份知识库文件已放入该目录。

**4) 回答里说「知识库中未检索到相关内容」？**
说明知识库确实没有覆盖该问题。这是刻意的设计——**不允许编造**，此时会建议联系官方售后。

**5) 前端提示无法连接后端？**
先启动后端，并在前端侧边栏确认「后端地址」为 `http://localhost:8000`。

**6) Windows 上 `python -m venv .venv` 报 `ensurepip` 失败？**
通常是 `%TEMP%` 目录不可写导致，可先把临时目录指向项目内再重试：

```bat
mkdir .tmp
set TMP=%CD%\.tmp
set TEMP=%CD%\.tmp
python -m venv .venv
```

**7) 启动时报 `WinError 1114 ... c10.dll`？**
说明当前进程环境无法加载 `torch` 的原生库——最常见的原因是**运行在受限沙箱/受限进程里**（例如被禁止创建命名管道、被限制加载 DLL），
其次是系统缺少 VC++ 运行库。处理办法：

- 首选：改用**普通终端**运行（`cmd`、PowerShell、VSCode 终端、Docker 容器），不要在被沙箱包裹的进程里启动；
- 若确实缺运行库：安装 [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe)，然后重建知识库以获得 `bge-m3` 的语义检索效果；
- 兜底：在 `.env` 里设 `EMBEDDING_ALLOW_DOWNLOAD=false`，直接用内置轻量向量器（不用 torch，启动也不再等待模型加载）。

无论哪种情况，项目都能正常启动和问答——Embedding 会自动降级。

**8) 启动日志出现 `自动改用内置 NumPy 向量库`？**
说明当前环境里 chromadb 的原生引擎不可用（写入会直接崩溃，同样多见于受限沙箱）。项目已在子进程里探测并自动切换到内置向量库，功能不受影响。
换到普通终端后它会自动重新探测并使用 Chroma（探测缓存 `vectorstore/.chroma_probe.json` 对失败结果只缓存 6 小时，会自动失效）；
也可用 `VECTOR_BACKEND=chroma|local` 强制指定。

**9) 修改了 Embedding 模型后检索变差/报维度不一致？**
必须重建知识库（`scripts/build_kb.py` 或前端「重建知识库」按钮），因为不同模型的向量维度不同。
`/api/kb/status` 的 `needs_rebuild` 字段会告诉你是否需要重建。

**10) 启动时卡在 `加载 Embedding 模型` 好几分钟？**
这是 `huggingface_hub` 的**联网更新检查**导致的：它会对模型目录里的**每个文件**发一次 HTTP HEAD，
当 huggingface.co 不可达时每个文件都要重试 5 次（退避最长 16 秒），累计就会卡住几分钟。

本项目默认 `HF_HUB_OFFLINE=auto`：**只要检测到本地已有完整模型缓存，就自动切到离线模式**（实测启动从数分钟降到约 30 秒）。
如果仍遇到卡顿，可以显式处理：

```bat
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
```
或者改用镜像站（首次下载模型时尤其有用）：

```env
HF_ENDPOINT=https://hf-mirror.com
```

**11) 回答里的小标题显示成字面 `**`（没有加粗）？**
这是 Markdown 的规则问题：`**` 作为**右定界符**时，后面**紧跟非空白字符则不生效**。
`**结论：**知识库中……` 里 `**` 后面直接是「知」，于是星号原样显示。

本项目做了**两层处理**，正常情况不会再出现：

1. **治本**：`core/prompts.py` 的输出格式要求里明确规定小标题写成 `**小标题：** 内容`，
   `**` 与正文之间必须有一个空格；
2. **兜底**：前端渲染前会规范化一次（`streamlit_app.py` 的 `normalize_markdown`），
   兼容任意模型的输出与历史消息，包括 `**结论**：正文` 这种冒号写在加粗外面的写法。
   流式渲染时也做了跨分块保护，不会因为 `**` 恰好被切在两个数据块之间而漏掉。

**12) 克隆仓库后后端起不来，报 `ModuleNotFoundError: No module named 'app.models'`？**
这是历史版本 `.gitignore` 用裸 `models/` 规则误伤了源码包 `backend/app/models/`，导致
`schemas.py` / `db.py` 没被提交。**当前版本已修复**（规则改成根锚定的 `/models/`，
应用包已纳入版本控制）。如果你从旧提交克隆，请拉取最新 `main`；详见
[二、目录结构](#二目录结构) 下的 `.gitignore` 根锚定说明。

自检一条命令即可确认：

```bash
git ls-files backend/app/models
# 期望看到：__init__.py  db.py  schemas.py
```

---

## 十、本次交付的验证情况

已在 Windows + Python 3.13 + `.venv`（全部依赖只装在项目内）下实测通过。

**可复现的一键验收**（不需要任何 API Key，LLM 走纯检索兜底模式）：

```bat
.venv\Scripts\python.exe tests\test_smoke.py
```

覆盖 6 项检查：模块导入、知识库与向量维度一致性、LangGraph 六节点图编译、Embedding 工厂、
端到端 `/api/chat`（含来源与工具调用）、风险问题安全兜底首句提示。当前结果：**通过 6 项，失败 0 项**
（384 维降级库与 1024 维 BGE-M3 库各跑一遍均通过）。

**完整能力验证（普通终端环境，`bge-m3` + Chroma + 真实 LLM）**

| 验证项 | 结果 |
| --- | --- |
| 6 个知识库文件切分 | ✅ 共 910 条，元数据字段零缺失（含 PDF：100 条） |
| 知识库构建 | ✅ `bge-m3`（1024 维）向量化 910 条入 Chroma；含 HF 联网检查重试的总耗时约 901 秒，纯向量化阶段约 250 秒 |
| 语义检索质量 | ✅ 「机器人开机无反应」→ `故障排除.txt` 第 1 条；「这个月怎么保养」→ 前 5 条全部来自 `维护保养.txt`；「滤网多久换一次」→ `滤网应该多久更换？` |
| **检索效果量化** | ✅ 34 道计分题：1024 维 BGE-M3 纯向量 **Recall@5 = 0.971 / MRR = 0.801**；详见[十一、检索效果评估](#十一检索效果评估实测数据) |
| 完整 Agent 链路 | ✅ 意图识别 → 工具调用 → 查询改写 → 混合检索 → 摘要 → 回答；湿度/天气类问题自动调用位置+时间+天气工具 |
| 真实 LLM 回答 | ✅ 输出「结论 → 原因 → 操作步骤 → 注意事项 → 参考来源」结构，并结合用户画像（机型/地板/宠物）给个性化建议 |
| 安全兜底 | ✅ 「电池鼓包了还能用吗」首句即提示立即停用、断电、联系售后 |
| 后端接口 | ✅ `/health`、`/api/kb/status`、`/api/kb/rebuild`、`/api/datetime`、`/api/weather`、`/api/location`、`/api/profile`、`/api/chat/history` |
| SSE 流式问答 | ✅ 事件序列 `start → delta... → sources → done` |
| 前端 | ✅ Streamlit 正常启动，`AppTest` 无异常；可流式渲染回答，工具调用与参考来源面板正常 |
| 启动性能 | ✅ 模型已缓存时自动切 HF 离线模式，加载从「卡住数分钟」降到 **约 29 秒** |
| 依赖降级路径 | ✅ 在受限沙箱中会自动降级为内置向量器 + NumPy 向量库并照常问答（该降级在普通终端下不会被触发） |
| 云端 Embedding | ✅ 双返回结构（OpenAI `data` / DashScope `output.embeddings`）解析、乱序按序号校正、分批、4xx 不重试、5xx 退避重试、云端不可达自动回落（均有单元级验证） |
| 回答加粗渲染 | ✅ `**结论：**正文` 经规范化后右定界符后为空白，可正常加粗；流式在任意位置切分结果一致，且不误伤句中加粗 |

> `vectorstore/`（向量库 + BM25 语料）与 `.env` 均已被 `.gitignore` 排除（注意规则是**根锚定**的
> `/vectorstore/`，不会误伤 `backend/app/vectorstore/`），不会提交到仓库；
> 克隆后按「五、构建知识库」执行一次即可生成属于自己的向量库，再填好 `.env` 中的 `OPENAI_API_KEY` 即可获得 LLM 总结式回答。

---

## 十一、检索效果评估（实测数据）

评测脚本与原始输出都在 [`eval/`](eval/)，方法与结论见 [`eval/README.md`](eval/README.md)。

**34 道人工计分题**（覆盖全部 6 份源文件，query 为口语化改写、刻意不等于条目标题原文；
标准答案用「来源文件 + 条目号」锁定）。同一套题、同一份代码，只换向量器：

| 融合策略 | 384 维哈希（降级路径） | 1024 维 BGE-M3（正常路径） |
| --- | --- | --- |
| A 纯向量 | 0.706 / 0.576 | **0.971 / 0.801** |
| B 纯 BM25 | 0.676 / 0.578 | 0.676 / 0.578 |
| C RRF 融合（k=60） | 0.735 / 0.619 | **0.971 / 0.754** |
| D C + 覆盖度加权（**线上配置**） | 0.735 / 0.619 | 0.853 / 0.728 |
| E C + 覆盖度加权 ×10 | 0.765 / 0.599 | 0.853 / 0.680 |
| F 纯向量 + 覆盖度加权 | 0.706 / 0.627 | 0.941 / 0.729 |

格式为 `Recall@5 / MRR`（BM25 一路与向量器无关，两列相同，可作自洽性交叉验证）。

**结论（含"哪个优化其实没用"）**

1. **换 Embedding 模型是收益最大的单项优化**：命中数 24/34 → 33/34，Recall@5 **0.706 → 0.971**，MRR **0.576 → 0.801**。
2. **覆盖度加权在当前系数下没有贡献，且在强向量器下有害**：384 维下 C 与 D 完全相同（0.735 / 0.619，该步没有产生效果）；1024 维下 C 0.971 → D 0.853（33 → 29 命中）。原因是 RRF 分数只有 ~0.03 量级，而加权项最大 0.010 且对比度极高（0 或满值），等于往已排好的序列里注入噪声；系数放大 10 倍仍不能挽回，说明该信号本身无效。
3. **它唯一起正面作用的地方**：384 维下「纯向量 + 覆盖度」把 MRR 从 0.576 提到 0.627 —— 能微调排序，不能提升召回。
4. **已知缺陷**：`HybridRetriever.retrieve()` **没有最低相关度阈值**，永远返回 top-k。4 道"知识库无答案"题在 4 种配置下全部返回了内容，即检索层没有拒答能力，"不知道就说不知道"目前只靠 Prompt 约束兜底。下一步：加相对阈值，或启用已预留的 `RERANK_ENABLED` + `bge-reranker-v2-m3`。

---

## 十二、免责声明

知识库内容用于演示，具体维修、保修与安全处置请以**品牌官方售后**为准。
涉及电池鼓包、漏液、冒烟、烧焦味等情形，请**立即停用并断电**，不要自行拆机。
