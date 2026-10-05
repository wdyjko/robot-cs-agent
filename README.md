# 智能扫地机器人客服 Agent

基于 **FastAPI + LangGraph + LangChain + Chroma + Streamlit** 的扫地 / 扫拖一体机器人智能客服。

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
│  │  │  ├─ embeddings.py        # Embedding 工厂（bge-m3，失败自动降级）
│  │  │  ├─ llm_service.py       # LLM 工厂（OpenAI 兼容，未配 Key 优雅降级）
│  │  │  ├─ vectorstore_service.py  # 向量库后端选择（子进程探测 Chroma / 自动切本地）
│  │  │  └─ local_vectorstore.py    # 内置 NumPy 向量库（Chroma 不可用时的兜底）
│  │  ├─ tools/
│  │  │  ├─ location.py          # get_user_location
│  │  │  ├─ weather.py           # get_weather
│  │  │  ├─ datetime_tool.py     # get_current_datetime
│  │  │  ├─ profile.py           # get_user_profile
│  │  │  └─ kb_tool.py           # search_knowledge_base
│  │  ├─ models/
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
├─ vectorstore/                  # Chroma 持久化目录（.gitignore 已忽略）
├─ scripts/
│  └─ build_kb.py                # 独立建库脚本
├─ docker-compose.yml
├─ .env.example
├─ .gitignore
├─ setup.sh                      # macOS / Linux 一键安装
├─ setup.bat                     # Windows 一键安装
└─ README.md
```

---

## 三、环境配置说明

### 3.1 前置要求

| 项目 | 要求 |
| --- | --- |
| Python | **3.10 ~ 3.13**（推荐 3.11，依赖预编译包最全） |
| 内存 | ≥ 8 GB（加载 bge-m3 时建议 16 GB） |
| 磁盘 | ≥ 6 GB（依赖约 3 GB + Embedding 模型约 2.3 GB） |
| 网络 | 首次安装依赖与下载 Embedding 模型需要联网 |

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

### 3.5 关键环境变量一览

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 空 | 留空则禁用 LLM（有 mock 兜底） |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI 兼容地址 |
| `LLM_MODEL` | `gpt-4o-mini` | 模型名 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 向量模型；加载失败自动降级为内置哈希向量器 |
| `EMBEDDING_MODEL_PATH` | 空 | 指向本地模型目录，避免重复下载 |
| `EMBEDDING_ALLOW_DOWNLOAD` | `true` | 为 `false` 时直接用内置哈希向量器（不需要 torch） |
| `CHROMA_PERSIST_DIR` | `./vectorstore` | 向量库持久化目录 |
| `CHROMA_COLLECTION` | `robot_customer_service` | 集合名称 |
| `VECTOR_BACKEND` | `auto` | `auto` 自动探测 / `chroma` 强制 Chroma / `local` 内置 NumPy 向量库 |
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
| `torch` / `sentence-transformers` | `WinError 1114 动态链接库(DLL)初始化例程失败`（加载 `c10.dll` 失败） | Embedding 自动降级为内置 `HashingEmbeddings`；`langchain_text_splitters` / `langchain_community.document_loaders` 改为可选依赖，改用内置的递归切分器与 pypdf 加载器 |
| `chromadb` 原生引擎 | 写入时进程直接**访问违例崩溃**（`0xC0000005`），`try/except` 抓不住 | `VECTOR_BACKEND=auto`（默认）会在**子进程**里探测一次 chromadb，探测失败就自动切到内置 NumPy 向量库（`<collection>.vectors.npy` + `<collection>.records.json`） |
| `onnxruntime` | 导入即崩溃 | 代码从不使用 Chroma 的默认 ONNX Embedding，始终注入自己的 Embedding 函数 |
| LLM 未配置或调用失败 | 无法总结/生成自然回答 | 仍会检索知识库并把原文要点整理成结构化回答，并**区分提示**「未配置 Key」与「调用失败/超时」，绝不编造 |
| 未配置天气 Key | 无法获取真实天气 | 返回稳定可复现的 mock 天气 + 环境适配建议 |

> 实测结论：上述 `torch` / `chromadb` 报错在**解除沙箱限制后会自动消失**——换到普通 `cmd`/`PowerShell`、
> VSCode 终端或 Docker 里运行，`bge-m3`（约 2.3 GB，首次自动下载）与 Chroma 均可正常工作，无需额外安装 VC++ 运行库。

**降级后的检索质量**：双路检索是「向量 Top10 + BM25 Top10 → RRF 融合 → 词面覆盖度微调 → Top5」。
使用内置哈希向量器时语义能力弱，系统会自动给 BM25 更高权重；等环境可加载 `bge-m3` 后，
**记得重建一次知识库**，`/api/kb/status` 会返回 `needs_rebuild: true` 提醒你，前端侧边栏也会显示警告。

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

```bash
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

---

## 十、本次交付的验证情况

已在 Windows + Python 3.13 + `.venv`（全部依赖只装在项目内）下实测通过。

**完整能力验证（普通终端环境，`bge-m3` + Chroma + 真实 LLM）**

| 验证项 | 结果 |
| --- | --- |
| 6 个知识库文件切分 | ✅ 共 910 条，元数据字段零缺失（含 PDF：100 条） |
| 知识库构建 | ✅ `bge-m3`（1024 维）向量化 910 条入 Chroma，CPU 约 250 秒 |
| 语义检索质量 | ✅ 「机器人开机无反应」→ `故障排除.txt` 第 1 条；「这个月怎么保养」→ 前 5 条全部来自 `维护保养.txt`；「滤网多久换一次」→ `滤网应该多久更换？` |
| 完整 Agent 链路 | ✅ 意图识别 → 工具调用 → 查询改写 → 混合检索 → 摘要 → 回答；湿度/天气类问题自动调用位置+时间+天气工具 |
| 真实 LLM 回答 | ✅ 输出「结论 → 原因 → 操作步骤 → 注意事项 → 参考来源」结构，并结合用户画像（机型/地板/宠物）给个性化建议 |
| 安全兜底 | ✅ 「电池鼓包了还能用吗」首句即提示立即停用、断电、联系售后 |
| 后端接口 | ✅ `/health`、`/api/kb/status`、`/api/kb/rebuild`、`/api/datetime`、`/api/weather`、`/api/location`、`/api/profile`、`/api/chat/history` |
| SSE 流式问答 | ✅ 事件序列 `start → delta... → sources → done` |
| 前端 | ✅ Streamlit 正常启动，`AppTest` 无异常；可流式渲染回答，工具调用与参考来源面板正常 |
| 依赖降级路径 | ✅ 在受限沙箱中会自动降级为内置向量器 + NumPy 向量库并照常问答（该降级在普通终端下不会被触发） |

> `vectorstore/`（Chroma 向量库 + BM25 语料）与 `.env` 均已被 `.gitignore` 排除，不会提交到仓库；
> 克隆后按「五、构建知识库」执行一次即可生成属于自己的向量库，再填好 `.env` 中的 `OPENAI_API_KEY` 即可获得 LLM 总结式回答。

---

## 十一、免责声明

知识库内容用于演示，具体维修、保修与安全处置请以**品牌官方售后**为准。
涉及电池鼓包、漏液、冒烟、烧焦味等情形，请**立即停用并断电**，不要自行拆机。
