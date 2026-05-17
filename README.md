# 深度学习课程学习平台

## 一、项目说明

本项目是一套面向深度学习课程的教学支持平台，提供课程问答、知识点练习、学习报告和历史记录管理等功能。平台基于课程资料构建课程知识库，并将知识库统一服务于问答、练习和学习反馈三个使用场景。

项目同时包含教师侧内容生产链路，即课程资料接入、知识库构建、题库生成与发布。

---

## 二、目录结构

```text
deep_learning/
├─ deep_learning_materials/      课程 PDF 材料
├─ deep_learning_portal/         Web 应用前后端
├─ deep_learning_rag/            知识库构建脚本与产物目录
├─ pg_support/                   PostgreSQL 适配层
├─ db.py                         数据库接入
├─ storage_backend.py            存储后端选择入口
├─ requirements.txt              Python 依赖
└─ README.md                     运行说明
```

---

## 三、运行环境

建议环境如下：

- 操作系统：Windows 或 Linux
- Python：3.10 及以上
- 数据库：PostgreSQL 或 Firebase
- 可选缓存：Redis

安装依赖：

```powershell
pip install -r .\requirements.txt
```

---

## 四、配置说明

项目运行依赖环境变量。仓库中不包含任何真实密钥、证书或服务账号文件，部署时需要由运行方自行提供。

核心配置项如下：

### 1. 大模型配置

示例：

```text
COURSE_LLM_API_KEY=your_llm_api_key
COURSE_LLM_BASE_URL=https://api.example.com/v1
COURSE_LLM_MODEL=your-model-name
```

- `COURSE_LLM_API_KEY`  
  大模型服务的访问密钥。

- `COURSE_LLM_BASE_URL`  
  大模型服务的接口地址。例如兼容 OpenAI 风格接口的服务，通常填写到 `/v1` 这一层。

- `COURSE_LLM_MODEL`  
  调用的大模型名称。例如 `kimi-k2-0711-preview`、`gpt-4.1` 或你实际部署时使用的模型名。

### 2. 课程材料与知识库配置

示例：

```text
DEEP_LEARNING_MATERIAL_ROOT=D:/digital_human/deep_learning/deep_learning_materials
DEEP_LEARNING_ARTIFACT_DIR=D:/digital_human/deep_learning/deep_learning_rag/artifacts_full_course
DEEP_LEARNING_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

- `DEEP_LEARNING_MATERIAL_ROOT`  
  课程材料目录。这里存放平台用于构建知识库的课程 PDF 文件。

- `DEEP_LEARNING_ARTIFACT_DIR`  
  知识库构建输出目录。向量索引、知识点产物、题库产物等会写到这里。

- `DEEP_LEARNING_EMBED_MODEL`  
  向量化所使用的嵌入模型名称。通常保持默认即可，除非你准备替换 embedding 模型。

### 3. 应用运行配置

示例：

```text
DEEP_LEARNING_PORTAL_SECRET=replace_with_a_random_secret
DEEP_LEARNING_STORAGE_BACKEND=postgres
```

- `DEEP_LEARNING_PORTAL_SECRET`  
  Web 应用的会话签名密钥，用于保护登录状态和 Session 数据。  
  不要直接使用示例值。部署时应改成一段随机、足够长的字符串，例如：

  ```text
  DEEP_LEARNING_PORTAL_SECRET=deep-learning-portal-2026-a-long-random-secret
  ```

- `DEEP_LEARNING_STORAGE_BACKEND`  
  指定平台使用哪种存储后端。当前支持两种取值：

  - `postgres` 或 `postgresql`：使用 PostgreSQL
  - `firebase`：使用 Firebase

  示例：

  ```text
  DEEP_LEARNING_STORAGE_BACKEND=postgres
  ```

### 4. 数据库配置

平台支持两类数据库接入方式：PostgreSQL 或 Firebase。二选一即可，不需要同时配置。

#### 4.1 使用 PostgreSQL

如使用 PostgreSQL，可采用单条连接串方式：

```text
DEEP_LEARNING_POSTGRES_DSN=postgresql://postgres:your_password@127.0.0.1:5432/deep_learning_portal
```

这条配置的含义如下：

- `postgresql://`：协议头
- `postgres`：数据库用户名
- `your_password`：数据库密码
- `127.0.0.1`：数据库主机
- `5432`：数据库端口
- `deep_learning_portal`：数据库名

如果你不想写成一整条连接串，也可以拆分配置：

```text
DEEP_LEARNING_PGHOST=127.0.0.1
DEEP_LEARNING_PGPORT=5432
DEEP_LEARNING_PGDATABASE=deep_learning_portal
DEEP_LEARNING_PGUSER=postgres
DEEP_LEARNING_PGPASSWORD=your_password
```

各字段含义如下：

- `DEEP_LEARNING_PGHOST`：数据库主机地址
- `DEEP_LEARNING_PGPORT`：数据库端口
- `DEEP_LEARNING_PGDATABASE`：数据库名
- `DEEP_LEARNING_PGUSER`：数据库用户名
- `DEEP_LEARNING_PGPASSWORD`：数据库密码

说明：

- 如果已经填写了 `DEEP_LEARNING_POSTGRES_DSN`，通常不需要再拆分写主机、端口、用户名、密码。
- 如果数据库不在本机，需要把 `127.0.0.1` 改成实际数据库地址。

#### 4.2 使用 Firebase

如使用 Firebase，需要提供 Firebase 服务账号 JSON 文件路径。可使用以下任一变量：

```text
FIREBASE_CREDENTIALS=D:/path/to/firebase-service-account.json
```

或：

```text
GOOGLE_APPLICATION_CREDENTIALS=D:/path/to/firebase-service-account.json
```

说明：

- `FIREBASE_CREDENTIALS`：项目内自定义读取的服务账号路径变量
- `GOOGLE_APPLICATION_CREDENTIALS`：Google 官方 SDK 常用的默认变量名

通常只需要配置其中一个即可。如果两个都配置，建议保持为同一个文件路径，避免混淆。

### 5. 缓存配置

示例：

```text
DEEP_LEARNING_REDIS_URL=redis://127.0.0.1:6379/0
```

- `DEEP_LEARNING_REDIS_URL`  
  Redis 连接地址，主要用于缓存、异步任务状态和高并发下的运行支撑。

这条地址的含义如下：

- `redis://`：Redis 协议头
- `127.0.0.1`：Redis 主机地址
- `6379`：Redis 默认端口
- `/0`：使用的 Redis 数据库编号，`0` 表示第 0 号库

如果 Redis 部署在其他机器上，可写成：

```text
DEEP_LEARNING_REDIS_URL=redis://192.168.1.20:6379/0
```

如果 Redis 设置了密码，可写成：

```text
DEEP_LEARNING_REDIS_URL=redis://:your_redis_password@127.0.0.1:6379/0
```

---

## 五、知识库构建

平台知识库的输入材料为课程 PDF 文件。默认情况下，项目读取 `deep_learning_materials/` 目录下按教学周次整理的 PDF 讲义，并将其解析后构建为课程知识库。

如需根据课程 PDF 重新构建知识库，可执行：

```powershell
python .\deep_learning_rag\pipeline.py
```

默认逻辑如下：

- 输入目录：`.\deep_learning_materials`
- 输入内容：课程 PDF 讲义文件
- 输出目录：`.\deep_learning_rag\artifacts_full_course`

该步骤会完成课程材料解析、文本切片、向量化索引构建以及知识点与题库相关产物生成。

---

## 六、数据库初始化

### 1. 使用 PostgreSQL

如采用 PostgreSQL，可先创建数据库并执行：

```powershell
psql -f .\pg_schema.sql
```

如需将既有 Firebase 数据迁移到 PostgreSQL，可使用：

```powershell
python .\pg_migrate_from_firebase.py
```

### 2. 使用 Firebase

如采用 Firebase，需要提供服务账号 JSON，并正确配置环境变量。仓库不包含该文件，需要部署方自行提供。

---

## 七、启动方式

### 推荐方式：一体化启动

该方式会同时启动 Web 服务和聊天 Worker，适合本地验收与正式部署前测试。

```powershell
python .\deep_learning_portal\run_deep_learning_all_in_one.py
```

默认访问入口：

- 登录页：`http://127.0.0.1:50225/login_deep_learning`
- 主界面：`http://127.0.0.1:50225/chatapi_deep_learning.html`

### 调试方式：仅启动门户

如仅需调试页面或接口，可执行：

```powershell
python .\deep_learning_portal\run_deep_learning_portal.py
```

注意：仅启动门户时，部分依赖后台 Worker 的问答任务可能无法完整执行，因此正式运行应优先使用一体化启动方式。
