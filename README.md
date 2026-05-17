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

- `COURSE_LLM_API_KEY`
- `COURSE_LLM_BASE_URL`
- `COURSE_LLM_MODEL`

### 2. 课程材料与知识库配置

- `DEEP_LEARNING_MATERIAL_ROOT`
- `DEEP_LEARNING_ARTIFACT_DIR`
- `DEEP_LEARNING_EMBED_MODEL`

### 3. 应用运行配置

- `DEEP_LEARNING_PORTAL_SECRET`
- `DEEP_LEARNING_STORAGE_BACKEND`

### 4. PostgreSQL 配置

可使用以下任一方式：

- `DEEP_LEARNING_POSTGRES_DSN`

或拆分配置：

- `DEEP_LEARNING_PGHOST`
- `DEEP_LEARNING_PGPORT`
- `DEEP_LEARNING_PGDATABASE`
- `DEEP_LEARNING_PGUSER`
- `DEEP_LEARNING_PGPASSWORD`

### 5. Firebase 配置

- `FIREBASE_CREDENTIALS`

或：

- `GOOGLE_APPLICATION_CREDENTIALS`

### 6. Redis 配置

- `DEEP_LEARNING_REDIS_URL`

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
