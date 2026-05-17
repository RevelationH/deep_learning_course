# 深度学习课程学习平台

## 一、项目说明

本项目是一套面向深度学习课程的教学支持平台，主要提供以下三类能力：

1. 课程问答  
   基于课程资料构建知识库，为学生提供课程范围内的问答服务，并支持来源定位。

2. 知识点练习  
   围绕课程知识点组织选择题练习，支持答案解析、图片题展示和来源回看。

3. 学习报告  
   根据学生作答记录生成学习反馈，用于识别薄弱知识点并给出复习建议。

项目同时包含教师侧内容生产链路，即课程资料接入、RAG 知识库构建、题库生成与发布。

---

## 二、目录结构

```text
deep_learning/
├─ deep_learning_materials/          课程 PDF 材料
├─ deep_learning_portal/             Web 应用前后端
├─ deep_learning_rag/                知识库构建脚本与产物目录
├─ pg_support/                       PostgreSQL 适配层
├─ project_report_20260429/          项目汇报材料
├─ db.py                             Firebase 相关接入
├─ storage_backend.py                存储后端选择入口
├─ requirements.txt                  Python 依赖
└─ README.md                         运行说明
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

项目运行依赖环境变量。可在系统环境变量中配置，或在项目根目录提供本地环境配置文件。仓库中不包含任何真实密钥、证书或服务账号文件。

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

如需根据课程 PDF 重新构建知识库，可执行：

```powershell
python .\deep_learning_rag\pipeline.py
```

默认逻辑：

- 输入目录：`.\deep_learning_materials`
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

---

## 八、审核与验收建议

审核人员可按以下顺序检查系统是否可用：

1. 启动应用后访问登录页  
   确认页面可正常打开。

2. 登录系统  
   使用已配置账户或注册账户进入平台。

3. 检查课程问答  
   提出课程相关问题，确认系统能够返回回答，并显示课程来源信息。

4. 检查知识点练习  
   打开练习界面，确认题目、选项、图片、答案解析和来源定位可正常展示。

5. 检查学习报告  
   完成一定数量练习后，确认系统能够生成学习报告并给出复习建议。

6. 检查历史对话  
   确认历史对话可保存并再次打开。

---

## 九、常见运行问题

### 1. 登录页可以打开，但登录失败

优先检查：

- Firebase 服务账号是否已配置
- PostgreSQL 或 Firebase 后端是否可连接

### 2. 页面能打开，但问答一直无返回

优先检查：

- 是否使用了 `run_deep_learning_all_in_one.py`
- LLM 配置是否完整
- Redis、数据库或后台 Worker 是否正常

### 3. 问答能返回，但没有课程来源

优先检查：

- `deep_learning_rag/artifacts_full_course` 是否已生成
- 课程材料路径是否正确
- 知识库构建步骤是否执行成功

### 4. 学习报告生成较慢

学习报告依赖作答记录汇总和后台任务处理，延迟通常与以下因素相关：

- 数据库连接性能
- Redis 是否启用
- Worker 数量配置
- LLM 响应时间

---

## 十、说明

1. 本仓库仅提交代码、配置模板、课程材料产物和正式文档，不提交真实密钥、服务账号、私钥证书等敏感信息。
2. 若用于服务器部署，建议结合 Nginx、进程守护和独立数据库服务进行部署，不建议直接以开发方式长期运行。
3. 如需对外提供稳定服务，应优先核查数据库配置、并发参数、Worker 数量和大模型供应商的并发额度。
