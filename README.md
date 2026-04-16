# 深度学习课程学习平台

这个目录是一套独立的新项目，基于你当前最新的课程平台结构重新搭建，但课程内容已经切换为 `D:\digital_human\2025` 下的深度学习讲义，界面文案也改为中文。

## 目录结构

- `deep_learning_portal/`
  - 中文前端与 Flask 后端
  - 包含登录、问答、练习、学习报告、历史对话
- `deep_learning_rag/`
  - 深度学习课程的 RAG 构建脚本与 artifacts
- `deep_learning_materials/`
  - 从 `D:\digital_human\2025` 复制并按 `Week1` 到 `Week13` 组织好的 PDF 讲义
- `db.py` 与 `user.py`
  - 预留 Firebase 用户体系接入
- `.env.example`
  - 需要你后续补充 Firebase 与 LLM 配置

## 当前已完成

- 已建立新的深度学习课程材料目录
- 已从 PDF 讲义构建新的知识库 artifacts
- 已生成 13 个课程知识点
- 已生成 65 道选择题
- 已为图片题渲染所需的 PDF 页面截图
- 已建立新的中文问答、练习、学习报告页面骨架
- 已把历史对话、练习记录的数据结构切换为新的深度学习课程命名空间

## 运行前需要补充

你后续只需要在项目根目录新增 `.env` 或 `.env.local`，填写：

- `FIREBASE_CREDENTIALS` 或 `GOOGLE_APPLICATION_CREDENTIALS`
- `COURSE_LLM_API_KEY`
- `COURSE_LLM_BASE_URL`
- `COURSE_LLM_MODEL`

可选项：

- `DEEP_LEARNING_MATERIAL_ROOT`
- `DEEP_LEARNING_ARTIFACT_DIR`
- `DEEP_LEARNING_EMBED_MODEL`
- `DEEP_LEARNING_PORTAL_SECRET`

## 本地构建知识库

```powershell
python .\deep_learning_rag\pipeline.py
```

默认会读取：

- 材料目录：`.\deep_learning_materials`
- 输出目录：`.\deep_learning_rag\artifacts_full_course`

## 启动平台

```powershell
python .\deep_learning_portal\run_deep_learning_portal.py
```

默认地址：

`http://127.0.0.1:50225/chatapi_deep_learning.html`

## 说明

- 当前项目保留了 Firebase 相关接入点，但没有内置任何密钥。
- 如果暂时没有配置 Firebase，应用仍然可以启动并打开登录页，但登录与注册会保持禁用状态，并提示你先补服务账号配置。
- 一旦你补齐 Firebase 服务账号和 LLM API 配置，这套系统就可以直接进入运行阶段。
- 当前健康检查可返回 `13` 个知识点和 `65` 道选择题，说明新课程知识库与练习机制已经完成构建。
