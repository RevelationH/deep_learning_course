from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from deep_learning_portal.kb_service import (
    CODE_RE,
    EXPLAIN_RE,
    GREETING_RE,
    HELP_RE,
    IDENTITY_RE,
    THANKS_RE,
    DeepLearningKnowledgeBase,
    clean_display_text,
    clean_multiline_text,
)


OUT_OF_SCOPE_GUIDANCE = [
    "这个问题和当前深度学习课程无关，所以我先不直接回答。你可以改问神经网络、CNN、RNN、Transformer、GAN、扩散模型或训练优化相关内容。",
    "这不是当前课程范围内的问题，我先不展开。若你愿意，可以继续问卷积网络、反向传播、LSTM、注意力机制或框架使用。",
    "这个问题超出了这门深度学习课程的问答边界，我先不直接作答。你可以切换到课程主题，我会继续帮你分析。",
]
TERM_HINTS = {
    "dit": "写法请使用 DiT（Diffusion Transformer）。它是扩散模型里的 Transformer 骨干，常见于图像生成等生成建模任务，不是 Dense layer，也不要把它回答成通用 NLP Transformer、语言模型或泛指序列建模模型",
    "vit": "写法请使用 ViT（Vision Transformer），主要用于视觉表征与图像分类等视觉任务",
    "vae": "VAE 指 Variational Autoencoder，属于概率生成模型",
    "mae": "MAE 指 Masked Autoencoder，常用于自监督表征学习",
    "clip": "CLIP 是图文对齐的多模态模型，不是单纯的分类头",
    "unet": "U-Net 常用于分割与扩散模型中的去噪骨干",
    "u-net": "U-Net 常用于分割与扩散模型中的去噪骨干",
}
MODEL_NAME_NORMALIZATIONS = {
    "DiT（Diffusion Transformer）": [
        "深度不变图（DiT）",
        "深度不变图（Diffusion Transformer）",
        "扩散变换器（DiT）",
        "扩散变换器（Diffusion Transformer）",
    ],
    "ViT（Vision Transformer）": [
        "视觉变换器（ViT）",
        "视觉变换器（Vision Transformer）",
    ],
}


def _history_text(history: Sequence[Dict[str, str]], limit: int = 6) -> str:
    rows = []
    for turn in list(history or [])[-limit:]:
        role = "学生" if turn.get("role") == "user" else "助教"
        content = clean_display_text(turn.get("content") or "")
        if not content:
            continue
        rows.append(f"{role}: {content}")
    return "\n".join(rows) if rows else "无"


def _pick_variety(seed_text: str, choices: Sequence[str]) -> str:
    if not choices:
        return ""
    seed = sum(ord(ch) for ch in clean_display_text(seed_text))
    return choices[seed % len(choices)]


def _term_hints(query: str) -> List[str]:
    lowered = clean_display_text(query).lower()
    rows: List[str] = []
    for key, value in TERM_HINTS.items():
        if key in lowered:
            rows.append(value)
    return rows


def _clean_list(items: Sequence[Any], limit: int = 6) -> List[str]:
    rows: List[str] = []
    for item in items or []:
        cleaned = clean_display_text(item)
        if cleaned and cleaned not in rows:
            rows.append(cleaned)
        if len(rows) >= limit:
            break
    return rows


def _normalize_model_names(text: str) -> str:
    normalized = str(text or "")
    normalized = re.sub(r"[\u4e00-\u9fff]{2,12}（DiT", "DiT（Diffusion Transformer", normalized)
    normalized = re.sub(r"[\u4e00-\u9fff]{2,12}（ViT", "ViT（Vision Transformer", normalized)
    normalized = normalized.replace("DiT，即Diffusion Transformer", "DiT（Diffusion Transformer）")
    normalized = normalized.replace("ViT，即Vision Transformer", "ViT（Vision Transformer）")
    for canonical, aliases in MODEL_NAME_NORMALIZATIONS.items():
        for alias in aliases:
            normalized = normalized.replace(alias, canonical)
    normalized = normalized.replace("DiT（Diffusion Transformer））", "DiT（Diffusion Transformer）")
    normalized = normalized.replace("ViT（Vision Transformer））", "ViT（Vision Transformer）")
    return clean_multiline_text(normalized)


class DeepLearningChatPipeline:
    def __init__(self, kb: DeepLearningKnowledgeBase) -> None:
        self.kb = kb

    def _special_response(self, query: str) -> Optional[str]:
        cleaned = clean_display_text(query)
        if not cleaned:
            return "请输入你想咨询的课程问题。"
        if GREETING_RE.match(cleaned):
            return _pick_variety(
                cleaned,
                [
                    "你好，这里是深度学习课程学习平台。你可以直接问我概念、模型结构、训练方法，或者让系统为你准备练习题。",
                    "你好，欢迎进入深度学习课程学习平台。你可以问我课程讲义内容、模型原理、代码实现思路，或继续做题复习。",
                    "你好，我可以帮你理解这门深度学习课程中的知识点，也可以配合练习区和学习报告一起复习。",
                ],
            )
        if IDENTITY_RE.match(cleaned):
            return _pick_variety(
                cleaned,
                [
                    "我是这门深度学习课程的学习助教，负责根据课程讲义和深度学习领域知识回答问题、组织练习，并辅助你查看学习报告。",
                    "我是面向这门深度学习课程的中文学习助教，可以帮助你理解概念、比较模型、查看讲义来源，并配合练习与学习报告使用。",
                    "我是这套深度学习课程平台里的课程助教，主要帮助学生做问答、练习和复习分析。",
                ],
            )
        if HELP_RE.search(cleaned):
            return "你可以直接问课程中的概念、模型结构、训练方法、代码实现思路，也可以进入左侧的练习区做题，或到学习报告查看当前强弱项分析。"
        if THANKS_RE.search(cleaned):
            return _pick_variety(
                cleaned,
                [
                    "不客气。你可以继续问某个知识点，也可以直接去做题或查看学习报告。",
                    "没问题。如果你愿意，我可以继续帮你梳理某个模型、某个训练技巧，或者给你一个代码示例。",
                    "不用客气。接下来你可以继续追问当前主题，也可以切换到练习区做题。",
                ],
            )
        return None

    def _default_plan(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
        session_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cleaned = clean_display_text(query)
        code_request = bool(CODE_RE.search(cleaned))
        explain_request = bool(EXPLAIN_RE.search(cleaned))
        in_domain = self.kb.looks_in_domain(cleaned)
        active_topic = clean_display_text((session_memory or {}).get("active_topic") or "")
        lowered = cleaned.lower()
        comparison_hint = any(token in lowered for token in ["区别", "比较", "对比", "相比", "比起", "优缺点", "vs", "versus"])
        retrieval_queries = [cleaned]
        for hint in _term_hints(cleaned):
            if hint not in retrieval_queries:
                retrieval_queries.append(hint)
        if active_topic and active_topic not in retrieval_queries:
            retrieval_queries.append(active_topic)
        if comparison_hint and active_topic and active_topic not in retrieval_queries:
            retrieval_queries.append(active_topic)
        intent = "comparison" if comparison_hint else "definition"
        if code_request and explain_request:
            intent = "mixed"
        elif code_request:
            intent = "code"
        elif not explain_request and not comparison_hint:
            intent = "other"
        return {
            "domain": "deep_learning" if in_domain else "out_of_scope",
            "intent": intent,
            "rewritten_query": cleaned,
            "retrieval_queries": retrieval_queries,
            "key_terms": [active_topic] if active_topic else [],
            "sub_questions": [cleaned],
            "comparison_axes": ["适用任务", "结构特点", "计算与数据代价"] if comparison_hint else [],
            "needs_general_knowledge": False,
            "code_request": code_request,
            "explain_request": explain_request,
            "only_code": code_request and not explain_request,
        }

    def _plan_query(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
        session_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        fallback = self._default_plan(query, history, session_memory=session_memory)
        if self.kb._llm_client is None or not self.kb._llm_model:
            return fallback

        history_block = _history_text(history)
        active_topic = clean_display_text((session_memory or {}).get("active_topic") or "")
        session_summary = clean_display_text((session_memory or {}).get("session_summary") or "")
        payload = self.kb.call_json_llm(
            "你是深度学习课程平台里的问答规划器。请只输出严格 JSON。",
            f"""
当前学生问题：
{clean_display_text(query)}

最近对话：
{history_block}

当前会话主题：
{active_topic or '无'}

当前会话摘要：
{session_summary or '无'}

请先判断这个问题是否属于“深度学习课程 / 深度学习领域可接受问题”。
注意：
1. 只要问题属于深度学习领域，即使讲义未直接覆盖，也应判定为 deep_learning。
2. 不要因为问题里出现课程外的具体模型名，就误判为 out_of_scope。
3. 如果问题明显和深度学习无关，例如政治、娱乐、体育、历史人物、实时新闻，才判定为 out_of_scope。
4. 如果问题包含多个概念或比较关系，必须在 key_terms 和 retrieval_queries 中同时保留这些概念。
5. 不要把学生写出的术语偷偷改成别的术语；若有可能存在常见含义，可在 rewritten_query 中保留原词并补充你的理解。
6. 如果问题其实包含多个子问，请拆成 sub_questions。
7. 如果问题是比较题，请给出 comparison_axes，优先考虑：适用任务、结构特点、计算/数据代价、局限性。

输出 JSON：
{{
  "domain": "deep_learning" 或 "out_of_scope",
  "intent": "definition" | "comparison" | "code" | "mixed" | "application" | "other",
  "rewritten_query": "...",
  "retrieval_queries": ["...", "..."],
  "key_terms": ["...", "..."],
  "sub_questions": ["...", "..."],
  "comparison_axes": ["...", "..."],
  "needs_general_knowledge": true 或 false
}}
""".strip(),
            temperature=0.1,
            max_tokens=420,
        )
        if not isinstance(payload, dict):
            return fallback

        domain = clean_display_text(payload.get("domain") or "").lower()
        if domain not in {"deep_learning", "out_of_scope"}:
            domain = fallback["domain"]
        if fallback["domain"] == "deep_learning":
            domain = "deep_learning"

        intent = clean_display_text(payload.get("intent") or "").lower()
        if intent not in {"definition", "comparison", "code", "mixed", "application", "other"}:
            intent = fallback["intent"]

        rewritten_query = clean_display_text(payload.get("rewritten_query") or "") or fallback["rewritten_query"]

        retrieval_queries = _clean_list(payload.get("retrieval_queries") or [], limit=8)
        if rewritten_query not in retrieval_queries:
            retrieval_queries.insert(0, rewritten_query)
        for hint in _term_hints(rewritten_query):
            if hint not in retrieval_queries:
                retrieval_queries.append(hint)

        key_terms = _clean_list(payload.get("key_terms") or [], limit=8)
        for hint in _term_hints(rewritten_query):
            if hint not in key_terms:
                key_terms.append(hint)

        sub_questions = _clean_list(payload.get("sub_questions") or [], limit=4)
        if not sub_questions:
            sub_questions = [rewritten_query]
        comparison_axes = _clean_list(payload.get("comparison_axes") or [], limit=4)
        for item in sub_questions:
            if item not in retrieval_queries:
                retrieval_queries.append(item)
        for item in key_terms:
            if item not in retrieval_queries:
                retrieval_queries.append(item)

        code_request = bool(CODE_RE.search(query)) or intent in {"code", "mixed"}
        explain_request = bool(EXPLAIN_RE.search(query)) or intent in {"definition", "comparison", "mixed", "application"}
        only_code = code_request and not bool(EXPLAIN_RE.search(query)) and intent == "code"

        return {
            "domain": domain,
            "intent": intent,
            "rewritten_query": rewritten_query,
            "retrieval_queries": retrieval_queries[:6],
            "key_terms": key_terms[:6],
            "sub_questions": sub_questions,
            "comparison_axes": comparison_axes,
            "needs_general_knowledge": bool(payload.get("needs_general_knowledge")),
            "code_request": code_request,
            "explain_request": explain_request,
            "only_code": only_code,
        }

    def _out_of_scope_response(self, query: str) -> str:
        topic = clean_display_text(query)[:26]
        return _pick_variety(
            query,
            [
                f"这个问题和当前深度学习课程无关，所以我先不直接回答“{topic}”。你可以改问神经网络、CNN、RNN、Transformer、GAN、扩散模型或训练优化相关内容。",
                f"“{topic}”不属于这门课的问答范围，我先不展开。若你愿意，可以继续问卷积网络、反向传播、LSTM、注意力机制或框架使用。",
                f"这不是当前深度学习课程内的问题，我先不直接作答“{topic}”。你可以切回课程主题，我会继续帮你分析。",
            ],
        )

    def _extract_used_citations(
        self,
        citations: Sequence[Dict[str, Any]],
        used_sources: Sequence[str],
    ) -> List[Dict[str, Any]]:
        allowed = {clean_display_text(item).upper() for item in used_sources if clean_display_text(item)}
        if not allowed:
            return []
        rows: List[Dict[str, Any]] = []
        for citation in citations:
            citation_id = clean_display_text(citation.get("citation_id") or "").upper()
            if citation_id in allowed:
                rows.append(citation)
        return rows[:3]

    def _build_session_memory(
        self,
        query: str,
        answer_text: str,
        retrieval: Dict[str, Any],
        planner: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        key_terms = list((planner or {}).get("key_terms") or [])
        related_kps = retrieval.get("related_kps") or []
        active_topic = clean_display_text(key_terms[0] if key_terms else "")
        if not active_topic:
            active_topic = clean_display_text((related_kps[0] or {}).get("name") if related_kps else "")
        if not active_topic:
            active_topic = clean_display_text(query)[:24]
        session_summary = f"当前对话主要围绕“{active_topic}”展开，最近问题是：{clean_display_text(query)[:60]}"
        return {
            "active_topic": active_topic,
            "session_summary": session_summary,
            "latest_answer_summary": clean_display_text(answer_text)[:140],
        }

    def _llm_answer(
        self,
        query: str,
        retrieval: Dict[str, Any],
        history: Sequence[Dict[str, str]],
        planner: Dict[str, Any],
        session_memory: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        citations = retrieval.get("citations") or []
        source_lines = []
        for citation in citations[:4]:
            source_lines.append(
                f"{citation['citation_id']}: {citation['source']}，{citation['location']}，主题：{citation['section']}，摘录：{citation['excerpt']}"
            )
        source_block = "\n".join(source_lines) if source_lines else "无直接讲义来源"
        history_block = _history_text(history)
        active_topic = clean_display_text((session_memory or {}).get("active_topic") or "")
        key_terms = [clean_display_text(item) for item in planner.get("key_terms") or [] if clean_display_text(item)]
        retrieval_queries = [clean_display_text(item) for item in retrieval.get("query_variants") or [] if clean_display_text(item)]
        sub_questions = [clean_display_text(item) for item in planner.get("sub_questions") or [] if clean_display_text(item)]
        comparison_axes = [clean_display_text(item) for item in planner.get("comparison_axes") or [] if clean_display_text(item)]
        term_hints = _term_hints(query)
        code_request = bool(planner.get("code_request"))
        explain_request = bool(planner.get("explain_request"))
        only_code = bool(planner.get("only_code"))
        prompt_body = f"""
你要回答一名学生关于深度学习课程的问题。

学生当前问题：
{clean_display_text(query)}

规划结果：
- 任务类型：{planner.get('intent') or 'other'}
- 关键概念：{', '.join(key_terms) if key_terms else '无'}
- 子问题：{'；'.join(sub_questions) if sub_questions else '无'}
- 比较维度：{'；'.join(comparison_axes) if comparison_axes else '无'}
- 检索查询：{'; '.join(retrieval_queries) if retrieval_queries else '无'}
- 是否允许使用深度学习通识知识补充：true
- 术语提示：{'; '.join(term_hints) if term_hints else '无'}

历史上下文：
{history_block}

当前会话主题：
{active_topic or '无'}

检索到的课程来源：
{source_block}

请遵守：
1. 第一段先直接回答学生真正想问的点，不要先铺垫空泛背景。
2. 回答必须自然、像老师或课程助教，不要像检索系统或百科词条。
2. 全部使用中文。
3. 如果问题里有多个概念、多个子问或比较关系，必须按顺序全部回答，不要只回答其中一半。
4. 如果某个术语属于深度学习领域，但讲义没有直接覆盖，可以依据可靠的深度学习知识回答；但不要把它偷偷改写成别的模型。
5. 若提供了术语提示，就必须按术语提示理解，不要改写成别的架构；如果术语提示里给出了典型任务场景，也必须沿这个场景回答。只有在术语确实仍然歧义、且当前没有明确术语提示时，才写“如果这里的 X 指 ...”。
6. 如果问题是比较题，必须明确写出双方分别更适合什么，不要写成单方面吹捧；至少比较 2 个维度，优先使用 comparison_axes。若双方本来属于不同任务范式，也要先说明“不能脱离任务目标笼统比较”。
7. 如果学生问“为什么 A 比 B 有优势”，要说明 A 的优势成立于什么前提，同时指出 B 更擅长的场景，避免回答得像营销文案；如果两者通常服务于不同任务，先把这一点说清楚。
8. 如果问题只要求代码，不要先写长篇概念介绍；先给代码块，再补 1 到 2 句必要说明。
9. 如果问题既要解释又要代码，先用 2 到 4 句讲清核心思路，再给代码块，最后补 1 到 2 句说明。
10. 只有真正支撑了回答的课程来源，才放入 used_sources；如果答案部分来自课程、部分来自通识知识，也应保留真正支撑课程部分的来源。
11. 不要在 answer 中直接写“来源”“Source”“course source”等字样，来源会由前端单独展示。
12. 少用“首先、其次、最后”这种模板化铺陈；优先用自然、直接的中文讲解。
13. 对于模型缩写或专有名词，如 CNN、DiT、ViT、GPT、BERT，优先保留英文缩写或英文全称，不要生造中文译名；推荐写法如 DiT（Diffusion Transformer）、ViT（Vision Transformer）。
14. definition / application / comparison 类型问题，正文控制在 2 到 4 个短段落或 4 到 6 个要点内。
15. `active_topic` 请用 2 到 12 个字概括最核心主题，尽量使用中文；`session_summary` 用一句中文概括学生的问题和结论。

输出 JSON 结构：
{{
  "answer": "...",
  "used_sources": ["S1", "S2"],
  "active_topic": "...",
  "session_summary": "..."
}}

附加判定：
- only_code = {str(only_code).lower()}
- code_request = {str(code_request).lower()}
- explain_request = {str(explain_request).lower()}
""".strip()

        payload = self.kb.call_json_llm(
            "你是商业化课程平台中的中文助教。请只输出严格 JSON。",
            prompt_body,
            temperature=0.42,
            max_tokens=1400 if code_request else 1100,
        )
        if isinstance(payload, dict):
            answer_text = _normalize_model_names(payload.get("answer") or "")
            if answer_text:
                return {
                    "answer": answer_text,
                    "citations": self._extract_used_citations(citations, payload.get("used_sources") or []),
                    "active_topic": clean_display_text(payload.get("active_topic") or ""),
                    "session_summary": clean_display_text(payload.get("session_summary") or ""),
                }

        text_answer = self.kb.call_text_llm(
            "你是商业化课程平台中的中文助教。请直接回答学生问题，不要输出 JSON，不要输出来源行。",
            f"""
学生当前问题：
{clean_display_text(query)}

任务类型：{planner.get('intent') or 'other'}
关键概念：{', '.join(key_terms) if key_terms else '无'}
术语提示：{'; '.join(term_hints) if term_hints else '无'}
最近对话：
{history_block}
当前主题：{active_topic or '无'}
课程来源：
{source_block}

请直接给出自然中文答案：
1. 第一段直接回答问题，不要先写空泛背景。
2. 多个概念、多个子问或比较关系要全部回答，且按学生提问顺序回答。
3. 深度学习领域但讲义未直接覆盖的部分，也可以用可靠的通识知识解释。
4. 如果提供了术语提示，就必须按术语提示理解；如果术语提示里给出了典型任务场景，也要按那个场景回答。不要把术语偷偷改成别的模型。只有在术语仍然歧义且当前没有明确术语提示时，才写“如果这里的 X 指 ...”。
5. 如果是比较题，要明确写出双方各自更适合的任务与局限，不要写成单方面吹捧；如果双方本来属于不同任务范式，要先点明“不能脱离任务目标笼统比较”。
6. 只要问题要求代码，就给出代码块；如果还要求解释，就在代码前后补简洁说明。
7. 对于模型缩写或专有名词，如 CNN、DiT、ViT、GPT、BERT，优先保留英文缩写或英文全称，不要生造中文译名；推荐写法如 DiT（Diffusion Transformer）、ViT（Vision Transformer）。
8. 语气要像老师讲解，不要像百科词条；不要输出 JSON，不要输出 used_sources，不要输出来源行。
""".strip(),
            temperature=0.35,
            max_tokens=1400 if code_request else 1100,
        )
        text_answer = _normalize_model_names(text_answer)
        if not text_answer:
            return None
        return {
            "answer": text_answer,
            "citations": [],
            "active_topic": key_terms[0] if key_terms else active_topic,
            "session_summary": f"最近围绕“{key_terms[0]}”进行了问答。" if key_terms else "",
        }

    def answer(
        self,
        query: str,
        *,
        history: Optional[Sequence[Dict[str, str]]] = None,
        top_k: int = 5,
        session_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw_query = clean_display_text(query)
        special = self._special_response(raw_query)
        if special:
            memory = self._build_session_memory(raw_query, special, {"related_kps": []})
            return {
                "answer": special,
                "citations": [],
                "related_kps": [],
                "mode": "assistant",
                "session_memory": memory,
            }

        planner = self._plan_query(raw_query, history or [], session_memory=session_memory)
        if planner.get("domain") == "out_of_scope":
            answer_text = self._out_of_scope_response(raw_query)
            memory = self._build_session_memory(raw_query, answer_text, {"related_kps": []}, planner=planner)
            return {
                "answer": answer_text,
                "citations": [],
                "related_kps": [],
                "mode": "assistant",
                "session_memory": memory,
            }

        retrieval = self.kb.retrieve(
            raw_query,
            history=history or [],
            session_memory=session_memory,
            top_k=top_k,
            extra_queries=planner.get("retrieval_queries") or planner.get("key_terms") or [],
        )
        llm_payload = self._llm_answer(raw_query, retrieval, history or [], planner, session_memory=session_memory)
        if llm_payload:
            answer_text = llm_payload["answer"]
            citations = llm_payload["citations"]
            memory = {
                "active_topic": llm_payload["active_topic"] or self._build_session_memory(raw_query, answer_text, retrieval, planner=planner)["active_topic"],
                "session_summary": llm_payload["session_summary"] or self._build_session_memory(raw_query, answer_text, retrieval, planner=planner)["session_summary"],
            }
        else:
            answer_text = self.kb.answer_from_chunks(raw_query, retrieval)
            citations = list((retrieval.get("citations") or [])[:2])
            memory = self._build_session_memory(raw_query, answer_text, retrieval, planner=planner)

        return {
            "answer": answer_text,
            "citations": citations,
            "related_kps": retrieval.get("related_kps") or [],
            "mode": "assistant",
            "session_memory": memory,
        }
