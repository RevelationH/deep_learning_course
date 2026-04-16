from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except Exception:
    HuggingFaceEmbeddings = None
    FAISS = None

from env_loader import load_project_env

load_project_env()


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = ROOT_DIR / "deep_learning_rag" / "artifacts_full_course"
DEFAULT_MATERIAL_ROOT = ROOT_DIR / "deep_learning_materials"
DEFAULT_EMBED_MODEL = os.getenv("DEEP_LEARNING_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*")
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
OPTION_RE = re.compile(r"^\s*([A-D])\.\s*(.+?)\s*$")
WS_RE = re.compile(r"\s+")
STOPWORDS = {
    "的", "了", "和", "与", "及", "在", "是", "对", "中", "上", "下", "把", "将", "并", "或",
    "the", "and", "for", "with", "from", "that", "this", "into", "about", "which", "what",
    "why", "how", "when", "where", "used", "using", "use", "course", "lecture", "week",
    "question", "questions", "deep", "learning", "model", "models",
}
DEEP_LEARNING_DOMAIN_TERMS = {
    "深度学习", "神经网络", "反向传播", "感知器", "卷积神经网络", "卷积", "池化",
    "cnn", "rnn", "lstm", "transformer", "attention", "自注意力", "位置编码",
    "gpt", "bert", "swin", "gan", "生成器", "判别器", "扩散模型", "自编码器",
    "rbm", "dbn", "dropout", "正则化", "优化", "梯度下降", "loss", "optimizer",
    "pytorch", "tensorflow", "keras", "paddlepaddle", "deepforest", "随机森林",
    "dit", "vit", "vae", "mae", "clip", "unet", "u-net", "diffusion transformer",
}
FOLLOW_UP_HINT_RE = re.compile(r"^(那|这个|这个概念|这个方法|它|那它|那这个|继续|再说|顺便|那如果|如果是这样|那代码呢|再举个例子)", re.IGNORECASE)
GREETING_RE = re.compile(r"^\s*(hi|hello|hey|你好|您好|早上好|下午好|晚上好)\s*[!.?。！？]*\s*$", re.IGNORECASE)
HELP_RE = re.compile(r"(你能做什么|你可以做什么|怎么用这个系统|help|what can you do)", re.IGNORECASE)
THANKS_RE = re.compile(r"(谢谢|感谢|thanks|thank you)", re.IGNORECASE)
IDENTITY_RE = re.compile(r"^\s*(who are you|你是谁|你是做什么的|介绍一下你自己)\s*[!.?。！？]*\s*$", re.IGNORECASE)
CODE_RE = re.compile(r"(代码|code|pytorch|tensorflow|python|示例|例子|demo|实现|implementation)", re.IGNORECASE)
EXPLAIN_RE = re.compile(r"(解释|explain|what is|什么是|区别|difference|compare|比较)", re.IGNORECASE)
DOMAIN_TERM_TOKENS = {token.lower() for token in DEEP_LEARNING_DOMAIN_TERMS if len(token) >= 2}
COURSE_INLINE_NOISE_RE = re.compile(
    r"(?i)(?:school of computer science and technology\s*计算机科学与技术学院\s*)+"
)
COURSE_PLAIN_HEADER_RE = re.compile(
    r"(?i)^(school of computer science and technology|计算机科学与技术学院|university of chinese academy of sciences|中国科学院大学)$"
)
COURSE_PAGE_NUMBER_RE = re.compile(r"^\d{1,3}$")
COURSE_LEADING_BULLET_RE = re.compile(r"^[qQ•·▪▫■□◆◇○●◦Ø\-–—]+\s*")
COURSE_TITLE_PREFIX_RE = re.compile(r"^(?:Ø|第[一二三四五六七八九十0-9]+章)\s*")
SHORT_TERM_RE = re.compile(r"\b(?:cnn|rnn|lstm|gan|gpt|bert|dit|vit|vae|mlp|rbm|dbn)\b", re.IGNORECASE)


def clean_display_text(text: Any) -> str:
    value = str(text or "").replace("\x00", " ")
    value = value.encode("utf-8", "ignore").decode("utf-8", "ignore")
    value = WS_RE.sub(" ", value)
    return value.strip()


def clean_multiline_text(text: Any) -> str:
    value = str(text or "").replace("\x00", " ")
    value = value.encode("utf-8", "ignore").decode("utf-8", "ignore")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def clean_course_line(text: Any) -> str:
    value = str(text or "").replace("\x00", " ")
    value = value.encode("utf-8", "ignore").decode("utf-8", "ignore")
    value = COURSE_INLINE_NOISE_RE.sub(" ", value)
    value = value.replace("SCHOOL OF COMPUTER SCIENCE AND TECHNOLOGY", " ")
    value = value.replace("计算机科学与技术学院", " ")
    value = value.replace("University of Chinese Academy of Sciences", " ")
    value = value.replace("中国科学院大学", " ")
    value = COURSE_TITLE_PREFIX_RE.sub("", value.strip())
    value = COURSE_LEADING_BULLET_RE.sub("", value.strip())
    value = WS_RE.sub(" ", value)
    return value.strip(" -|:;,.")


def is_noise_line(text: Any) -> bool:
    value = clean_course_line(text)
    if not value:
        return True
    lowered = value.lower()
    if COURSE_PAGE_NUMBER_RE.fullmatch(value):
        return True
    if COURSE_PLAIN_HEADER_RE.fullmatch(lowered):
        return True
    if len(value) <= 1:
        return True
    return False


def course_lines(text: Any, limit: int = 12) -> List[str]:
    rows: List[str] = []
    seen: set[str] = set()
    for raw_line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = clean_course_line(raw_line)
        if is_noise_line(line):
            continue
        normalized = line.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        rows.append(line)
        if len(rows) >= limit:
            break
    return rows


def clean_course_excerpt(text: Any, max_chars: int = 280) -> str:
    lines = course_lines(text, limit=8)
    excerpt = " ".join(lines)
    excerpt = WS_RE.sub(" ", excerpt).strip()
    if len(excerpt) <= max_chars:
        return excerpt
    return excerpt[: max_chars - 1].rstrip() + "…"


def derive_section_label(title: Any, text: Any, fallback: str = "相关讲义部分") -> str:
    for candidate in [title, *course_lines(text, limit=6)]:
        label = clean_course_line(candidate)
        if is_noise_line(label):
            continue
        if 2 <= len(label) <= 80:
            return label
    return fallback


def tokenize(text: Any) -> List[str]:
    lowered = clean_display_text(text).lower()
    tokens: List[str] = []
    tokens.extend(WORD_RE.findall(lowered))
    for match in CJK_RE.findall(lowered):
        if len(match) == 1:
            tokens.append(match)
            continue
        for index in range(len(match) - 1):
            tokens.append(match[index:index + 2])
    return [token for token in tokens if token and token not in STOPWORDS]


def parse_options(options: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for option in options:
        match = OPTION_RE.match(str(option))
        if not match:
            continue
        rows.append({"label": match.group(1).upper(), "text": clean_display_text(match.group(2))})
    return rows


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_jsonish_text(text: Any) -> Any:
    candidate = clean_multiline_text(text)
    if not candidate:
        return None

    variants: List[str] = [candidate]
    if candidate.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
        if stripped:
            variants.append(stripped)

    decoder = json.JSONDecoder()
    seen: set[str] = set()
    for variant in variants:
        if not variant or variant in seen:
            continue
        seen.add(variant)
        try:
            return json.loads(variant)
        except Exception:
            pass
        for start_index in [0] + [match.start() for match in re.finditer(r"[\{\[]", variant)]:
            snippet = variant[start_index:].strip()
            if not snippet or snippet in seen:
                continue
            seen.add(snippet)
            try:
                parsed, _ = decoder.raw_decode(snippet)
                return parsed
            except Exception:
                continue
    return None


def lexical_rank(query: str, items: Sequence[Any], text_getter, top_k: int = 8) -> List[Tuple[float, Any]]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    tokenized_docs = [tokenize(text_getter(item)) for item in items]
    if not tokenized_docs:
        return []

    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    doc_count = len(tokenized_docs)
    avg_len = max(sum(len(tokens) for tokens in tokenized_docs) / max(doc_count, 1), 1.0)
    hits: List[Tuple[float, Any]] = []
    for item, tokens in zip(items, tokenized_docs):
        if not tokens:
            continue
        term_freq = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in set(query_tokens):
            freq = term_freq.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = freq + 1.5 * (1 - 0.75 + 0.75 * doc_len / avg_len)
            score += idf * (freq * 2.5 / denom)
        if score > 0:
            hits.append((score, item))
    hits.sort(key=lambda pair: pair[0], reverse=True)
    return hits[:top_k]


def _dense_stack_available() -> bool:
    return HuggingFaceEmbeddings is not None and FAISS is not None


def _resolve_llm_client() -> Tuple[Optional[Any], Optional[str]]:
    if OpenAI is None:
        return None, None
    api_key = (
        os.getenv("COURSE_LLM_API_KEY")
        or os.getenv("DEEP_LEARNING_LLM_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("KIMI_API_KEY")
    )
    base_url = (
        os.getenv("COURSE_LLM_BASE_URL")
        or os.getenv("DEEP_LEARNING_LLM_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or os.getenv("KIMI_API_BASE")
    )
    model = (
        os.getenv("COURSE_LLM_MODEL")
        or os.getenv("DEEP_LEARNING_LLM_MODEL")
        or os.getenv("DEEPSEEK_CHAT_MODEL")
        or os.getenv("KIMI_MODEL")
        or "deepseek-chat"
    )
    if not api_key:
        return None, None
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    return client, model


class DeepLearningKnowledgeBase:
    def __init__(self) -> None:
        artifact_dir = os.getenv("DEEP_LEARNING_ARTIFACT_DIR", "").strip()
        material_root = os.getenv("DEEP_LEARNING_MATERIAL_ROOT", "").strip()
        self.artifact_dir = Path(artifact_dir) if artifact_dir else DEFAULT_ARTIFACT_DIR
        self.material_root = Path(material_root) if material_root else DEFAULT_MATERIAL_ROOT
        if not self.artifact_dir.exists():
            raise FileNotFoundError(f"未找到深度学习课程 artifacts：{self.artifact_dir}")

        self.kps: List[Dict[str, Any]] = read_json(self.artifact_dir / "knowledge_points.json")
        self.questions: List[Dict[str, Any]] = read_json(self.artifact_dir / "questions.json")
        self.chunks: List[Dict[str, Any]] = read_jsonl(self.artifact_dir / "chunks.jsonl")
        self.build_meta: Dict[str, Any] = read_json(self.artifact_dir / "build_meta.json")
        self.kp_by_id = {item["kp_id"]: dict(item) for item in self.kps}
        self.chunks_by_id = {item["chunk_id"]: dict(item) for item in self.chunks}
        self.questions_by_kp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.questions_by_id: Dict[str, Dict[str, Any]] = {}
        self.chunk_family_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for chunk in self.chunks:
            self.chunk_family_by_source[str(chunk.get("relative_path") or "")].append(chunk)

        for question in self.questions:
            row = dict(question)
            row["parsed_options"] = parse_options(row.get("options") or [])
            self.questions_by_kp[row["kp_id"]].append(row)
            self.questions_by_id[row["question_id"]] = row

        for kp in self.kps:
            kp_id = kp["kp_id"]
            self.kp_by_id[kp_id]["questions"] = list(self.questions_by_kp.get(kp_id, []))
            self.kp_by_id[kp_id]["question_count"] = len(self.questions_by_kp.get(kp_id, []))

        self._llm_client, self._llm_model = _resolve_llm_client()
        self._dense_store = None
        self.dense_enabled = False
        self._load_dense_index()

    def _load_dense_index(self) -> None:
        if os.getenv("DEEP_LEARNING_DISABLE_DENSE", "").strip() == "1":
            return
        if not _dense_stack_available():
            return
        index_dir = self.artifact_dir / "chunk_index"
        if not index_dir.exists():
            return
        try:
            embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})
            self._dense_store = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
            self.dense_enabled = True
        except Exception:
            self._dense_store = None
            self.dense_enabled = False

    def list_knowledge_points(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in self.kps:
            row = dict(item)
            row["question_count"] = len(self.questions_by_kp.get(item["kp_id"], []))
            row["questions"] = list(self.questions_by_kp.get(item["kp_id"], []))
            rows.append(row)
        return rows

    def get_kp(self, kp_id: str) -> Optional[Dict[str, Any]]:
        kp = self.kp_by_id.get(kp_id)
        return dict(kp) if kp else None

    def related_questions(self, kp_id: str) -> List[Dict[str, Any]]:
        return [dict(item) for item in self.questions_by_kp.get(kp_id, [])]

    def kp_review_refs(self, kp_id: str, limit: int = 2) -> List[Dict[str, Any]]:
        kp = self.kp_by_id.get(kp_id)
        if not kp:
            return []
        refs: List[Dict[str, Any]] = []
        seen: set[Tuple[str, int]] = set()
        for chunk_id in kp.get("support_chunk_ids", []):
            chunk = self.chunks_by_id.get(chunk_id)
            if not chunk:
                continue
            key = (str(chunk.get("relative_path") or ""), int(chunk.get("unit_index") or 0))
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                {
                    "source": chunk["relative_path"],
                    "unit_type": chunk.get("unit_type") or "page",
                    "unit_index": int(chunk.get("unit_index") or 0),
                    "chunk_index": int(chunk.get("chunk_index") or 0),
                    "section": derive_section_label(chunk.get("title"), chunk.get("text")),
                    "location": f"第 {int(chunk.get('unit_index') or 0)} 页",
                }
            )
            if len(refs) >= limit:
                break
        return refs

    def reference_context(self, source: str, unit_type: str = "", unit_index: int = 0, chunk_index: int = 0) -> Optional[Dict[str, Any]]:
        cleaned_source = str(source or "").replace("\\", "/").strip()
        if not cleaned_source:
            return None
        candidates = self.chunk_family_by_source.get(cleaned_source, [])
        best = None
        for chunk in candidates:
            if unit_index and int(chunk.get("unit_index") or 0) != int(unit_index):
                continue
            if chunk_index and int(chunk.get("chunk_index") or 0) != int(chunk_index):
                continue
            best = chunk
            break
        if best is None and candidates:
            best = candidates[0]
        if not best:
            return None
        return {
            "source": cleaned_source,
            "display_source": Path(cleaned_source).name,
            "unit_type": best.get("unit_type") or unit_type or "page",
            "unit_index": int(best.get("unit_index") or unit_index or 0),
            "chunk_index": int(best.get("chunk_index") or chunk_index or 0),
            "section": derive_section_label(best.get("title"), best.get("text")),
            "excerpt": clean_course_excerpt(best.get("text") or ""),
            "location": f"第 {int(best.get('unit_index') or unit_index or 0)} 页",
        }

    def suggest_session_title(self, user_message: str, assistant_message: str = "") -> str:
        text = clean_display_text(user_message)
        text = re.sub(r"[？?。！!]+$", "", text)
        text = re.sub(r"^(请问|请你|请帮我|帮我|我想知道|能不能|可以不可以|可以|请解释一下)", "", text).strip()
        if not text:
            return "新对话"
        if self._llm_client is not None and self._llm_model:
            try:
                response = self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[
                        {"role": "system", "content": "请根据学生提问生成一个简短中文标题，不超过12个字，不要标点，不要解释。"},
                        {"role": "user", "content": f"学生提问：{text}\n回答摘要：{clean_display_text(assistant_message)[:120]}"},
                    ],
                    temperature=0.2,
                    max_tokens=30,
                )
                title = clean_display_text(response.choices[0].message.content or "")
                title = re.sub(r"[，。；：、“”\"'`]+", "", title).strip()
                if 2 <= len(title) <= 16:
                    return title
            except Exception:
                pass
        return text[:16] or "新对话"

    def _topic_labels(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        ranked = lexical_rank(query, self.kps, lambda item: " ".join([item["name"], item["description"], " ".join(item.get("keywords", []))]), top_k=limit)
        return [
            {
                "kp_id": item["kp_id"],
                "name": item["name"],
                "description": item["description"],
                "score": round(score, 4),
            }
            for score, item in ranked
        ]

    def looks_in_domain(self, query: str) -> bool:
        lowered = clean_display_text(query).lower()
        if not lowered:
            return False
        if any(term.lower() in lowered for term in DEEP_LEARNING_DOMAIN_TERMS):
            return True
        if SHORT_TERM_RE.search(lowered):
            return True
        token_set = set(tokenize(lowered))
        if token_set & DOMAIN_TERM_TOKENS:
            return True
        related_kps = self._topic_labels(lowered, limit=1)
        return bool(related_kps and related_kps[0]["score"] >= 1.0)

    def _resolve_follow_up_query(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
        session_memory: Optional[Dict[str, Any]] = None,
    ) -> str:
        cleaned = clean_display_text(query)
        if not cleaned:
            return ""
        if not FOLLOW_UP_HINT_RE.search(cleaned):
            return cleaned
        previous_topic = clean_display_text((session_memory or {}).get("active_topic") or "")
        for turn in reversed(list(history or [])):
            content = clean_display_text(turn.get("content") or "")
            if turn.get("role") == "user" and content:
                if previous_topic and previous_topic not in cleaned:
                    return f"{cleaned}，主题仍然是：{previous_topic}。上一个问题是：{content}"
                return f"{cleaned}。请结合上一个问题：{content}"
        if previous_topic:
            return f"{cleaned}，主题仍然是：{previous_topic}"
        return cleaned

    def retrieve(
        self,
        query: str,
        *,
        history: Optional[Sequence[Dict[str, str]]] = None,
        session_memory: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        extra_queries: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        resolved_query = self._resolve_follow_up_query(query, history or [], session_memory=session_memory)
        combined_scores: Dict[str, float] = {}
        query_variants: List[str] = [resolved_query]
        for item in extra_queries or []:
            cleaned = clean_display_text(item)
            if cleaned and cleaned not in query_variants:
                query_variants.append(cleaned)

        for query_index, current_query in enumerate(query_variants, start=1):
            lexical_hits = lexical_rank(
                current_query,
                self.chunks,
                lambda item: " ".join([str(item.get("title") or ""), str(item.get("text") or "")]),
                top_k=max(top_k * 3, 10),
            )
            for rank, (score, item) in enumerate(lexical_hits, start=1):
                weight = 1.0 if query_index == 1 else 0.7
                combined_scores[item["chunk_id"]] = combined_scores.get(item["chunk_id"], 0.0) + (score + (1.0 / rank)) * weight

            if self.dense_enabled and self._dense_store is not None:
                try:
                    dense_hits = self._dense_store.similarity_search_with_score(current_query, k=max(top_k * 3, 10))
                    for rank, (doc, score) in enumerate(dense_hits, start=1):
                        chunk_id = str(doc.metadata.get("chunk_id") or "")
                        if not chunk_id:
                            continue
                        similarity = 1.0 / (1.0 + float(score))
                        weight = 1.0 if query_index == 1 else 0.7
                        combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + (similarity + (1.0 / rank)) * weight
                except Exception:
                    pass

        ranked_chunks = []
        for chunk_id, score in combined_scores.items():
            chunk = self.chunks_by_id.get(chunk_id)
            if not chunk:
                continue
            ranked_chunks.append((score, chunk))
        ranked_chunks.sort(key=lambda pair: pair[0], reverse=True)
        hits = [dict(item) for _score, item in ranked_chunks[:top_k]]

        citations: List[Dict[str, Any]] = []
        seen_units: set[Tuple[str, int]] = set()
        for item in hits:
            key = (str(item.get("relative_path") or ""), int(item.get("unit_index") or 0))
            if key in seen_units:
                continue
            seen_units.add(key)
            citations.append(
                {
                    "citation_id": f"S{len(citations) + 1}",
                    "source": item["relative_path"],
                    "unit_type": item.get("unit_type") or "page",
                    "unit_index": int(item.get("unit_index") or 0),
                    "chunk_index": int(item.get("chunk_index") or 0),
                    "section": derive_section_label(item.get("title"), item.get("text")),
                    "excerpt": clean_course_excerpt(item.get("text") or ""),
                    "location": f"第 {int(item.get('unit_index') or 0)} 页",
                }
            )

        related_kps = self._topic_labels(" ".join(query_variants), limit=3)
        coverage_level = "none"
        if hits:
            top_score = ranked_chunks[0][0]
            coverage_level = "direct" if top_score >= 2.8 else "related"
        return {
            "resolved_query": resolved_query,
            "query_variants": query_variants,
            "hits": hits,
            "citations": citations,
            "related_kps": related_kps,
            "coverage_level": coverage_level,
        }

    def answer_from_chunks(self, query: str, retrieval: Dict[str, Any]) -> str:
        hits = retrieval.get("hits") or []
        if not hits:
            return "这个问题和深度学习课程相关，但我暂时没有在当前讲义中找到足够直接的支持内容。你可以换一种更具体的问法，比如指定某个模型、机制或训练方法。"

        opening = "结合当前课程讲义，可以先这样理解："
        top = hits[0]
        summary = clean_course_excerpt(top.get("text") or "", max_chars=320)
        if not summary:
            summary = derive_section_label(top.get("title"), top.get("text"))
        if CODE_RE.search(query):
            return (
                "你的问题与本课程相关，但当前环境还没有可用的大模型生成更自然的代码讲解。"
                " 先给你一个基于讲义方向的要点提示：\n\n"
                f"{summary}"
            )
        return f"{opening}\n\n{summary}"

    def call_json_llm(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.3, max_tokens: int = 900) -> Optional[Dict[str, Any]]:
        if self._llm_client is None or not self._llm_model:
            return None
        response = self._llm_client.chat.completions.create(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content or ""
        parsed = parse_jsonish_text(raw)
        return parsed if isinstance(parsed, (dict, list)) else None

    def call_text_llm(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.3, max_tokens: int = 900) -> str:
        if self._llm_client is None or not self._llm_model:
            return ""
        response = self._llm_client.chat.completions.create(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return clean_multiline_text(response.choices[0].message.content or "")

    def generate_follow_up_variants(
        self,
        kp_id: str,
        weak_reasons: Sequence[str],
        existing_questions: Sequence[Dict[str, Any]],
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        kp = self.kp_by_id.get(kp_id)
        if not kp:
            return []
        base_refs = self.kp_review_refs(kp_id, limit=2)
        if self._llm_client is None or not self._llm_model:
            fallback = []
            for question in list(existing_questions)[:limit]:
                item = dict(question)
                item["review_refs"] = item.get("review_refs") or base_refs
                fallback.append(item)
            return fallback[:limit]

        existing_stems = "\n".join(f"- {clean_display_text(item.get('question') or '')}" for item in existing_questions[:5])
        weak_lines = "\n".join(f"- {clean_display_text(item)}" for item in weak_reasons if clean_display_text(item)) or "- 学生在这个知识点上稳定性不足。"
        payload = self.call_json_llm(
            "你是面向高校学生的中文助教。请为学习报告生成自然、像老师出的选择题。只输出严格 JSON。",
            f"""
知识点：{kp['name']}
知识点说明：{kp['description']}
薄弱原因：
{weak_lines}

现有题目示例：
{existing_stems}

请生成 {limit} 道新的中文单选题，要求：
1. 风格要像高校课程测验，不要像模板题或百科问答。
2. 每题包含 question, options, correct_option, answer, explanation。
3. options 必须是 4 个，格式为 A. / B. / C. / D.
4. 干扰项要像学生真实会混淆的选项，不要使用明显荒谬、滑稽或一眼排除的错误选项。
5. 优先考察概念区分、条件变化下的判断、公式或结构含义，不要大量使用“最准确概括”“最符合”这类空泛模板。
6. explanation 必须简洁，并指出学生应复习的思路或概念。
7. 只输出一个 JSON 数组。
""".strip(),
            temperature=0.5,
            max_tokens=1400,
        )
        if not isinstance(payload, list):
            return self.generate_follow_up_variants(kp_id, weak_reasons, existing_questions, limit=limit)[:limit] if self._llm_client is None else []

        rows: List[Dict[str, Any]] = []
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                continue
            options = item.get("options") or []
            parsed = parse_options(options)
            if len(parsed) != 4:
                continue
            correct_option = clean_display_text(item.get("correct_option") or "").upper()
            if correct_option not in {"A", "B", "C", "D"}:
                continue
            rows.append(
                {
                    "question_id": f"{kp_id}-followup-{index}",
                    "kp_id": kp_id,
                    "kp_name": kp["name"],
                    "question_type": "multiple_choice",
                    "question": clean_display_text(item.get("question") or ""),
                    "options": [clean_display_text(option) for option in options],
                    "parsed_options": parsed,
                    "correct_option": correct_option,
                    "answer": clean_display_text(item.get("answer") or parsed[ord(correct_option) - ord('A')]["text"]),
                    "explanation": clean_display_text(item.get("explanation") or ""),
                    "review_refs": base_refs,
                    "source_chunk_ids": list(kp.get("support_chunk_ids") or []),
                    "source_files": list(kp.get("source_files") or []),
                }
            )
        return rows[:limit]
