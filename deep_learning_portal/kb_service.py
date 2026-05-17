from __future__ import annotations

import json
import math
import os
import random
import re
import textwrap
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
    "深度学习", "机器学习", "神经网络", "反向传播", "前向传播", "感知器", "多层感知机",
    "全连接", "全连接层", "全连接网络", "fully connected", "linear layer", "mlp",
    "卷积神经网络", "卷积", "池化", "激活函数", "损失函数", "交叉熵", "批归一化",
    "归一化", "残差连接", "注意力", "自注意力", "位置编码", "嵌入", "预训练",
    "微调", "迁移学习", "蒸馏", "对比学习", "表示学习", "自监督", "监督学习",
    "生成模型", "判别模型", "扩散模型", "去噪", "分类", "回归", "分割",
    "cnn", "rnn", "gru", "lstm", "mlp", "transformer", "attention", "encoder",
    "decoder", "gpt", "bert", "swin", "gan", "diffusion", "diffusion model",
    "generator", "discriminator", "autoencoder", "rbm", "dbn", "dropout",
    "regularization", "loss", "optimizer", "sgd", "adam", "adamw", "softmax",
    "relu", "gelu", "batchnorm", "layernorm", "pytorch", "tensorflow", "keras",
    "paddlepaddle", "deepforest", "随机森林", "dit", "vit", "vae", "mae", "clip",
    "unet", "u-net", "diffusion transformer", "lora", "low-rank adaptation",
    "peft", "parameter-efficient fine-tuning", "prompt tuning", "prefix tuning",
    "instruction tuning", "alignment", "rlhf", "dpo", "sft",
    "moe", "mixture of experts", "expert routing", "router", "sparse activation",
    "llm", "large language model", "tokenizer", "embedding layer",
}
DEEP_LEARNING_DOMAIN_PHRASES = {
    "mixture of experts",
    "parameter-efficient fine-tuning",
    "prompt tuning",
    "prefix tuning",
    "instruction tuning",
    "large language model",
    "diffusion transformer",
    "low-rank adaptation",
    "fully connected",
    "linear layer",
    "multi-layer perceptron",
    "全连接网络",
    "全连接层",
    "参数高效微调",
    "提示微调",
    "指令微调",
    "混合专家",
    "大语言模型",
}
DEEP_LEARNING_STRONG_SINGLE_TERM_ANCHORS = {
    "cnn",
    "rnn",
    "lstm",
    "gru",
    "mlp",
    "transformer",
    "batchnorm",
    "layernorm",
    "dropout",
    "attention",
    "vit",
    "dit",
    "vae",
    "gan",
    "clip",
    "mae",
    "lora",
    "peft",
    "moe",
    "pytorch",
    "tensorflow",
    "卷积",
    "池化",
    "反向传播",
    "梯度",
    "归一化",
    "扩散模型",
    "全连接",
}
FOLLOW_UP_HINT_RE = re.compile(
    r"(?:^|\b)(那|这个|这个概念|这个方法|它|它的|它们|那它|那这个|继续|再说|顺便|那如果|如果是这样|那代码呢|再举个例子|前者|后者|刚才那个|上面这个)",
    re.IGNORECASE,
)
GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|你好(?:啊|呀|哈)?|您好|早上好|下午好|晚上好)\s*[!.?。！？~～]*\s*$",
    re.IGNORECASE,
)
HELP_RE = re.compile(
    r"^\s*(?:"
    r"help"
    r"|what can you do"
    r"|how (?:do|can) i use (?:this )?(?:system|platform)"
    r"|how to use (?:this )?(?:system|platform)"
    r"|how does this (?:system|platform) work"
    r"|你能做什么"
    r"|你可以做什么"
    r"|怎么用(?:这个|本)?(?:系统|平台)"
    r"|如何使用(?:这个|本)?(?:系统|平台)"
    r"|这个(?:系统|平台)怎么用"
    r")\s*[.!?。！？~～]*\s*$",
    re.IGNORECASE,
)
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
QUERY_INTENT_STOPWORDS = {
    "什么", "什么是", "怎么", "如何", "为什么", "区别", "比较", "对比", "relation", "relationship",
    "difference", "compare", "explain", "define", "show", "give", "tell", "about", "lecture",
    "lectures", "review", "section", "page", "where", "which", "code", "example", "examples",
    "复习", "讲义", "哪部分", "哪里", "代码", "示例", "例子", "部分", "课程", "通常", "考试", "考题",
}
LOW_SIGNAL_MATCH_TOKENS = {"normalization", "归一", "一化"}
GENERIC_COURSE_TOKENS = {
    "深度学习", "机器学习", "model", "models", "learning", "deep", "network", "networks", "course",
}
NOISY_CHUNK_HEADINGS = {
    "school of computer science and technology",
    "谢谢",
    "thanks",
}
FRAMEWORK_QUERY_RE = re.compile(r"\b(pytorch|tensorflow|keras|paddlepaddle)\b|代码|实现|demo|api", re.IGNORECASE)
TOOL_OR_PLATFORM_TEXT_RE = re.compile(r"(tools|tool|platform|pytorch|tensorflow|keras|paddlepaddle|开发平台)", re.IGNORECASE)
LOW_VALUE_SECTION_RE = re.compile(r"(中英文术语对照|术语对照|glossary)", re.IGNORECASE)
REFERENCE_STYLE_QUERY_RE = re.compile(r"(哪.*讲|哪里讲|哪部分|哪一页|哪一讲|先看哪|复习|review|which page|which section|where)", re.IGNORECASE)
DEEP_LEARNING_QUERY_ALIAS_RULES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bbatch\s*norm(?:alization)?\b|\bbn\b", re.IGNORECASE), "batchnorm batch normalization 批归一化"),
    (re.compile(r"\blayer\s*norm(?:alization)?\b|\bln\b", re.IGNORECASE), "layernorm layer normalization 层归一化"),
    (re.compile(r"\bdrop\s*out\b", re.IGNORECASE), "dropout"),
    (re.compile(r"\bweight\s*decay\b", re.IGNORECASE), "weight decay 权重衰减"),
    (re.compile(r"\bback\s*prop(?:agation)?\b|\bbp\b", re.IGNORECASE), "backpropagation 反向传播"),
    (re.compile(r"\bgradient\s*descent\b", re.IGNORECASE), "gradient descent 梯度下降"),
    (re.compile(r"\bself[- ]attention\b", re.IGNORECASE), "self-attention 自注意力"),
    (re.compile(r"\bpositional?\s*encoding\b", re.IGNORECASE), "positional encoding 位置编码"),
    (re.compile(r"\bpadding\b", re.IGNORECASE), "padding 填充 same padding valid padding"),
    (re.compile(r"\bstride\b", re.IGNORECASE), "stride 步长"),
    (re.compile(r"\bpool(?:ing)?\b", re.IGNORECASE), "pooling 池化"),
    (re.compile(r"\bconv(?:olution|olutional)?\b", re.IGNORECASE), "convolution 卷积"),
    (re.compile(r"\bcnn\b", re.IGNORECASE), "cnn 卷积神经网络"),
    (re.compile(r"\brnn\b", re.IGNORECASE), "rnn 循环神经网络"),
    (re.compile(r"\blstm\b", re.IGNORECASE), "lstm"),
    (re.compile(r"\bgru\b", re.IGNORECASE), "gru"),
    (re.compile(r"\btransformer\b", re.IGNORECASE), "transformer"),
    (re.compile(r"\battention\b", re.IGNORECASE), "attention 注意力"),
    (re.compile(r"\bclip\b", re.IGNORECASE), "clip"),
    (re.compile(r"\bmae\b", re.IGNORECASE), "mae masked autoencoder"),
    (re.compile(r"\bvit\b", re.IGNORECASE), "vit vision transformer"),
    (re.compile(r"\bdit\b", re.IGNORECASE), "dit diffusion transformer"),
    (re.compile(r"\bu-?net\b", re.IGNORECASE), "unet u-net"),
    (re.compile(r"\bvae\b", re.IGNORECASE), "vae variational autoencoder"),
    (re.compile(r"\bgan\b", re.IGNORECASE), "gan generative adversarial network"),
)
TOPIC_ALIAS_GROUPS: Dict[str, Tuple[str, ...]] = {
    "batchnorm": ("batchnorm", "batch normalization", "批归一化"),
    "layernorm": ("layernorm", "layer normalization", "层归一化"),
    "fully_connected": ("fully connected", "full connected", "linear layer", "全连接", "全连接层", "全连接网络"),
    "mlp": ("mlp", "multi-layer perceptron", "多层感知机"),
    "padding": ("padding", "填充", "same padding", "valid padding"),
    "dropout": ("dropout",),
    "convolution": ("convolution", "卷积"),
    "pooling": ("pooling", "池化"),
    "cnn": ("cnn", "卷积神经网络"),
    "rnn": ("rnn", "循环神经网络"),
    "lstm": ("lstm",),
    "gru": ("gru",),
    "transformer": ("transformer",),
    "self_attention": ("self-attention", "自注意力"),
    "attention": ("attention", "注意力"),
    "backpropagation": ("backpropagation", "backprop", "反向传播"),
    "gradient_descent": ("gradient descent", "梯度下降"),
    "weight_decay": ("weight decay", "权重衰减"),
    "gan": ("gan", "生成对抗网络"),
    "vae": ("vae", "variational autoencoder", "变分自编码器"),
    "diffusion": ("diffusion", "diffusion model", "扩散模型"),
    "unet": ("unet", "u-net"),
    "vit": ("vit", "vision transformer"),
    "dit": ("dit", "diffusion transformer"),
    "clip": ("clip",),
    "lora": ("lora", "low-rank adaptation"),
    "peft": ("peft", "parameter-efficient fine-tuning", "参数高效微调"),
    "prompt_tuning": ("prompt tuning", "提示微调"),
    "moe": ("moe", "mixture of experts", "混合专家"),
    "llm": ("llm", "large language model", "大语言模型"),
}
TOPIC_DISPLAY_LABELS: Dict[str, str] = {
    "batchnorm": "BatchNorm",
    "layernorm": "LayerNorm",
    "fully_connected": "全连接层",
    "mlp": "多层感知机",
    "padding": "padding",
    "dropout": "Dropout",
    "convolution": "卷积",
    "pooling": "池化",
    "cnn": "CNN",
    "rnn": "RNN",
    "lstm": "LSTM",
    "gru": "GRU",
    "transformer": "Transformer",
    "self_attention": "自注意力",
    "attention": "注意力机制",
    "backpropagation": "反向传播",
    "gradient_descent": "梯度下降",
    "weight_decay": "权重衰减",
    "gan": "GAN",
    "vae": "VAE",
    "diffusion": "扩散模型",
    "unet": "U-Net",
    "vit": "ViT",
    "dit": "DiT",
    "clip": "CLIP",
    "lora": "LoRA",
    "peft": "参数高效微调",
    "prompt_tuning": "Prompt Tuning",
    "moe": "MoE",
    "llm": "大语言模型",
}


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


def expand_query_aliases(text: Any) -> str:
    raw = clean_display_text(text)
    if not raw:
        return ""
    expanded = raw.lower()
    for pattern, canonical in DEEP_LEARNING_QUERY_ALIAS_RULES:
        def repl(match: re.Match[str]) -> str:
            matched = clean_display_text(match.group(0)).lower().strip()
            return matched if canonical in matched else f"{matched} {canonical}".strip()

        expanded = pattern.sub(repl, expanded)
    return WS_RE.sub(" ", expanded).strip()


def meaningful_query_tokens(text: Any) -> List[str]:
    return [
        token
        for token in tokenize(expand_query_aliases(text))
        if len(token) > 1 and token not in QUERY_INTENT_STOPWORDS
    ]


def query_phrases(text: Any, max_ngram: int = 4) -> List[str]:
    tokens = meaningful_query_tokens(text)
    phrases: List[str] = []
    seen: set[str] = set()
    if not tokens:
        return phrases
    max_width = min(max_ngram, len(tokens))
    for width in range(max_width, 1, -1):
        for index in range(len(tokens) - width + 1):
            phrase = " ".join(tokens[index : index + width]).strip()
            if len(phrase) < 4 or phrase in seen:
                continue
            seen.add(phrase)
            phrases.append(phrase)
    return phrases


def query_anchor_aliases(text: Any) -> List[Tuple[str, Tuple[str, ...]]]:
    expanded = expand_query_aliases(text)
    anchors: List[Tuple[str, Tuple[str, ...]]] = []
    for key, aliases in TOPIC_ALIAS_GROUPS.items():
        if any(alias in expanded for alias in aliases):
            anchors.append((key, aliases))
    return anchors


def chunk_quality_adjustment(chunk: Dict[str, Any]) -> float:
    title = clean_course_line(chunk.get("title") or "").lower()
    lines = course_lines(chunk.get("text") or "", limit=4)
    summary = clean_display_text(" ".join(lines)).lower()
    score = 0.0
    if title and title not in NOISY_CHUNK_HEADINGS and not is_noise_line(title):
        score += 0.35
    else:
        score -= 0.6
    if len(summary) >= 24:
        score += 0.2
    elif len(summary) < 12:
        score -= 0.45
    if "提纲" in summary or "outline" in summary:
        score -= 0.2
    if LOW_VALUE_SECTION_RE.search(title) or LOW_VALUE_SECTION_RE.search(summary):
        score -= 1.1
    if title in NOISY_CHUNK_HEADINGS or "谢谢" in summary:
        score -= 1.2
    return score


def looks_like_follow_up(text: Any) -> bool:
    cleaned = clean_display_text(text)
    if not cleaned:
        return False
    if FOLLOW_UP_HINT_RE.search(cleaned):
        return True
    lowered = cleaned.lower()
    if len(cleaned) <= 40 and any(token in cleaned for token in ["它", "它的", "它们", "这个", "那个", "前者", "后者"]):
        return True
    return len(cleaned) <= 60 and bool(re.search(r"\b(it|its|this|that|these|those|former|latter)\b", lowered))


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
        self.chunk_search_texts: Dict[str, str] = {}
        self.chunk_term_sets: Dict[str, set[str]] = {}
        self.kp_search_texts: Dict[str, str] = {}
        self.kp_term_sets: Dict[str, set[str]] = {}
        self.kp_heading_texts: Dict[str, str] = {}
        self.kp_heading_term_sets: Dict[str, set[str]] = {}

        for chunk in self.chunks:
            self.chunk_family_by_source[str(chunk.get("relative_path") or "")].append(chunk)
            chunk_id = str(chunk.get("chunk_id") or "")
            if not chunk_id:
                continue
            search_text = expand_query_aliases(
                " ".join(
                    [
                        str(chunk.get("relative_path") or ""),
                        str(chunk.get("title") or ""),
                        str(chunk.get("text") or ""),
                    ]
                )
            )
            self.chunk_search_texts[chunk_id] = search_text
            self.chunk_term_sets[chunk_id] = {
                token for token in meaningful_query_tokens(search_text)
                if token not in GENERIC_COURSE_TOKENS
            }

        for question in self.questions:
            row = dict(question)
            row["parsed_options"] = parse_options(row.get("options") or [])
            self.questions_by_kp[row["kp_id"]].append(row)
            self.questions_by_id[row["question_id"]] = row

        for kp in self.kps:
            kp_id = kp["kp_id"]
            self.kp_by_id[kp_id]["questions"] = list(self.questions_by_kp.get(kp_id, []))
            self.kp_by_id[kp_id]["question_count"] = len(self.questions_by_kp.get(kp_id, []))
            source_context_parts: List[str] = []
            heading_parts: List[str] = []
            for source in kp.get("source_files") or []:
                for chunk in self.chunk_family_by_source.get(str(source), [])[:80]:
                    source_context_parts.append(str(chunk.get("title") or ""))
                    source_context_parts.extend(course_lines(chunk.get("text") or "", limit=2))
                    heading_parts.append(derive_section_label(chunk.get("title"), chunk.get("text")))
                    if len(source_context_parts) >= 160:
                        break
                if len(source_context_parts) >= 160:
                    break
            search_text = expand_query_aliases(
                " ".join(
                    [
                        str(kp.get("name") or ""),
                        str(kp.get("description") or ""),
                        " ".join(str(item) for item in (kp.get("keywords") or [])),
                        str(kp.get("scenario") or ""),
                        " ".join(str(item) for item in (kp.get("source_files") or [])),
                        " ".join(str(item) for item in (kp.get("support_preview") or [])),
                        " ".join(source_context_parts),
                    ]
                )
            )
            self.kp_search_texts[kp_id] = search_text
            self.kp_term_sets[kp_id] = {
                token for token in meaningful_query_tokens(search_text)
                if token not in GENERIC_COURSE_TOKENS
            }
            heading_text = expand_query_aliases(" ".join(heading_parts))
            self.kp_heading_texts[kp_id] = heading_text
            self.kp_heading_term_sets[kp_id] = {
                token for token in meaningful_query_tokens(heading_text)
                if token not in GENERIC_COURSE_TOKENS
            }

        self._llm_client, self._llm_model = _resolve_llm_client()
        self._dl_scope_cache: Dict[str, bool] = {}
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
        expanded_query = expand_query_aliases(query)
        ranked = lexical_rank(
            expanded_query,
            self.kps,
            lambda item: self.kp_search_texts.get(item["kp_id"], ""),
            top_k=max(len(self.kps), limit * 4, 10),
        )
        if not ranked:
            return []

        query_tokens = {
            token for token in meaningful_query_tokens(expanded_query)
            if token not in GENERIC_COURSE_TOKENS
        }
        query_phrase_list = query_phrases(expanded_query, max_ngram=4)
        query_anchors = query_anchor_aliases(expanded_query)
        if query_anchors:
            query_tokens = {token for token in query_tokens if token not in LOW_SIGNAL_MATCH_TOKENS}
        framework_query = bool(FRAMEWORK_QUERY_RE.search(query))
        rescored: List[Tuple[float, Dict[str, Any]]] = []
        for base_score, item in ranked:
            kp_id = item["kp_id"]
            kp_terms = self.kp_term_sets.get(kp_id, set())
            kp_text = self.kp_search_texts.get(kp_id, "")
            heading_text = self.kp_heading_texts.get(kp_id, "")
            heading_terms = self.kp_heading_term_sets.get(kp_id, set())
            score = float(base_score)
            overlap = query_tokens & kp_terms
            heading_overlap = query_tokens & heading_terms
            score += min(len(overlap) * 0.95, 6.0)
            if overlap:
                score += 0.4
            score += min(len(heading_overlap) * 2.5, 10.0)
            for phrase in query_phrase_list:
                if phrase in kp_text:
                    score += 2.0 if len(phrase.split()) >= 2 else 0.8
                if phrase in heading_text:
                    score += 6.0 if len(phrase.split()) >= 2 else 2.0
            for keyword in item.get("keywords", []):
                keyword_text = expand_query_aliases(keyword)
                if keyword_text and keyword_text in expanded_query:
                    score += 1.2
            for _anchor_key, aliases in query_anchors:
                if any(alias in heading_text for alias in aliases):
                    score += 10.0
                elif any(alias in kp_text for alias in aliases):
                    score += 4.5
            if not framework_query:
                kp_identity = " ".join(
                    [
                        clean_display_text(item.get("name") or ""),
                        clean_display_text(item.get("description") or ""),
                        " ".join(clean_display_text(source) for source in (item.get("source_files") or [])),
                    ]
                )
                if TOOL_OR_PLATFORM_TEXT_RE.search(kp_identity):
                    score -= 40.0
            rescored.append((score, item))

        rescored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "kp_id": item["kp_id"],
                "name": item["name"],
                "description": item["description"],
                "score": round(score, 4),
            }
            for score, item in rescored[:limit]
        ]

    def _looks_dl_request_by_terms(self, query: str) -> bool:
        lowered = expand_query_aliases(query)
        if not lowered:
            return False
        if any(phrase in lowered for phrase in DEEP_LEARNING_DOMAIN_PHRASES):
            return True
        if any(term.lower() in lowered for term in DEEP_LEARNING_DOMAIN_TERMS):
            return True
        if SHORT_TERM_RE.search(lowered):
            return True
        focus_tokens = meaningful_query_tokens(lowered)
        token_set = set(focus_tokens)
        domain_hits = token_set & DOMAIN_TERM_TOKENS
        strong_hits = domain_hits & DEEP_LEARNING_STRONG_SINGLE_TERM_ANCHORS
        if len(domain_hits) >= 2:
            return True
        if strong_hits and len(focus_tokens) <= 4:
            return True
        related_kps = self._topic_labels(lowered, limit=2)
        if related_kps and float(related_kps[0].get("score", 0.0) or 0.0) >= 3.2:
            return True
        return False

    def _parse_dl_scope_payload(self, content: str) -> Optional[bool]:
        candidate = clean_display_text(content).strip()
        if not candidate:
            return None
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\s*```$", "", candidate).strip()
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict) and "is_dl_related" in payload:
                return bool(payload.get("is_dl_related"))
        except Exception:
            pass
        match = re.search(r'"is_dl_related"\s*:\s*(true|false)', candidate, re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
        lowered = candidate.lower()
        if lowered in {"true", "yes", "deep learning", "dl"}:
            return True
        if lowered in {"false", "no", "not deep learning", "not dl"}:
            return False
        return None

    def _llm_dl_scope_decision(self, query: str) -> Optional[bool]:
        cleaned = clean_display_text(query)
        if not cleaned or self._llm_client is None or not self._llm_model:
            return None
        cache_key = cleaned.lower()
        if cache_key in self._dl_scope_cache:
            return self._dl_scope_cache[cache_key]

        prompt = textwrap.dedent(
            f"""
            Decide whether the student's question is about deep learning or a closely related model-training topic.

            Return JSON only:
            {{"is_dl_related": true}}
            or
            {{"is_dl_related": false}}

            Mark true for questions about:
            - neural networks, CNNs, RNNs, Transformers, attention, embeddings, normalization
            - optimization, loss functions, training stability, fine-tuning, distillation, transfer learning
            - generative models such as GAN, VAE, diffusion, DiT, U-Net
            - modern deep-learning topics such as LoRA, PEFT, prompt tuning, MoE, LLM training
            - code requests when the code is clearly about building or training a deep-learning model

            Mark false for:
            - politics, biography, weather, finance, restaurant, travel, daily life, or other unrelated chat
            - generic programming questions with no deep-learning context

            If the question could reasonably be answered in a deep-learning course discussion, prefer true.

            Student question:
            {cleaned}
            """
        ).strip()

        try:
            response = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict JSON classifier. Reply with JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=40,
            )
            decision = self._parse_dl_scope_payload(response.choices[0].message.content or "")
            if decision is not None:
                self._dl_scope_cache[cache_key] = decision
                return decision
        except Exception:
            pass
        return None

    def _looks_dl_request(self, query: str) -> bool:
        if self._looks_dl_request_by_terms(query):
            return True
        decision = self._llm_dl_scope_decision(query)
        if decision is not None:
            return decision
        return False

    def looks_in_domain(self, query: str) -> bool:
        return self._looks_dl_request(query)

    def _concept_payload(
        self,
        *,
        label: str,
        aliases: Sequence[str],
    ) -> Dict[str, Any]:
        phrase_rows = {
            clean_display_text(alias).lower()
            for alias in aliases
            if clean_display_text(alias)
        }
        token_rows = {
            token
            for token in meaningful_query_tokens(" ".join(phrase_rows))
            if token not in GENERIC_COURSE_TOKENS
        }
        return {
            "label": clean_display_text(label),
            "phrases": sorted(phrase_rows),
            "tokens": sorted(token_rows),
        }

    def _query_target_concepts(
        self,
        query: str,
        kp_context: Optional[Sequence[Dict[str, Any]]] = None,
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for key, aliases in query_anchor_aliases(query):
            label = clean_display_text(TOPIC_DISPLAY_LABELS.get(key) or key)
            if not label or label in seen:
                continue
            seen.add(label)
            rows.append(self._concept_payload(label=label, aliases=aliases))
            if len(rows) >= limit:
                return rows

        for hit in list(kp_context or [])[:6]:
            score = float(hit.get("score", 0.0) or 0.0)
            if score < 3.6:
                continue
            item = self.kp_by_id.get(str(hit.get("kp_id") or ""))
            if not item:
                continue
            label = clean_display_text(item.get("name") or "")
            if not label or label in seen:
                continue
            aliases: List[str] = [label]
            aliases.extend(clean_display_text(keyword) for keyword in (item.get("keywords") or []) if clean_display_text(keyword))
            rows.append(self._concept_payload(label=label, aliases=aliases))
            seen.add(label)
            if len(rows) >= limit:
                break
        return rows[:limit]

    def _concept_matches_chunk(self, concept: Dict[str, Any], chunk: Dict[str, Any]) -> bool:
        haystack = expand_query_aliases(
            " ".join(
                [
                    clean_display_text(chunk.get("relative_path") or ""),
                    clean_display_text(chunk.get("title") or ""),
                    clean_display_text(chunk.get("text") or ""),
                ]
            )
        )
        if not haystack:
            return False
        phrases = [
            clean_display_text(item).lower()
            for item in concept.get("phrases") or []
            if clean_display_text(item)
        ]
        if any(phrase in haystack for phrase in phrases if len(phrase) >= 3):
            return True
        chunk_terms = self.chunk_term_sets.get(str(chunk.get("chunk_id") or ""), set())
        concept_tokens = {
            clean_display_text(item).lower()
            for item in concept.get("tokens") or []
            if clean_display_text(item)
        }
        if not concept_tokens:
            return False
        overlap = len(chunk_terms & concept_tokens)
        if len(concept_tokens) == 1:
            return overlap >= 1
        return overlap >= min(2, len(concept_tokens))

    def _course_coverage_level(
        self,
        query: str,
        chunk_hits: Sequence[Dict[str, Any]],
        *,
        target_concepts: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> str:
        concepts = list(target_concepts or self._query_target_concepts(query, limit=4))
        if concepts:
            covered_labels = []
            for concept in concepts[:4]:
                if any(self._concept_matches_chunk(concept, item) for item in chunk_hits[:4]):
                    covered_labels.append(clean_display_text(concept.get("label") or ""))
            if len(concepts) >= 2:
                if len(covered_labels) >= min(2, len(concepts)):
                    return "direct"
                if covered_labels:
                    return "related"
                return "none"
            if covered_labels:
                return "direct"

        query_tokens = {
            token for token in meaningful_query_tokens(query)
            if token not in GENERIC_COURSE_TOKENS
        }
        query_phrase_list = query_phrases(query, max_ngram=4)
        related = False
        for chunk in list(chunk_hits or [])[:4]:
            chunk_id = str(chunk.get("chunk_id") or "")
            chunk_terms = self.chunk_term_sets.get(chunk_id, set())
            chunk_text = self.chunk_search_texts.get(chunk_id, "")
            overlap = len(query_tokens & chunk_terms)
            phrase_hit = any(phrase in chunk_text for phrase in query_phrase_list)
            if phrase_hit and overlap >= 1:
                return "direct"
            if overlap >= 2:
                return "direct"
            if phrase_hit or overlap >= 1:
                related = True
        return "related" if related else "none"

    def _resolve_follow_up_query(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
        session_memory: Optional[Dict[str, Any]] = None,
    ) -> str:
        cleaned = clean_display_text(query)
        if not cleaned:
            return ""
        if not looks_like_follow_up(cleaned):
            return cleaned
        previous_topic = clean_display_text((session_memory or {}).get("active_topic") or "")
        session_summary = clean_display_text((session_memory or {}).get("session_summary") or "")
        anchor_query = ""
        for turn in reversed(list(history or [])):
            content = clean_display_text(turn.get("content") or "")
            if turn.get("role") == "user" and content:
                anchor_query = content
                break

        supplements: List[str] = []
        lowered = cleaned.lower()
        if previous_topic and previous_topic.lower() not in lowered:
            supplements.append(f"主题：{previous_topic}")
        if anchor_query and anchor_query != cleaned and anchor_query.lower() not in lowered:
            supplements.append(f"上一问：{anchor_query}")
        if not supplements and session_summary:
            supplements.append(f"会话上下文：{session_summary[:80]}")
        if not supplements:
            return cleaned
        return clean_display_text(f"{cleaned}；{'；'.join(supplements)}")

    def resolve_query_context(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
        session_memory: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._resolve_follow_up_query(query, history, session_memory=session_memory)

    def query_uses_context(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
        session_memory: Optional[Dict[str, Any]] = None,
    ) -> bool:
        cleaned = clean_display_text(query)
        if not cleaned:
            return False
        return self.resolve_query_context(cleaned, history, session_memory=session_memory) != cleaned

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

        expanded_variants = [expand_query_aliases(item) for item in query_variants if expand_query_aliases(item)]
        topic_query = " ".join(expanded_variants or query_variants)
        related_kps = self._topic_labels(topic_query, limit=4)
        target_concepts = self._query_target_concepts(topic_query, related_kps, limit=4)
        query_tokens = {
            token for token in meaningful_query_tokens(topic_query)
            if token not in GENERIC_COURSE_TOKENS
        }
        query_phrase_list = query_phrases(topic_query, max_ngram=4)
        query_anchors = query_anchor_aliases(topic_query)
        if query_anchors:
            query_tokens = {token for token in query_tokens if token not in LOW_SIGNAL_MATCH_TOKENS}
        framework_query = bool(FRAMEWORK_QUERY_RE.search(topic_query))
        reference_query = bool(REFERENCE_STYLE_QUERY_RE.search(topic_query))

        for query_index, current_query in enumerate(query_variants, start=1):
            lexical_query = expand_query_aliases(current_query) or current_query
            lexical_hits = lexical_rank(
                lexical_query,
                self.chunks,
                lambda item: self.chunk_search_texts.get(str(item.get("chunk_id") or ""), ""),
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

        for kp_rank, hit in enumerate(related_kps, start=1):
            kp = self.kp_by_id.get(hit["kp_id"])
            if not kp:
                continue
            kp_boost = max(0.0, float(hit.get("score") or 0.0))
            boost_weight = 1.5 if kp_rank == 1 else 1.0
            for chunk_id in kp.get("support_chunk_ids", []) or []:
                if not chunk_id:
                    continue
                combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + kp_boost * boost_weight
            for source in kp.get("source_files", []) or []:
                for chunk in self.chunk_family_by_source.get(str(source), [])[:80]:
                    chunk_id = str(chunk.get("chunk_id") or "")
                    if not chunk_id:
                        continue
                    combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + kp_boost * 0.08 * boost_weight

        ranked_chunks = []
        for chunk_id, score in combined_scores.items():
            chunk = self.chunks_by_id.get(chunk_id)
            if not chunk:
                continue
            chunk_text = self.chunk_search_texts.get(chunk_id, "")
            chunk_terms = self.chunk_term_sets.get(chunk_id, set())
            section_text = expand_query_aliases(derive_section_label(chunk.get("title"), chunk.get("text")))
            section_terms = {
                token for token in meaningful_query_tokens(section_text)
                if token not in GENERIC_COURSE_TOKENS
            }
            overlap = query_tokens & chunk_terms
            section_overlap = query_tokens & section_terms
            score += min(len(overlap) * 0.55, 4.0)
            score += min(len(section_overlap) * 2.8, 10.0)
            section_phrase_hit = False
            for phrase in query_phrase_list:
                if phrase in chunk_text:
                    score += 1.2 if len(phrase.split()) >= 2 else 0.5
                if phrase in section_text:
                    section_phrase_hit = True
                    score += 6.0 if len(phrase.split()) >= 2 else 1.8
            section_anchor_hit = False
            chunk_anchor_hit = False
            for _anchor_key, aliases in query_anchors:
                if any(alias in section_text for alias in aliases):
                    section_anchor_hit = True
                    score += 12.0
                elif any(alias in chunk_text for alias in aliases):
                    chunk_anchor_hit = True
                    score += 4.5
            if not framework_query and TOOL_OR_PLATFORM_TEXT_RE.search(str(chunk.get("relative_path") or "")):
                score -= 14.0
            if reference_query and not section_overlap and not section_phrase_hit and not section_anchor_hit:
                score -= 3.0
            if query_anchors and not section_anchor_hit and not chunk_anchor_hit and reference_query:
                score -= 3.5
            score += chunk_quality_adjustment(chunk)
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

        coverage_level = self._course_coverage_level(
            topic_query,
            hits,
            target_concepts=target_concepts,
        )
        return {
            "resolved_query": resolved_query,
            "query_variants": query_variants,
            "hits": hits,
            "citations": citations,
            "related_kps": related_kps[:3],
            "target_concepts": target_concepts,
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
        try:
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
        except Exception:
            return None

    def call_text_llm(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.3, max_tokens: int = 900) -> str:
        if self._llm_client is None or not self._llm_model:
            return ""
        try:
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
        except Exception:
            return ""

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
