from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from deep_learning_portal.chat_code_templates import build_code_template
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
    expand_query_aliases,
    looks_like_follow_up,
    parse_jsonish_text,
    tokenize,
)


OUT_OF_SCOPE_GUIDANCE = [
    "这个问题和当前深度学习课程无关，所以我先不直接回答。你可以改问神经网络、CNN、Transformer、扩散模型、训练优化或课程讲义相关内容。",
    "这不属于当前课程的问答范围，我先不展开。若你愿意，可以继续问模型结构、训练机制、代码实现或课程复习相关问题。",
    "这个问题超出了这门深度学习课程的问答边界，我先不直接作答。你可以切回课程主题，我会继续帮你分析。",
]
TERM_HINTS = {
    "dit": "写法请使用 DiT（Diffusion Transformer）。它是扩散模型里的 Transformer 骨干，常见于图像生成等生成建模任务，不是 Dense layer，也不要把它回答成通用 NLP Transformer、语言模型或泛指序列建模模型。",
    "vit": "写法请使用 ViT（Vision Transformer），主要用于视觉表征与图像分类等视觉任务。",
    "vae": "VAE 指 Variational Autoencoder，属于概率生成模型。",
    "mae": "MAE 指 Masked Autoencoder，常用于自监督表征学习。",
    "clip": "CLIP 是图文对齐的多模态模型，不是单纯的分类头。",
    "unet": "U-Net 常用于分割与扩散模型中的去噪骨干。",
    "u-net": "U-Net 常用于分割与扩散模型中的去噪骨干。",
    "batchnorm": "BatchNorm 是基于 mini-batch 统计量的归一化方法，重点是 batch 统计是否稳定，而不是“因为需要固定长度输入”。",
    "layernorm": "LayerNorm 是对单个样本的特征维度做归一化，不依赖 mini-batch 统计量，在 Transformer 中更常见。",
    "padding": "padding 是在卷积前对边界做填充，主要影响输出尺寸和边缘位置的特征保留。",
    "lora": "LoRA 指 Low-Rank Adaptation，是参数高效微调方法。只回答深度学习里的这个含义，不要讨论同名缩写在其他领域或其他模型里的意思。",
    "peft": "PEFT 指 Parameter-Efficient Fine-Tuning，重点是只训练少量附加参数来降低微调成本。不要把它扩写成无关概念。",
    "moe": "MoE 指 Mixture of Experts，核心是多个 expert 子网络加上 router 或 gating 机制的稀疏激活架构。不要把它写成泛泛而谈的其他缩写含义。",
}
TERM_AMBIGUITY_GUARDS = {
    "lora": (
        re.compile(r"无线|通信|radio|long[- ]range", re.IGNORECASE),
        re.compile(r"龙猫"),
    ),
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
NON_COURSE_SIGNAL_PATTERNS = (
    re.compile(r"\b(weather|restaurant|restaurants|bitcoin|crypto|stock|stocks|finance|financial)\b|天气|餐厅|饭店|比特币|加密货币|股票|股市|金融", re.IGNORECASE),
    re.compile(r"\b(president|prime minister|election|government|politics|political)\b|总统|首相|选举|政府|政治", re.IGNORECASE),
    re.compile(r"\b(joke|funny|married|wife|husband|boyfriend|girlfriend|dating)\b|笑话|结婚|老婆|老公|男朋友|女朋友|恋爱", re.IGNORECASE),
    re.compile(r"\b(travel|flight|hotel|visa|restaurant|driving test|driver'?s test)\b|旅行|航班|酒店|签证|驾照考试|驾驶证考试|考驾照|驾校", re.IGNORECASE),
    re.compile(r"\b(job application|resume|curriculum vitae|cover letter|linkedin)\b|简历|求职|求职信|领英", re.IGNORECASE),
)
DOMAIN_HINT_TERMS = {
    "深度学习", "机器学习", "神经网络", "反向传播", "前向传播", "卷积", "池化", "梯度",
    "优化器", "损失函数", "交叉熵", "归一化", "批归一化", "残差", "注意力", "自注意力",
    "位置编码", "预训练", "微调", "蒸馏", "迁移学习", "自监督", "生成模型", "扩散模型",
    "cnn", "rnn", "gru", "lstm", "transformer", "attention", "gpt", "bert", "vit",
    "dit", "vae", "gan", "unet", "pytorch", "tensorflow", "keras", "dropout", "adam",
    "adamw", "softmax", "relu", "gelu", "batchnorm", "layernorm", "embedding", "padding",
    "lora", "low-rank adaptation", "peft", "parameter-efficient fine-tuning", "prompt tuning",
    "moe", "mixture of experts", "全连接", "全连接层", "全连接网络", "mlp", "linear layer",
}
COMPARISON_RE = re.compile(r"(区别|比较|对比|相比|比起|relation|relationship|difference|compare|vs\b|versus)", re.IGNORECASE)
REFERENCE_RE = re.compile(
    r"(哪一页|哪份讲义|哪一讲|哪里讲了|在哪里讲了|哪部分讲了|"
    r"去哪看|复习哪部分|先看哪部分|先从哪部分|应该先看哪部分|"
    r"最适合复习|复习入口|哪里复习|哪部分最适合复习|"
    r"which lecture|which page|which section|what should i review|where in the course|where should i review|where is .* covered)",
    re.IGNORECASE,
)
CODE_ONLY_RE = re.compile(r"(只给代码|only code|just code|仅代码|不要解释|无需解释)", re.IGNORECASE)
SOURCE_TOKEN_RE = re.compile(r"@@COURSE_SOURCE_\d+@@")
SOURCE_LINE_RE = re.compile(r"(?im)^\s*(course source|source|used_sources)\s*:.*$")
JSON_LIKE_RE = re.compile(r"^\s*[\{\[]")
SHORT_FOLLOW_UP_RE = re.compile(r"^(为什么|怎么做|怎么理解|那呢|然后呢|继续|展开讲讲|举个例子|代码呢|再说一下)[？?！!。.]?$")
SHORT_CONTEXT_REQUEST_RE = re.compile(
    r"^(?:那|再|现在|就|先)?(?:只给代码|只要代码|给我(?:一个|一段)?代码(?:例子|示例)?|给个(?:代码)?例子|写(?:一段|一个|个)?代码|代码(?:呢|怎么写)|再解释一下|解释一下(?:上面|这段|这个)?(?:代码)?(?:里|里的)?|现在解释一下(?:上面|这段|这个)?(?:代码)?(?:里|里的)?|总结一下|再总结一下|再用一句话总结一下|一句话总结一下|三句话总结一下|继续|展开讲讲|再说一下)$",
    re.IGNORECASE,
)
CODE_REQUEST_TRIGGER_RE = re.compile(
    r"(给我|给个|写(?:一段|一个|个)?|实现|怎么写|代码呢|示例|demo|block|训练循环|sample code|code example)",
    re.IGNORECASE,
)
EXISTING_CODE_REFERENCE_RE = re.compile(
    r"(上面(?:的)?代码|上面的例子|这段代码|代码里|代码中的|这份代码|上一段代码|刚才那段代码|上面代码里|上面代码中的)",
    re.IGNORECASE,
)
SUMMARY_REQUEST_RE = re.compile(r"(总结|概括|一句话|三句话|简要|briefly|summari[sz]e|summary)", re.IGNORECASE)
NON_TOPIC_FOCUS_TERMS = {"这", "那个", "这个", "它", "它们", "前者", "后者", "那", "其"}
INLINE_SOURCE_DETAIL_RE = re.compile(r"(?i)\b[\w./-]+\.(pdf|ppt|pptx)\b|第\s*\d+\s*页|\bpage\s*\d+\b")
INLINE_SOURCE_ID_RE = re.compile(
    r"(?:(?:在|如|见|参考|例如)\s*)?(?:\(?\s*S\d+(?:\s*[、,，和及]\s*S\d+)*\s*\)?)(?:\s*(?:中|里|提到|显示|给出|所示))?",
    re.IGNORECASE,
)
NON_SPECIFIC_TOPIC_TERMS = {"深度学习", "机器学习", "pytorch", "tensorflow", "keras", "optimizer", "loss"}
SUPPORTING_FOCUS_LABELS = {"PyTorch", "TensorFlow", "Keras"}
SUPPRESSED_FOCUS_LABELS = {
    "归一化": {"BatchNorm", "LayerNorm"},
    "注意力机制": {"自注意力"},
    "神经网络": {"卷积神经网络"},
}
FOCUS_LABEL_ALIASES = {
    "BatchNorm": ("batchnorm", "batch normalization", "批归一化"),
    "LayerNorm": ("layernorm", "layer normalization", "层归一化"),
    "全连接层": ("fully connected", "full connected", "linear layer", "全连接", "全连接层", "全连接网络", "fc"),
    "多层感知机": ("mlp", "multi-layer perceptron", "多层感知机"),
    "卷积": ("convolution", "conv", "卷积"),
    "卷积神经网络": ("cnn", "卷积神经网络"),
    "反向传播": ("backpropagation", "backprop", "反向传播"),
    "梯度下降": ("gradient descent", "梯度下降"),
    "优化器": ("optimizer", "优化器"),
    "损失函数": ("loss function", "loss", "损失函数"),
    "归一化": ("normalization", "归一化"),
    "池化": ("pooling", "池化"),
    "感知器": ("perceptron", "感知器"),
    "神经网络": ("neural network", "神经网络"),
    "RNN": ("rnn", "recurrent neural network"),
    "LSTM": ("lstm",),
    "GRU": ("gru",),
    "Transformer": ("transformer",),
    "自注意力": ("self-attention", "自注意力"),
    "注意力机制": ("attention", "注意力"),
    "位置编码": ("positional encoding", "位置编码"),
    "嵌入": ("embedding", "嵌入"),
    "预训练": ("pre-training", "pretraining", "预训练"),
    "微调": ("fine-tuning", "finetuning", "微调"),
    "参数高效微调": ("peft", "parameter-efficient fine-tuning", "参数高效微调"),
    "Prompt Tuning": ("prompt tuning", "提示微调"),
    "迁移学习": ("transfer learning", "迁移学习"),
    "蒸馏": ("distillation", "蒸馏"),
    "对比学习": ("contrastive learning", "对比学习"),
    "自监督学习": ("self-supervised", "自监督"),
    "GAN": ("gan", "生成对抗网络"),
    "扩散模型": ("diffusion", "diffusion model", "扩散模型"),
    "VAE": ("vae", "variational autoencoder", "变分自编码器"),
    "U-Net": ("u-net", "unet"),
    "DiT": ("dit", "diffusion transformer"),
    "ViT": ("vit", "vision transformer"),
    "CLIP": ("clip",),
    "LoRA": ("lora", "low-rank adaptation"),
    "MoE": ("moe", "mixture of experts", "混合专家"),
    "PyTorch": ("pytorch",),
    "TensorFlow": ("tensorflow",),
}
QUERY_NOISE_TOKENS = {
    "what", "is", "the", "a", "an", "of", "to", "for", "in", "on", "and", "or", "with",
    "please", "show", "give", "tell", "me", "about", "how", "why", "explain", "example",
    "examples", "code", "just", "only", "need", "want", "具体", "一下", "一个", "一段", "代码",
    "例子", "展示", "解释", "说明", "比较", "区别", "关系", "请问", "请", "给我", "给个", "一下子",
}
QUERY_INTENT_TOKENS = {
    "show", "give", "tell", "explain", "compare", "difference", "relation", "relationship",
    "how", "why", "what", "just", "only", "代码", "例子", "解释", "比较", "区别", "关系", "说明",
}
FOLLOWUP_NON_TOPIC_TOKENS = {
    "then", "also", "so", "continue", "regarding", "about", "that", "this", "those", "these",
    "why", "how", "what", "again", "继续", "然后", "再", "那", "这个", "那个", "这样", "例子", "代码",
}
FOCUS_LABEL_PATTERNS = (
    ("BatchNorm", re.compile(r"\b(batchnorm|batch normalization)\b|BatchNorm|批归一化", re.IGNORECASE)),
    ("LayerNorm", re.compile(r"\b(layernorm|layer normalization)\b|LayerNorm|层归一化", re.IGNORECASE)),
    ("全连接层", re.compile(r"\b(fully connected|full connected|linear layer|fc layer)\b|全连接层|全连接网络|全连接", re.IGNORECASE)),
    ("多层感知机", re.compile(r"\bmlp\b|multi-layer perceptron|多层感知机", re.IGNORECASE)),
    ("卷积", re.compile(r"\b(convolution|conv|convolutional)\b|卷积", re.IGNORECASE)),
    ("卷积神经网络", re.compile(r"\bcnn\b|卷积神经网络", re.IGNORECASE)),
    ("反向传播", re.compile(r"\bbackprop(?:agation)?\b|反向传播", re.IGNORECASE)),
    ("梯度下降", re.compile(r"\bgradient descent\b|梯度下降", re.IGNORECASE)),
    ("优化器", re.compile(r"\boptimizer\b|优化器", re.IGNORECASE)),
    ("损失函数", re.compile(r"\b(loss function|loss)\b|损失函数", re.IGNORECASE)),
    ("归一化", re.compile(r"\b(normalization|batchnorm|batch normalization|layernorm|layer normalization)\b|归一化|批归一化|层归一化", re.IGNORECASE)),
    ("池化", re.compile(r"\b(pooling|max pooling|average pooling)\b|池化", re.IGNORECASE)),
    ("感知器", re.compile(r"\bperceptron\b|感知器", re.IGNORECASE)),
    ("神经网络", re.compile(r"\b(neural network|mlp)\b|神经网络|多层感知机", re.IGNORECASE)),
    ("RNN", re.compile(r"\b(rnn|recurrent neural network)\b|循环神经网络", re.IGNORECASE)),
    ("LSTM", re.compile(r"\blstm\b", re.IGNORECASE)),
    ("GRU", re.compile(r"\bgru\b", re.IGNORECASE)),
    ("Transformer", re.compile(r"\btransformer\b|Transformer", re.IGNORECASE)),
    ("自注意力", re.compile(r"\bself-attention\b|自注意力", re.IGNORECASE)),
    ("注意力机制", re.compile(r"\battention\b|注意力", re.IGNORECASE)),
    ("位置编码", re.compile(r"\bpositional encoding\b|位置编码", re.IGNORECASE)),
    ("嵌入", re.compile(r"\bembedding\b|嵌入", re.IGNORECASE)),
    ("预训练", re.compile(r"\bpre-?training\b|预训练", re.IGNORECASE)),
    ("微调", re.compile(r"\bfine-?tuning\b|微调", re.IGNORECASE)),
    ("迁移学习", re.compile(r"\btransfer learning\b|迁移学习", re.IGNORECASE)),
    ("蒸馏", re.compile(r"\bdistillation\b|蒸馏", re.IGNORECASE)),
    ("对比学习", re.compile(r"\bcontrastive learning\b|对比学习", re.IGNORECASE)),
    ("自监督学习", re.compile(r"\bself-supervised\b|自监督", re.IGNORECASE)),
    ("GAN", re.compile(r"\bgan\b|生成对抗网络", re.IGNORECASE)),
    ("扩散模型", re.compile(r"\b(diffusion|diffusion model)\b|扩散模型", re.IGNORECASE)),
    ("VAE", re.compile(r"\bvae\b|变分自编码器", re.IGNORECASE)),
    ("U-Net", re.compile(r"\bu-?net\b|U-Net", re.IGNORECASE)),
    ("DiT", re.compile(r"\bdit\b|Diffusion Transformer", re.IGNORECASE)),
    ("ViT", re.compile(r"\bvit\b|Vision Transformer", re.IGNORECASE)),
    ("CLIP", re.compile(r"\bclip\b", re.IGNORECASE)),
    ("LoRA", re.compile(r"\blora\b|Low-Rank Adaptation", re.IGNORECASE)),
    ("参数高效微调", re.compile(r"\b(peft|parameter-efficient fine-tuning)\b|参数高效微调", re.IGNORECASE)),
    ("Prompt Tuning", re.compile(r"\bprompt tuning\b|提示微调", re.IGNORECASE)),
    ("MoE", re.compile(r"\b(moe|mixture of experts)\b|混合专家", re.IGNORECASE)),
    ("PyTorch", re.compile(r"\bpytorch\b", re.IGNORECASE)),
    ("TensorFlow", re.compile(r"\btensorflow\b", re.IGNORECASE)),
)


def _history_text(history: Sequence[Dict[str, str]], limit: int = 6) -> str:
    rows: List[str] = []
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


def _clean_focus_label(text: str) -> str:
    value = clean_display_text(text)
    value = re.sub(r"^(什么是|请解释|解释一下|介绍一下|比较|对比|说明一下|给我一个|给我一段)", "", value)
    value = value.split("，", 1)[0].split(",", 1)[0]
    value = re.sub(r"(是什么|有什么区别|有什么关系|有哪些区别)$", "", value)
    return clean_display_text(value.strip("：:；;。.!?？"))


@dataclass(frozen=True)
class ChatTaskPlan:
    intent: str = "other"
    paired_focus: Sequence[str] = ()
    focus_labels: Sequence[str] = ()
    is_comparison: bool = False
    is_code_request: bool = False
    is_explain_request: bool = False
    is_code_only_request: bool = False
    is_reference_request: bool = False
    is_summary_request: bool = False
    references_existing_code: bool = False
    needs_multi_concept_coverage: bool = False

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def prompt_flags(self, *, include_existing_code: bool = False) -> str:
        rows = [
            ("任务类型", self.intent or "other"),
            ("是否比较题", str(bool(self.is_comparison)).lower()),
            ("是否代码请求", str(bool(self.is_code_request)).lower()),
            ("是否只要代码", str(bool(self.is_code_only_request)).lower()),
            ("是否讲义定位请求", str(bool(self.is_reference_request)).lower()),
            ("需要覆盖多个概念", str(bool(self.needs_multi_concept_coverage)).lower()),
        ]
        if include_existing_code:
            rows.insert(4, ("是否在解释上文代码", str(bool(self.references_existing_code)).lower()))
        if self.paired_focus:
            rows.append(("对比焦点", "；".join(self.paired_focus)))
        if self.focus_labels:
            rows.append(("重点概念", "；".join(self.focus_labels)))
        return "\n".join(f"- {label}: {value}" for label, value in rows)


class DeepLearningChatPipeline:
    """Unified chat pipeline for the deep learning course.

    The goal is to keep one stable answer chain:
    1. resolve the user query with session context
    2. decide scope once
    3. retrieve once with preserved context
    4. generate one student-facing answer
    5. finalize citations and session memory consistently
    """

    def __init__(self, kb: DeepLearningKnowledgeBase) -> None:
        self.kb = kb

    def _repeat_count(self, query: str, history: Sequence[Dict[str, str]]) -> int:
        cleaned = clean_display_text(query).lower()
        if not cleaned:
            return 0
        count = 0
        for turn in history:
            if str(turn.get("role") or "").lower() != "user":
                continue
            if clean_display_text(turn.get("content") or "").lower() == cleaned:
                count += 1
        return count

    def _assistant_intent(self, query: str) -> Optional[str]:
        cleaned = clean_display_text(query)
        lowered = cleaned.lower()
        if not cleaned:
            return "empty"
        if GREETING_RE.match(cleaned) or cleaned.startswith(("你好", "您好")):
            return "greeting"
        if IDENTITY_RE.search(lowered):
            return "identity"
        if HELP_RE.search(lowered):
            return "help"
        if THANKS_RE.search(lowered):
            return "thanks"
        return None

    def _response_kind_for_intent(self, intent: Optional[str]) -> str:
        mapping = {
            "empty": "empty_input",
            "greeting": "greeting",
            "identity": "identity",
            "help": "meta_help",
            "thanks": "thanks",
        }
        return mapping.get(str(intent or "").strip(), "assistant_meta")

    def _assistant_response(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
        *,
        intent: Optional[str] = None,
    ) -> Optional[str]:
        intent = intent or self._assistant_intent(query)
        if not intent:
            return None
        seed_text = f"{clean_display_text(query)}::{self._repeat_count(query, history)}::{len(history)}"

        if intent == "empty":
            return "请输入你想咨询的课程问题。"
        if intent == "greeting":
            choices = [
                "你好，这里是深度学习课程学习平台。你可以直接问我概念、模型结构、训练方法、代码实现思路，或者继续做题复习。",
                "你好，欢迎进入深度学习课程学习平台。你可以问我课程讲义内容、模型原理、训练机制，也可以配合 Quiz 和 Learning Report 一起复习。",
                "你好，我可以帮你理解这门深度学习课程中的知识点，也可以结合讲义来源、练习和学习报告来辅助复习。",
            ]
            return _pick_variety(
                f"{seed_text}::{self._repeat_count(query, history)}",
                choices,
            )
        if intent == "identity":
            choices = [
                "我是这门深度学习课程的学习助教，负责根据课程讲义和深度学习领域知识回答问题、组织练习，并辅助你查看学习报告。",
                "我是面向这门深度学习课程的中文学习助教，可以帮助你理解概念、比较模型、查看讲义来源，并配合练习与学习报告使用。",
                "我是这套深度学习课程平台里的课程助教，主要帮助学生做问答、练习和复习分析。",
            ]
            return choices[self._repeat_count(query, history) % len(choices)]
        if intent == "help":
            choices = [
                "你可以直接问课程中的概念、模型结构、训练方法、代码实现思路，也可以进入 Quiz 做题，或到 Learning Report 查看当前强弱项分析。",
                "你可以在这里继续追问讲义内容、模型差异、训练细节和代码示例；如果想系统复习，也可以切换到 Quiz 和 Learning Report。",
                "你可以把这里当作课程问答入口来用：概念解释、模型比较、训练机制、代码示例、讲义定位和复习建议都可以继续问。",
            ]
            return choices[self._repeat_count(query, history) % len(choices)]
        if intent == "thanks":
            choices = [
                "不客气。你可以继续追问当前主题，也可以切换到 Quiz 或 Learning Report。",
                "没问题。如果你愿意，我可以继续帮你梳理某个模型、某个训练技巧，或者给你一个代码示例。",
                "不用客气。接下来你可以继续问讲义里的知识点，也可以直接去做题复习。",
            ]
            return choices[self._repeat_count(query, history) % len(choices)]
        return None

    def _content_tokens(self, query: str) -> List[str]:
        return [
            token
            for token in tokenize(clean_display_text(query).lower())
            if len(token) > 1 and token not in QUERY_NOISE_TOKENS
        ]

    def _query_focus_tokens(self, query: str) -> List[str]:
        return [
            token
            for token in self._content_tokens(query)
            if token not in QUERY_INTENT_TOKENS
        ]

    def _query_has_strong_focus(self, query: str) -> bool:
        cleaned = clean_display_text(query)
        if not cleaned:
            return False
        if self._query_is_context_dependent(cleaned):
            return False
        if self._paired_focus_from_query(cleaned):
            return True
        if self._query_focus_labels(cleaned, limit=2):
            return True
        if self._raw_dl_signal_score(cleaned) >= 3:
            return True
        focus_tokens = self._query_focus_tokens(cleaned)
        return len(focus_tokens) >= 2 and self.kb.looks_in_domain(cleaned)

    def _query_language_hint(self, text: str) -> str:
        lowered = clean_display_text(text).lower()
        if "pytorch" in lowered:
            return "PyTorch"
        if "tensorflow" in lowered:
            return "TensorFlow"
        if "python" in lowered:
            return "Python"
        if "keras" in lowered:
            return "Keras"
        if "c++" in lowered or re.search(r"\bcpp\b", lowered):
            return "C++"
        return ""

    def _topic_components(self, text: str, limit: int = 3) -> List[str]:
        rows: List[str] = []
        for item in re.split(r"\s*/\s*", clean_display_text(text)):
            cleaned = clean_display_text(item)
            if cleaned and cleaned not in rows:
                rows.append(cleaned)
            if len(rows) >= limit:
                break
        return rows

    def _merge_focus_concepts(
        self,
        primary: Sequence[str],
        secondary: Sequence[str],
        *,
        limit: int = 3,
    ) -> List[str]:
        rows: List[str] = []
        for source in [primary or [], secondary or []]:
            for item in source:
                cleaned = clean_display_text(item)
                if not cleaned or cleaned.lower() in NON_TOPIC_FOCUS_TERMS:
                    continue
                if cleaned not in rows:
                    rows.append(cleaned)
                if len(rows) >= limit:
                    return rows
        return rows

    def _recent_history_focus_concepts(
        self,
        history: Sequence[Dict[str, str]],
        limit: int = 3,
    ) -> List[str]:
        rows: List[str] = []
        for turn in reversed(list(history or [])):
            if str(turn.get("role") or "").lower() != "user":
                continue
            content = clean_display_text(turn.get("content") or "")
            if not content or self._assistant_intent(content):
                continue
            labels = self._paired_focus_from_query(content) or self._query_focus_labels(content, limit=3)
            for label in labels:
                cleaned = clean_display_text(label)
                if not cleaned or cleaned.lower() in NON_TOPIC_FOCUS_TERMS:
                    continue
                if cleaned not in rows:
                    rows.append(cleaned)
                if len(rows) >= limit:
                    return rows
        return rows

    def _query_is_context_dependent(self, query: str) -> bool:
        cleaned = clean_display_text(query)
        if not cleaned:
            return False
        if self._looks_like_contextual_request(cleaned) or looks_like_follow_up(cleaned):
            return True
        if self._paired_focus_from_query(cleaned):
            return False
        focus_labels = self._query_focus_labels(cleaned, limit=3)
        semantic_focus_tokens = [
            token
            for token in self._query_focus_tokens(cleaned)
            if token not in FOLLOWUP_NON_TOPIC_TOKENS
        ]
        lowered = cleaned.lower()
        has_pronoun = bool(re.search(r"\b(it|its|them|this|that|these|those|former|latter)\b", lowered)) or any(
            marker in cleaned for marker in ("它", "它们", "这个", "那个", "前者", "后者")
        )
        if has_pronoun and len(focus_labels) <= 1 and len(semantic_focus_tokens) <= 8:
            return True
        return False

    def _conversation_focus(
        self,
        history: Sequence[Dict[str, str]],
        session_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        turns = list(history or [])
        previous_memory = dict(session_memory or {})
        active_topic = clean_display_text(previous_memory.get("active_topic") or "")
        session_summary = clean_display_text(previous_memory.get("session_summary") or "")
        recent_focus_concepts = self._recent_history_focus_concepts(turns, limit=3)

        if not turns:
            active_components = self._topic_components(active_topic, limit=3)
            return {
                "active_topic": active_topic,
                "session_summary": session_summary,
                "anchor_query": "",
                "anchor_answer": "",
                "focus_concept": active_topic,
                "focus_concepts": active_components or ([active_topic] if active_topic else []),
                "language_hint": "",
            }

        for index in range(len(turns) - 1, -1, -1):
            turn = turns[index]
            if str(turn.get("role") or "").lower() != "user":
                continue
            anchor_query = clean_display_text(turn.get("content") or "")
            if not anchor_query or self._assistant_intent(anchor_query):
                continue
            if not self._query_has_strong_focus(anchor_query):
                continue

            anchor_answer = ""
            for follow_turn in turns[index + 1 :]:
                follow_role = str(follow_turn.get("role") or "").lower()
                if follow_role == "assistant":
                    anchor_answer = clean_multiline_text(str(follow_turn.get("content") or ""))
                    break
                if follow_role == "user":
                    break

            focus_concepts = self._query_focus_labels(anchor_query, limit=2)
            if not focus_concepts:
                related = self.kb._topic_labels(anchor_query, limit=2)
                focus_concepts = [
                    clean_display_text((item or {}).get("name") or "")
                    for item in related
                    if clean_display_text((item or {}).get("name") or "")
                ]
            focus_concepts = self._merge_focus_concepts(focus_concepts, recent_focus_concepts, limit=3)
            focus_concept = ""
            if len(focus_concepts) >= 2:
                focus_concept = " / ".join(focus_concepts[:2])
            elif focus_concepts:
                focus_concept = focus_concepts[0]
            elif active_topic:
                focus_concept = active_topic

            return {
                "active_topic": active_topic,
                "session_summary": session_summary,
                "anchor_query": anchor_query,
                "anchor_answer": anchor_answer[:260],
                "focus_concept": focus_concept,
                "focus_concepts": focus_concepts[:2],
                "language_hint": self._query_language_hint(anchor_query) or self._query_language_hint(anchor_answer),
            }

        return {
            "active_topic": active_topic,
            "session_summary": session_summary,
            "anchor_query": "",
            "anchor_answer": "",
            "focus_concept": active_topic,
            "focus_concepts": self._topic_components(active_topic, limit=3) or ([active_topic] if active_topic else []),
            "language_hint": "",
        }

    def _session_context_focus(
        self,
        history: Sequence[Dict[str, str]],
        session_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        focus = self._conversation_focus(history, session_memory=session_memory)
        focus_concept = clean_display_text(focus.get("focus_concept") or "") or clean_display_text(focus.get("active_topic") or "")
        focus_concepts = [
            clean_display_text(item)
            for item in (focus.get("focus_concepts") or [])
            if clean_display_text(item)
        ]
        if not focus_concepts and focus_concept:
            focus_concepts = [focus_concept]
        merged = dict(focus)
        merged["focus_concept"] = focus_concept
        merged["focus_concepts"] = focus_concepts[:2]
        merged["active_topic"] = clean_display_text(focus.get("active_topic") or "")
        merged["session_summary"] = clean_display_text(focus.get("session_summary") or "")
        return merged

    def _is_existing_code_reference(self, query: str) -> bool:
        return bool(EXISTING_CODE_REFERENCE_RE.search(clean_display_text(query)))

    def _looks_like_contextual_request(self, query: str) -> bool:
        cleaned = clean_display_text(query)
        if not cleaned:
            return False
        if SHORT_CONTEXT_REQUEST_RE.search(cleaned):
            return True
        if len(cleaned) <= 24 and CODE_ONLY_RE.search(cleaned):
            return True
        if len(cleaned) <= 24 and SUMMARY_REQUEST_RE.search(cleaned):
            return True
        return len(cleaned) <= 24 and self._is_existing_code_reference(cleaned)

    def _detect_followup_query(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
    ) -> bool:
        current = clean_display_text(query)
        if not current or not history:
            return False
        if self._assistant_intent(current):
            return False
        if self._query_is_context_dependent(current):
            return True
        if self._query_has_strong_focus(current):
            return False

        lowered = current.lower()
        semantic_focus_tokens = [
            token
            for token in self._query_focus_tokens(current)
            if token not in FOLLOWUP_NON_TOPIC_TOKENS
        ]
        content_tokens = self._content_tokens(current)
        starts_like_followup = bool(looks_like_follow_up(current))
        has_contextual_shape = self._looks_like_contextual_request(current)
        has_code_or_explain_shape = bool(
            self._is_code_generation_request(current)
            or self._is_explanation_request(current)
            or COMPARISON_RE.search(current)
        )
        short_query = len(content_tokens) <= 8
        if has_contextual_shape and short_query:
            return True
        if starts_like_followup and short_query:
            return True
        if has_code_or_explain_shape and len(semantic_focus_tokens) == 0:
            return True
        if lowered.startswith(("and ", "also ", "then ", "so ", "what about ", "how about ", "那", "再", "然后")):
            return True
        return len(content_tokens) <= 10 and len(semantic_focus_tokens) == 0 and has_contextual_shape

    def _rule_based_followup_resolution(self, query: str, focus: Dict[str, Any]) -> str:
        current = clean_display_text(query)
        focus_concepts = [
            clean_display_text(item)
            for item in (focus.get("focus_concepts") or [])
            if clean_display_text(item)
        ]
        focus_concept = clean_display_text(focus.get("focus_concept") or "")
        scope_labels = [
            label
            for label in self._query_focus_labels(current, limit=2)
            if clean_display_text(label) and clean_display_text(label) not in focus_concepts
        ]
        subject = ""
        if len(focus_concepts) >= 2:
            subject = f"{focus_concepts[0]}和{focus_concepts[1]}"
        else:
            subject = focus_concept
        if scope_labels:
            subject = f"{scope_labels[0]}中的{subject}" if subject else scope_labels[0]
        if not current or not subject:
            return current

        language_hint = self._query_language_hint(current) or clean_display_text(focus.get("language_hint") or "")
        lowered = current.lower()
        if self._is_code_generation_request(current):
            if self._is_explanation_request(current):
                if language_hint:
                    return f"请用 {language_hint} 给出一个关于{subject}的代码示例，并解释它的含义。"
                return f"请给出一个关于{subject}的代码示例，并解释它的含义。"
            if language_hint:
                return f"请用 {language_hint} 给出一个关于{subject}的代码示例。"
            return f"请给出一个关于{subject}的代码示例。"
        if self._is_explanation_request(current):
            return f"请继续解释{subject}，并回答：{current}"
        if COMPARISON_RE.search(lowered):
            return f"围绕{subject}，继续回答：{current}"
        return f"围绕{subject}，继续回答：{current}"

    def _followup_needs_topic_injection(self, query: str, focus_concept: str) -> bool:
        current = clean_display_text(query)
        if not current or not clean_display_text(focus_concept):
            return False
        if self._query_is_context_dependent(current):
            return True
        if self._query_has_strong_focus(current):
            return False
        if self._query_focus_labels(current, limit=2):
            return False
        if self._looks_like_contextual_request(current):
            return True
        if self._detect_followup_query(current, history=[{"role": "user", "content": focus_concept}]):
            return True
        return len(self._content_tokens(current)) <= 10

    def _context_resolution_candidate_valid(
        self,
        raw_query: str,
        candidate_query: str,
        *,
        followup_detected: bool,
    ) -> bool:
        raw = clean_display_text(raw_query)
        candidate = clean_display_text(candidate_query)
        if not raw or not candidate:
            return False
        if self._is_code_generation_request(raw) and not self._is_code_generation_request(candidate):
            return False
        if self._query_language_hint(raw) and self._is_code_generation_request(raw):
            if self._query_language_hint(candidate) not in {"", self._query_language_hint(raw)}:
                return False
        if self._query_has_strong_focus(raw) and not followup_detected:
            return candidate == raw
        return True

    def _llm_resolve_query_with_context(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
        focus: Dict[str, Any],
        session_memory: Optional[Dict[str, Any]],
        rule_based_candidate: str,
    ) -> Optional[Dict[str, Any]]:
        cleaned = clean_display_text(query)
        if not cleaned or self.kb._llm_client is None or not self.kb._llm_model:
            return None

        previous_topic = clean_display_text((session_memory or {}).get("active_topic") or "")
        previous_summary = clean_display_text((session_memory or {}).get("session_summary") or "")
        recent_history = "\n".join(
            f"{turn.get('role', 'user')}: {clean_display_text(turn.get('content', ''))}"
            for turn in list(history)[-6:]
            if clean_display_text(turn.get("content", ""))
        ) or "无"
        prompt = textwrap.dedent(
            f"""
            请判断学生最新一句话是否依赖前文。如果依赖，请把它改写成一个自然、完整、可独立回答的问题；如果本身已经完整，就保持原样。

            只输出严格 JSON：
            {{"resolved_query": "...", "focus_concept": "...", "used_history": true}}

            规则：
            1. 不要回答问题本身。
            2. 保留学生要求的输出形式，例如代码示例、解释、比较、例子。
            3. 如果学生已经明确提出一个新主题，不要硬带回旧主题。
            4. 如果学生只是短追问，例如“给我代码”“举个例子”“那为什么”“继续解释一下”，就结合上下文补全。
            5. 改写后要自然，像学生本来就在问这个完整问题。

            学生最新一句：
            {cleaned}

            会话记忆：
            - session_summary: {previous_summary or '无'}
            - active_topic: {previous_topic or '无'}

            当前上下文焦点：
            - anchor_query: {clean_display_text(focus.get('anchor_query', '')) or '无'}
            - focus_concept: {clean_display_text(focus.get('focus_concept', '')) or '无'}
            - focus_concepts: {"；".join(clean_display_text(item) for item in (focus.get('focus_concepts') or []) if clean_display_text(item)) or '无'}
            - anchor_answer: {clean_display_text(focus.get('anchor_answer', '')) or '无'}

            规则候选改写：
            {clean_display_text(rule_based_candidate) or cleaned}

            最近对话：
            {recent_history}
            """
        ).strip()
        payload = self.kb.call_json_llm(
            "你负责把依赖上下文的学生追问改写成完整问题。只输出 JSON。",
            prompt,
            temperature=0.0,
            max_tokens=200,
        )
        if not isinstance(payload, dict):
            return None
        resolved_query = clean_display_text(payload.get("resolved_query") or "")
        focus_concept = clean_display_text(payload.get("focus_concept") or "") or clean_display_text(focus.get("focus_concept") or "")
        used_history = bool(payload.get("used_history") or payload.get("used_context"))
        if not resolved_query:
            return None
        return {
            "resolved_query": resolved_query,
            "focus_concept": focus_concept,
            "used_history": used_history,
        }

    def _is_code_generation_request(self, query: str) -> bool:
        cleaned = clean_display_text(query)
        lowered = cleaned.lower()
        if not cleaned:
            return False
        if CODE_ONLY_RE.search(cleaned):
            return True
        if self._is_existing_code_reference(cleaned):
            return False
        has_code_words = bool(CODE_RE.search(cleaned))
        has_framework_words = bool(re.search(r"\b(pytorch|tensorflow|python|keras)\b", lowered))
        has_generation_trigger = bool(CODE_REQUEST_TRIGGER_RE.search(cleaned))
        return has_framework_words or (has_code_words and has_generation_trigger)

    def _is_explanation_request(self, query: str) -> bool:
        cleaned = clean_display_text(query)
        if not cleaned:
            return False
        return bool(
            EXPLAIN_RE.search(cleaned)
            or SUMMARY_REQUEST_RE.search(cleaned)
            or self._is_existing_code_reference(cleaned)
        )

    def _resolve_query_with_context(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
        session_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw_query = clean_display_text(query)
        if not raw_query:
            return {
                "raw_query": raw_query,
                "resolved_query": raw_query,
                "followup_detected": False,
                "used_history": False,
                "focus_concept": "",
                "focus_concepts": self._topic_components(clean_display_text((session_memory or {}).get("active_topic") or ""), limit=3),
                "anchor_query": "",
                "anchor_answer": "",
                "active_topic": clean_display_text((session_memory or {}).get("active_topic") or ""),
                "session_summary": clean_display_text((session_memory or {}).get("session_summary") or ""),
            }
        if self._assistant_intent(raw_query):
            return {
                "raw_query": raw_query,
                "resolved_query": raw_query,
                "followup_detected": False,
                "used_history": False,
                "focus_concept": "",
                "focus_concepts": self._topic_components(clean_display_text((session_memory or {}).get("active_topic") or ""), limit=3),
                "anchor_query": "",
                "anchor_answer": "",
                "active_topic": clean_display_text((session_memory or {}).get("active_topic") or ""),
                "session_summary": clean_display_text((session_memory or {}).get("session_summary") or ""),
            }

        focus = self._session_context_focus(history, session_memory=session_memory)
        followup_detected = self._detect_followup_query(raw_query, history)
        focus_concept = clean_display_text(focus.get("focus_concept") or "")
        focus_concepts = [
            clean_display_text(item)
            for item in (focus.get("focus_concepts") or [])
            if clean_display_text(item)
        ]
        rule_based_candidate = self._rule_based_followup_resolution(raw_query, focus)
        needs_injection = self._followup_needs_topic_injection(raw_query, focus_concept)

        resolved_query = raw_query
        used_history = False
        has_session_context = bool(history or clean_display_text(focus.get("session_summary") or "") or focus_concept)
        if has_session_context and not self._query_has_strong_focus(raw_query):
            llm_resolution = self._llm_resolve_query_with_context(
                raw_query,
                history,
                focus,
                session_memory,
                rule_based_candidate,
            )
            if llm_resolution:
                candidate = clean_display_text(llm_resolution.get("resolved_query") or "")
                if candidate and self._context_resolution_candidate_valid(
                    raw_query,
                    candidate,
                    followup_detected=followup_detected,
                ):
                    resolved_query = candidate
                    used_history = bool(llm_resolution.get("used_history"))
                    focus_concept = clean_display_text(llm_resolution.get("focus_concept") or "") or focus_concept
            focus_concepts = self._merge_focus_concepts(
                focus_concepts,
                self._query_focus_labels(resolved_query, limit=3),
                limit=3,
            )
            if resolved_query == raw_query and needs_injection and clean_display_text(rule_based_candidate) and clean_display_text(rule_based_candidate) != raw_query:
                resolved_query = clean_display_text(rule_based_candidate)
                used_history = True

        active_topic = self._derive_active_topic(
            resolved_query,
            {
                **focus,
                "focus_concept": focus_concept,
                "focus_concepts": focus_concepts,
            },
            {"related_kps": []},
            self._task_profile(raw_query, resolved_query),
            previous_memory=session_memory,
        )

        return {
            "raw_query": raw_query,
            "resolved_query": resolved_query or raw_query,
            "followup_detected": followup_detected,
            "used_history": used_history,
            "focus_concept": active_topic or focus_concept,
            "focus_concepts": focus_concepts,
            "anchor_query": clean_display_text(focus.get("anchor_query") or ""),
            "anchor_answer": clean_display_text(focus.get("anchor_answer") or ""),
            "active_topic": active_topic,
            "session_summary": clean_display_text(focus.get("session_summary") or ""),
        }

    def _raw_dl_signal_score(self, query: str) -> int:
        cleaned = clean_display_text(query)
        lowered = cleaned.lower()
        if not lowered:
            return 0
        score = 0
        if self.kb.looks_in_domain(cleaned):
            score += 4
        phrase_hits = sum(1 for term in DOMAIN_HINT_TERMS if term in lowered)
        score += min(phrase_hits, 6)
        token_hits = len(set(tokenize(lowered)) & DOMAIN_HINT_TERMS)
        score += min(token_hits, 4)
        return score

    def _non_course_signal_score(self, query: str) -> int:
        lowered = clean_display_text(query).lower()
        if not lowered:
            return 0
        score = 0
        for pattern in NON_COURSE_SIGNAL_PATTERNS:
            if pattern.search(lowered):
                score += 2
        if re.search(r"\bwho am i\b", lowered, re.IGNORECASE):
            score += 2
        return score

    def _is_in_scope(
        self,
        raw_query: str,
        resolved_query: str,
        resolution_context: Dict[str, Any],
    ) -> bool:
        focus_concept = clean_display_text(resolution_context.get("focus_concept") or "")
        active_topic = clean_display_text(resolution_context.get("active_topic") or "")
        followup_detected = bool(resolution_context.get("followup_detected"))
        positive_signal = max(
            self._raw_dl_signal_score(raw_query),
            self._raw_dl_signal_score(resolved_query),
            self._raw_dl_signal_score(focus_concept) if followup_detected else 0,
            self._raw_dl_signal_score(active_topic) if followup_detected else 0,
        )
        negative_signal = max(
            self._non_course_signal_score(raw_query),
            self._non_course_signal_score(resolved_query),
        )
        if negative_signal >= 2 and positive_signal == 0:
            return False
        if positive_signal > 0:
            return True
        if followup_detected and (focus_concept or active_topic):
            return max(self._raw_dl_signal_score(focus_concept), self._raw_dl_signal_score(active_topic)) > 0
        return False

    def _paired_focus_from_query(self, query: str) -> List[str]:
        cleaned = clean_display_text(query)
        if not cleaned:
            return []
        patterns = [
            re.compile(r"(?P<left>.+?)和(?P<right>.+?)(有什么)?(区别|关系|联系|差异)"),
            re.compile(r"比较(?P<left>.+?)和(?P<right>.+?)(?:$|[？?。.!])"),
            re.compile(r"compare\s+(?P<left>.+?)\s+and\s+(?P<right>.+?)(?:$|[?.!])", re.IGNORECASE),
            re.compile(r"difference\s+between\s+(?P<left>.+?)\s+and\s+(?P<right>.+?)(?:$|[?.!])", re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(cleaned)
            if not match:
                continue
            left = _clean_focus_label(match.group("left"))
            right = _clean_focus_label(match.group("right"))
            rows = []
            for item in [left, right]:
                lowered = item.lower()
                if item and lowered not in NON_TOPIC_FOCUS_TERMS:
                    rows.append(item)
            if rows:
                return rows[:2]
        return []

    def _query_focus_labels(self, query: str, limit: int = 2) -> List[str]:
        cleaned = clean_display_text(query)
        lowered = cleaned.lower()
        rows: List[str] = []
        for label, pattern in FOCUS_LABEL_PATTERNS:
            if pattern.search(cleaned):
                if label not in rows:
                    rows.append(label)
        for term in sorted(DOMAIN_HINT_TERMS, key=len, reverse=True):
            if term in NON_SPECIFIC_TOPIC_TERMS:
                continue
            if term not in lowered:
                continue
            label = clean_display_text(term)
            if label and label not in rows:
                rows.append(label)
        filtered_rows: List[str] = []
        for label in rows:
            suppressors = SUPPRESSED_FOCUS_LABELS.get(label, set())
            if suppressors and any(item in rows for item in suppressors):
                continue
            filtered_rows.append(label)
            if len(filtered_rows) >= limit:
                break
        return filtered_rows[:limit]

    def _task_focus_labels(
        self,
        raw_query: str,
        resolved_query: str,
        limit: int = 4,
    ) -> List[str]:
        rows = self._merge_focus_concepts(
            self._query_focus_labels(raw_query, limit=limit),
            self._query_focus_labels(resolved_query, limit=limit),
            limit=limit,
        )
        if len(rows) > 1:
            meaningful = [label for label in rows if label not in SUPPORTING_FOCUS_LABELS]
            if meaningful:
                rows = meaningful
        return rows[:limit]

    def _task_profile(
        self,
        raw_query: str,
        resolved_query: str,
    ) -> ChatTaskPlan:
        explicit_focus_labels = self._task_focus_labels(raw_query, resolved_query, limit=4)
        paired_focus = self._paired_focus_from_query(raw_query) or self._paired_focus_from_query(resolved_query)
        if len(paired_focus) < 2 and len(explicit_focus_labels) >= 2:
            paired_focus = explicit_focus_labels[:2]
        comparison = bool(COMPARISON_RE.search(raw_query)) or len(paired_focus) >= 2
        existing_code_reference = self._is_existing_code_reference(raw_query)
        code_request = self._is_code_generation_request(raw_query)
        explain_request = self._is_explanation_request(raw_query) or comparison
        reference_request = bool(REFERENCE_RE.search(raw_query))
        summary_request = bool(SUMMARY_REQUEST_RE.search(raw_query))
        only_code = code_request and bool(CODE_ONLY_RE.search(raw_query))
        if only_code:
            intent = "code"
        elif code_request and explain_request:
            intent = "mixed"
        elif code_request:
            intent = "code"
        elif comparison:
            intent = "comparison"
        elif explain_request:
            intent = "definition"
        else:
            intent = "other"
        return ChatTaskPlan(
            intent=intent,
            paired_focus=tuple(paired_focus[:2]),
            focus_labels=tuple(explicit_focus_labels[:4]),
            is_comparison=comparison,
            is_code_request=code_request,
            is_explain_request=explain_request,
            is_code_only_request=only_code,
            is_reference_request=reference_request,
            is_summary_request=summary_request,
            references_existing_code=existing_code_reference,
            needs_multi_concept_coverage=comparison or len(explicit_focus_labels) >= 2,
        )

    def _effective_retrieval_query(
        self,
        raw_query: str,
        resolution_context: Dict[str, Any],
        task_profile: ChatTaskPlan,
    ) -> str:
        resolved_query = clean_display_text(resolution_context.get("resolved_query") or raw_query)
        if task_profile.get("is_reference_request"):
            explicit_reference_terms = self._merge_focus_concepts(
                list(task_profile.get("paired_focus") or []),
                self._query_focus_labels(resolved_query or raw_query, limit=3),
                limit=4,
            )
            use_context = bool(
                resolution_context.get("followup_detected")
                or resolution_context.get("used_history")
            )
            reference_terms = list(explicit_reference_terms)
            if not reference_terms or (use_context and not self._query_has_strong_focus(raw_query)):
                reference_terms = self._merge_focus_concepts(
                    reference_terms,
                    resolution_context.get("focus_concepts") or [],
                    limit=4,
                )
            active_topic = clean_display_text(resolution_context.get("active_topic") or "")
            if active_topic and (not explicit_reference_terms or use_context):
                reference_terms = self._merge_focus_concepts(
                    reference_terms,
                    self._topic_components(active_topic, limit=3),
                    limit=4,
                )
            if reference_terms:
                return " ".join(reference_terms)
        focus_concept = clean_display_text(resolution_context.get("focus_concept") or "")
        if (
            resolution_context.get("followup_detected")
            and focus_concept
            and focus_concept.lower() not in resolved_query.lower()
        ):
            return clean_display_text(f"{resolved_query}；主题：{focus_concept}")
        return resolved_query

    def _retrieval_extra_queries(
        self,
        raw_query: str,
        resolution_context: Dict[str, Any],
        task_profile: ChatTaskPlan,
    ) -> List[str]:
        rows: List[str] = []
        use_context = bool(
            resolution_context.get("followup_detected")
            or resolution_context.get("used_history")
        )
        candidates: List[Any] = [
            raw_query,
            resolution_context.get("resolved_query") or "",
            *list(task_profile.get("paired_focus") or []),
        ]
        if use_context:
            candidates.extend(
                [
                    resolution_context.get("focus_concept") or "",
                    resolution_context.get("anchor_query") or "",
                    *(resolution_context.get("focus_concepts") or []),
                ]
            )
        for candidate in candidates:
            cleaned = clean_display_text(candidate)
            if cleaned and cleaned not in rows:
                rows.append(cleaned)
        return rows[:6]

    def _citation_score(self, query: str, citation: Dict[str, Any], rank: int) -> float:
        tokens = [token for token in tokenize(expand_query_aliases(query).lower()) if len(token) >= 2]
        if not tokens:
            return 0.0
        haystack = " ".join(
            [
                clean_display_text(citation.get("source") or "").lower(),
                clean_display_text(citation.get("section") or "").lower(),
                clean_display_text(citation.get("excerpt") or "").lower(),
            ]
        )
        score = 0.0
        for token in tokens:
            if token in haystack:
                score += 1.0
        for label in self._query_focus_labels(query, limit=3):
            if clean_display_text(label).lower() in haystack:
                score += 4.0
        score += max(0.0, 0.6 - 0.15 * rank)
        return score

    def _reference_guidance_payload(
        self,
        raw_query: str,
        retrieval_query: str,
        retrieval: Dict[str, Any],
        resolution_context: Dict[str, Any],
        task_profile: ChatTaskPlan,
    ) -> Dict[str, Any]:
        citations = self._reference_citations(retrieval_query, retrieval)
        topic = self._derive_active_topic(
            retrieval_query,
            resolution_context,
            retrieval,
            task_profile,
            previous_memory=None,
        ) or "相关知识点"
        if not citations:
            return {
                "answer": f"如果你现在是为了复习“{topic}”，建议把问题再具体一点，例如直接点出模型、机制或公式，这样我能更准确地给你定位到讲义入口。",
                "used_sources": [],
                "active_topic": topic,
                "session_summary": f"最近在定位“{topic}”对应的讲义位置。",
            }

        primary_section = clean_display_text((citations[0] or {}).get("section") or "") or topic
        secondary_section = clean_display_text((citations[1] or {}).get("section") or "") if len(citations) > 1 else ""
        answer = f"如果你现在要复习“{topic}”，先从右侧第一条来源中的“{primary_section}”开始。"
        if secondary_section and secondary_section != primary_section:
            answer += f" 如果想把前后的背景和应用串起来，再接着看第二条来源里的“{secondary_section}”。"
        answer += " 右侧的来源按钮会直接带你打开对应 PDF。"
        return {
            "answer": answer,
            "used_sources": [
                clean_display_text(item.get("citation_id") or "").upper()
                for item in citations[:2]
                if clean_display_text(item.get("citation_id") or "")
            ],
            "active_topic": topic,
            "session_summary": f"最近在定位“{topic}”对应的讲义复习入口。",
        }

    def _reference_citations(
        self,
        query_text: str,
        retrieval: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen: set[tuple[str, int, int, str]] = set()

        def add_citation(citation: Dict[str, Any]) -> None:
            source = clean_display_text(citation.get("source") or "")
            if not source:
                return
            key = (
                source,
                int(citation.get("unit_index") or 0),
                int(citation.get("chunk_index") or 0),
                clean_display_text(citation.get("section") or ""),
            )
            if key in seen:
                return
            seen.add(key)
            rows.append(citation)

        for item in retrieval.get("citations") or []:
            add_citation(dict(item))

        for related_kp in list(retrieval.get("related_kps") or [])[:2]:
            kp_id = clean_display_text((related_kp or {}).get("kp_id") or "")
            if not kp_id:
                continue
            for ref in self.kb.kp_review_refs(kp_id, limit=8):
                add_citation(
                    {
                        "citation_id": f"R{len(rows) + 1}",
                        "source": clean_display_text(ref.get("source") or ""),
                        "unit_type": clean_display_text(ref.get("unit_type") or "page"),
                        "unit_index": int(ref.get("unit_index") or 0),
                        "chunk_index": int(ref.get("chunk_index") or 0),
                        "section": clean_display_text(ref.get("section") or ""),
                        "excerpt": "",
                        "location": clean_display_text(ref.get("location") or ""),
                    }
                )

        rows.sort(
            key=lambda citation: (
                -self._citation_score(query_text, citation, 0),
                clean_display_text(citation.get("source") or ""),
                int(citation.get("unit_index") or 0),
            )
        )
        return rows[:2]

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

    def _autofill_used_sources(
        self,
        payload: Dict[str, Any],
        retrieval: Dict[str, Any],
        query: str,
        task_profile: ChatTaskPlan,
    ) -> Dict[str, Any]:
        normalized = dict(payload or {})
        citations = list(retrieval.get("citations") or [])
        if not citations:
            normalized["used_sources"] = []
            return normalized

        allowed = {
            clean_display_text(item.get("citation_id") or "").upper()
            for item in citations
            if clean_display_text(item.get("citation_id") or "")
        }
        scored_map = {
            clean_display_text(citation.get("citation_id") or "").upper(): self._citation_score(query, citation, index)
            for index, citation in enumerate(citations)
            if clean_display_text(citation.get("citation_id") or "")
        }
        explicit: List[str] = []
        for item in normalized.get("used_sources") or []:
            source_id = clean_display_text(item).upper()
            if source_id and source_id in allowed and source_id not in explicit:
                explicit.append(source_id)
        if explicit:
            explicit_best = max((float(scored_map.get(item, 0.0)) for item in explicit), default=0.0)
            candidate_best = max((float(score) for score in scored_map.values()), default=0.0)
            if explicit_best >= max(0.3, candidate_best - 0.6):
                normalized["used_sources"] = explicit
                return normalized

        if clean_display_text(retrieval.get("coverage_level") or "") == "none":
            normalized["used_sources"] = []
            return normalized

        coverage_level = clean_display_text(retrieval.get("coverage_level") or "")
        scored_rows: List[Dict[str, Any]] = []
        for index, citation in enumerate(citations):
            scored_rows.append(
                {
                    "source_id": clean_display_text(citation.get("citation_id") or "").upper(),
                    "score": self._citation_score(query, citation, index),
                    "index": index,
                }
            )
        scored_rows = [row for row in scored_rows if row["source_id"]]
        scored_rows.sort(key=lambda row: (-float(row["score"]), int(row["index"])))
        desired_count = 1
        if task_profile.get("needs_multi_concept_coverage"):
            desired_count = min(2, len(scored_rows))
        positive_rows = [row for row in scored_rows if float(row["score"]) > 0.0]
        if coverage_level == "direct":
            normalized["used_sources"] = [row["source_id"] for row in positive_rows[:desired_count]]
        else:
            normalized["used_sources"] = [row["source_id"] for row in positive_rows[:desired_count] if row["score"] >= 2.2]
        if not normalized["used_sources"] and citations and coverage_level == "direct":
            normalized["used_sources"] = [clean_display_text(citations[0].get("citation_id") or "").upper()]
        return normalized

    def _strip_internal_source_ids(self, text: str) -> str:
        parts = re.split(r"(```[\s\S]*?```)", text or "")
        cleaned_parts: List[str] = []
        for index, part in enumerate(parts):
            if index % 2 == 1:
                cleaned_parts.append(part)
                continue
            segment = INLINE_SOURCE_ID_RE.sub("", part)
            segment = re.sub(r"(在|如|见|参考|例如)\s*(中|里)", "", segment)
            segment = re.sub(r"[（(]\s*[)）]", "", segment)
            segment = re.sub(r"\s{2,}", " ", segment)
            segment = re.sub(r"([，。；,;])\s*([，。；,;])+", r"\1", segment)
            cleaned_parts.append(segment)
        return "".join(cleaned_parts)

    def _sanitize_answer_text(self, text: Any) -> str:
        value = clean_multiline_text(text)
        if not value:
            return ""
        for _ in range(3):
            extracted = ""
            if JSON_LIKE_RE.match(value):
                parsed = parse_jsonish_text(value)
                if isinstance(parsed, dict) and clean_display_text(parsed.get("answer") or ""):
                    extracted = clean_multiline_text(parsed.get("answer") or "")
            if not extracted and '"answer"' in value:
                match = re.search(r'"answer"\s*:\s*"', value)
                if match:
                    cursor = match.end()
                    buffer: List[str] = []
                    escaped = False
                    for char in value[cursor:]:
                        if escaped:
                            buffer.append(char)
                            escaped = False
                            continue
                        if char == "\\":
                            buffer.append(char)
                            escaped = True
                            continue
                        if char == '"':
                            break
                        buffer.append(char)
                    raw_answer = "".join(buffer).strip()
                    if raw_answer:
                        try:
                            extracted = clean_multiline_text(json.loads(f'"{raw_answer}"'))
                        except Exception:
                            extracted = clean_multiline_text(
                                raw_answer
                                .replace("\\n", "\n")
                                .replace('\\"', '"')
                                .replace("\\\\", "\\")
                            )
            if not extracted or extracted == value:
                break
            value = extracted
        value = SOURCE_TOKEN_RE.sub("", value)
        value = SOURCE_LINE_RE.sub("", value)
        value = self._strip_internal_source_ids(value)
        value = re.sub(r"(?im)^```json\s*$", "```", value)
        value = re.sub(r"```([A-Za-z0-9_+-]+)[ \t]+", r"```\1\n", value)
        value = re.sub(r"(?<!\n)```([A-Za-z0-9_+-]+)?", lambda match: ("\n" if match.start() > 0 else "") + match.group(0), value)
        if value.count("```") % 2 == 1:
            value = value.rstrip() + "\n```"
        value = re.sub(r"\n{3,}", "\n\n", value)
        return _normalize_model_names(value).strip()

    def _normalize_payload(self, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        normalized = dict(payload or {})
        normalized["answer"] = self._sanitize_answer_text(normalized.get("answer") or "")
        used_sources: List[str] = []
        for item in normalized.get("used_sources") or []:
            source_id = clean_display_text(item).upper()
            if source_id and source_id not in used_sources:
                used_sources.append(source_id)
        normalized["used_sources"] = used_sources[:6]
        normalized["active_topic"] = clean_display_text(normalized.get("active_topic") or "")
        normalized["session_summary"] = clean_display_text(normalized.get("session_summary") or "")
        return normalized

    def _answer_mentions_focus_label(self, answer: str, label: str) -> bool:
        cleaned_label = clean_display_text(label)
        if not cleaned_label:
            return False
        lowered_answer = expand_query_aliases(answer).lower()
        aliases = FOCUS_LABEL_ALIASES.get(cleaned_label, (cleaned_label,))
        alias_rows = [clean_display_text(item).lower() for item in aliases if clean_display_text(item)]
        if any(alias in lowered_answer for alias in alias_rows if len(alias) >= 2):
            return True
        answer_tokens = set(tokenize(lowered_answer))
        alias_tokens = {
            token
            for token in tokenize(" ".join(alias_rows))
            if len(token) >= 2 and token not in QUERY_NOISE_TOKENS
        }
        if not alias_tokens:
            return False
        overlap = len(answer_tokens & alias_tokens)
        if len(alias_tokens) == 1:
            return overlap >= 1
        return overlap >= min(2, len(alias_tokens))

    def _answer_issues(
        self,
        payload: Dict[str, Any],
        query: str,
        task_profile: ChatTaskPlan,
    ) -> List[str]:
        issues: List[str] = []
        answer = self._sanitize_answer_text(payload.get("answer") or "")
        if not answer:
            issues.append("answer_is_empty")
            return issues
        lowered = answer.lower()
        fenced_code_blocks = re.findall(r"```(?:[A-Za-z0-9_+-]+)?\n[\s\S]*?```", answer)
        plain = re.sub(r"\s+", "", re.sub(r"```[\s\S]*?```", " ", answer))
        if task_profile.get("is_code_request") and not fenced_code_blocks:
            issues.append("missing_code_block")
        if task_profile.get("is_code_only_request"):
            stripped = answer.strip()
            if not (stripped.startswith("```") and stripped.endswith("```")):
                issues.append("code_only_response_not_pure")
        if (
            task_profile.get("is_code_request")
            and task_profile.get("is_explain_request")
            and not task_profile.get("is_code_only_request")
            and len(plain) < 28
        ):
            issues.append("mixed_request_missing_explanation")
        if task_profile.get("references_existing_code") and "```" in answer and len(plain) < 24:
            issues.append("existing_code_explanation_misread")
        if task_profile.get("is_summary_request") and len(plain) < 10:
            issues.append("summary_missing")
        if task_profile.get("is_code_request") and task_profile.get("is_explain_request") and not task_profile.get("is_code_only_request"):
            stripped = answer.strip()
            if stripped.startswith("```") and stripped.endswith("```") and "mixed_request_missing_explanation" not in issues:
                issues.append("mixed_request_missing_explanation")
        focus_labels = [
            clean_display_text(item)
            for item in task_profile.get("focus_labels") or []
            if clean_display_text(item)
        ]
        if task_profile.get("needs_multi_concept_coverage"):
            relevant_focus_labels = [label for label in focus_labels if label not in SUPPORTING_FOCUS_LABELS]
            mentioned = sum(
                1 for label in relevant_focus_labels[:3]
                if self._answer_mentions_focus_label(answer, label)
            )
            if relevant_focus_labels and mentioned < min(2, len(relevant_focus_labels)):
                issues.append("multi_concept_coverage_missing")
        elif focus_labels:
            primary_label = next((label for label in focus_labels if label not in SUPPORTING_FOCUS_LABELS), focus_labels[0])
            if (
                (task_profile.get("is_code_request") or task_profile.get("is_explain_request") or task_profile.get("is_summary_request"))
                and not self._answer_mentions_focus_label(answer, primary_label)
            ):
                issues.append("primary_concept_missing")
        if INLINE_SOURCE_DETAIL_RE.search(answer) or INLINE_SOURCE_ID_RE.search(answer):
            issues.append("inline_source_detail_leaked")
        if SOURCE_TOKEN_RE.search(answer) or "used_sources" in lowered or "\"answer\"" in lowered:
            issues.append("raw_control_text_leaked")
        if "根据检索" in answer or "我检索到" in answer or "系统" in answer and "学习系统" not in answer:
            issues.append("retrieval_tone_leaked")
        lowered_query = clean_display_text(query).lower()
        for token, patterns in TERM_AMBIGUITY_GUARDS.items():
            if token not in lowered_query:
                continue
            if any(pattern.search(answer) for pattern in patterns):
                issues.append("term_disambiguation_failed")
                break
        return issues

    def _repair_payload(
        self,
        *,
        raw_query: str,
        resolved_query: str,
        history: Sequence[Dict[str, str]],
        retrieval: Dict[str, Any],
        task_profile: ChatTaskPlan,
        current_payload: Dict[str, Any],
        issues: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        if self.kb._llm_client is None or not self.kb._llm_model:
            return None

        citations = list(retrieval.get("citations") or [])
        source_block = "\n".join(
            f"{clean_display_text(item.get('citation_id') or '')}: {clean_display_text(item.get('source') or '')}，{clean_display_text(item.get('location') or '')}，主题：{clean_display_text(item.get('section') or '')}"
            for item in citations[:4]
        ) or "无"
        repair_prompt = f"""
你要修复一条已经生成过的学生端回答。请只输出严格 JSON：
{{
  "answer": "...",
  "used_sources": ["S1", "S2"],
  "active_topic": "...",
  "session_summary": "..."
}}

当前学生问题：
{clean_display_text(raw_query)}

独立问题：
{clean_display_text(resolved_query)}

最近对话：
{_history_text(history)}

任务类型：
{task_profile.get('intent') or 'other'}

引用候选：
{source_block}

当前回答：
{clean_multiline_text(current_payload.get("answer") or "")}

待修复问题：
{", ".join(str(item) for item in issues)}

修复规则：
1. 回答要自然、像课程助教，不要像系统提示或检索提示。
2. 如果是代码题，必须使用 fenced Markdown code block。
3. 如果是只要代码的题，输出只能是一个代码块，不要额外解释。
4. 如果学生问了多个概念或比较关系，必须全部回答。
5. 不要输出 source、used_sources、JSON 解释、系统说明或控制标记到 answer 正文。
6. 只有真正支撑回答的来源才放进 used_sources。
""".strip()

        payload = self.kb.call_json_llm(
            "你是商业化课程平台中的中文助教。请只输出严格 JSON。",
            repair_prompt,
            temperature=0.08,
            max_tokens=1400,
        )
        if isinstance(payload, dict):
            sanitized = self._normalize_payload(payload)
            if not self._answer_issues(sanitized, raw_query, task_profile):
                return sanitized

        text_answer = self.kb.call_text_llm(
            "你是商业化课程平台中的中文助教。请只输出修复后的最终学生答案正文，不要 JSON，不要来源行，不要控制字段。",
            f"""
学生当前问题：
{clean_display_text(raw_query)}

独立问题：
{clean_display_text(resolved_query)}

最近对话：
{_history_text(history)}

当前有问题的回答：
{clean_multiline_text(current_payload.get("answer") or "")}

待修复问题：
{", ".join(str(item) for item in issues)}

修复规则：
1. 直接输出最终答案正文，不要输出 JSON，不要输出 used_sources、active_topic、session_summary。
2. 如果学生这轮是在追问上面那段代码，就解释现有代码，不要额外新写一段无关代码。
3. 如果这轮既要解释又要代码，先简洁解释，再给 fenced Markdown code block。
4. 如果学生只要代码，就只输出一个 fenced Markdown code block。
5. 不要在正文中写文件名、页码、citation id、source 或系统提示。
""".strip(),
            temperature=0.08,
            max_tokens=1400,
        )
        if clean_display_text(text_answer):
            fallback_payload = self._normalize_payload({
                "answer": text_answer,
                "used_sources": list(current_payload.get("used_sources") or []),
                "active_topic": clean_display_text(current_payload.get("active_topic") or ""),
                "session_summary": clean_display_text(current_payload.get("session_summary") or ""),
            })
            if not self._answer_issues(fallback_payload, raw_query, task_profile):
                return fallback_payload
        return None

    def _force_code_answer(
        self,
        *,
        raw_query: str,
        resolved_query: str,
        history: Sequence[Dict[str, str]],
        retrieval: Dict[str, Any],
        task_profile: ChatTaskPlan,
        current_payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self.kb._llm_client is None or not self.kb._llm_model:
            return None

        citations = list(retrieval.get("citations") or [])
        source_ids = [
            clean_display_text(item.get("citation_id") or "").upper()
            for item in citations[:3]
            if clean_display_text(item.get("citation_id") or "")
        ]
        text_answer = self.kb.call_text_llm(
            "你是商业化课程平台中的中文助教。学生这轮明确要代码，请输出最终答案正文。",
            f"""
学生当前问题：
{clean_display_text(raw_query)}

独立问题：
{clean_display_text(resolved_query)}

最近对话：
{_history_text(history)}

课程来源摘要：
{chr(10).join(clean_display_text(item.get("excerpt") or "") for item in citations[:3]) or '无直接摘要'}

输出规则：
1. 这是一道代码请求，最终答案必须包含 fenced Markdown code block。
2. 如果学生只要代码，就只输出一个 fenced Markdown code block。
3. 如果学生既要解释又要代码，先用 2 到 3 句中文说明，再给一个 fenced Markdown code block。
4. 代码以最小可运行、最小可读示例为准；如果学生没指定框架，默认用 PyTorch。
5. 不要输出 JSON，不要输出来源、页码、文件名、citation id 或系统说明。
""".strip(),
            temperature=0.08,
            max_tokens=1600,
        )
        if not clean_display_text(text_answer):
            return None
        payload = self._normalize_payload({
            "answer": self._sanitize_answer_text(text_answer),
            "used_sources": source_ids or list(current_payload.get("used_sources") or []),
            "active_topic": clean_display_text(current_payload.get("active_topic") or ""),
            "session_summary": clean_display_text(current_payload.get("session_summary") or ""),
        })
        if self._answer_issues(payload, raw_query, task_profile):
            return None
        return payload

    def _stabilize_payload(
        self,
        *,
        raw_query: str,
        resolved_query: str,
        history: Sequence[Dict[str, str]],
        retrieval: Dict[str, Any],
        task_profile: ChatTaskPlan,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        working_payload = self._normalize_payload(payload)
        issues = self._answer_issues(working_payload, raw_query, task_profile)
        if issues:
            repaired = self._repair_payload(
                raw_query=raw_query,
                resolved_query=resolved_query,
                history=history,
                retrieval=retrieval,
                task_profile=task_profile,
                current_payload=working_payload,
                issues=issues,
            )
            if isinstance(repaired, dict):
                working_payload = self._normalize_payload(repaired)
                issues = self._answer_issues(working_payload, raw_query, task_profile)
        if issues and task_profile.is_code_request:
            forced_code_payload = self._force_code_answer(
                raw_query=raw_query,
                resolved_query=resolved_query,
                history=history,
                retrieval=retrieval,
                task_profile=task_profile,
                current_payload=working_payload,
            )
            if isinstance(forced_code_payload, dict):
                working_payload = self._normalize_payload(forced_code_payload)
        return self._normalize_payload(working_payload)

    def _finalize_course_payload(
        self,
        *,
        raw_query: str,
        retrieval_query: str,
        history: Sequence[Dict[str, str]],
        retrieval: Dict[str, Any],
        task_profile: ChatTaskPlan,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        working_payload = self._stabilize_payload(
            raw_query=raw_query,
            resolved_query=retrieval_query,
            history=history,
            retrieval=retrieval,
            task_profile=task_profile,
            payload=payload,
        )
        working_payload = self._normalize_payload(
            self._autofill_used_sources(working_payload, retrieval, retrieval_query, task_profile)
        )
        answer_text = working_payload.get("answer") or ""
        if not answer_text:
            answer_text = self._sanitize_answer_text(self.kb.answer_from_chunks(retrieval_query, retrieval))

        final_citations = self._extract_used_citations(
            retrieval.get("citations") or [],
            working_payload.get("used_sources") or [],
        )
        if not final_citations and clean_display_text(retrieval.get("coverage_level") or "") == "direct":
            final_citations = list((retrieval.get("citations") or [])[:1])

        return {
            "answer": answer_text,
            "citations": final_citations,
            "related_kps": retrieval.get("related_kps") or [],
            "mode": "llm_unified",
            "response_kind": "course_answer",
        }

    def _generate_payload(
        self,
        *,
        raw_query: str,
        resolution_context: Dict[str, Any],
        retrieval: Dict[str, Any],
        task_profile: ChatTaskPlan,
        history: Sequence[Dict[str, str]],
    ) -> Optional[Dict[str, Any]]:
        citations = list(retrieval.get("citations") or [])
        related_kps = list(retrieval.get("related_kps") or [])
        source_block = "\n".join(
            f"{clean_display_text(item.get('citation_id') or '')}: {clean_display_text(item.get('source') or '')}，{clean_display_text(item.get('location') or '')}，主题：{clean_display_text(item.get('section') or '')}，摘录：{clean_display_text(item.get('excerpt') or '')}"
            for item in citations[:4]
        ) or "无直接讲义来源"
        related_text = "；".join(
            clean_display_text(item.get("name") or "")
            for item in related_kps
            if clean_display_text(item.get("name") or "")
        ) or "无"
        resolved_query = clean_display_text(resolution_context.get("resolved_query") or raw_query)
        history_block = _history_text(history if resolution_context.get("followup_detected") else [])
        repeat_count = self._repeat_count(raw_query, history)
        repeat_instruction = "请换一种讲解路径和措辞，但事实保持一致。" if repeat_count else "正常自然作答即可。"
        term_hints = _clean_list(_term_hints(resolved_query), limit=4)

        prompt = f"""
你要回答一名学生关于深度学习课程的问题。请只输出严格 JSON：
{{
  "answer": "...",
  "used_sources": ["S1", "S2"],
  "active_topic": "...",
  "session_summary": "..."
}}

学生当前问题：
{clean_display_text(raw_query)}

独立问题：
{resolved_query}

上下文信息：
- follow_up: {str(bool(resolution_context.get("followup_detected"))).lower()}
- 使用历史上下文: {str(bool(resolution_context.get("used_history"))).lower()}
- 当前会话主题: {clean_display_text(resolution_context.get("active_topic") or '') or '无'}
- 当前会话摘要: {clean_display_text(resolution_context.get("session_summary") or '') or '无'}
- 追问焦点: {clean_display_text(resolution_context.get("focus_concept") or '') or '无'}
- 上一问: {clean_display_text(resolution_context.get("anchor_query") or '') or '无'}

最近对话：
{history_block}

任务要求：
{task_profile.prompt_flags()}
- 术语提示: {'；'.join(term_hints) if term_hints else '无'}
- 历史重复回答要求: {repeat_instruction}

课程检索结果：
- coverage_level: {clean_display_text(retrieval.get("coverage_level") or '') or 'none'}
- related_kps: {related_text}
- citations:
{source_block}

作答规则：
1. 第一段先直接回答学生真正想问的点，不要先铺垫空泛背景。
2. 语气要像老师或课程助教，不要像检索系统、百科词条或演示系统。
3. 全部使用中文，但代码本身按正常编程语言书写。
4. 如果问题属于深度学习领域，但讲义没有直接覆盖，也可以依据可靠的深度学习知识自然回答；不要因为讲义里没有原句就拒答。
5. 如果是追问，继承必要上下文；如果不是追问，就把这轮当作独立问题，不要被旧主题带偏。
6. 如果学生问了多个概念、多个子问或比较关系，必须全部回答，不要只答一半。
7. 如果是比较题，必须明确写出双方的任务定位、关键差异和适用场景，不要写成单方面吹捧。
8. 如果学生只要代码，answer 必须只包含一个 fenced Markdown code block，不要额外解释。
9. 如果既要解释又要代码，先用 2 到 4 句讲清思路，再给一个 fenced Markdown code block，最后可补 1 到 2 句说明。
10. 只有真正支撑回答的课程来源，才放入 used_sources。不要在 answer 正文里写 source、页码、文件名、citation id 或系统提示；这些会由前端单独展示。
11. 对于 DiT、ViT、CLIP、U-Net 等术语，若有提示就按提示理解，不要偷偷改写成别的模型。
12. active_topic 要短，session_summary 用一句中文概括当前轮次给下轮对话续接。
13. 如果学生是在追问“上面代码里的某个概念/参数/操作”，说明那段已有代码在做什么，不要误判成“再生成一份新代码”。
""".strip()

        payload = self.kb.call_json_llm(
            "你是商业化课程平台中的中文助教。请只输出严格 JSON。",
            prompt,
            temperature=0.24,
            max_tokens=1800,
        )
        if isinstance(payload, dict):
            return payload

        text_prompt = f"""
你是一名商业化课程平台里的中文课程助教，请直接回答学生，不要输出 JSON，不要输出来源行。

学生当前问题：
{clean_display_text(raw_query)}

独立问题：
{resolved_query}

最近对话：
{history_block}

任务要求：
{task_profile.prompt_flags(include_existing_code=True)}

课程检索线索：
- related_kps: {related_text}
- citations:
{source_block}

回答规则：
1. 先直接回答学生真正的问题，不要先说系统说明。
2. 如果问题属于深度学习领域，但讲义没有直接原句，也可以基于可靠的深度学习知识自然回答。
3. 如果这轮既要解释又要代码，先用 2 到 4 句讲清楚，再给 fenced Markdown code block。
4. 如果学生只要代码，就只输出一个 fenced Markdown code block。
5. 如果学生是在追问上面那段代码里的某个点，就解释那段已有代码，不要改成重新生成一份新代码。
6. 如果是比较题或多概念题，必须把所有对象都答全。
7. 不要在正文中写 source、页码、文件名、citation id 或 JSON 控制字段。
""".strip()
        text_answer = self.kb.call_text_llm(
            "你是商业化课程平台中的中文助教。请直接回答学生问题，不要输出 JSON，不要输出来源行。",
            text_prompt,
            temperature=0.24,
            max_tokens=1800,
        )
        if not clean_display_text(text_answer):
            return None
        return self._normalize_payload({
            "answer": text_answer,
            "used_sources": [],
            "active_topic": "",
            "session_summary": "",
        })

    def _generate_payload(
        self,
        *,
        raw_query: str,
        resolution_context: Dict[str, Any],
        retrieval: Dict[str, Any],
        task_profile: ChatTaskPlan,
        history: Sequence[Dict[str, str]],
    ) -> Optional[Dict[str, Any]]:
        citations = list(retrieval.get("citations") or [])
        related_kps = list(retrieval.get("related_kps") or [])
        retrieval_target_concepts = [
            clean_display_text(item.get("label") or "")
            for item in retrieval.get("target_concepts") or []
            if clean_display_text(item.get("label") or "")
        ]
        prompt_focus_labels = self._merge_focus_concepts(
            list(task_profile.get("focus_labels") or []),
            retrieval_target_concepts,
            limit=4,
        )
        target_concepts_text = "；".join(prompt_focus_labels) or "无"
        allowed_source_ids = ", ".join(
            clean_display_text(item.get("citation_id") or "").upper()
            for item in citations
            if clean_display_text(item.get("citation_id") or "")
        ) or "无"
        source_block = "\n".join(
            f"{clean_display_text(item.get('citation_id') or '')}: {clean_display_text(item.get('source') or '')}，{clean_display_text(item.get('location') or '')}，主题：{clean_display_text(item.get('section') or '')}，摘录：{clean_display_text(item.get('excerpt') or '')}"
            for item in citations[:4]
        ) or "无直接讲义来源"
        related_text = "；".join(
            clean_display_text(item.get("name") or "")
            for item in related_kps
            if clean_display_text(item.get("name") or "")
        ) or "无"
        resolved_query = clean_display_text(resolution_context.get("resolved_query") or raw_query)
        history_block = _history_text(history if resolution_context.get("followup_detected") else [])
        repeat_count = self._repeat_count(raw_query, history)
        repeat_instruction = "请换一种讲解路径和措辞，但事实保持一致。" if repeat_count else "正常自然作答即可。"
        term_hints = _clean_list(_term_hints(resolved_query), limit=4)

        prompt = f"""
你是商业化深度学习课程平台里的学生端中文助教。请只输出严格 JSON：
{{
  "answer": "...",
  "used_sources": ["S1", "S2"],
  "active_topic": "...",
  "session_summary": "..."
}}

学生当前问题：
{clean_display_text(raw_query)}

独立问题：
{resolved_query}

上下文信息：
- follow_up: {str(bool(resolution_context.get("followup_detected"))).lower()}
- 使用历史上下文: {str(bool(resolution_context.get("used_history"))).lower()}
- 当前会话主题: {clean_display_text(resolution_context.get("active_topic") or '') or '无'}
- 当前会话摘要: {clean_display_text(resolution_context.get("session_summary") or '') or '无'}
- 追问焦点: {clean_display_text(resolution_context.get("focus_concept") or '') or '无'}
- 上一问: {clean_display_text(resolution_context.get("anchor_query") or '') or '无'}

最近对话：
{history_block}

任务要求：
{task_profile.prompt_flags()}
- 重点概念: {target_concepts_text}
- 术语提示: {'；'.join(term_hints) if term_hints else '无'}
- 历史重复回答要求: {repeat_instruction}

课程检索结果：
- coverage_level: {clean_display_text(retrieval.get("coverage_level") or '') or 'none'}
- related_kps: {related_text}
- allowed_source_ids: {allowed_source_ids}
- citations:
{source_block}

作答规则：
1. 先直接回答学生真正的问题，不要先铺垫空泛背景。
2. 语气要像助教，保持自然、稳定，不要像检索系统、百科词条或 demo。
3. 全部使用中文，但代码本身按正常编程语言书写。
4. 如果问题属于深度学习领域，即使讲义没有直接覆盖，也要基于可靠的深度学习知识自然回答，不要机械拒答。
5. 如果这轮是追问，只继承必要上下文；如果学生已经切到新主题，不要硬拉回旧主题。
6. 如果学生问了多个概念、多个子问或比较关系，必须全部覆盖。
7. 如果学生点名了具体模型、模块或训练方法，就回答那个对象本身，不要偷偷换成邻近概念。
8. 如果是比较题，明确写出双方的作用、差异和适用场景。
9. 如果学生只要代码，answer 必须只包含一个 fenced Markdown code block，不要额外解释。
10. 如果既要解释又要代码，先用 2 到 4 句解释，再给一个 fenced Markdown code block，不要只给其一。
11. 只有真正支撑答案的课程来源才放入 used_sources；如果 coverage_level 不是 direct，就不要勉强挂来源。
12. 不要在 answer 正文里写 source、页码、文件名、citation id、JSON 字段名或系统提示。
13. 对于 DiT、ViT、CLIP、U-Net、LoRA、MoE、PEFT 等术语，若有提示就按提示理解，不要偷偷改写成别的模型，也不要引入这些缩写在其他领域的含义。
14. active_topic 要短，session_summary 要能帮下一轮延续对话。
15. 如果学生是在追问上文代码中的参数、模块或操作，就解释那段已有代码，不要误判成重新生成一份无关的新代码。
""".strip()

        payload = self.kb.call_json_llm(
            "你是商业化深度学习课程平台中的中文助教。请只输出严格 JSON。",
            prompt,
            temperature=0.08 if task_profile.get("is_code_request") else 0.22,
            max_tokens=1800,
        )
        if isinstance(payload, dict):
            return payload

        text_prompt = f"""
你是一名商业化课程平台里的中文课程助教，请直接回答学生，不要输出 JSON，不要输出来源行。

学生当前问题：
{clean_display_text(raw_query)}

独立问题：
{resolved_query}

最近对话：
{history_block}

任务要求：
{task_profile.prompt_flags(include_existing_code=True)}
- 重点概念: {target_concepts_text}

课程检索线索：
- related_kps: {related_text}
- coverage_level: {clean_display_text(retrieval.get("coverage_level") or '') or 'none'}
- citations:
{source_block}

回答规则：
1. 先直接回答学生真正的问题，不要先说系统说明。
2. 如果问题属于深度学习领域，但讲义没有直接原句，也可以基于可靠的深度学习知识自然回答。
3. 如果这轮既要解释又要代码，先用 2 到 4 句讲清楚，再给 fenced Markdown code block。
4. 如果学生只要代码，就只输出一个 fenced Markdown code block。
5. 如果学生是在追问上面那段代码里的某个点，就解释那段已有代码，不要改成重新生成一份新代码。
6. 如果是比较题或多概念题，必须把所有对象都答全。
7. 如果学生点名了具体模型、模块或训练方法，就回答那个对象本身，不要换成邻近概念。
8. 不要在正文中写 source、页码、文件名、citation id 或 JSON 控制字段。
""".strip()
        text_answer = self.kb.call_text_llm(
            "你是商业化深度学习课程平台中的中文助教。请直接回答学生问题，不要输出 JSON，不要输出来源行。",
            text_prompt,
            temperature=0.08 if task_profile.get("is_code_request") else 0.22,
            max_tokens=1800,
        )
        if not clean_display_text(text_answer):
            return None
        return self._normalize_payload({
            "answer": text_answer,
            "used_sources": [],
            "active_topic": "",
            "session_summary": "",
        })

    def _derive_active_topic(
        self,
        raw_query: str,
        resolution_context: Dict[str, Any],
        retrieval: Dict[str, Any],
        task_profile: ChatTaskPlan,
        previous_memory: Optional[Dict[str, Any]] = None,
    ) -> str:
        carried_focus = [
            clean_display_text(item)
            for item in (resolution_context.get("focus_concepts") or [])
            if clean_display_text(item)
        ]
        if len(carried_focus) >= 2 and resolution_context.get("followup_detected"):
            return " / ".join(carried_focus[:2])[:32]
        if carried_focus and resolution_context.get("followup_detected"):
            return carried_focus[0][:24]
        explicit_labels = [
            clean_display_text(item)
            for item in task_profile.get("focus_labels") or []
            if clean_display_text(item)
        ] or self._query_focus_labels(raw_query, limit=2)
        if len(explicit_labels) >= 2:
            return " / ".join(explicit_labels[:2])[:32]
        if explicit_labels:
            return explicit_labels[0][:24]
        paired_focus = [
            clean_display_text(item)
            for item in task_profile.get("paired_focus") or []
            if clean_display_text(item) and clean_display_text(item).lower() not in NON_TOPIC_FOCUS_TERMS
        ]
        if len(paired_focus) >= 2:
            return " / ".join(paired_focus[:2])[:32]
        focus_concept = clean_display_text(resolution_context.get("focus_concept") or "")
        if resolution_context.get("followup_detected") and focus_concept:
            return focus_concept[:24]
        focus_labels = self._query_focus_labels(raw_query, limit=2)
        if len(focus_labels) >= 2:
            return " / ".join(focus_labels[:2])[:32]
        if focus_labels:
            return focus_labels[0][:24]
        related_kps = list(retrieval.get("related_kps") or [])
        if related_kps:
            return clean_display_text((related_kps[0] or {}).get("name") or "")[:24]
        previous_topic = clean_display_text((previous_memory or {}).get("active_topic") or "")
        if previous_topic and resolution_context.get("followup_detected"):
            return previous_topic[:24]
        return clean_display_text(raw_query)[:24]

    def _topic_signature(self, text: str) -> set[str]:
        return {
            token
            for token in self._query_focus_tokens(text)
            if len(token) > 1 and token not in FOLLOWUP_NON_TOPIC_TOKENS
        }

    def _topic_shifted(self, previous_topic: str, new_topic: str) -> bool:
        previous_clean = clean_display_text(previous_topic).lower()
        new_clean = clean_display_text(new_topic).lower()
        if not previous_clean or not new_clean or previous_clean == new_clean:
            return False
        previous_signature = self._topic_signature(previous_clean)
        new_signature = self._topic_signature(new_clean)
        if not previous_signature or not new_signature:
            return previous_clean != new_clean
        if previous_signature.issubset(new_signature) or new_signature.issubset(previous_signature):
            return False
        overlap = len(previous_signature & new_signature) / max(len(previous_signature), len(new_signature))
        return overlap < 0.5

    def _compose_session_memory_summary(
        self,
        raw_query: str,
        effective_query: str,
        answer: str,
        previous_memory: Optional[Dict[str, Any]],
        *,
        active_topic: str,
        llm_summary: str = "",
    ) -> str:
        previous_topic = clean_display_text((previous_memory or {}).get("active_topic") or "")
        previous_summary = clean_display_text((previous_memory or {}).get("session_summary") or "")
        request_hint = "代码示例" if self._is_code_generation_request(raw_query) else "概念理解"
        if COMPARISON_RE.search(raw_query):
            request_hint = "模型比较"
        if REFERENCE_RE.search(raw_query):
            request_hint = "讲义定位"
        answer_hint = clean_display_text(answer).replace("\n", " ")[:120]

        parts: List[str] = []
        if active_topic:
            parts.append(f"当前围绕“{active_topic}”继续复习。")
        if self._topic_shifted(previous_topic, active_topic):
            parts.append(f"讨论主题已从“{previous_topic}”切换到“{active_topic}”。")
        if effective_query:
            parts.append(f"最近的独立问题是：{effective_query[:120]}。")
        parts.append(f"最近问题偏向{request_hint}。")
        if answer_hint and not self._is_code_generation_request(raw_query):
            parts.append(f"最近结论：{answer_hint}")
        if llm_summary:
            parts.append(clean_display_text(llm_summary))

        summary = clean_display_text(" ".join(part for part in parts if part))
        if not summary:
            summary = previous_summary
        return summary[:560]

    def _fallback_session_memory_update(
        self,
        *,
        raw_query: str,
        answer_text: str,
        resolution_context: Dict[str, Any],
        retrieval: Dict[str, Any],
        task_profile: ChatTaskPlan,
        previous_memory: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        active_topic = self._derive_active_topic(
            clean_display_text(resolution_context.get("resolved_query") or raw_query),
            resolution_context,
            retrieval,
            task_profile,
            previous_memory=previous_memory,
        )
        return {
            "active_topic": active_topic,
            "session_summary": self._compose_session_memory_summary(
                raw_query,
                clean_display_text(resolution_context.get("resolved_query") or raw_query),
                answer_text,
                previous_memory,
                active_topic=active_topic,
            ),
        }

    def _build_session_memory(
        self,
        *,
        raw_query: str,
        answer_text: str,
        resolution_context: Dict[str, Any],
        retrieval: Dict[str, Any],
        task_profile: ChatTaskPlan,
        previous_memory: Optional[Dict[str, Any]] = None,
        mode: str = "assistant",
    ) -> Dict[str, Any]:
        previous = {
            "active_topic": clean_display_text((previous_memory or {}).get("active_topic") or ""),
            "session_summary": clean_display_text((previous_memory or {}).get("session_summary") or ""),
        }
        if mode == "assistant":
            return previous
        fallback = self._fallback_session_memory_update(
            raw_query=raw_query,
            answer_text=answer_text,
            resolution_context=resolution_context,
            retrieval=retrieval,
            task_profile=task_profile,
            previous_memory=previous_memory,
        )
        if self.kb._llm_client is None or not self.kb._llm_model:
            return fallback

        recent_context = textwrap.dedent(
            f"""
            最近独立问题：{clean_display_text(resolution_context.get("resolved_query") or raw_query) or '无'}
            当前解析出的主题：{clean_display_text(fallback.get("active_topic") or '') or '无'}
            旧主题：{previous['active_topic'] or '无'}
            旧摘要：{previous['session_summary'] or '无'}
            最新回答摘要：{clean_display_text(answer_text).replace(chr(10), ' ')[:220] or '无'}
            """
        ).strip()
        payload = self.kb.call_json_llm(
            "你负责维护课程问答会话的紧凑记忆。只输出 JSON。",
            f"""
请为当前会话生成记忆状态。只输出严格 JSON：
{{"session_summary":"...","active_topic":"..."}}

规则：
1. active_topic 必须是短语，准确描述当前会话主题，不要过宽。
2. session_summary 用 1 到 3 句中文，概括下一轮继续对话所需的上下文。
3. 如果这轮是短追问，优先继承并细化当前主题，而不是丢失主题。
4. 如果这轮明显切换到新课程主题，可以更新主题。
5. 不要输出文件名、页码、source id 或系统说明。

上下文：
{recent_context}
""".strip(),
            temperature=0.0,
            max_tokens=180,
        )
        if not isinstance(payload, dict):
            return fallback

        llm_topic = clean_display_text(payload.get("active_topic") or "")
        llm_summary = clean_display_text(payload.get("session_summary") or "")
        active_topic = llm_topic or clean_display_text(fallback.get("active_topic") or "")
        if not active_topic:
            active_topic = previous["active_topic"]
        session_summary = self._compose_session_memory_summary(
            raw_query,
            clean_display_text(resolution_context.get("resolved_query") or raw_query),
            answer_text,
            previous_memory,
            active_topic=active_topic,
            llm_summary=llm_summary,
        )
        return {
            "active_topic": active_topic[:120],
            "session_summary": session_summary[:560],
        }

    def _out_of_scope_response(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
    ) -> str:
        topic = clean_display_text(query)[:26]
        seed_text = f"{topic}::{self._repeat_count(query, history)}::{len(history)}"
        return _pick_variety(
            seed_text,
            [
                f"这个问题和当前深度学习课程无关，所以我先不直接回答“{topic}”。如果你愿意，可以继续问神经网络、CNN、Transformer、扩散模型或训练优化相关内容。",
                f"“{topic}”不属于这门课的问答范围，我先不展开。若你愿意，可以继续问卷积网络、反向传播、注意力机制、生成模型或框架使用。",
                f"这不是当前深度学习课程内的问题，我先不直接作答“{topic}”。你可以切回课程主题，我会继续帮你分析。",
            ],
        )

    def _build_result(
        self,
        *,
        raw_query: str,
        answer_text: str,
        citations: Sequence[Dict[str, Any]],
        related_kps: Sequence[Dict[str, Any]],
        mode: str,
        response_kind: str,
        resolution_context: Dict[str, Any],
        retrieval: Dict[str, Any],
        task_profile: ChatTaskPlan,
        previous_memory: Optional[Dict[str, Any]],
        session_mode: str,
    ) -> Dict[str, Any]:
        final_answer = self._sanitize_answer_text(answer_text)
        return {
            "answer": final_answer,
            "citations": list(citations or []),
            "related_kps": list(related_kps or []),
            "mode": mode,
            "response_kind": response_kind,
            "session_memory": self._build_session_memory(
                raw_query=raw_query,
                answer_text=final_answer,
                resolution_context=resolution_context,
                retrieval=retrieval,
                task_profile=task_profile,
                previous_memory=previous_memory,
                mode=session_mode,
            ),
        }

    def answer(
        self,
        query: str,
        *,
        history: Optional[Sequence[Dict[str, str]]] = None,
        top_k: int = 5,
        session_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        history = list(history or [])
        raw_query = clean_display_text(query)
        resolution_context = self._resolve_query_with_context(
            raw_query,
            history,
            session_memory=session_memory,
        )
        resolved_query = clean_display_text(resolution_context.get("resolved_query") or raw_query)

        special_intent = self._assistant_intent(raw_query)
        special_response = self._assistant_response(raw_query, history, intent=special_intent)
        if special_response:
            return self._build_result(
                raw_query=raw_query,
                answer_text=special_response,
                citations=[],
                related_kps=[],
                mode="assistant",
                response_kind=self._response_kind_for_intent(special_intent),
                resolution_context=resolution_context,
                retrieval={},
                task_profile=ChatTaskPlan(intent="assistant"),
                previous_memory=session_memory,
                session_mode="assistant",
            )

        if not self._is_in_scope(raw_query, resolved_query, resolution_context):
            return self._build_result(
                raw_query=raw_query,
                answer_text=self._out_of_scope_response(raw_query, history),
                citations=[],
                related_kps=[],
                mode="assistant",
                response_kind="out_of_scope",
                resolution_context=resolution_context,
                retrieval={},
                task_profile=ChatTaskPlan(intent="assistant"),
                previous_memory=session_memory,
                session_mode="assistant",
            )

        task_profile = self._task_profile(raw_query, resolved_query)
        retrieval_query = self._effective_retrieval_query(raw_query, resolution_context, task_profile)
        retrieval = self.kb.retrieve(
            retrieval_query,
            history=history,
            session_memory=session_memory,
            top_k=top_k,
            extra_queries=self._retrieval_extra_queries(raw_query, resolution_context, task_profile),
        )

        if task_profile.is_reference_request:
            reference_citations = self._reference_citations(retrieval_query, retrieval)
            payload = dict(
                self._reference_guidance_payload(
                    raw_query,
                    retrieval_query,
                    retrieval,
                    resolution_context,
                    task_profile,
                )
            )
            payload["used_sources"] = [
                clean_display_text(item.get("citation_id") or "").upper()
                for item in reference_citations
                if clean_display_text(item.get("citation_id") or "")
            ]
            return self._build_result(
                raw_query=raw_query,
                answer_text=payload.get("answer") or "",
                citations=reference_citations,
                related_kps=retrieval.get("related_kps") or [],
                mode="reference_guidance",
                response_kind="reference_guidance",
                resolution_context=resolution_context,
                retrieval=retrieval,
                task_profile=task_profile,
                previous_memory=session_memory,
                session_mode="student",
            )

        if task_profile.is_code_request:
            template_payload = build_code_template(
                f"{raw_query}\n{retrieval_query}",
                code_only=bool(task_profile.get("is_code_only_request")),
                variant_index=self._repeat_count(raw_query, history),
            )
            if template_payload:
                return self._build_result(
                    raw_query=raw_query,
                    answer_text=template_payload.get("answer") or "",
                    citations=[],
                    related_kps=retrieval.get("related_kps") or [],
                    mode="template_code",
                    response_kind="course_answer",
                    resolution_context=resolution_context,
                    retrieval=retrieval,
                    task_profile=task_profile,
                    previous_memory=session_memory,
                    session_mode="student",
                )

        payload = self._generate_payload(
            raw_query=raw_query,
            resolution_context={**resolution_context, "resolved_query": retrieval_query},
            retrieval=retrieval,
            task_profile=task_profile,
            history=history,
        )
        if not payload:
            return self._build_result(
                raw_query=raw_query,
                answer_text=self.kb.answer_from_chunks(retrieval_query, retrieval),
                citations=list((retrieval.get("citations") or [])[:1]),
                related_kps=retrieval.get("related_kps") or [],
                mode="fallback_unified",
                response_kind="course_answer",
                resolution_context=resolution_context,
                retrieval=retrieval,
                task_profile=task_profile,
                previous_memory=session_memory,
                session_mode="student",
            )

        finalized = self._finalize_course_payload(
            raw_query=raw_query,
            retrieval_query=retrieval_query,
            history=history,
            retrieval=retrieval,
            task_profile=task_profile,
            payload=payload,
        )
        return self._build_result(
            raw_query=raw_query,
            answer_text=finalized.get("answer") or "",
            citations=finalized.get("citations") or [],
            related_kps=finalized.get("related_kps") or [],
            mode=clean_display_text(finalized.get("mode") or "") or "llm_unified",
            response_kind=clean_display_text(finalized.get("response_kind") or "") or "course_answer",
            resolution_context=resolution_context,
            retrieval=retrieval,
            task_profile=task_profile,
            previous_memory=session_memory,
            session_mode="student",
        )
