from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import fitz
from pypdf import PdfReader

try:
    from langchain_core.documents import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except Exception:
    Document = None
    HuggingFaceEmbeddings = None
    FAISS = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env_loader import load_project_env
from deep_learning_rag.course_content import COURSE_TAXONOMY
from deep_learning_rag.question_bank import QUESTION_BLUEPRINTS

load_project_env()


DEFAULT_COURSE_ROOT = ROOT_DIR / "deep_learning_materials"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts_full_course"
DEFAULT_EMBED_MODEL = os.getenv("DEEP_LEARNING_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SUPPORTED_EXTENSIONS = {".pdf"}
CHUNK_SIZE = 880
CHUNK_OVERLAP = 140
MIN_QUESTION_COUNT = 6
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*")
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
WS_RE = re.compile(r"\s+")
WEEK_RE = re.compile(r"week\s*(\d+)", re.IGNORECASE)
DISPLAY_REPLACEMENTS = {
    "\u2022": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00a0": " ",
    "\u2212": "-",
}
STOPWORDS = {
    "的", "了", "和", "与", "及", "在", "是", "对", "中", "上", "下", "把", "将", "并", "或",
    "the", "and", "for", "with", "from", "that", "this", "into", "about", "which", "what",
    "why", "how", "when", "where", "used", "using", "use", "course", "lecture", "week",
    "question", "questions", "deep", "learning", "model", "models",
}


@dataclass
class RawDocument:
    doc_id: str
    relative_path: str
    source_type: str
    week: str
    unit_type: str
    unit_index: int
    title: str
    text: str


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    relative_path: str
    source_type: str
    week: str
    title: str
    unit_type: str
    unit_index: int
    chunk_index: int
    text: str


@dataclass
class KnowledgePoint:
    kp_id: str
    name: str
    description: str
    weeks: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    scenario: str = ""
    source_files: List[str] = field(default_factory=list)
    support_chunk_ids: List[str] = field(default_factory=list)
    support_preview: List[str] = field(default_factory=list)


@dataclass
class QuestionRecord:
    question_id: str
    kp_id: str
    kp_name: str
    question_type: str
    question: str
    answer: str
    explanation: str
    options: List[str] = field(default_factory=list)
    correct_option: Optional[str] = None
    source_chunk_ids: List[str] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)
    review_refs: List[Dict[str, Any]] = field(default_factory=list)
    image_path: Optional[str] = None
    image_caption: Optional[str] = None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def short_hash(text: str, length: int = 12) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", text.lower()).strip("-")
    return cleaned or short_hash(text)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(_sanitize_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_sanitize_jsonable(row), ensure_ascii=False) + "\n")


def _strip_invalid_unicode(text: str) -> str:
    return str(text or "").encode("utf-8", "ignore").decode("utf-8", "ignore")


def _sanitize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_jsonable(item) for item in value]
    if isinstance(value, str):
        return _strip_invalid_unicode(value)
    return value


def clean_display_text(text: Any) -> str:
    value = _strip_invalid_unicode(str(text or ""))
    if not value:
        return ""
    for source, target in DISPLAY_REPLACEMENTS.items():
        value = value.replace(source, target)
    value = value.replace("\x00", " ")
    value = WS_RE.sub(" ", value)
    return value.strip()


def normalize_text(text: Any) -> str:
    lines: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = clean_display_text(raw_line)
        if not line:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def safe_snippet(text: Any, limit: int = 180) -> str:
    compact = clean_display_text(text)
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


REVIEW_NOISE_PATTERNS = (
    "school of computer science and technology",
    "计算机科学与技术学院",
    "university of chinese academy of sciences",
    "中国科学院大学",
)


def review_clean_lines(text: Any, limit: int = 6) -> List[str]:
    rows: List[str] = []
    seen: set[str] = set()
    for raw_line in normalize_text(text).splitlines():
        line = clean_display_text(raw_line).strip(" -|:;,.")
        if not line:
            continue
        lowered = line.lower()
        if lowered.isdigit():
            continue
        if any(pattern in lowered for pattern in REVIEW_NOISE_PATTERNS):
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        rows.append(line)
        if len(rows) >= limit:
            break
    return rows


def review_section_label(chunk: "ChunkRecord") -> str:
    title = clean_display_text(getattr(chunk, "title", "")).strip(" -|:;,.")
    if title:
        lowered = title.lower()
        if not any(pattern in lowered for pattern in REVIEW_NOISE_PATTERNS):
            return safe_snippet(title, limit=80)
    text_lines = review_clean_lines(getattr(chunk, "text", ""), limit=4)
    if text_lines:
        return safe_snippet(text_lines[0], limit=80)
    return "相关讲义部分"


def review_excerpt_text(chunk: "ChunkRecord", limit: int = 220) -> str:
    lines = review_clean_lines(getattr(chunk, "text", ""), limit=4)
    excerpt = " ".join(lines)
    if not excerpt:
        excerpt = clean_display_text(getattr(chunk, "text", ""))
    return safe_snippet(excerpt, limit=limit)


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


def parse_week_number(label: Any) -> int:
    match = WEEK_RE.search(str(label or ""))
    return int(match.group(1)) if match else 0


def iter_course_files(course_root: Path, max_week: Optional[int] = None) -> List[Path]:
    rows: List[Path] = []
    for path in sorted(course_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        relative = path.relative_to(course_root)
        week_no = parse_week_number(relative.parts[0] if relative.parts else "")
        if max_week is not None and week_no and week_no > max_week:
            continue
        rows.append(path)
    return rows


def detect_week(relative_path: Path) -> str:
    for part in relative_path.parts:
        if WEEK_RE.fullmatch(part):
            return part
    return "Week0"


def guess_title(text: str, fallback: str) -> str:
    for line in text.splitlines()[:8]:
        cleaned = line.strip(" -:|")
        if 3 <= len(cleaned) <= 120:
            return cleaned
    return fallback


def extract_pdf_units(file_path: Path, relative_path: Path) -> List[RawDocument]:
    reader = PdfReader(str(file_path))
    week = detect_week(relative_path)
    rows: List[RawDocument] = []
    for index, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if not text:
            continue
        rows.append(
            RawDocument(
                doc_id=f"raw-{short_hash(f'{relative_path.as_posix()}|{index}')}",
                relative_path=relative_path.as_posix(),
                source_type="pdf",
                week=week,
                unit_type="page",
                unit_index=index,
                title=guess_title(text, file_path.stem),
                text=text,
            )
        )
    return rows


def extract_raw_documents(course_root: Path, max_week: Optional[int] = None) -> Tuple[List[RawDocument], List[Dict[str, str]]]:
    raw_documents: List[RawDocument] = []
    issues: List[Dict[str, str]] = []
    for file_path in iter_course_files(course_root, max_week=max_week):
        relative_path = file_path.relative_to(course_root)
        try:
            docs = extract_pdf_units(file_path, relative_path)
            raw_documents.extend(docs)
            if not docs:
                issues.append({"relative_path": relative_path.as_posix(), "issue": "No extractable text found"})
        except Exception as exc:
            issues.append({"relative_path": relative_path.as_posix(), "issue": f"{type(exc).__name__}: {exc}"})
    return raw_documents, issues


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    compact = normalize_text(text)
    if not compact:
        return []
    if len(compact) <= chunk_size:
        return [compact]
    pieces: List[str] = []
    start = 0
    while start < len(compact):
        end = min(len(compact), start + chunk_size)
        window = compact[start:end]
        if end < len(compact):
            split_at = max(window.rfind("\n"), window.rfind(" "), window.rfind("。"))
            if split_at > int(chunk_size * 0.55):
                end = start + split_at + 1
                window = compact[start:end]
        pieces.append(window.strip())
        if end >= len(compact):
            break
        start = max(0, end - overlap)
    return [piece for piece in pieces if piece]


def build_chunk_records(raw_documents: Sequence[RawDocument]) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []
    for doc in raw_documents:
        for chunk_index, piece in enumerate(split_text(doc.text), start=1):
            records.append(
                ChunkRecord(
                    chunk_id=f"chunk-{short_hash(f'{doc.doc_id}|{chunk_index}')}",
                    doc_id=doc.doc_id,
                    relative_path=doc.relative_path,
                    source_type=doc.source_type,
                    week=doc.week,
                    title=doc.title,
                    unit_type=doc.unit_type,
                    unit_index=doc.unit_index,
                    chunk_index=chunk_index,
                    text=piece,
                )
            )
    return records


def support_sort_key(score: float, chunk: ChunkRecord) -> Tuple[float, int, int, str]:
    return (-score, parse_week_number(chunk.week), int(chunk.unit_index or 0), chunk.relative_path)


def build_knowledge_points(chunk_records: Sequence[ChunkRecord]) -> List[KnowledgePoint]:
    knowledge_points: List[KnowledgePoint] = []
    for entry in COURSE_TAXONOMY:
        scored: List[Tuple[float, ChunkRecord]] = []
        for chunk in chunk_records:
            blob = "\n".join([chunk.title, chunk.relative_path, chunk.text]).lower()
            title_blob = "\n".join([chunk.title, chunk.relative_path]).lower()
            hits = 0
            score = 0.0
            for keyword in entry["keywords"]:
                term = str(keyword).lower()
                if term in blob:
                    hits += 1
                    score += 3.0 if term in title_blob else 1.5
            if not hits:
                continue
            if chunk.week in entry.get("weeks", []):
                score += 2.0
            else:
                score -= abs(parse_week_number(chunk.week) - parse_week_number(entry.get("weeks", ["Week0"])[0])) * 0.35
            scored.append((score, chunk))

        if not scored:
            continue

        scored.sort(key=lambda pair: support_sort_key(pair[0], pair[1]))
        selected: List[ChunkRecord] = []
        seen_units: set[Tuple[str, int]] = set()
        for score, chunk in scored:
            unit_key = (chunk.relative_path, int(chunk.unit_index or 0))
            if unit_key in seen_units:
                continue
            selected.append(chunk)
            seen_units.add(unit_key)
            if len(selected) >= 10:
                break

        knowledge_points.append(
            KnowledgePoint(
                kp_id=f"kp-{slugify(entry['name'])}",
                name=entry["name"],
                description=entry["description"],
                weeks=list(entry.get("weeks", [])),
                keywords=list(entry.get("keywords", [])),
                scenario=str(entry.get("scenario") or ""),
                source_files=sorted({chunk.relative_path for chunk in selected}),
                support_chunk_ids=[chunk.chunk_id for chunk in selected],
                support_preview=[safe_snippet(chunk.text, limit=180) for chunk in selected[:3]],
            )
        )
    return knowledge_points


def lexical_rank(query: str, items: Sequence[Any], text_getter, top_k: int = 6) -> List[Tuple[float, Any]]:
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


def build_options(correct_text: str, distractors: Sequence[str], seed_key: str) -> Tuple[List[str], str]:
    unique: List[str] = []
    seen: set[str] = set()
    for item in [correct_text, *distractors]:
        cleaned = clean_display_text(item)
        if not cleaned or cleaned in seen:
            continue
        unique.append(cleaned)
        seen.add(cleaned)
    if clean_display_text(correct_text) not in seen:
        unique.insert(0, clean_display_text(correct_text))
    cleaned_correct = clean_display_text(correct_text)
    pool = [item for item in unique if item != cleaned_correct]
    while len(pool) < 3:
        pool.append(f"干扰项 {len(pool) + 1}")
    rng = random.Random(short_hash(seed_key, length=8))
    choices = [cleaned_correct, *pool[:3]]
    rng.shuffle(choices)
    labels = ["A", "B", "C", "D"]
    options = [f"{label}. {text}" for label, text in zip(labels, choices)]
    correct_option = labels[choices.index(cleaned_correct)]
    return options, correct_option


def parsed_option_lookup(options: Sequence[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for option in options:
        match = re.match(r"^\s*([A-D])\.\s*(.+?)\s*$", str(option))
        if match:
            lookup[match.group(1).upper()] = match.group(2).strip()
    return lookup


def build_review_refs(
    question_text: str,
    answer_text: str,
    explanation: str,
    review_chunks: Sequence[ChunkRecord],
    review_terms: Optional[Sequence[str]] = None,
    limit: int = 2,
) -> List[Dict[str, Any]]:
    query = " ".join(
        item for item in [
            clean_display_text(question_text),
            clean_display_text(answer_text),
            clean_display_text(explanation),
            " ".join(clean_display_text(term) for term in (review_terms or [])),
        ]
        if item
    )
    ranked = lexical_rank(query, review_chunks, lambda chunk: f"{chunk.title}\n{chunk.text}", top_k=max(limit * 3, 6))
    refs: List[Dict[str, Any]] = []
    seen_units: set[Tuple[str, int]] = set()
    for _score, chunk in ranked:
        unit_key = (chunk.relative_path, int(chunk.unit_index or 0))
        if unit_key in seen_units:
            continue
        seen_units.add(unit_key)
        refs.append(
            {
                "source": chunk.relative_path,
                "unit_type": chunk.unit_type,
                "unit_index": int(chunk.unit_index or 0),
                "chunk_index": int(chunk.chunk_index or 0),
                "section": review_section_label(chunk),
                "excerpt": review_excerpt_text(chunk, limit=220),
                "location": f"第 {int(chunk.unit_index or 0)} 页",
            }
        )
        if len(refs) >= limit:
            break
    return refs


def find_best_review_chunk(review_chunks: Sequence[ChunkRecord], review_terms: Sequence[str]) -> Optional[ChunkRecord]:
    if not review_chunks:
        return None
    if not review_terms:
        return review_chunks[0]
    query = " ".join(clean_display_text(item) for item in review_terms if clean_display_text(item))
    ranked = lexical_rank(query, review_chunks, lambda chunk: f"{chunk.title}\n{chunk.text}", top_k=1)
    return ranked[0][1] if ranked else review_chunks[0]


def build_explicit_review_ref(
    review_chunks: Sequence[ChunkRecord],
    relative_path: str,
    page_number: int,
) -> Optional[Dict[str, Any]]:
    target_path = clean_display_text(relative_path)
    target_page = int(page_number or 0)
    if not target_path or target_page <= 0:
        return None
    for chunk in review_chunks:
        if (
            chunk.unit_type == "page"
            and clean_display_text(chunk.relative_path) == target_path
            and int(chunk.unit_index or 0) == target_page
        ):
            return {
                "source": chunk.relative_path,
                "unit_type": chunk.unit_type,
                "unit_index": int(chunk.unit_index or 0),
                "chunk_index": int(chunk.chunk_index or 0),
                "section": review_section_label(chunk),
                "excerpt": review_excerpt_text(chunk, limit=220),
                "location": f"第 {int(chunk.unit_index or 0)} 页",
            }
    return {
        "source": target_path,
        "unit_type": "page",
        "unit_index": target_page,
        "chunk_index": 0,
        "section": f"{Path(target_path).name} 第 {target_page} 页",
        "excerpt": "",
        "location": f"第 {target_page} 页",
    }


def render_pdf_page_image(course_root: Path, relative_path: str, page_number: int, artifact_dir: Path) -> Optional[str]:
    try:
        pdf_path = course_root / Path(relative_path)
        if not pdf_path.exists():
            return None
        output_dir = ensure_dir(artifact_dir / "images" / slugify(Path(relative_path).stem))
        output_path = output_dir / f"page-{int(page_number):03d}.png"
        if not output_path.exists():
            doc = fitz.open(pdf_path)
            try:
                page = doc.load_page(max(int(page_number) - 1, 0))
                pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
                pix.save(output_path)
            finally:
                doc.close()
        return output_path.relative_to(artifact_dir).as_posix()
    except Exception:
        return None


def build_question_from_blueprint(
    course_root: Path,
    artifact_dir: Path,
    kp: KnowledgePoint,
    blueprint: Dict[str, Any],
    index: int,
    review_chunks: Sequence[ChunkRecord],
    source_chunk_ids: Sequence[str],
    source_files: Sequence[str],
) -> QuestionRecord:
    prompt = clean_display_text(blueprint.get("prompt"))
    correct = clean_display_text(blueprint.get("correct"))
    distractors = [clean_display_text(item) for item in blueprint.get("distractors", []) if clean_display_text(item)]
    explanation = clean_display_text(blueprint.get("explanation"))
    review_terms = [clean_display_text(item) for item in blueprint.get("review_terms", []) if clean_display_text(item)]
    image_review_terms = [
        clean_display_text(item)
        for item in blueprint.get("image_review_terms", [])
        if clean_display_text(item)
    ] or review_terms
    options, correct_option = build_options(correct, distractors, f"{kp.kp_id}|{index}|{prompt}")
    review_refs = build_review_refs(prompt, correct, explanation, review_chunks, review_terms=review_terms)
    image_path = None
    image_caption = None
    if blueprint.get("use_image"):
        explicit_image_source = clean_display_text(blueprint.get("image_source"))
        explicit_image_page = int(blueprint.get("image_page") or 0)
        explicit_image_caption = clean_display_text(blueprint.get("image_caption"))
        explicit_ref = build_explicit_review_ref(review_chunks, explicit_image_source, explicit_image_page)
        if explicit_ref:
            deduped_refs = [explicit_ref]
            explicit_key = (clean_display_text(explicit_ref.get("source")), int(explicit_ref.get("unit_index") or 0))
            for ref in review_refs:
                unit_key = (clean_display_text(ref.get("source")), int(ref.get("unit_index") or 0))
                if unit_key == explicit_key:
                    continue
                deduped_refs.append(ref)
            review_refs = deduped_refs[: max(len(review_refs), 2)]
        if explicit_image_source and explicit_image_page > 0:
            image_path = render_pdf_page_image(course_root, explicit_image_source, explicit_image_page, artifact_dir)
            if image_path:
                image_caption = explicit_image_caption or f"{Path(explicit_image_source).name} 第 {explicit_image_page} 页"
        if not image_path:
            best_chunk = find_best_review_chunk(review_chunks, image_review_terms)
            if best_chunk is not None and best_chunk.unit_type == "page":
                image_path = render_pdf_page_image(course_root, best_chunk.relative_path, best_chunk.unit_index, artifact_dir)
                if image_path:
                    image_caption = explicit_image_caption or f"{Path(best_chunk.relative_path).name} 第 {best_chunk.unit_index} 页"
    return QuestionRecord(
        question_id=f"{kp.kp_id}-q{index}",
        kp_id=kp.kp_id,
        kp_name=kp.name,
        question_type="multiple_choice",
        question=prompt,
        answer=correct,
        explanation=explanation,
        options=options,
        correct_option=correct_option,
        source_chunk_ids=list(source_chunk_ids),
        source_files=list(source_files),
        review_refs=review_refs,
        image_path=image_path,
        image_caption=image_caption,
    )


def choose_other_kps(current_kp: KnowledgePoint, knowledge_points: Sequence[KnowledgePoint]) -> List[KnowledgePoint]:
    return [item for item in knowledge_points if item.kp_id != current_kp.kp_id]


def choose_distractor_texts(correct_text: str, candidates: Sequence[str], seed_key: str, limit: int = 3) -> List[str]:
    cleaned_correct = clean_display_text(correct_text)
    pool = []
    seen: set[str] = {cleaned_correct}
    for candidate in candidates:
        cleaned = clean_display_text(candidate)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        pool.append(cleaned)
    rng = random.Random(short_hash(seed_key, length=8))
    rng.shuffle(pool)
    return pool[:limit]


def fallback_questions_for_kp(
    kp: KnowledgePoint,
    knowledge_points: Sequence[KnowledgePoint],
    review_chunks: Sequence[ChunkRecord],
    source_chunk_ids: Sequence[str],
    source_files: Sequence[str],
    start_index: int,
    question_count: int,
) -> List[QuestionRecord]:
    other_kps = choose_other_kps(kp, knowledge_points)
    other_names = [item.name for item in other_kps]
    other_scenarios = [item.scenario for item in other_kps if item.scenario]
    other_descriptions = [item.description for item in other_kps]
    keyword_candidates = [
        "、".join(item.keywords[:3]) for item in other_kps if item.keywords[:3]
    ]

    specs = [
        {
            "prompt": f"关于“{kp.name}”，下列说法哪一项最准确？",
            "correct": kp.description,
            "distractors": choose_distractor_texts(kp.description, other_descriptions, kp.kp_id + "-desc"),
            "explanation": f"{kp.name} 这一知识点在课程中的核心内容就是：{kp.description}",
        },
        {
            "prompt": f"如果一个任务的核心需求是“{kp.scenario}”，最应优先复习哪个知识点？",
            "correct": kp.name,
            "distractors": choose_distractor_texts(kp.name, other_names, kp.kp_id + "-name"),
            "explanation": f"该场景与 {kp.name} 的课程定位直接对应。",
        },
        {
            "prompt": f"在本课程中，{kp.name} 最直接服务于哪一类问题？",
            "correct": kp.scenario or kp.description,
            "distractors": choose_distractor_texts(kp.scenario or kp.description, other_scenarios or other_descriptions, kp.kp_id + "-scenario"),
            "explanation": f"这是 {kp.name} 在课程中的主要应用场景。",
        },
        {
            "prompt": f"下面哪组关键词与“{kp.name}”最相关？",
            "correct": "、".join(kp.keywords[:3]),
            "distractors": choose_distractor_texts("、".join(kp.keywords[:3]), keyword_candidates, kp.kp_id + "-keywords"),
            "explanation": f"{kp.name} 的复习应优先抓住这些关键词：{'、'.join(kp.keywords[:3])}。",
        },
        {
            "prompt": f"下列哪一项最能概括“{kp.name}”的核心复习重点？",
            "correct": kp.description,
            "distractors": choose_distractor_texts(kp.description, other_descriptions, kp.kp_id + "-review"),
            "explanation": f"这项内容最能概括 {kp.name} 的课程重点。",
        },
    ]

    questions: List[QuestionRecord] = []
    next_index = start_index
    for spec in specs:
        options, correct_option = build_options(spec["correct"], spec["distractors"], f"{kp.kp_id}|fallback|{next_index}")
        questions.append(
            QuestionRecord(
                question_id=f"{kp.kp_id}-q{next_index}",
                kp_id=kp.kp_id,
                kp_name=kp.name,
                question_type="multiple_choice",
                question=spec["prompt"],
                answer=spec["correct"],
                explanation=spec["explanation"],
                options=options,
                correct_option=correct_option,
                source_chunk_ids=list(source_chunk_ids),
                source_files=list(source_files),
                review_refs=build_review_refs(spec["prompt"], spec["correct"], spec["explanation"], review_chunks, review_terms=kp.keywords[:4]),
            )
        )
        next_index += 1
        if len(questions) >= question_count:
            break
    return questions


def generate_questions(
    course_root: Path,
    artifact_dir: Path,
    knowledge_points: Sequence[KnowledgePoint],
    chunk_records: Sequence[ChunkRecord],
    question_count: int,
) -> List[QuestionRecord]:
    chunk_map = {chunk.chunk_id: chunk for chunk in chunk_records}
    questions: List[QuestionRecord] = []
    for kp in knowledge_points:
        support_chunks = [chunk_map[item] for item in kp.support_chunk_ids if item in chunk_map]
        review_chunks = [chunk for chunk in chunk_records if chunk.relative_path in set(kp.source_files)] or support_chunks
        source_chunk_ids = [chunk.chunk_id for chunk in support_chunks[:8]]
        source_files = sorted({chunk.relative_path for chunk in support_chunks[:8]})
        built: List[QuestionRecord] = []
        for index, blueprint in enumerate(QUESTION_BLUEPRINTS.get(kp.name, []), start=1):
            built.append(
                build_question_from_blueprint(
                    course_root=course_root,
                    artifact_dir=artifact_dir,
                    kp=kp,
                    blueprint=blueprint,
                    index=index,
                    review_chunks=review_chunks,
                    source_chunk_ids=source_chunk_ids,
                    source_files=source_files,
                )
            )
            if len(built) >= question_count:
                break
        if len(built) < question_count:
            built.extend(
                fallback_questions_for_kp(
                    kp=kp,
                    knowledge_points=knowledge_points,
                    review_chunks=review_chunks,
                    source_chunk_ids=source_chunk_ids,
                    source_files=source_files,
                    start_index=len(built) + 1,
                    question_count=question_count - len(built),
                )
            )
        questions.extend(built[:question_count])
    return questions


def dense_stack_available() -> bool:
    return all(item is not None for item in [Document, HuggingFaceEmbeddings, FAISS])


def get_embeddings(model_name: str) -> Any:
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})


def build_dense_index(chunk_records: Sequence[ChunkRecord], artifact_dir: Path, model_name: str) -> Dict[str, Any]:
    index_dir = artifact_dir / "chunk_index"
    if not dense_stack_available():
        return {"status": "skipped", "reason": "langchain/faiss not available", "index_dir": str(index_dir)}
    try:
        documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "relative_path": chunk.relative_path,
                    "week": chunk.week,
                    "title": chunk.title,
                    "unit_type": chunk.unit_type,
                    "unit_index": chunk.unit_index,
                    "chunk_index": chunk.chunk_index,
                    "source_type": chunk.source_type,
                },
            )
            for chunk in chunk_records
        ]
        vector_store = FAISS.from_documents(documents, get_embeddings(model_name))
        ensure_dir(index_dir)
        vector_store.save_local(str(index_dir))
        return {"status": "built", "count": len(documents), "index_dir": str(index_dir), "model_name": model_name}
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}", "index_dir": str(index_dir), "model_name": model_name}


def inventory_summary(course_root: Path, raw_documents: Sequence[RawDocument], issues: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    week_counts = Counter(doc.week for doc in raw_documents)
    source_counts = Counter(Path(doc.relative_path).name for doc in raw_documents)
    return {
        "course_root": str(course_root),
        "file_count": len(iter_course_files(course_root)),
        "raw_document_count": len(raw_documents),
        "weeks": dict(sorted(week_counts.items())),
        "sources": dict(sorted(source_counts.items())),
        "issues": list(issues),
    }


def build_pipeline(
    course_root: Path,
    artifact_dir: Path,
    question_count: int = MIN_QUESTION_COUNT,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    max_week: Optional[int] = None,
) -> Dict[str, Any]:
    question_count = max(MIN_QUESTION_COUNT, int(question_count or MIN_QUESTION_COUNT))
    ensure_dir(artifact_dir)

    raw_documents, issues = extract_raw_documents(course_root, max_week=max_week)
    chunk_records = build_chunk_records(raw_documents)
    knowledge_points = build_knowledge_points(chunk_records)
    questions = generate_questions(course_root, artifact_dir, knowledge_points, chunk_records, question_count)
    dense_meta = build_dense_index(chunk_records, artifact_dir, embedding_model)

    inventory = inventory_summary(course_root, raw_documents, issues)
    write_json(artifact_dir / "inventory.json", inventory)
    write_jsonl(artifact_dir / "raw_documents.jsonl", (asdict(item) for item in raw_documents))
    write_jsonl(artifact_dir / "chunks.jsonl", (asdict(item) for item in chunk_records))
    write_json(artifact_dir / "knowledge_points.json", [asdict(item) for item in knowledge_points])
    write_json(artifact_dir / "questions.json", [asdict(item) for item in questions])
    write_json(
        artifact_dir / "build_meta.json",
        {
            "course_root": str(course_root),
            "artifact_dir": str(artifact_dir),
            "knowledge_point_count": len(knowledge_points),
            "question_count": len(questions),
            "dense_index": dense_meta,
            "embedding_model": embedding_model,
            "max_week": max_week,
        },
    )

    return {
        "raw_document_count": len(raw_documents),
        "chunk_count": len(chunk_records),
        "knowledge_point_count": len(knowledge_points),
        "question_count": len(questions),
        "issue_count": len(issues),
        "dense_index": dense_meta,
        "artifact_dir": str(artifact_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="构建深度学习课程的 RAG 知识库与题库数据。")
    parser.add_argument("--course-root", default=str(DEFAULT_COURSE_ROOT))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--question-count", type=int, default=MIN_QUESTION_COUNT)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--max-week", type=int, default=None)
    args = parser.parse_args()

    result = build_pipeline(
        course_root=Path(args.course_root),
        artifact_dir=Path(args.artifact_dir),
        question_count=args.question_count,
        embedding_model=args.embedding_model,
        max_week=args.max_week,
    )
    sys.stdout.buffer.write((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8", "replace"))


if __name__ == "__main__":
    main()
