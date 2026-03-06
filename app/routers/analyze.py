from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import require_app_key
from app.services.pdf_extract import extract_text_from_pdf

router = APIRouter(prefix="/v1", tags=["analyze"])

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+.#/-]*")
_SPACED_LETTERS = re.compile(r"(?:\b[A-Za-z]\b(?:\s+|$)){3,}")

_STOP = {
    "and", "or", "the", "a", "an", "to", "of", "in", "on", "for", "with", "as", "is", "are", "be", "by", "at",
    "we", "you", "your", "our", "they", "their", "need", "needs", "required", "requirements", "must", "should", "will",
    "experience", "years", "year", "role", "job", "work", "working", "ability", "skills", "skill", "strong", "hands", "knowledge",
    "looking", "seek", "seeking", "hiring", "candidate", "position", "responsibilities", "responsibility", "plus", "preferred",
    "team", "teams", "using", "use", "build", "built", "develop", "developed", "support", "supporting", "good", "excellent",
}

TECH_PATTERNS: dict[str, tuple[str, ...]] = {
    "python": (r"\bpython\b",),
    "java": (r"\bjava\b",),
    "kotlin": (r"\bkotlin\b",),
    "javascript": (r"\bjavascript\b", r"\bjs\b"),
    "typescript": (r"\btypescript\b", r"\bts\b"),
    "sql": (r"\bsql\b", r"structured query language"),
    "fastapi": (r"\bfastapi\b",),
    "flask": (r"\bflask\b",),
    "django": (r"\bdjango\b",),
    "node.js": (r"\bnode\b", r"\bnode\.js\b", r"\bnodejs\b"),
    "react": (r"\breact\b",),
    "android": (r"\bandroid\b",),
    "jetpack compose": (r"jetpack compose", r"\bcompose\b"),
    "retrofit": (r"\bretrofit\b",),
    "okhttp": (r"\bokhttp\b",),
    "coroutines": (r"\bcoroutines\b",),
    "rest api": (r"\brest\b", r"\brest api\b", r"\brestful\b"),
    "graphql": (r"\bgraphql\b",),
    "postgresql": (r"\bpostgres(?:ql)?\b",),
    "mysql": (r"\bmysql\b",),
    "sqlite": (r"\bsqlite\b",),
    "mongodb": (r"\bmongodb\b",),
    "redis": (r"\bredis\b",),
    "docker": (r"\bdocker\b",),
    "kubernetes": (r"\bkubernetes\b", r"\bk8s\b"),
    "aws": (r"\baws\b", r"amazon web services"),
    "gcp": (r"\bgcp\b", r"google cloud"),
    "azure": (r"\bazure\b",),
    "linux": (r"\blinux\b",),
    "git": (r"\bgit\b",),
    "github actions": (r"github actions",),
    "ci/cd": (r"\bci/cd\b", r"continuous integration", r"continuous delivery", r"continuous deployment"),
    "microservices": (r"\bmicroservices?\b",),
    "distributed systems": (r"distributed systems?", r"distributed architecture"),
    "machine learning": (r"machine learning", r"\bml\b"),
    "llm": (r"\bllm\b", r"large language model"),
    "rag": (r"\brag\b", r"retrieval augmented generation"),
    "vector db": (r"vector db", r"vector database", r"embedding"),
    "tableau": (r"\btableau\b",),
    "power bi": (r"power bi",),
}

SOFT_PATTERNS: dict[str, tuple[str, ...]] = {
    "communication": (r"communication", r"communicate", r"stakeholder", r"presentation", r"present findings"),
    "leadership": (r"leadership", r"led team", r"mentor", r"managed team", r"ownership"),
    "teamwork": (r"teamwork", r"collaborat", r"cross-functional"),
    "problem solving": (r"problem solving", r"solve", r"debug", r"troubleshoot"),
    "analytical thinking": (r"analytical", r"analysis", r"data-driven"),
    "adaptability": (r"adapt", r"fast-paced", r"learn quickly"),
}

TRANSFERABLE_HINTS: dict[str, tuple[str, ...]] = {
    "distributed systems": ("microservices", "rest api", "docker", "postgresql", "backend"),
    "leadership": ("mentor", "managed", "ownership", "led"),
    "communication": ("stakeholder", "collaborated", "presented"),
    "problem solving": ("debug", "optimiz", "improv", "fix"),
}


class AnalyzeRequest(BaseModel):
    resume_text: str
    jd_text: str


@router.post("/analyze")
def analyze(payload: AnalyzeRequest, _=Depends(require_app_key)):
    return _analyze_text(payload.resume_text, payload.jd_text)


@router.post("/analyze/pdf")
async def analyze_pdf(
    jd_text: str = Form(...),
    debug: bool = Form(False),
    resume: UploadFile = File(...),
    _=Depends(require_app_key),
):
    if resume.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Resume must be a PDF")

    pdf_bytes = await resume.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    resume_text = extract_text_from_pdf(pdf_bytes)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    result = _analyze_text(resume_text, jd_text)
    result["resume_text_length"] = len(resume_text)

    if debug and settings.analyze_debug_enabled:
        result["debug"] = {
            "resume_text_preview": resume_text[:500],
            "resume_detected_skills": _extract_matches(resume_text, TECH_PATTERNS),
            "jd_detected_skills": _extract_matches(jd_text, TECH_PATTERNS),
            "jd_keywords_sample": _extract_keywords(jd_text)[:30],
        }

    return result



def _collapse_spaced_letters(text: str) -> str:
    s = text or ""
    for _ in range(3):
        s2 = _SPACED_LETTERS.sub(lambda m: m.group(0).replace(" ", ""), s)
        if s2 == s:
            break
        s = s2
    return s



def _normalize_text(text: str) -> str:
    return _collapse_spaced_letters(text or "").lower()



def _tokenize(text: str) -> list[str]:
    tokens = []
    seen: set[str] = set()
    for token in _WORD_RE.findall(_normalize_text(text)):
        if len(token) < 3 or token in _STOP:
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens



def _extract_matches(text: str, patterns: dict[str, tuple[str, ...]]) -> list[str]:
    norm = _normalize_text(text)
    found: list[str] = []
    for label, exprs in patterns.items():
        if any(re.search(expr, norm) for expr in exprs):
            found.append(label)
    return found



def _extract_lines(text: str) -> list[str]:
    lines = []
    for raw in (text or "").replace("\r", "").split("\n"):
        line = raw.strip(" \t•-*;")
        if line:
            lines.append(line)
    return lines



def _extract_keywords(jd_text: str) -> list[str]:
    tokens = _tokenize(jd_text)
    counts = Counter(_WORD_RE.findall(_normalize_text(jd_text)))
    frequent = [tok for tok, count in counts.items() if tok not in _STOP and len(tok) >= 4 and count >= 2]
    combined = _extract_matches(jd_text, TECH_PATTERNS) + _extract_matches(jd_text, SOFT_PATTERNS) + frequent + tokens[:20]
    seen: set[str] = set()
    out: list[str] = []
    for item in combined:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out[:40]



def _extract_priority_lines(jd_text: str) -> tuple[list[str], list[str]]:
    must_lines: list[str] = []
    nice_lines: list[str] = []
    for line in _extract_lines(jd_text):
        norm = _normalize_text(line)
        if any(key in norm for key in ("must", "required", "requirement", "need to", "needs", "strong ", "experience with", "hands-on")):
            must_lines.append(line)
        elif any(key in norm for key in ("preferred", "nice to have", "plus", "bonus", "good to have")):
            nice_lines.append(line)
    return must_lines[:15], nice_lines[:15]



def _score_overlap(items: Iterable[str], resume_text: str) -> tuple[list[str], list[str]]:
    norm_resume = _normalize_text(resume_text)
    matched: list[str] = []
    missing: list[str] = []
    for item in items:
        norm_item = item.lower()
        if norm_item and norm_item in norm_resume:
            matched.append(item)
        else:
            missing.append(item)
    return matched, missing



def _transferable_matches(missing_items: Iterable[str], resume_text: str) -> list[dict[str, str]]:
    norm_resume = _normalize_text(resume_text)
    results: list[dict[str, str]] = []
    for item in missing_items:
        hints = TRANSFERABLE_HINTS.get(item.lower(), ())
        hint = next((candidate for candidate in hints if candidate in norm_resume), None)
        if hint:
            results.append({"requirement": item, "evidence": hint})
    return results



def _extract_year_requirements(jd_text: str) -> list[str]:
    found = re.findall(r"(\d+\+?\s+years?[^\n,.;]*)", jd_text, flags=re.IGNORECASE)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in found:
        cleaned = " ".join(item.split())
        if cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            deduped.append(cleaned)
    return deduped[:5]



def _recommendations(
    must_have_missing: list[str],
    nice_to_have_missing: list[str],
    missing_soft_skills: list[str],
    resume_text: str,
) -> list[str]:
    recs: list[str] = []
    if must_have_missing:
        recs.append(
            "Add direct evidence for the most important missing requirements, or de-emphasize less relevant content."
        )
    if nice_to_have_missing:
        recs.append(
            "If you have related exposure, surface transferable experience for the preferred skills instead of leaving them invisible."
        )
    if missing_soft_skills:
        recs.append(
            "Use bullet points that explicitly show communication, ownership, collaboration, or leadership rather than listing soft skills alone."
        )
    if not re.search(r"\b\d+(?:%|x|\+)?\b", resume_text):
        recs.append(
            "Quantify achievements with metrics, percentages, time saved, scale, or impact wherever possible."
        )
    if not re.search(r"\b(led|built|implemented|optimized|created|analyzed|designed|improved)\b", _normalize_text(resume_text)):
        recs.append(
            "Rewrite experience bullets using action-verb format: Action + What + How/Why + Result."
        )
    return recs[:5]



def _analyze_text(resume_text: str, jd_text: str) -> dict:
    jd_text = jd_text or ""
    resume_text = resume_text or ""

    jd_tokens = set(_tokenize(jd_text))
    resume_tokens = set(_tokenize(resume_text))

    if not jd_tokens:
        return {
            "score": 0,
            "matched_count": 0,
            "missing_count": 0,
            "matched_top": [],
            "missing_top": [],
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "soft_skills": [],
            "keywords": [],
            "matched_requirements": [],
            "transferable_matches": [],
            "gaps": [],
            "score_breakdown": {"must_have": 0, "nice_to_have": 0, "soft_skills": 0, "keywords": 0},
            "recommendations": [],
        }

    jd_tech = _extract_matches(jd_text, TECH_PATTERNS)
    jd_soft = _extract_matches(jd_text, SOFT_PATTERNS)
    resume_tech = _extract_matches(resume_text, TECH_PATTERNS)
    resume_soft = _extract_matches(resume_text, SOFT_PATTERNS)

    must_lines, nice_lines = _extract_priority_lines(jd_text)
    nice_norm = _normalize_text("\n".join(nice_lines))

    must_have_skills: list[str] = []
    nice_to_have_skills: list[str] = []
    for skill in jd_tech:
        if skill in nice_norm:
            nice_to_have_skills.append(skill)
        else:
            must_have_skills.append(skill)

    must_have_skills = list(dict.fromkeys(must_have_skills))
    nice_to_have_skills = list(dict.fromkeys([s for s in nice_to_have_skills if s not in must_have_skills]))

    matched_must, missing_must = _score_overlap(must_have_skills, resume_text)
    matched_nice, missing_nice = _score_overlap(nice_to_have_skills, resume_text)
    matched_soft, missing_soft = _score_overlap(jd_soft, resume_text)

    matched_keywords = sorted(jd_tokens.intersection(resume_tokens))
    missing_keywords = sorted(jd_tokens.difference(resume_tokens))

    transferable = _transferable_matches(missing_must + missing_soft, resume_text)

    def pct(found: int, total: int) -> int:
        return int((found / total) * 100) if total else 0

    keyword_score = pct(len(matched_keywords), len(jd_tokens))
    must_score = pct(len(matched_must), len(must_have_skills))
    nice_score = pct(len(matched_nice), len(nice_to_have_skills))
    soft_score = pct(len(matched_soft), len(jd_soft))

    score = int(round((must_score * 0.5) + (nice_score * 0.15) + (keyword_score * 0.25) + (soft_score * 0.10)))

    gaps = list(dict.fromkeys(missing_must + missing_soft + missing_nice))
    keywords = _extract_keywords(jd_text)

    matched_requirements = [
        {"requirement": item, "type": "must_have"} for item in matched_must
    ] + [
        {"requirement": item, "type": "soft_skill"} for item in matched_soft
    ] + [
        {"requirement": item, "type": "nice_to_have"} for item in matched_nice
    ]

    recommendations = _recommendations(missing_must, missing_nice, missing_soft, resume_text)

    return {
        "score": max(0, min(score, 100)),
        "matched_count": len(matched_keywords),
        "missing_count": len(missing_keywords),
        "matched_top": matched_keywords[:30],
        "missing_top": missing_keywords[:30],
        "must_have_skills": must_have_skills,
        "nice_to_have_skills": nice_to_have_skills,
        "soft_skills": jd_soft,
        "keywords": keywords,
        "must_have_lines": must_lines,
        "nice_to_have_lines": nice_lines,
        "matched_requirements": matched_requirements,
        "transferable_matches": transferable,
        "gaps": gaps,
        "score_breakdown": {
            "must_have": must_score,
            "nice_to_have": nice_score,
            "soft_skills": soft_score,
            "keywords": keyword_score,
        },
        "resume_detected_skills": resume_tech,
        "resume_detected_soft_skills": resume_soft,
        "year_requirements": _extract_year_requirements(jd_text),
        "recommendations": recommendations,
    }
