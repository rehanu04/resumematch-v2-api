from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from app.core.security import require_app_key`nfrom app.core.config import settings
import re

from app.services.pdf_extract import extract_text_from_pdf

router = APIRouter(prefix="/v1", tags=["analyze"])

_word_re = re.compile(r"[a-z0-9]+")
_STOP = {
    "and","or","the","a","an","to","of","in","on","for","with","as","is","are","be","by","at",
    "we","you","your","our","they","their","need","needs","required","requirements","must",
    "experience","years","year","role","job","work","working","ability","skills","skill",
    "looking","seek","seeking","hiring","candidate","position","responsibilities","responsibility",
    "strong","hands","knowledge"
}

# Collapse spaced-out letters: "P y t h o n" -> "Python"
_spaced_letters = re.compile(r"(?:\b[A-Za-z]\b(?:\s+|$)){3,}")

def _collapse_spaced_letters(s: str) -> str:
    def repl(m):
        return m.group(0).replace(" ", "")
    # Apply a few times because PDFs can be weird
    for _ in range(3):
        s2 = _spaced_letters.sub(lambda m: m.group(0).replace(" ", ""), s)
        if s2 == s:
            break
        s = s2
    return s

def tokenize_list(s: str) -> list[str]:
    s = _collapse_spaced_letters(s)
    s = s.lower()
    toks = [w for w in _word_re.findall(s) if len(w) >= 3 and w not in _STOP]
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

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
        resume_tokens = tokenize_list(resume_text)
        jd_tokens = tokenize_list(jd_text)
        result["debug"] = {
            "resume_text_preview": resume_text[:500],
            "resume_tokens_sample": resume_tokens[:30],
            "jd_tokens_sample": jd_tokens[:30],
        }

    return result

def _analyze_text(resume_text: str, jd_text: str):
    jd_tokens = set(tokenize_list(jd_text))
    resume_tokens = set(tokenize_list(resume_text))

    if not jd_tokens:
        return {"score": 0, "matched_count": 0, "missing_count": 0, "matched_top": [], "missing_top": []}

    matched = sorted(jd_tokens.intersection(resume_tokens))
    missing = sorted(jd_tokens.difference(resume_tokens))
    score = int((len(matched) / len(jd_tokens)) * 100)

    return {
        "score": score,
        "matched_count": len(matched),
        "missing_count": len(missing),
        "matched_top": matched[:30],
        "missing_top": missing[:30],
    }
