from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from app.core.security import require_app_key
import re

from app.services.pdf_extract import extract_text_from_pdf

router = APIRouter(prefix="/v1", tags=["analyze"])

_word_re = re.compile(r"[a-z0-9]+")
_STOP = {
    "and","or","the","a","an","to","of","in","on","for","with","as","is","are","be","by","at",
    "we","you","your","our","they","their","need","needs","required","requirements","must",
    "experience","years","year","role","job","work","working","ability","skills","skill"
}

def tokenize(s: str) -> set[str]:
    s = s.lower()
    toks = {w for w in _word_re.findall(s) if len(w) >= 3}
    return {w for w in toks if w not in _STOP}

class AnalyzeRequest(BaseModel):
    resume_text: str
    jd_text: str

@router.post("/analyze")
def analyze(payload: AnalyzeRequest, _=Depends(require_app_key)):
    return _analyze_text(payload.resume_text, payload.jd_text)

@router.post("/analyze/pdf")
async def analyze_pdf(
    jd_text: str = Form(...),
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
    return result

def _analyze_text(resume_text: str, jd_text: str):
    jd_tokens = tokenize(jd_text)
    resume_tokens = tokenize(resume_text)

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
