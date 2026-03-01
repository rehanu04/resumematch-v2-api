from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.core.security import require_app_key
import re

router = APIRouter(prefix="/v1", tags=["analyze"])

class AnalyzeRequest(BaseModel):
    resume_text: str
    jd_text: str

_word_re = re.compile(r"[a-z0-9]+")  # simple, robust for v0

def tokenize(s: str) -> set[str]:
    s = s.lower()
    return {w for w in _word_re.findall(s) if len(w) >= 3}

@router.post("/analyze")
def analyze(payload: AnalyzeRequest, _=Depends(require_app_key)):
    jd_tokens = tokenize(payload.jd_text)
    resume_tokens = tokenize(payload.resume_text)

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
