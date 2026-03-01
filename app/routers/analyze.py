from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.core.security import require_app_key

router = APIRouter(prefix="/v1", tags=["analyze"])

class AnalyzeRequest(BaseModel):
    resume_text: str
    jd_text: str

@router.post("/analyze")
def analyze(payload: AnalyzeRequest, _=Depends(require_app_key)):
    resume = payload.resume_text.lower()
    jd = payload.jd_text.lower()

    jd_tokens = {t for t in jd.split() if len(t) >= 3}
    resume_tokens = {t for t in resume.split() if len(t) >= 3}

    if not jd_tokens:
        return {"score": 0, "missing": [], "matched": []}

    matched = sorted(list(jd_tokens.intersection(resume_tokens)))
    missing = sorted(list(jd_tokens.difference(resume_tokens)))

    score = int((len(matched) / len(jd_tokens)) * 100)

    return {
        "score": score,
        "matched_count": len(matched),
        "missing_count": len(missing),
        "matched_top": matched[:30],
        "missing_top": missing[:30],
    }
