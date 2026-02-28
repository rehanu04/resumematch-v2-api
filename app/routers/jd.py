from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl
import httpx

from app.core.security import require_app_key

router = APIRouter(prefix="/v1/jd", tags=["jd"])

class JDExtractRequest(BaseModel):
    url: HttpUrl

@router.post("/extract")
async def extract_jd(payload: JDExtractRequest, _=Depends(require_app_key)):
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            r = await client.get(str(payload.url), headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code >= 400:
                raise HTTPException(status_code=400, detail=f"Fetch failed: {r.status_code}")
            # v0: return first N chars of raw HTML (we'll improve parsing next)
            text = r.text
            return {"url": str(payload.url), "length": len(text), "preview": text[:2000]}
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Request error: {e.__class__.__name__}")
