from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl
import httpx

from app.core.security import require_app_key
from app.core.config import settings

router = APIRouter(prefix="/v1/jd", tags=["jd"])

class JDExtractRequest(BaseModel):
    url: HttpUrl

@router.post("/extract")
async def extract_jd(payload: JDExtractRequest, _=Depends(require_app_key)):
    proxy_url = settings.jd_proxy_url
    proxy_key = settings.jd_proxy_key

    if not proxy_url or not proxy_key:
        raise HTTPException(status_code=500, detail="JD proxy not configured")

    try:
        async with httpx.AsyncClient(timeout=25) as client:
            r = await client.post(
                proxy_url,
                headers={"X-Proxy-Key": proxy_key},
                json={"url": str(payload.url)},
            )

        if r.status_code == 401:
            raise HTTPException(status_code=500, detail="JD proxy auth failed")
        if r.status_code >= 400:
            raise HTTPException(status_code=400, detail=f"JD proxy fetch failed: {r.status_code}")

        data = r.json()

        # Prefer already-cleaned text from worker
        text = (data.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Failed to extract JD text")

        # Return only what backend needs (keep payload small)
        return {
            "url": data.get("url", str(payload.url)),
            "status": data.get("status"),
            "content_type": data.get("content_type"),
            "text": text,
        }

    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Proxy request error: {e.__class__.__name__}")
