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
    proxy_url = getattr(settings, "jd_proxy_url", None)
    proxy_key = getattr(settings, "jd_proxy_key", None)

    if not proxy_url or not proxy_key:
        raise HTTPException(status_code=500, detail="JD proxy not configured")

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                str(proxy_url),
                headers={"X-Proxy-Key": proxy_key, "Content-Type": "application/json"},
                json={"url": str(payload.url)},
            )
            if r.status_code >= 400:
                raise HTTPException(status_code=400, detail=f"Proxy fetch failed: {r.status_code}")
            return r.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Proxy request error: {e.__class__.__name__}")
