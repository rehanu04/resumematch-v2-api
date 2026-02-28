from fastapi import APIRouter, Depends, HTTPException
import httpx
from app.core.security import require_app_key

router = APIRouter(prefix="/v1/net", tags=["net"])

@router.get("/check")
async def check(_=Depends(require_app_key)):
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            r = await client.get("https://example.com", headers={"User-Agent":"Mozilla/5.0"})
            return {"ok": True, "status_code": r.status_code}
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Request error: {e.__class__.__name__}")
