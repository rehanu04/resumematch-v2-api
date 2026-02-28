from fastapi import APIRouter, Depends
from app.core.security import require_app_key

router = APIRouter(prefix="/v1", tags=["ping"])

@router.get("/ping")
def ping(_=Depends(require_app_key)):
    return {"pong": True}
