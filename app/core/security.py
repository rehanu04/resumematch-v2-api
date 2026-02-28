from fastapi import Header, HTTPException
from app.core.config import settings

def require_app_key(x_app_key: str = Header(default="")):
    if not x_app_key or x_app_key != settings.app_backend_key:
        raise HTTPException(status_code=401, detail="Invalid X-App-Key")
