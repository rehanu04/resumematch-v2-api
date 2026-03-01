from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routers.health import router as health_router
from app.routers.ping import router as ping_router
from app.routers.jd import router as jd_router
from app.routers.net import router as net_router
from app.routers.analyze import router as analyze_router

from app.core.ratelimit import check_and_record

app = FastAPI(title="ResumeMatch v2 API", version="0.1.0")

@app.middleware("http")
async def ratelimit_middleware(request: Request, call_next):
    key = request.headers.get("X-App-Key", "no-key")
    if not check_and_record(key):
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    return await call_next(request)

app.include_router(health_router)
app.include_router(ping_router)
app.include_router(jd_router)
app.include_router(net_router)
app.include_router(analyze_router)
