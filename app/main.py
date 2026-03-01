from fastapi import FastAPI, Request
from app.routers.health import router as health_router
from app.routers.ping import router as ping_router
from app.routers.jd import router as jd_router
from app.routers.net import router as net_router
from app.routers.rl import router as rl_router
from app.core.ratelimit import rate_limit

app = FastAPI(title="ResumeMatch v2 API", version="0.1.0")

@app.middleware("http")
async def ratelimit_middleware(request: Request, call_next):
    await rate_limit(request)
    return await call_next(request)

app.include_router(health_router)
app.include_router(ping_router)
app.include_router(jd_router)
app.include_router(net_router)
app.include_router(rl_router)
