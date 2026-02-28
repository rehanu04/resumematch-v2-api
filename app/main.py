from fastapi import FastAPI
from app.routers.health import router as health_router
from app.routers.ping import router as ping_router
from app.routers.jd import router as jd_router
from app.routers.net import router as net_router

app = FastAPI(title="ResumeMatch v2 API", version="0.1.0")

app.include_router(health_router)
app.include_router(ping_router)
app.include_router(jd_router)
app.include_router(net_router)
