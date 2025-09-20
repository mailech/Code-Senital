from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.logging_setup import setup_logging, logger
from app.config import settings
from app.routes.health import router as health_router
from app.routes.webhooks import router as webhooks_router
from app.routes.dashboard import router as dashboard_router
from app.db import init_db


setup_logging()
app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(webhooks_router)
app.include_router(dashboard_router)


@app.on_event("startup")
async def on_startup():
    init_db()
    logger.info("app_startup", env=settings.environment)


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("app_shutdown")


# Optional root endpoint
@app.get("/")
async def root():
    return {"message": f"{settings.app_name} running"}
