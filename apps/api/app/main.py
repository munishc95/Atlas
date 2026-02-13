from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from app.api.routes import router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.db.bootstrap import seed_defaults
from app.db.session import engine, init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    init_db()
    with Session(engine) as session:
        seed_defaults(session, settings)
    yield


app = FastAPI(
    title="Atlas API",
    version="0.1.0",
    description="Adaptive Swing Trading Research and Paper Trading API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(router)
