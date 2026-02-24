from __future__ import annotations

from sqlmodel import Session

from app.core.config import Settings
from app.providers.base import BaseProvider
from app.providers.mock_provider import MockProvider
from app.providers.nse_eod_provider import NseEodProvider
from app.providers.upstox_provider import UpstoxProvider
from app.services.data_store import DataStore


def build_provider(
    *,
    kind: str,
    session: Session,
    settings: Settings,
    store: DataStore,
) -> BaseProvider:
    token = str(kind or "UPSTOX").strip().upper()
    if token == "UPSTOX":
        return UpstoxProvider(session=session, settings=settings, store=store)
    if token == "NSE_EOD":
        return NseEodProvider(session=session, settings=settings, store=store)
    if token == "MOCK":
        return MockProvider(
            seed=int(settings.fast_mode_seed),
            session=session,
            store=store,
        )
    raise ValueError(f"Unsupported provider kind: {kind}")
