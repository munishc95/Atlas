from __future__ import annotations

from redis import Redis
from rq import Queue

from app.core.config import Settings


def get_redis_connection(settings: Settings) -> Redis:
    return Redis.from_url(settings.redis_url)


def get_queue(settings: Settings) -> Queue:
    return Queue(settings.rq_queue_name, connection=get_redis_connection(settings))
