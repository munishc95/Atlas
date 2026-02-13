from __future__ import annotations

import os

from redis import Redis
from rq import Queue, SimpleWorker, Worker

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    redis_url = os.getenv("ATLAS_REDIS_URL", settings.redis_url)
    conn = Redis.from_url(redis_url)
    queue = Queue(settings.rq_queue_name, connection=conn)
    worker_cls = SimpleWorker if os.name == "nt" else Worker
    worker = worker_cls([queue], connection=conn)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
