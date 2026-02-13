from __future__ import annotations

import os

from rq import Queue, SimpleWorker, Worker

from app.core.config import get_settings
from app.jobs.queue import get_redis_connection


def main() -> None:
    settings = get_settings()
    conn = get_redis_connection(settings)
    queue = Queue(settings.rq_queue_name, connection=conn)
    worker_cls = SimpleWorker if os.name == "nt" else Worker
    worker = worker_cls([queue], connection=conn)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
