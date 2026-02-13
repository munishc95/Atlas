from __future__ import annotations

import os
import threading

from rq import Queue, SimpleWorker, Worker

from app.core.config import get_settings
from app.jobs.queue import get_redis_connection
from app.services.operate_scheduler import scheduler_loop


def main() -> None:
    settings = get_settings()
    conn = get_redis_connection(settings)
    queue = Queue(settings.rq_queue_name, connection=conn)
    worker_cls = SimpleWorker if os.name == "nt" else Worker
    stop_event = threading.Event()
    scheduler_thread = threading.Thread(
        target=scheduler_loop,
        kwargs={"stop_event": stop_event, "queue": queue, "poll_seconds": 20},
        daemon=True,
        name="atlas-operate-scheduler",
    )
    scheduler_thread.start()
    worker = worker_cls([queue], connection=conn)
    try:
        worker.work(with_scheduler=True)
    finally:
        stop_event.set()
        scheduler_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
