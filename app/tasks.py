"""
Celery task queue for asynchronous ingestion.

The FastAPI /ingest endpoint enqueues ingest_task onto Redis; a separate Celery
worker picks it up and runs incremental_ingest (only re-embedding changed files).
This decouples the (potentially slow) ingestion from the request/response cycle
and gives retries on failure.

Broker + result backend: Redis (REDIS_URL, default redis://localhost:6379/0).

Run the worker (from project root):
    celery -A app.tasks worker --loglevel=info --pool=solo

(--pool=solo avoids a known macOS fork crash; fine for single-worker dev.)
"""

import os
from celery import Celery

from app.ingest import incremental_ingest

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("rag_ingest", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_track_started=True,
    result_expires=3600,
)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def ingest_task(self, kind="openai"):
    """Run incremental ingestion as a background job, retrying on transient
    failures (e.g. a hiccup reaching the embedding API)."""
    try:
        return incremental_ingest(kind=kind)
    except Exception as exc:
        # retry with backoff; after max_retries the task is marked FAILURE
        raise self.retry(exc=exc)