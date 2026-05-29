"""Celery application factory."""

from celery import Celery

from medscan.config import get_settings

settings = get_settings()


def create_celery_app() -> Celery:
    """Create and configure the Celery application."""
    app = Celery(
        "medscan",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=["medscan.tasks.document_tasks"],
    )

    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,  # One task at a time (heavy GPU workloads)
        task_routes={
            "medscan.tasks.document_tasks.*": {"queue": "pipeline"},
        },
        task_soft_time_limit=600,   # 10 min soft limit
        task_time_limit=900,         # 15 min hard limit
        result_expires=86400,        # Results expire after 24 hours
    )

    return app


celery_app = create_celery_app()


def run_worker() -> None:
    """Entry point for the Celery worker (used by pyproject.toml script)."""
    celery_app.start(
        argv=["worker", "--loglevel=info", "-Q", "pipeline", "--concurrency=2"]
    )
